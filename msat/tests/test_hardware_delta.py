from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msat.hardware.delta import (
    compare_stores,
    confound_verdict,
    probe_plan,
    select_probe_keys,
)
from msat.metrics.normalisation import Normaliser
from msat.store import Condition, ResponseStore, StoredResponse, cache_key

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
COND = Condition(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")


@pytest.fixture(scope="module")
def norm() -> Normaliser:
    spec = yaml.safe_load((CONFIGS / "prompts" / "normalisation.yaml").read_text(encoding="utf-8"))
    return Normaliser.from_config(spec)


def _store(tmp_path: Path, name: str, texts: dict[str, str]) -> ResponseStore:
    store = ResponseStore(tmp_path / name)
    for i, (key, text) in enumerate(texts.items()):
        req = {
            "model": "google/gemma-4-E4B-it",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "temperature": 0.0,
        }
        store.put(
            StoredResponse(
                key=key,
                condition=COND.as_dict(),
                request=req,
                text=text,
                finish_reason="stop",
                model_version="v1",
                prompt_tokens=1000,
                completion_tokens=4,
                latency_ms=10.0,
            )
        )
    return store


def test_a_store_compared_to_itself_reports_exactly_zero(tmp_path, norm) -> None:
    texts = {f"k{i}": f"answer {i}" for i in range(20)}
    store = _store(tmp_path, "a", texts)
    r = compare_stores(
        store, store, sorted(texts), gold_by_key={}, norm=norm, label_a="x", label_b="x"
    )
    assert r.n_compared == 20
    assert r.n_text_identical == 20
    assert r.n_pairs_moved_total == 0
    assert r.em_delta == 0.0
    assert r.f1_delta == 0.0


def test_identical_stores_are_never_confounded(tmp_path, norm) -> None:
    texts = {f"k{i}": "paris" for i in range(20)}
    a, b = _store(tmp_path, "a", texts), _store(tmp_path, "b", texts)
    r = compare_stores(a, b, sorted(texts), gold_by_key=dict.fromkeys(texts, "paris"), norm=norm)
    v = confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=0.15)
    assert v["verdict"].startswith("NEGLIGIBLE")


def test_whitespace_drift_is_invisible_after_normalisation(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris", "k2": "london"})
    b = _store(tmp_path, "b", {"k1": "  paris  ", "k2": "London"})
    r = compare_stores(a, b, ["k1", "k2"], gold_by_key={}, norm=norm)
    assert r.n_text_identical == 0
    assert r.n_normalised_identical == 2


def test_a_real_disagreement_shows_at_the_metric_layer(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris"})
    b = _store(tmp_path, "b", {"k1": "lyon"})
    r = compare_stores(a, b, ["k1"], gold_by_key={"k1": "paris"}, norm=norm)
    assert r.n_text_identical == 0
    assert r.n_normalised_identical == 0
    assert r.n_metric_equal == 0
    assert r.em_a == 1.0
    assert r.em_b == 0.0
    assert r.em_delta == -1.0


def test_a_key_missing_from_box_b_is_counted_not_scored(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris", "k2": "london"})
    b = _store(tmp_path, "b", {"k1": "paris"})
    r = compare_stores(a, b, ["k1", "k2"], gold_by_key={"k1": "paris", "k2": "london"}, norm=norm)
    assert r.n_missing_in_b == 1
    assert r.n_compared == 1
    assert r.em_delta == 0.0


def test_a_key_with_no_gold_is_text_only(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "x"})
    b = _store(tmp_path, "b", {"k1": "y"})
    r = compare_stores(a, b, ["k1"], gold_by_key={}, norm=norm)
    assert r.n_text_identical == 0
    assert r.n_compared == 1
    assert r.n_pairs_moved_total == 0
    assert r.em_delta == 0.0


def test_cancelling_pairs_are_reported_even_though_the_mean_is_zero(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris", "k2": "lyon"})
    b = _store(tmp_path, "b", {"k1": "lyon", "k2": "paris"})
    r = compare_stores(a, b, ["k1", "k2"], gold_by_key={"k1": "paris", "k2": "paris"}, norm=norm)
    assert r.em_delta == pytest.approx(0.0)
    assert r.n_pairs_moved_total == 2


def test_too_many_moving_pairs_makes_the_aggregate_untrustworthy(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {f"k{i}": "paris" for i in range(20)})
    b = _store(tmp_path, "b", {f"k{i}": ("lyon" if i % 2 == 0 else "paris") for i in range(20)})
    r = compare_stores(
        a,
        b,
        sorted(a and [x.key for x in a.iter_all()]),
        gold_by_key={x.key: "paris" for x in a.iter_all()},
        norm=norm,
    )
    v = confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=0.15)
    assert v["aggregate_delta_is_trustworthy"] is False
    assert v["verdict"].startswith("TOO MANY PAIRS MOVED")


def test_drift_below_the_mde_is_negligible(tmp_path, norm) -> None:
    texts = {f"k{i}": ("paris" if i < 99 else "lyon") for i in range(200)}
    a = _store(tmp_path, "a", texts)
    b = _store(tmp_path, "b", dict(texts))
    r = compare_stores(a, b, sorted(texts), gold_by_key=dict.fromkeys(texts, "paris"), norm=norm)
    v = confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=0.15)
    assert v["verdict"].startswith("NEGLIGIBLE")
    assert v["drift_pp"] < v["mde_pp_at_p50"]


def test_drift_above_the_mde_is_confounded(tmp_path, norm) -> None:
    texts = {f"k{i:03d}": "paris" for i in range(100)}
    other = dict(texts)
    other["k000"] = "lyon"
    a, b = _store(tmp_path, "a", texts), _store(tmp_path, "b", other)
    keys = sorted(texts)
    r = compare_stores(a, b, keys, gold_by_key=dict.fromkeys(keys, "paris"), norm=norm)
    v = confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=0.15)
    assert v["aggregate_delta_is_trustworthy"] is True
    assert v["verdict"].startswith("CONFOUNDED")
    assert v["drift_pp"] == pytest.approx(1.0)
    assert "corrected for or reported as separate" in v["verdict"]


def test_an_unknown_mde_refuses_to_render_a_verdict(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris"})
    b = _store(tmp_path, "b", {"k1": "lyon"})
    r = compare_stores(a, b, ["k1"], gold_by_key={"k1": "paris"}, norm=norm)
    v = confound_verdict(r, mde_log_odds=None, mde_pp_at_p50=None)
    assert "MDE unknown" in v["verdict"]


def test_the_mde_threshold_loosens_with_a_weaker_design(tmp_path, norm) -> None:
    a = _store(tmp_path, "a", {"k1": "paris"})
    b = _store(tmp_path, "b", {"k1": "lyon"})
    r = compare_stores(a, b, ["k1"], gold_by_key={"k1": "paris"}, norm=norm)
    assert r.em_delta == -1.0
    assert confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=150.0)["verdict"].startswith(
        "NEGLIGIBLE"
    )
    assert confound_verdict(r, mde_log_odds=0.006, mde_pp_at_p50=50.0)["verdict"].startswith(
        "CONFOUNDED"
    )


def test_probe_selection_is_deterministic(tmp_path) -> None:
    store = _store(tmp_path, "a", {f"k{i:03d}": "x" for i in range(500)})
    a = select_probe_keys(store, n=100, seed=7)
    b = select_probe_keys(store, n=100, seed=7)
    assert a == b
    assert len(a) == 100


def test_probe_selection_is_independent_of_directory_order(tmp_path) -> None:
    store = _store(tmp_path, "a", {f"k{i:03d}": "x" for i in range(50)})
    assert select_probe_keys(store, n=10, seed=1) == sorted(select_probe_keys(store, n=10, seed=1))


def test_a_probe_smaller_than_the_store_returns_everything(tmp_path) -> None:
    store = _store(tmp_path, "a", {"k1": "x", "k2": "y"})
    assert len(select_probe_keys(store, n=99, seed=1)) == 2


def test_the_probe_plan_reports_prompt_tokens(tmp_path) -> None:
    store = _store(tmp_path, "a", {f"k{i:03d}": "x" for i in range(40)})
    plan = probe_plan(store, n=10, seed=3)
    assert plan["n_keys"] == 10
    assert plan["prompt_tokens"] == 10_000
    assert "Deterministic selection is the point" in plan["note"]


def test_the_probe_prefers_a_named_arm_but_survives_its_absence(tmp_path) -> None:
    store = _store(tmp_path, "a", {f"k{i:03d}": "x" for i in range(40)})
    assert len(select_probe_keys(store, n=5, seed=1, prefer="zzz")) == 5


def test_two_different_machines_share_a_cache_key() -> None:
    req = {
        "model": "google/gemma-4-E4B-it",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    assert cache_key(COND, req) == cache_key(COND, dict(req))
