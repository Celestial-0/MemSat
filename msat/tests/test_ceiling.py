from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.ceiling import (
    CEILING_GRID,
    build_ceiling_artifact,
    build_ceiling_multiscale,
    build_corpus_fraction_artifact,
    by_type_table,
    ceiling_curve,
    corpus_fraction,
    degeneracy_fraction,
    dump_canonical,
    partial_credit_curve,
    saturation_k,
    survivors_at,
    verify_artifact,
)
from msat.datasets.schema import CorpusStats, Question
from tests.conftest import requires_m, requires_s


def _q(qid: str, n_sessions: int, n_gold: int, ability: str = "multi-session") -> Question:
    return Question(
        qid=qid,
        scale="S",
        question="q",
        question_date="",
        gold_answers=("a",),
        gold_session_ids=tuple(f"g{i}" for i in range(n_gold)),
        ability=ability,
        answerable=not qid.endswith("_abs"),
        corpus=CorpusStats(n_sessions=n_sessions, n_gold_sessions=n_gold),
    )


def test_partial_credit_honours_its_grid() -> None:
    gold = [1, 2, 3, 4]
    grid = (1, 2, 4, 8)
    partial = partial_credit_curve(gold, grid)
    assert set(partial) == {"1", "2", "4", "8"}


def test_partial_credit_values_are_correct_by_hand() -> None:
    gold = [1, 2, 3]
    assert partial_credit_curve(gold, (1,))["1"] == pytest.approx((1 + 0.5 + 1 / 3) / 3)
    assert partial_credit_curve(gold, (3,))["3"] == pytest.approx(1.0)


def test_ceiling_curve_is_monotone_non_decreasing() -> None:
    curve = ceiling_curve([1, 2, 3, 6], CEILING_GRID)
    values = [curve[str(k)] for k in sorted(CEILING_GRID)]
    assert values == sorted(values)


def test_saturation_k_is_the_min_crossing() -> None:
    gold = [1, 1, 2, 6]
    curve = ceiling_curve(gold, CEILING_GRID)
    assert saturation_k(curve) == 6


def test_saturation_k_returns_none_when_grid_is_too_short() -> None:
    assert saturation_k(ceiling_curve([9], (1, 2))) is None


def test_survivors_at_and_degeneracy_overlap_only_at_equality() -> None:
    sizes = [10, 20, 30, 40]
    for k in (1, 10, 25, 40):
        surv = survivors_at(sizes, (k,))[str(k)]
        degen = degeneracy_fraction(sizes, k)
        frac_equal = sum(1 for s in sizes if s == k) / len(sizes)
        assert surv + degen - frac_equal == pytest.approx(1.0)


def test_degeneracy_is_per_question_not_corpus_wide() -> None:
    sizes = [10, 90]
    assert degeneracy_fraction(sizes, 32) == pytest.approx(0.5)
    assert sum(sizes) / len(sizes) == 50.0


def test_k_equal_to_corpus_is_degenerate() -> None:
    assert degeneracy_fraction([40], 40) == pytest.approx(1.0)


def test_by_type_classification() -> None:
    rows = [
        ("single-session-user", 1),
        ("single-session-user", 1),
        ("multi-session", 3),
        ("multi-session", 4),
    ]
    table = by_type_table(rows, CEILING_GRID)
    assert table["single-session-user"]["classification"] == "retriever-independent"
    assert table["multi-session"]["classification"] == "retriever-limited"
    assert table["multi-session"]["probe"]["1"] == pytest.approx(0.0)


def test_by_type_mixed_when_some_but_not_all_fit_in_one_session() -> None:
    table = by_type_table([("temporal-reasoning", 1), ("temporal-reasoning", 3)], CEILING_GRID)
    assert table["temporal-reasoning"]["classification"] == "mixed"


def test_corpus_fraction_detects_full_context() -> None:
    result = corpus_fraction([8, 16], (8, 16))
    assert result["8"]["frac_full_context"] == pytest.approx(0.5)
    assert result["16"]["frac_full_context"] == pytest.approx(0.5)


def test_corpus_fraction_reports_k_greater_than_corpus() -> None:
    result = corpus_fraction([10], (64,))
    assert result["64"]["frac_k_gt_corpus"] == pytest.approx(1.0)
    assert result["64"]["frac_full_context"] == pytest.approx(0.0)
    assert result["64"]["quantiles"]["max"] > 1.0


def test_same_k_is_a_different_system_at_different_scales() -> None:
    s_sizes = [40, 48, 50]
    m_sizes = [460, 475, 490]
    assert corpus_fraction(s_sizes, (64,))["64"]["frac_k_gt_corpus"] == pytest.approx(1.0)
    assert corpus_fraction(m_sizes, (64,))["64"]["frac_full_context"] == pytest.approx(0.0)


def test_dump_canonical_is_byte_stable() -> None:
    artifact = build_ceiling_artifact([_q("a", 50, 1), _q("b", 50, 3)])
    first = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    second = (
        json.dumps(
            build_ceiling_artifact([_q("a", 50, 1), _q("b", 50, 3)]),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    assert first == second


def test_verify_detects_drift_and_points_at_the_byte(tmp_path: Path) -> None:
    artifact = build_ceiling_artifact([_q("a", 50, 1)])
    path = tmp_path / "ceiling.json"
    dump_canonical(artifact, path)

    ok, detail = verify_artifact(artifact, path)
    assert ok, detail

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["ceiling"]["1"] = 0.123456
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok, detail = verify_artifact(artifact, path)
    assert not ok
    assert "DRIFT" in detail
    assert "first difference at byte" in detail


def test_verify_reports_a_missing_artifact(tmp_path: Path) -> None:
    ok, detail = verify_artifact({"a": 1}, tmp_path / "nope.json")
    assert not ok
    assert "missing" in detail


def test_artifact_carries_key_semantics() -> None:
    artifact = build_ceiling_artifact([_q("a", 50, 1)])
    semantics = artifact["_key_semantics"]
    assert "survivors_at" in semantics
    assert "NOT" in semantics["survivors_at"]
    assert artifact["_provenance"]["api_calls"] == 0


def test_multiscale_reports_scale_invariance() -> None:
    artifact = build_ceiling_multiscale(
        {"S": [_q("a", 50, 1), _q("b", 50, 2)], "M": [_q("a", 475, 1), _q("b", 475, 2)]}
    )
    assert artifact["scale_invariance"]["identical_across_scales"] is True
    assert artifact["scales"]["S"]["analytic_saturation_k"] == 2


def _committed(project_root: Path, name: str) -> Path:
    return project_root / "results" / name


@requires_s
@requires_m
def test_committed_ceiling_reproduces_byte_for_byte(
    project_root: Path, path_s: Path, path_m: Path
) -> None:
    from msat.datasets import iter_metadata, raw_path

    scales = {s: list(iter_metadata(raw_path(s))) for s in ("S", "M")}
    artifact = build_ceiling_multiscale(scales)
    path = _committed(project_root, "ceiling.json")
    if not path.is_file():
        pytest.fail(f"{path} is not committed — run `make ceiling`")
    ok, detail = verify_artifact(artifact, path)
    assert ok, detail


@requires_s
@requires_m
def test_committed_corpus_fraction_reproduces_byte_for_byte(project_root: Path) -> None:
    from msat.datasets import iter_metadata, raw_path

    scales = {s: list(iter_metadata(raw_path(s))) for s in ("S", "M")}
    artifact = build_corpus_fraction_artifact(scales)
    path = _committed(project_root, "corpus_fraction.json")
    if not path.is_file():
        pytest.fail(f"{path} is not committed — run `make ceiling`")
    ok, detail = verify_artifact(artifact, path)
    assert ok, detail


def test_ceiling_grid_is_a_superset_of_any_design_grid() -> None:
    design = Path(__file__).resolve().parent.parent / "configs" / "design.yaml"
    if not design.is_file():
        pytest.skip("configs/design.yaml lands in Phase 2")
    import yaml

    k_grid = tuple(yaml.safe_load(design.read_text(encoding="utf-8"))["K_grid"])
    missing = [k for k in k_grid if k not in CEILING_GRID]
    assert not missing, f"design K_grid points absent from CEILING_GRID: {missing}"
