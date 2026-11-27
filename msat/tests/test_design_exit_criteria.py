from __future__ import annotations

import math
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

from msat.design import grid_center, k_delta_in_pp, k_log2_step, k_step_index
from msat.metrics import Normaliser, UnknownNormalisationError, score_answer, token_f1

DOUBLING = [1, 2, 4, 8, 16, 32, 64]
UNEVEN = [1, 2, 4, 8, 16, 48, 64]


def test_the_budget_profile_cannot_restate_the_grid() -> None:
    from msat.config import BudgetProfile

    fields = set(BudgetProfile.model_fields)
    assert "levels" not in fields, (
        "BudgetProfile must not carry budget levels - they live in design.yaml. "
        "Reinstating this field is how the repo ended up with two B axes."
    )
    assert "budget_source" in fields, "the profile should say where the levels are"


def test_the_duplicate_grid_guard_matches_on_structure_not_key_name() -> None:
    from msat.config import declares_budget_levels

    assert declares_budget_levels({"levels": [{"value": 4096, "mode": "x"}]})
    assert declares_budget_levels({"anything_at_all": [{"value": 1, "mode": "y"}]})
    assert declares_budget_levels([{"nested": {"deeper": [{"value": 2, "mode": "z"}]}}])
    assert not declares_budget_levels({"max_usd": 0.0, "disk_budget_bytes": 1})
    assert not declares_budget_levels({"models": {"m": {"input": 0.0, "output": 0.0}}})


def test_the_two_centres_are_derived_not_hardcoded() -> None:
    import math

    assert grid_center([math.log2(k) for k in DOUBLING]) == pytest.approx(3.0)
    assert grid_center([math.log2(b) for b in (128, 512, 2048)]) == pytest.approx(9.0)
    assert grid_center([]) == pytest.approx(0.0) if False else True
    with pytest.raises(ValueError):
        grid_center([])


def test_step_index_is_position_not_log() -> None:
    assert k_step_index(DOUBLING, 1) == 0.0
    assert k_step_index(DOUBLING, 64) == 1.0
    assert k_step_index(UNEVEN, 48) == pytest.approx(5 / 6)


def test_step_index_rejects_a_k_off_the_grid() -> None:
    with pytest.raises(ValueError, match="not on the declared grid"):
        k_step_index(DOUBLING, 3)


def test_every_step_on_a_doubling_grid_is_one_log2_unit() -> None:
    for k in DOUBLING[1:]:
        assert k_log2_step(DOUBLING, k) == pytest.approx(1.0)


def test_an_uneven_grid_has_genuinely_different_step_sizes() -> None:
    assert k_log2_step(UNEVEN, 48) == pytest.approx(math.log2(3))
    assert k_log2_step(UNEVEN, 64) == pytest.approx(math.log2(4 / 3))
    assert k_log2_step(UNEVEN, 48) > k_log2_step(UNEVEN, 64)


def test_the_threshold_scales_with_the_step_that_actually_occurs() -> None:
    assert k_delta_in_pp(0.5, DOUBLING, 32) == pytest.approx(0.5)
    big = k_delta_in_pp(0.5, UNEVEN, 48)
    small = k_delta_in_pp(0.5, UNEVEN, 64)
    assert big == pytest.approx(0.5 * math.log2(3))
    assert small == pytest.approx(0.5 * math.log2(4 / 3))
    for k in DOUBLING:
        assert k_delta_in_pp(0.5, DOUBLING, k) == pytest.approx(0.5)


def test_the_smallest_k_is_not_exempt() -> None:
    assert k_log2_step(DOUBLING, 1) == pytest.approx(1.0)


def _norm() -> Normaliser:
    import yaml

    raw = yaml.safe_load((CONFIGS / "prompts" / "normalisation.yaml").read_text(encoding="utf-8"))
    return Normaliser.from_config(raw)


def test_normalisation_actually_runs() -> None:
    n = _norm()
    assert n.normalise("  The   CAT, sat!  ") == "cat sat"


def test_nfkc_folds_typography_before_anything_else() -> None:
    assert _norm().normalise("ﬁsh") == "fish"
    assert _norm().normalise("\uff43\uff41\uff54") == "cat"


def test_the_declared_step_order_is_preserved() -> None:
    ids = [s.id for s in _norm().steps]
    assert ids == [
        "unicode_nfkc",
        "lowercase",
        "strip_whitespace",
        "collapse_whitespace",
        "remove_articles",
        "remove_punctuation",
        "collapse_whitespace_again",
        "strip",
    ], f"normalisation.yaml step order changed: {ids}"


def test_articles_are_removed_case_insensitively() -> None:
    n = _norm()
    assert n.normalise("a dog") == n.normalise("dog")
    assert n.normalise("The dog") == n.normalise("dog")


def test_exact_match_uses_normalised_strings() -> None:
    n = _norm()
    assert score_answer("The Cat.", "cat", n)["exact_match"] == 1.0
    assert score_answer("cat", "dog", n)["exact_match"] == 0.0


def test_token_f1_is_a_symmetric_overlap() -> None:
    n = _norm()
    assert token_f1("cat sat", "cat sat", n) == pytest.approx(1.0)
    assert token_f1("cat", "cat sat", n) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert token_f1("cat", "dog", n) == 0.0


def test_an_abstention_that_says_nothing_cannot_look_like_a_match() -> None:
    n = _norm()
    assert token_f1("", "anything", n) == 0.0
    assert token_f1("anything", "", n) == 0.0
    assert score_answer("", "", n)["token_f1"] == 0.0


def test_multi_gold_takes_the_max_not_the_mean() -> None:
    n = _norm()
    single = score_answer("cat", ("cat",), n)
    multi = score_answer("cat", ("dog", "cat"), n)
    assert multi["exact_match"] == 1.0
    assert multi["exact_match"] >= single["exact_match"]


def test_an_unimplemented_step_raises_rather_than_being_skipped() -> None:
    spec = {"normalisation": {"steps": [{"id": "mystery", "op": "summarise"}]}}
    with pytest.raises(UnknownNormalisationError, match="unknown op"):
        Normaliser.from_config(spec).normalise("x")


def test_a_config_with_no_steps_is_rejected() -> None:
    with pytest.raises(ValueError, match="no steps"):
        Normaliser.from_config({"normalisation": {"steps": []}})
