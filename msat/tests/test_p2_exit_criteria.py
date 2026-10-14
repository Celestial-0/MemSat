"""P2 exit criteria, asserted directly.

Architecture §P2 lists six deliverables and an exit condition. These tests read
as the checklist, because a phase is complete when its criteria are met and not
when someone says so — P2 was reported complete twice while items 2.4 and 2.5 were
still open.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

from msat.design import grid_center, k_delta_in_pp, k_log2_step, k_step_index
from msat.metrics import Normaliser, UnknownNormalisationError, score_answer, token_f1

DOUBLING = [1, 2, 4, 8, 16, 32, 64]
UNEVEN = [1, 2, 4, 8, 16, 48, 64]


# ── P2.4 — ONE grid, one source ───────────────────────────────────────────────
def test_the_budget_profile_cannot_restate_the_grid() -> None:
    """The regression that let two B axes exist at once.

    `budget.yaml` carried a `levels:` list in `absolute_tokens` while
    `design.yaml` declared a different one in `per_memory_tokens`. The guard
    missed it because it grepped for the key `B_grid` and that file said
    `levels`. The model now has no field to restate the grid in, so the
    duplication is not expressible rather than merely discouraged.
    """
    from msat.config import BudgetProfile

    fields = set(BudgetProfile.model_fields)
    assert "levels" not in fields, (
        "BudgetProfile must not carry budget levels - they live in design.yaml. "
        "Reinstating this field is how the repo ended up with two B axes."
    )
    assert "budget_source" in fields, "the profile should say where the levels are"


def test_the_duplicate_grid_guard_matches_on_structure_not_key_name() -> None:
    """A renamed key must not bypass the guard.

    The old guard grepped for `K_grid|B_grid|scales|readers`. Writing the same
    grid under `levels:` walked straight past it. The new check walks the parsed
    document and flags any `{value, mode}` pair outside design.yaml.
    """
    from msat.config import _declares_budget_levels

    # Whatever it is called, a B level is `{value, mode}`.
    assert _declares_budget_levels({"levels": [{"value": 4096, "mode": "x"}]})
    assert _declares_budget_levels({"anything_at_all": [{"value": 1, "mode": "y"}]})
    assert _declares_budget_levels([{"nested": {"deeper": [{"value": 2, "mode": "z"}]}}])
    # And unrelated config must not trip it.
    assert not _declares_budget_levels({"max_usd": 0.0, "disk_budget_bytes": 1})
    assert not _declares_budget_levels({"models": {"m": {"input": 0.0, "output": 0.0}}})


def test_the_two_centres_are_derived_not_hardcoded() -> None:
    """A hardcoded centre cannot notice that its grid changed.

    `B_CENTER = 14.0` was the mean of `[4096, 16384, 65536]`. When the B axis
    moved to `[128, 512, 2048]` it silently became the mean of a grid that no
    longer existed, and nothing complained.
    """
    import math

    assert grid_center([math.log2(k) for k in DOUBLING]) == pytest.approx(3.0)
    # The CURRENT B grid, which the old constant got wrong.
    assert grid_center([math.log2(b) for b in (128, 512, 2048)]) == pytest.approx(9.0)
    assert grid_center([]) == pytest.approx(0.0) if False else True  # raises below
    with pytest.raises(ValueError):
        grid_center([])


# ── P2.5 — K_delta on the actual grid (PRD §R9.4) ─────────────────────────────
def test_step_index_is_position_not_log() -> None:
    assert k_step_index(DOUBLING, 1) == 0.0
    assert k_step_index(DOUBLING, 64) == 1.0
    assert k_step_index(UNEVEN, 48) == pytest.approx(5 / 6)


def test_step_index_rejects_a_k_off_the_grid() -> None:
    """Silently interpolating an undeclared K is how a figure ends up describing
    a condition the design never ran."""
    with pytest.raises(ValueError, match="not on the declared grid"):
        k_step_index(DOUBLING, 3)


def test_every_step_on_a_doubling_grid_is_one_log2_unit() -> None:
    for k in DOUBLING[1:]:
        assert k_log2_step(DOUBLING, k) == pytest.approx(1.0)


def test_an_uneven_grid_has_genuinely_different_step_sizes() -> None:
    # 16 -> 48 is 1.58 log2 units; 48 -> 64 is 0.42. A raw per-step percentage
    # threshold means something three times larger at one than at the other.
    assert k_log2_step(UNEVEN, 48) == pytest.approx(math.log2(3))
    assert k_log2_step(UNEVEN, 64) == pytest.approx(math.log2(4 / 3))
    assert k_log2_step(UNEVEN, 48) > k_log2_step(UNEVEN, 64)


def test_the_threshold_scales_with_the_step_that_actually_occurs() -> None:
    """The whole point of R9.4: 0.5pp means the same thing at every point."""
    assert k_delta_in_pp(0.5, DOUBLING, 32) == pytest.approx(0.5)
    big = k_delta_in_pp(0.5, UNEVEN, 48)
    small = k_delta_in_pp(0.5, UNEVEN, 64)
    assert big == pytest.approx(0.5 * math.log2(3))
    assert small == pytest.approx(0.5 * math.log2(4 / 3))
    # On the shipped grid the rule is unchanged - this is a generalisation, not a
    # change of threshold.
    for k in DOUBLING:
        assert k_delta_in_pp(0.5, DOUBLING, k) == pytest.approx(0.5)


def test_the_smallest_k_is_not_exempt() -> None:
    """K=1 has no arriving step; using the departing one stops the first cell
    escaping every rule."""
    assert k_log2_step(DOUBLING, 1) == pytest.approx(1.0)


# ── P2.3 — the answer-normalisation function ──────────────────────────────────
def _norm() -> Normaliser:
    """Build from the SHIPPED `configs/prompts/normalisation.yaml`.

    Deliberately not a hand-copied fixture. A copy duplicates the config and then
    drifts from it, and the first version of this test did exactly that - it
    omitted the final `collapse_whitespace_again` and `strip`, so it failed while
    the real normaliser was correct. Testing the shipped file means a change to
    the config changes the test, which is the whole point of R1.2's "documented".
    """
    import yaml

    raw = yaml.safe_load((CONFIGS / "prompts" / "normalisation.yaml").read_text(encoding="utf-8"))
    return Normaliser.from_config(raw)


def test_normalisation_actually_runs() -> None:
    n = _norm()
    assert n.normalise("  The   CAT, sat!  ") == "cat sat"


def test_nfkc_folds_typography_before_anything_else() -> None:
    # A typographic difference must never be scored as a semantic one.
    assert _norm().normalise("ﬁsh") == "fish"
    assert _norm().normalise("\uff43\uff41\uff54") == "cat"  # fullwidth


def test_the_declared_step_order_is_preserved() -> None:
    """Order is semantic: it changes results on the hardest items."""
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
    """An empty prediction scores 0, not 1 - otherwise abstaining would be
    rewarded on any empty gold."""
    n = _norm()
    assert token_f1("", "anything", n) == 0.0
    assert token_f1("anything", "", n) == 0.0
    assert score_answer("", "", n)["token_f1"] == 0.0


def test_multi_gold_takes_the_max_not_the_mean() -> None:
    """`any_match`: several acceptable answers. Averaging would let a two-option
    gold beat a one-option gold by half-matching."""
    n = _norm()
    single = score_answer("cat", ("cat",), n)
    multi = score_answer("cat", ("dog", "cat"), n)
    assert multi["exact_match"] == 1.0
    assert multi["exact_match"] >= single["exact_match"]


def test_an_unimplemented_step_raises_rather_than_being_skipped() -> None:
    """A silently skipped step computes something other than the config says,
    and the difference is invisible in the results."""
    spec = {"normalisation": {"steps": [{"id": "mystery", "op": "summarise"}]}}
    with pytest.raises(UnknownNormalisationError, match="unknown op"):
        Normaliser.from_config(spec).normalise("x")


def test_a_config_with_no_steps_is_rejected() -> None:
    with pytest.raises(ValueError, match="no steps"):
        Normaliser.from_config({"normalisation": {"steps": []}})
