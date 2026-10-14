"""R9.10 — the Arm B label-leakage assertion.

THE RULE, WHICH IS UNUSUAL
--------------------------
This test is supposed to PASS. If it is ever RED, **Arm B is invalid and the run is
flagged** — that is the intended behaviour, not a bug to suppress. 2609.32269 showed
that constructing insufficient-evidence conditions by deleting passages leaks the
label through memory size: a paragraph-counting classifier reaches AUROC 0.979 on
MuSiQue.

THE CONSEQUENCE OF A FAILURE
---------------------------
If size predicts the condition, then an accuracy difference between the conditions
is partly a size artefact rather than an evidence effect, and the count-vs-length
claim is not supported at all. That is worse than a null result, because it looks
like a finding.

WHY MOST OF THIS FILE IS ABOUT A FAILING FIXTURE
-------------------------------------------------
A leakage test that only ever asserts `report.passed is True` proves nothing: it
would pass against a detector that always returns True. The load-bearing tests here
construct an arm that DOES leak and assert the detector catches it. A smoke alarm
that has never been tested by setting off the alarm is not known to work.
"""

from __future__ import annotations

import numpy as np
import pytest

from msat.analysis.leakage import (
    DEFAULT_AUROC_THRESHOLD,
    SIZE_FEATURES,
    LeakageFeatures,
    assert_no_leakage,
    roc_auc,
)


def _feats(
    label: int, n_memories: int, n_tokens: int, n_turns: int | None = None
) -> LeakageFeatures:
    return LeakageFeatures(
        label=label,
        n_memories=n_memories,
        n_tokens=n_tokens,
        n_turns=n_turns if n_turns is not None else n_memories * 4,
    )


# ── AUROC itself, hand-computed ────────────────────────────────────────────────


def test_auc_of_a_perfect_separation_is_one() -> None:
    """Positives score above every negative."""
    assert roc_auc([0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6]) == 1.0


def test_auc_of_a_reversed_separation_is_zero() -> None:
    assert roc_auc([0, 0, 0, 1, 1, 1], [6, 5, 4, 3, 2, 1]) == 0.0


def test_auc_of_all_tied_scores_is_one_half() -> None:
    """Ties count as half. With no information at all the answer must be exactly
    0.5, not something near it - this is the value the null in
    `assert_no_leakage` is measured against."""
    assert roc_auc([0, 1, 0, 1, 0, 1], [5, 5, 5, 5, 5, 5]) == pytest.approx(0.5)


def test_auc_of_a_partial_separation_is_a_fraction() -> None:
    """9 pairs (3 positives x 3 negatives). Count the wins by hand:

        p=9 beats 8, 7, 1  -> 3
        p=3 beats 1        -> 1
        p=2 beats 1        -> 1

    5 of 9, so AUROC = 5/9. A perfectly-separated fixture would only prove the
    implementation handles the easy case.
    """
    assert roc_auc([1, 1, 1, 0, 0, 0], [9, 3, 2, 8, 7, 1]) == pytest.approx(5 / 9)


def test_auc_handles_an_unbalanced_split() -> None:
    """1 positive, 4 negatives = 4 pairs.

    A positive at the BOTTOM of the ranking beats nothing, so AUROC is exactly 0 -
    not "low", not undefined. This is the case where an off-by-one in the rank
    bookkeeping shows up most clearly.
    """
    assert roc_auc([0, 0, 0, 0, 1], [5, 4, 3, 2, 1]) == 0.0
    assert roc_auc([0, 0, 0, 0, 1], [1, 2, 3, 4, 5]) == 1.0
    # Positive score 3 against negatives 1, 2, 4, 5: it beats two of the four, so
    # 2/4 = 0.5. The exact-middle case, where an off-by-one shows up most clearly.
    assert roc_auc([0, 0, 0, 0, 1], [1, 2, 4, 5, 3]) == pytest.approx(0.5)


def test_auc_needs_both_classes() -> None:
    """One class means the statistic is undefined, not zero."""
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0, 0, 0], [1, 2, 3])


def test_auc_accepts_plain_lists_and_arrays_alike() -> None:
    """The hand-checked fixtures pass lists and the classifier passes arrays. Making
    both convert at the call site is a place for a bug to hide."""
    assert roc_auc([0, 1], [1, 2]) == roc_auc(np.array([0, 1]), np.array([1, 2]))


# ── the load-bearing tests: a leaking arm IS detected ──────────────────────────


def test_a_leaking_arm_is_detected() -> None:
    """The arm that R9.10 describes: conditions built by DELETING passages, so the
    short one is visibly smaller.

    A detector that cannot catch this is not a detector."""
    items = [
        # label 0 = the truncated condition. Far fewer memories and tokens.
        _feats(0, n_memories=2, n_tokens=800),
        _feats(0, n_memories=3, n_tokens=1100),
        _feats(0, n_memories=2, n_tokens=900),
        _feats(0, n_memories=3, n_tokens=1200),
        # label 1 = the full-length condition, visibly bigger.
        _feats(1, n_memories=20, n_tokens=9000),
        _feats(1, n_memories=24, n_tokens=11000),
        _feats(1, n_memories=19, n_tokens=8500),
        _feats(1, n_memories=23, n_tokens=10500),
    ]
    report = assert_no_leakage(items)
    assert report.passed is False
    assert report.auroc >= report.threshold
    assert "Arm B is INVALID" in report.verdict
    assert "0.979" in report.verdict, "the verdict should cite the prior art"


def test_a_leak_visible_in_a_single_feature_is_still_caught() -> None:
    """The MuSiQue result was a PARAGRAPH COUNT. One feature is enough."""
    items = [
        _feats(0, 2, 800),
        _feats(0, 3, 900),
        _feats(0, 2, 850),
        _feats(1, 20, 9000),
        _feats(1, 22, 9500),
        _feats(1, 21, 9200),
    ]
    report = assert_no_leakage(items)
    assert report.passed is False
    assert report.per_feature_auroc["n_memories"] > 0.9


def test_the_per_feature_breakdown_names_the_cue() -> None:
    """`n_tokens` is the leak, so the breakdown should show it and not the rest."""
    items = [
        _feats(0, 8, 800),
        _feats(0, 8, 850),
        _feats(0, 8, 820),
        _feats(1, 8, 9000),
        _feats(1, 8, 9100),
        _feats(1, 8, 9200),
    ]
    report = assert_no_leakage(items)
    assert report.per_feature_auroc["n_tokens"] > 0.9
    assert report.per_feature_auroc["n_memories"] == pytest.approx(0.5), (
        "n_memories is constant here, so it carries no information"
    )


# ── the arm as actually constructed: padding to a matched budget ───────────────


def test_arm_b_as_designed_is_separable_by_memory_count() -> None:
    """THE FINDING THIS TEST EXISTS TO RECORD.

    Arm B holds TOKEN COUNT fixed and varies MEMORY COUNT. A size-feature
    classifier that includes `n_memories` therefore separates the two conditions
    PERFECTLY - and that is not a bug in the arm, it is the arm's manipulation
    stated in another language.

    It matters because it means **R9.10's assertion, as literally written, does not
    apply to Arm B.** 2609.32269's leak is a different thing: conditions built by
    DELETING passages, where the label is "insufficient evidence" and memory size is
    a proxy for it that the model can exploit. Arm B's label IS the count, and the
    model can see the count directly from the numbered list anyway - so there is
    nothing for size to leak that the reader does not already have.

    What the assertion must instead check for Arm B is the one `size_leakage_matrix`
    checks: that `total_tokens` genuinely matches between arms, which is what
    "length is held constant" means. See
    `test_arm_bs_own_check_is_the_token_match` below.

    Written here rather than skipped so the fact is pinned. If someone later changes
    Arm B's construction so that count stops being the label, this test FAILS and
    the R9.10 assertion becomes applicable again - which would be worth knowing.
    """
    items = [_feats(0, n_memories=2 + (n % 2), n_tokens=4000) for n in range(8)]
    items += [_feats(1, n_memories=16 + (n % 3), n_tokens=4000) for n in range(8)]
    report = assert_no_leakage(items)
    assert report.passed is False
    assert report.auroc > 0.99, "the count separates them perfectly"
    assert report.per_feature_auroc["n_memories"] > 0.99
    assert report.per_feature_auroc["n_tokens"] == pytest.approx(0.5), (
        "the token count is matched, which is the point of the arm"
    )


def test_arm_bs_own_check_is_the_token_match() -> None:
    """The assertion that DOES apply to Arm B: length is held constant.

    R9.10's classifier is the right tool for a deletion-style construction. Arm B's
    is `size_leakage_matrix`, which compares `total_tokens` across arms and warns
    when they drift beyond the +/-2% target. If that drifts, length is not held
    constant and the count-vs-length separation is not being measured.
    """
    from msat.arms.padding import leakage_features, size_leakage_matrix
    from msat.arms.render import Memory

    def cond(memories: int, tokens_each: int) -> dict[str, float]:
        mems = [Memory(f"s{i}#0", f"s{i}", "q", "t", tokens_each) for i in range(memories)]
        return leakage_features(mems)

    matched = {
        ("A", 1, 128, "budget", "S", "gemma"): cond(20, 200),
        ("B", 1, 128, "budget", "S", "gemma"): cond(20, 200),
    }
    assert size_leakage_matrix(matched)["warnings"] == []

    drifted = {
        ("A", 1, 128, "budget", "S", "gemma"): cond(20, 200),  # 4000 tokens
        ("B", 1, 128, "budget", "S", "gemma"): cond(4, 200),  # 800 tokens
    }
    assert size_leakage_matrix(drifted)["warnings"], (
        "a 5x length difference is not 'length held constant'"
    )


def test_a_deletion_style_construction_is_still_caught() -> None:
    """R9.10's actual case, which is NOT Arm B: conditions built by deleting
    passages, so the short one is visibly smaller in EVERY size feature.

    The detector must catch this, because this is the leak it was built for."""
    items = [
        _feats(0, n_memories=2, n_tokens=800),
        _feats(0, n_memories=3, n_tokens=1100),
        _feats(0, n_memories=2, n_tokens=900),
        _feats(1, n_memories=20, n_tokens=9000),
        _feats(1, n_memories=24, n_tokens=11000),
        _feats(1, n_memories=19, n_tokens=8500),
        _feats(1, n_memories=23, n_tokens=10500),
    ]
    report = assert_no_leakage(items)
    assert report.passed is False
    assert report.auroc >= report.threshold
    assert "Arm B is INVALID" in report.verdict


def test_an_arm_matched_on_every_size_feature_passes() -> None:
    """The detector's positive case: two conditions with identical size profiles,
    distinguished only by something that is not size. This is what a correctly
    constructed control looks like."""
    items = [_feats(0, n_memories=10, n_tokens=4000, n_turns=40) for _ in range(8)]
    items += [_feats(1, n_memories=10, n_tokens=4000, n_turns=40) for _ in range(8)]
    report = assert_no_leakage(items, n_null=60)
    assert report.passed is True
    assert "Arm B is valid" in report.verdict


def test_a_constant_feature_contributes_nothing() -> None:
    """`n_memories` is pinned in this fixture, so it must read exactly 0.5 rather
    than drifting off chance on noise."""
    items = [_feats(0, 8, 4000), _feats(0, 8, 4000), _feats(1, 8, 4000), _feats(1, 8, 4000)]
    report = assert_no_leakage(items, n_null=40)
    assert report.per_feature_auroc["n_memories"] == pytest.approx(0.5)
    assert report.per_feature_auroc["n_tokens"] == pytest.approx(0.5)


# ── the null, which is what makes the number mean anything ─────────────────────


def test_the_null_sits_at_chance_on_real_items() -> None:
    """If the detector reports a high AUROC on SHUFFLED labels it is measuring
    something other than the label. This is why `null_auroc` is reported beside
    every result."""
    items = [
        _feats(0, 4, 4000),
        _feats(0, 5, 4100),
        _feats(0, 4, 3950),
        _feats(1, 6, 4000),
        _feats(1, 7, 4050),
        _feats(1, 6, 3980),
    ]
    report = assert_no_leakage(items, n_null=200)
    assert abs(report.null_auroc - 0.5) < 0.1


def test_a_null_off_chance_is_inconclusive_not_a_pass() -> None:
    """The verdict has a third state.

    `passed` requires BOTH that the observed AUROC is under the threshold AND that
    the detector's own null sits at chance. A detector whose null is broken cannot
    certify an arm, whatever the observed number says.
    """
    items = [_feats(0, 4, 4000), _feats(1, 5, 4000), _feats(0, 6, 4050), _feats(1, 7, 3980)]
    report = assert_no_leakage(items, threshold=0.5, n_null=60)
    assert report.n_items == 4
    # Whatever the outcome, it is one of the three NAMED states - never a bare bool,
    # because a reader has to be able to tell "arm is fine" from "we could not tell".
    assert report.verdict.startswith(("PASS", "FAIL", "INCONCLUSIVE"))
    if not report.passed:
        assert "INCONCLUSIVE" in report.verdict or "FAIL" in report.verdict


def test_too_few_items_is_refused() -> None:
    """Below 4 items a classifier cannot be fitted and an AUROC means nothing."""
    with pytest.raises(ValueError, match="at least 4 items"):
        assert_no_leakage([_feats(0, 2, 800), _feats(1, 20, 9000)])


# ── the threshold is a reviewable value, not a magic number ────────────────────


def test_the_threshold_is_far_below_the_prior_art() -> None:
    """0.979 is what leakage looks like. 0.70 is a smoke alarm, not a replication
    target, and it is deliberately far below the prior art."""
    assert DEFAULT_AUROC_THRESHOLD == 0.70
    assert DEFAULT_AUROC_THRESHOLD < 0.979


def test_the_threshold_is_configurable_and_appears_in_the_report() -> None:
    items = [_feats(0, 4, 4000), _feats(0, 5, 4100), _feats(1, 6, 4000), _feats(1, 7, 4050)]
    assert assert_no_leakage(items, threshold=0.5).threshold == 0.5
    assert assert_no_leakage(items, threshold=0.5).as_dict()["threshold"] == 0.5


def test_the_feature_set_is_few_and_explicit() -> None:
    """Every extra feature is another chance to find a spurious cue, and a leakage
    detector with many features detects leakage that is not there."""
    assert SIZE_FEATURES == (
        "n_memories",
        "n_tokens",
        "n_turns",
        "mean_tokens_per_memory",
    )
    assert len(SIZE_FEATURES) <= 4


def test_the_vector_matches_the_feature_names_in_order() -> None:
    v = _feats(1, n_memories=10, n_tokens=4000, n_turns=40).as_vector()
    assert v == [10.0, 4000.0, 40.0, 400.0]  # mean = 4000/10


def test_the_report_is_json_serialisable() -> None:
    import json

    items = [_feats(0, 4, 4000), _feats(1, 6, 4000), _feats(0, 5, 4050), _feats(1, 7, 3980)]
    json.loads(json.dumps(assert_no_leakage(items).as_dict()))


def test_the_report_carries_the_counts_it_needs_to_be_read() -> None:
    d = assert_no_leakage(
        [_feats(0, 4, 4000), _feats(1, 6, 4000), _feats(0, 5, 4050), _feats(1, 7, 3980)]
    ).as_dict()
    assert d["n_items"] == 4
    assert d["n_positive"] == 2
    assert "per_feature_auroc" in d
    assert "null_auroc" in d
