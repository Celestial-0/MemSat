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


def test_auc_of_a_perfect_separation_is_one() -> None:
    assert roc_auc([0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6]) == 1.0


def test_auc_of_a_reversed_separation_is_zero() -> None:
    assert roc_auc([0, 0, 0, 1, 1, 1], [6, 5, 4, 3, 2, 1]) == 0.0


def test_auc_of_all_tied_scores_is_one_half() -> None:
    assert roc_auc([0, 1, 0, 1, 0, 1], [5, 5, 5, 5, 5, 5]) == pytest.approx(0.5)


def test_auc_of_a_partial_separation_is_a_fraction() -> None:
    assert roc_auc([1, 1, 1, 0, 0, 0], [9, 3, 2, 8, 7, 1]) == pytest.approx(5 / 9)


def test_auc_handles_an_unbalanced_split() -> None:
    assert roc_auc([0, 0, 0, 0, 1], [5, 4, 3, 2, 1]) == 0.0
    assert roc_auc([0, 0, 0, 0, 1], [1, 2, 3, 4, 5]) == 1.0
    assert roc_auc([0, 0, 0, 0, 1], [1, 2, 4, 5, 3]) == pytest.approx(0.5)


def test_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0, 0, 0], [1, 2, 3])


def test_auc_accepts_plain_lists_and_arrays_alike() -> None:
    assert roc_auc([0, 1], [1, 2]) == roc_auc(np.array([0, 1]), np.array([1, 2]))


def test_a_leaking_arm_is_detected() -> None:
    items = [
        _feats(0, n_memories=2, n_tokens=800),
        _feats(0, n_memories=3, n_tokens=1100),
        _feats(0, n_memories=2, n_tokens=900),
        _feats(0, n_memories=3, n_tokens=1200),
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


def test_arm_b_as_designed_is_separable_by_memory_count() -> None:
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
        ("A", 1, 128, "budget", "S", "gemma"): cond(20, 200),
        ("B", 1, 128, "budget", "S", "gemma"): cond(4, 200),
    }
    assert size_leakage_matrix(drifted)["warnings"], (
        "a 5x length difference is not 'length held constant'"
    )


def test_a_deletion_style_construction_is_still_caught() -> None:
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
    items = [_feats(0, n_memories=10, n_tokens=4000, n_turns=40) for _ in range(8)]
    items += [_feats(1, n_memories=10, n_tokens=4000, n_turns=40) for _ in range(8)]
    report = assert_no_leakage(items, n_null=60)
    assert report.passed is True
    assert "Arm B is valid" in report.verdict


def test_a_constant_feature_contributes_nothing() -> None:
    items = [_feats(0, 8, 4000), _feats(0, 8, 4000), _feats(1, 8, 4000), _feats(1, 8, 4000)]
    report = assert_no_leakage(items, n_null=40)
    assert report.per_feature_auroc["n_memories"] == pytest.approx(0.5)
    assert report.per_feature_auroc["n_tokens"] == pytest.approx(0.5)


def test_the_null_sits_at_chance_on_real_items() -> None:
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
    items = [_feats(0, 4, 4000), _feats(1, 5, 4000), _feats(0, 6, 4050), _feats(1, 7, 3980)]
    report = assert_no_leakage(items, threshold=0.5, n_null=60)
    assert report.n_items == 4
    assert report.verdict.startswith(("PASS", "FAIL", "INCONCLUSIVE"))
    if not report.passed:
        assert "INCONCLUSIVE" in report.verdict or "FAIL" in report.verdict


def test_too_few_items_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 4 items"):
        assert_no_leakage([_feats(0, 2, 800), _feats(1, 20, 9000)])


def test_the_threshold_is_far_below_the_prior_art() -> None:
    assert DEFAULT_AUROC_THRESHOLD == 0.70
    assert DEFAULT_AUROC_THRESHOLD < 0.979


def test_the_threshold_is_configurable_and_appears_in_the_report() -> None:
    items = [_feats(0, 4, 4000), _feats(0, 5, 4100), _feats(1, 6, 4000), _feats(1, 7, 4050)]
    assert assert_no_leakage(items, threshold=0.5).threshold == 0.5
    assert assert_no_leakage(items, threshold=0.5).as_dict()["threshold"] == 0.5


def test_the_feature_set_is_few_and_explicit() -> None:
    assert SIZE_FEATURES == (
        "n_memories",
        "n_tokens",
        "n_turns",
        "mean_tokens_per_memory",
    )
    assert len(SIZE_FEATURES) <= 4


def test_the_vector_matches_the_feature_names_in_order() -> None:
    v = _feats(1, n_memories=10, n_tokens=4000, n_turns=40).as_vector()
    assert v == [10.0, 4000.0, 40.0, 400.0]


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
