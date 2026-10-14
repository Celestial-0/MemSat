"""R9.12 - depth-response sign disagreement between reader families.

WHAT THIS FILE IS FOR
--------------------
2606.21807 found compression gain *decreases* with reader baseline and flips 31% of
pairwise rankings. R9.12 makes that a headline for this study, because a depth curve
measured at one reader is not a property of retrieval.

The tests are built around one recurring failure of this project: **an aggregate that
cancels**. R8.6's "479 vs 475" hid 8 flipped rows; hardware-delta pairs cancelled;
gate branches cancelled. R9.12 is the same shape, and the module is built so it
cannot recur:

    if half the settings run one way and half the other, the MEAN depth slope
    difference is ~0 and "readers agree" - while a third of the pairs genuinely
    disagree.

So the load-bearing tests below construct a panel whose flips CANCEL in the mean,
and assert the report still says the readers disagree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pytest

from msat.analysis.reader_agreement import (
    FALLS,
    FLAT,
    MIN_POINTS_FOR_SLOPE,
    REFERENCE_FLIP_RATE,
    RISES,
    SYNTHETIC_STRENGTHS,
    CellScore,
    DepthResponse,
    ReaderAgreementReport,
    UnsupportedComparisonError,
    compare_readers,
    depth_response,
    responses_from_cells,
    synthetic_cells,
)


def _curve(
    reader: str,
    slope: float,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    b: float = 128.0,
    scale: str = "S",
    start: float = 0.40,
) -> list[CellScore]:
    """One reader's depth curve with the given per-doubling slope."""
    return [
        CellScore(
            reader=reader,
            k=k,
            b=b,
            scale=scale,
            score=max(0.0, min(1.0, start + slope * i)),
            n_questions=500,
        )
        for i, k in enumerate(ks)
    ]


def _panel(readers: Mapping[str, float], **kw: object) -> list[CellScore]:
    """Build one curve per named reader.

    Taking an explicit MAPPING rather than fixed gemma/qwen/phi slots matters:
    two calls at the same (reader, b, scale) MERGE into a single averaged curve, so a
    helper that always emits three readers cannot express "these two readers only" -
    and quietly averaged the effect away in an earlier draft of this file.
    """
    out: list[CellScore] = []
    for reader, slope in readers.items():
        out += _curve(reader, slope, **kw)  # type: ignore[arg-type]
    return out


def _report(cells: list[CellScore], **kwargs: object) -> ReaderAgreementReport:
    return compare_readers(responses_from_cells(cells), **kwargs)  # type: ignore[arg-type]


# ── the headline: a COUNT, never a mean ───────────────────────────────────────


def test_disagreeing_readers_are_reported_as_disagreeing() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    assert rep.n_flips > 0
    assert rep.flip_rate is not None and rep.flip_rate > 0


def test_flips_that_cancel_in_the_mean_are_still_flips() -> None:
    """THE test this module exists for.

    gemma falls at one budget and rises at another; qwen does the opposite. Every
    signed average over the panel is near zero. The mean of the two readers' depth
    slopes differs by ~0.00, which would read as "the readers agree" - while half
    the pairs disagree in sign.

    R8.6's shape exactly: 479 vs 475 is a net difference of 4 and 8 changed rows.
    """
    cells: list[CellScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0):
            rising = b < 300.0
            cells += _panel(
                {"gemma": +0.05 if rising else -0.05, "qwen": -0.05 if rising else +0.05},
                b=b,
                scale=scale,
            )

    rep = _report(cells)
    assert rep.n_flips == 4, "gemma|qwen flips at both budgets, in both scales"

    # The aggregate that would have hidden it: two flips each way, so the readers'
    # mean slopes coincide.
    by_reader: dict[str, list[float]] = {}
    for r in rep.responses:
        by_reader.setdefault(r.reader, []).append(r.slope or 0.0)
    mean_gemma = sum(by_reader["gemma"]) / len(by_reader["gemma"])
    mean_qwen = sum(by_reader["qwen"]) / len(by_reader["qwen"])
    assert abs(mean_gemma - mean_qwen) < 0.02, (
        "the aggregate really does look like agreement - which is the point"
    )
    assert rep.flip_rate == 1.0, "and every single comparable pair disagrees"
    assert rep.net_delta == 0, "the flips cancel in the signed aggregate, as expected"
    assert rep.directional_balance == 0.0, "and they are perfectly balanced, which is why"


def test_the_report_never_leaves_the_flip_count_as_the_only_number() -> None:
    """`n_flips` is necessary but not sufficient: 10 flips all one way and 10 flips
    balanced are different findings with different fixes. Both are reported."""
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.as_dict()
    assert d["n_flips"] == rep.n_flips
    assert "net_delta" in d
    assert "directional_balance" in d
    assert d["directional_balance"] is not None


def test_lopsided_and_balanced_flips_are_distinguishable() -> None:
    """Same headline count, different findings, different fixes.

    Lopsided: the weaker reader is systematically better at depth. Balanced: the two
    are near-mirror images. `n_flips` alone cannot tell them apart, so both numbers
    exist.
    """
    lopsided = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    assert lopsided.n_flips == 1
    assert lopsided.directional_balance == 1.0

    # gemma falls at one budget and rises at the other, against a rising qwen.
    cells: list[CellScore] = []
    for b in (128.0, 512.0):
        cells += _panel({"gemma": -0.05 if b < 300.0 else +0.05, "qwen": +0.05}, b=b)
    balanced = _report(cells)
    assert balanced.n_flips == 1, "only the budget where gemma falls"
    assert balanced.directional_balance == 1.0

    # The genuinely balanced case needs a flip in EACH direction.
    cells = []
    for b in (128.0, 512.0):
        rising = b < 300.0
        cells += _panel(
            {"gemma": +0.05 if rising else -0.05, "qwen": -0.05 if rising else +0.05}, b=b
        )
    both_ways = _report(cells)
    assert both_ways.n_flips == 2
    assert both_ways.directional_balance == 0.0
    assert both_ways.net_delta == 0


def test_a_purely_agreeing_panel_reports_zero_and_not_none() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": +0.05}))
    assert rep.n_flips == 0
    assert rep.flip_rate == 0.0
    assert rep.directional_balance is None, (
        "the balance of nothing is not zero; None says there were no flips"
    )


def test_no_comparable_pairs_is_none_not_zero() -> None:
    """ "No pairs" and "no flips" must not print the same. A 0.0 flip rate on an empty
    panel reads as a measured zero, which is a finding."""
    rep = ReaderAgreementReport()
    assert rep.flip_rate is None
    assert rep.n_pairs == 0
    assert "No reader pairs" in rep.headline()


def test_net_delta_is_context_not_the_headline() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.as_dict()
    assert d["flip_rate"] > 0
    assert "net_delta" in d


# ── flat is a third state, never rounded to a sign ────────────────────────────


def test_a_flat_reader_agrees_with_nobody_but_flips_with_nobody() -> None:
    """A flat response has no ranking to flip.

    Forcing a sign onto zero would inflate the flip rate with non-events - the
    mirror image of the cancellation bug: here the number would look alarming for a
    reason that has nothing to do with reader dependence."""
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05, "phi": 0.0}))
    assert rep.n_flips == 1, "only the gemma|qwen pair"
    assert rep.n_flat_involved == 2, "gemma|phi and qwen|phi are flat_involved"
    assert rep.n_agree == 0
    assert rep.flip_rate == pytest.approx(1 / 3), (
        "the flat pairs are excluded from the rate AND counted, never silently dropped"
    )
    verdicts = {c.verdict for c in rep.comparisons}
    assert verdicts == {"flip", "flat_involved"}


def test_two_flat_readers_agree() -> None:
    rep = _report(_panel({"gemma": 0.0, "qwen": 0.0}))
    assert rep.n_agree > 0
    assert rep.n_flips == 0
    assert rep.n_flat_involved == 0, "both flat is agreement, not a missing sign"


def test_an_exactly_flat_response_is_flat() -> None:
    r = depth_response([(1, 0.5), (2, 0.5), (4, 0.5)], reader="x", b=1.0, scale="S")
    assert r.slope == 0.0
    assert r.sign == FLAT
    assert r.sign_name == "flat"
    assert r.defined is False


# ── a slope needs enough points to be a slope ──────────────────────────────────


def test_two_points_do_not_yield_a_slope() -> None:
    """Two points always have a slope, and reporting it would let a single noisy
    measurement decide a sign."""
    r = depth_response([(1, 0.30), (2, 0.90)], reader="x", b=1.0, scale="S")
    assert r.slope is None
    assert r.sign == FLAT
    assert r.n_points == 2


def test_one_point_does_not_yield_a_slope() -> None:
    r = depth_response([(4, 0.5)], reader="x", b=1.0, scale="S")
    assert r.slope is None
    assert r.sign == FLAT


def test_three_points_is_enough() -> None:
    assert MIN_POINTS_FOR_SLOPE == 3
    r = depth_response([(1, 0.2), (2, 0.4), (4, 0.6)], reader="x", b=1.0, scale="S")
    assert r.slope is not None and r.slope > 0
    assert r.sign == RISES


def test_a_reader_with_a_truncated_curve_is_refused_before_it_can_flip() -> None:
    """A reader that only managed K = [1, 2] has no trend - and it cannot produce a
    coin-flip sign either.

    The support guard fires FIRST, which is the stronger behaviour: a 2-point reader
    is not merely unreliable, it is answering over a different range than its
    partner, so comparing them is refused outright rather than producing a
    `flat_involved` that a reader might mistake for a measured indifference.
    """
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2))
    with pytest.raises(UnsupportedComparisonError):
        compare_readers(responses_from_cells(cells))

    rep = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    assert rep.flips_by_pair() == {}, "a two-point reader cannot generate a sign flip"
    assert rep.n_pairs == 0
    assert rep.flip_rate is None, "and the rate is None, not a measured 0.0"


def test_two_readers_at_the_same_short_support_compare_normally() -> None:
    """Truncation is a difference, not a property. Two readers that BOTH stopped at
    K = [1, 2, 4] share their support and are comparable - they are just flat, and
    that is reported as flat rather than refused."""
    rep = compare_readers(
        responses_from_cells(_panel({"gemma": +0.05, "qwen": -0.05}, ks=(1, 2, 4)))
    )
    assert rep.n_pairs == 1
    assert rep.n_flips == 1
    assert rep.unsupported == []


# ── the slope is in log2(K), because R9.1's model is ───────────────────────────


def test_the_slope_is_per_doubling_not_per_raw_unit() -> None:
    """A slope in raw K weights each doubling differently, so two readers with
    different supports - which the 8 GB card guarantees - get slopes on different
    scales and their SIGNS stop being comparable."""
    pts = [(1, 0.20), (2, 0.40), (4, 0.60), (8, 0.80)]
    r = depth_response(pts, reader="x", b=1.0, scale="S")
    assert r.slope == pytest.approx(0.20), "each doubling adds 0.20"

    # In raw K the same data gives a very different number.
    xs = [float(k) for k, _ in pts]
    ys = [s for _, s in pts]
    xbar, ybar = sum(xs) / 4, sum(ys) / 4
    raw = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - xbar) ** 2 for x in xs
    )
    assert abs(raw - 0.20) > 0.05, "the two scales genuinely differ here"


def test_the_slope_matches_a_hand_computed_regression() -> None:
    # log2 K = 0, 1, 2, 3 ; y = 0, 1, 2, 3  -> perfect fit, slope 1
    r = depth_response([(1, 0.0), (2, 1.0), (4, 2.0), (8, 3.0)], reader="x", b=1.0, scale="S")
    assert r.slope == pytest.approx(1.0)


def test_k_zero_is_excluded_rather_than_logging_it() -> None:
    """`log2(0)` is undefined, and K = 0 is not a retrieval condition."""
    r = depth_response([(0, 0.0), (1, 0.2), (2, 0.4), (4, 0.6)], reader="x", b=1.0, scale="S")
    assert 0 not in r.ks
    assert r.ks == (1, 2, 4)


def test_an_all_one_k_curve_is_flat_not_a_division_error() -> None:
    """Unreachable through the public path (distinct `ks` by construction) but a
    duplicated K with different scores would land here, and dividing would raise
    rather than answer."""
    r = depth_response([(4, 0.2), (4, 0.9), (4, 0.5)], reader="x", b=1.0, scale="S", min_points=2)
    assert r.slope is None
    assert r.sign == FLAT


def test_math_log2_agrees_with_the_fixture_grid() -> None:
    """Guards the hand-computed expectations above against a change in the scale."""
    for k in (1, 2, 4, 8, 16, 32):
        assert math.log2(k) == float(k.bit_length() - 1)


# ── UNEQUAL SUPPORT IS REFUSED, NOT SKIPPED IN SILENCE ─────────────────────────


def test_a_reader_that_cannot_reach_the_deep_cells_blocks_the_comparison() -> None:
    """`configs/readers.yaml` documents the fact: on 8 GB, qwen tops out near 21k
    tokens and cannot serve the deepest cells. Comparing a 6-point gemma curve to a
    3-point qwen one answers two different questions."""
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    with pytest.raises(UnsupportedComparisonError, match="unequal K supports"):
        compare_readers(responses_from_cells(cells))


def test_skipping_still_records_what_was_skipped() -> None:
    """ "Skip" changes the default, not the bookkeeping. The unsupported comparisons
    stay in the report, because a discarded comparison is a hole in the factorial and
    nobody would see it."""
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    assert len(rep.unsupported) > 0
    assert rep.unsupported_rate > 0
    assert all(c.support != "common" for c in rep.unsupported)


def test_count_and_skip_agree_on_the_bookkeeping() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4))
    cells += _panel({"qwen": -0.05}, ks=(1, 2))
    a = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    b = compare_readers(responses_from_cells(cells), on_unsupported="count")
    assert len(a.unsupported) == len(b.unsupported)
    assert a.unsupported_rate == b.unsupported_rate


def test_the_shared_support_is_named_which_reader_was_narrower() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    kinds = {c.support for c in rep.unsupported}
    assert kinds <= {"truncated_a", "truncated_b", "truncated_both", "partial_overlap"}


def test_the_unsupported_rate_reaches_the_serialised_report() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    d = compare_readers(responses_from_cells(cells), on_unsupported="skip").as_dict()
    assert d["n_unsupported"] > 0
    assert d["unsupported_rate"] > 0
    assert d["unsupported"], "the individual skipped comparisons are listed"


def test_the_headline_mentions_the_truncation() -> None:
    """A third reader keeps the panel partly comparable, so the refusal is reported
    as a REDUCTION in coverage rather than as "nothing to say" - which is the
    difference between a hardware limitation and an absence of evidence."""
    cells = _panel({"gemma": +0.05, "phi": +0.04}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    assert rep.n_pairs > 0, "gemma|phi is comparable"
    assert rep.unsupported, "gemma|qwen and phi|qwen are not"
    assert "refused" in rep.headline()
    assert "narrower depth range" in rep.headline()


def test_a_fully_uncomparable_panel_says_so_rather_than_reporting_zero() -> None:
    """ "No pairs" and "no flips" must not print the same: the first is an absence of
    evidence, the second is a finding."""
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_cells(cells), on_unsupported="skip")
    assert rep.n_pairs == 0
    assert rep.flip_rate is None
    assert "not attributable across readers" in rep.headline()


def test_an_unknown_support_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_unsupported"):
        compare_readers(
            responses_from_cells(_panel({"gemma": 0.1, "qwen": -0.1})), on_unsupported="ignore"
        )


# ── the reference is a comparison point, never a threshold ────────────────────


def test_the_2606_reference_is_carried_as_a_comparison() -> None:
    assert REFERENCE_FLIP_RATE == 0.31
    d = _report(_panel({"gemma": +0.05, "qwen": -0.05})).as_dict()
    assert d["reference"]["flip_rate"] == 0.31
    assert "not a threshold" in d["reference"]["note"]


def test_a_flip_rate_wildly_different_from_the_reference_is_still_reported() -> None:
    """If 31% were a gate, a detector tuned to reproduce another paper's number would
    be indistinguishable from one measuring this panel. Nothing here compares to it."""
    rep = _report(_panel({"gemma": +0.10, "qwen": -0.10}))
    d = rep.as_dict()
    assert d["flip_rate"] != REFERENCE_FLIP_RATE
    assert d["flip_rate"] is not None and d["flip_rate"] > 0


# ── the strength-direction test (2606.21807's actual claim) ───────────────────


def test_the_stronger_reader_falling_is_the_claim_under_test() -> None:
    """2606.21807: compression gain DECREASES with reader baseline. qwen (0.80) is
    the stronger reader here, and it is the one that falls."""
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}), strengths=SYNTHETIC_STRENGTHS)
    d = rep.by_strength_direction()
    assert d["stronger_falls"] == rep.n_flips
    assert d["stronger_rises"] == 0


def test_the_synthetic_panel_reproduces_that_pattern() -> None:
    rep = _report(synthetic_cells(), strengths=SYNTHETIC_STRENGTHS)
    assert rep.n_flips > 0
    assert rep.by_strength_direction()["stronger_falls"] == rep.n_flips


def test_missing_strengths_land_in_unknown_not_in_equal() -> None:
    """Without strength data every flip would otherwise fall into `equal_strength`
    and read as "reader strength makes no difference" - the exact opposite of what
    is being tested."""
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.by_strength_direction()
    assert d["unknown"] == rep.n_flips
    assert d["equal_strength"] == 0


def test_a_partial_strength_table_does_not_silently_pair() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}), strengths={"gemma": 0.78})
    assert rep.by_strength_direction()["unknown"] == rep.n_flips


def test_the_strength_breakdown_counts_flips_only() -> None:
    """Agreed pairs have no direction to attribute."""
    rep = _report(
        _panel({"gemma": +0.05, "qwen": -0.05, "phi": +0.05}),
        strengths=SYNTHETIC_STRENGTHS,
    )
    total = sum(rep.by_strength_direction().values())
    assert total == rep.n_flips


# ── strata: the flip count must survive being broken down ─────────────────────


def test_flips_are_kept_when_stratified() -> None:
    """The aggregate-hides-instability trap survives stratification too, so the
    per-stratum flip counts are first-class and not derivable from the total."""
    cells: list[CellScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0):
            rising = scale == "S"
            cells += _panel(
                {"gemma": +0.05 if rising else -0.05, "qwen": -0.05 if rising else +0.05},
                b=b,
                scale=scale,
            )
    rep = _report(cells)
    assert sum(rep.flips_by_scale().values()) == rep.n_flips
    assert sum(rep.flips_by_budget().values()) == rep.n_flips
    assert set(rep.flips_by_scale()) == {"S", "M"}


def test_flips_are_attributed_to_a_reader_pair() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05, "phi": 0.0}))
    assert sum(rep.flips_by_pair().values()) == rep.n_flips
    assert "gemma|qwen" in rep.flips_by_pair()
    assert "gemma|phi" not in rep.flips_by_pair()


def test_the_counts_and_the_flips_are_the_same_population() -> None:
    """`by_*` counts what was compared; `flips_by_*` counts what disagreed. If these
    partitions differ, one of them is lying."""
    rep = _report(synthetic_cells())
    assert sum(rep.by_scale().values()) == rep.n_pairs
    for scale, n in rep.by_scale().items():
        assert rep.flips_by_scale().get(scale, 0) <= n


# ── bookkeeping ───────────────────────────────────────────────────────────────


def test_duplicate_rows_at_one_k_are_averaged_not_double_counted() -> None:
    """Should not happen - `score_store` emits one row per (condition, qid) - but if
    two rows shared a K, keeping both would weight that K twice in the regression."""
    cells = [
        CellScore(reader="gemma", k=1, b=128.0, scale="S", score=0.20),
        CellScore(reader="gemma", k=1, b=128.0, scale="S", score=0.40),
        CellScore(reader="gemma", k=2, b=128.0, scale="S", score=0.50),
        CellScore(reader="gemma", k=4, b=128.0, scale="S", score=0.70),
    ]
    out = responses_from_cells(cells)
    assert len(out) == 1
    assert out[0].n_points == 3, "three distinct K, not four rows"


def test_responses_are_grouped_by_reader_budget_and_scale() -> None:
    cells: list[CellScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0):
            cells += _panel({"gemma": 0.05, "qwen": 0.05}, b=b, scale=scale)
    out = responses_from_cells(cells)
    assert len(out) == 2 * 2 * 2
    assert {(r.b, r.scale, r.reader) for r in out} == {
        (b, scale, reader)
        for scale in ("S", "M")
        for b in (128.0, 512.0)
        for reader in ("gemma", "qwen")
    }


def test_arms_are_kept_apart() -> None:
    """Arm A and Arm B are different conditions at the same (K, B). Pooling them
    would compare a K-curve that mixes two experiments."""
    cells = _panel({"gemma": +0.05, "qwen": -0.05})
    cells += [
        CellScore(reader=c.reader, k=c.k, b=c.b, scale=c.scale, score=c.score, arm="B")
        for c in _panel({"gemma": -0.05, "qwen": +0.05})
    ]
    out = responses_from_cells(cells)
    assert {(r.reader, r.arm) for r in out} == {
        (reader, arm) for reader in ("gemma", "qwen") for arm in ("A", "B")
    }


def test_a_response_serialises_with_its_ks() -> None:
    """The K support travels with the number. A slope without the K values it was
    computed from cannot be checked."""
    r = depth_response([(1, 0.2), (2, 0.4), (4, 0.6)], reader="gemma", b=128.0, scale="S", arm="A")
    d = r.as_dict()
    assert d["ks"] == [1, 2, 4]
    assert d["n_points"] == 3
    assert d["sign_name"] == "rises"


def test_the_report_is_json_serialisable() -> None:
    json.dumps(_report(synthetic_cells(), strengths=SYNTHETIC_STRENGTHS).as_dict())


def test_the_report_is_deterministic() -> None:
    """`responses_from_cells` must not depend on dict iteration order, or two runs
    of the same data produce different files and the artifact diff is noise."""
    a = _report(synthetic_cells(), strengths=SYNTHETIC_STRENGTHS).as_dict()
    b = _report(list(reversed(synthetic_cells())), strengths=SYNTHETIC_STRENGTHS).as_dict()
    assert a == b


# ── the sign convention is named, not assumed ─────────────────────────────────


def test_the_three_signs_are_the_three_named_states() -> None:
    assert (FALLS, FLAT, RISES) == (-1, 0, 1)


def test_higher_is_better_is_stated_in_the_sign_names() -> None:
    """`rises`/`falls` are only meaningful if rising means a better score. A reader
    seeing `sign_name: "rises"` must not have to check which direction is good."""
    up = depth_response([(1, 0.2), (2, 0.4), (4, 0.6)], reader="x", b=1.0, scale="S")
    assert up.sign == RISES and up.sign_name == "rises"
    down = depth_response([(1, 0.6), (2, 0.4), (4, 0.2)], reader="x", b=1.0, scale="S")
    assert down.sign == FALLS and down.sign_name == "falls"


def test_a_depth_response_can_be_built_directly_for_a_test() -> None:
    """The dataclass is public so a caller can supply a response from an artifact
    rather than re-deriving it."""
    r = DepthResponse(reader="a", b=1.0, scale="S", arm="A", ks=(1, 2, 4), slope=0.1, sign=RISES)
    rep = compare_readers(
        [
            r,
            DepthResponse(
                reader="b", b=1.0, scale="S", arm="A", ks=(1, 2, 4), slope=-0.1, sign=FALLS
            ),
        ]
    )
    assert rep.n_flips == 1
