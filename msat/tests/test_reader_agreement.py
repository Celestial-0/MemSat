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
    ConditionScore,
    DepthResponse,
    ReaderAgreementReport,
    UnsupportedComparisonError,
    compare_readers,
    depth_response,
    responses_from_conditions,
    synthetic_conditions,
)


def _curve(
    reader: str,
    slope: float,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    b: float = 128.0,
    scale: str = "S",
    start: float = 0.40,
) -> list[ConditionScore]:
    return [
        ConditionScore(
            reader=reader,
            k=k,
            b=b,
            scale=scale,
            score=max(0.0, min(1.0, start + slope * i)),
            n_questions=500,
        )
        for i, k in enumerate(ks)
    ]


def _panel(readers: Mapping[str, float], **kw: object) -> list[ConditionScore]:
    out: list[ConditionScore] = []
    for reader, slope in readers.items():
        out += _curve(reader, slope, **kw)  # type: ignore[arg-type]
    return out


def _report(cells: list[ConditionScore], **kwargs: object) -> ReaderAgreementReport:
    return compare_readers(responses_from_conditions(cells), **kwargs)  # type: ignore[arg-type]


def test_disagreeing_readers_are_reported_as_disagreeing() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    assert rep.n_flips > 0
    assert rep.flip_rate is not None and rep.flip_rate > 0


def test_flips_that_cancel_in_the_mean_are_still_flips() -> None:
    cells: list[ConditionScore] = []
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
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.as_dict()
    assert d["n_flips"] == rep.n_flips
    assert "net_delta" in d
    assert "directional_balance" in d
    assert d["directional_balance"] is not None


def test_lopsided_and_balanced_flips_are_distinguishable() -> None:
    lopsided = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    assert lopsided.n_flips == 1
    assert lopsided.directional_balance == 1.0

    cells: list[ConditionScore] = []
    for b in (128.0, 512.0):
        cells += _panel({"gemma": -0.05 if b < 300.0 else +0.05, "qwen": +0.05}, b=b)
    balanced = _report(cells)
    assert balanced.n_flips == 1, "only the budget where gemma falls"
    assert balanced.directional_balance == 1.0

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
    rep = ReaderAgreementReport()
    assert rep.flip_rate is None
    assert rep.n_pairs == 0
    assert "No reader pairs" in rep.headline()


def test_net_delta_is_context_not_the_headline() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.as_dict()
    assert d["flip_rate"] > 0
    assert "net_delta" in d


def test_a_flat_reader_agrees_with_nobody_but_flips_with_nobody() -> None:
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


def test_two_points_do_not_yield_a_slope() -> None:
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
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2))
    with pytest.raises(UnsupportedComparisonError):
        compare_readers(responses_from_conditions(cells))

    rep = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    assert rep.flips_by_pair() == {}, "a two-point reader cannot generate a sign flip"
    assert rep.n_pairs == 0
    assert rep.flip_rate is None, "and the rate is None, not a measured 0.0"


def test_two_readers_at_the_same_short_support_compare_normally() -> None:
    rep = compare_readers(
        responses_from_conditions(_panel({"gemma": +0.05, "qwen": -0.05}, ks=(1, 2, 4)))
    )
    assert rep.n_pairs == 1
    assert rep.n_flips == 1
    assert rep.unsupported == []


def test_the_slope_is_per_doubling_not_per_raw_unit() -> None:
    pts = [(1, 0.20), (2, 0.40), (4, 0.60), (8, 0.80)]
    r = depth_response(pts, reader="x", b=1.0, scale="S")
    assert r.slope == pytest.approx(0.20), "each doubling adds 0.20"

    xs = [float(k) for k, _ in pts]
    ys = [s for _, s in pts]
    xbar, ybar = sum(xs) / 4, sum(ys) / 4
    raw = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - xbar) ** 2 for x in xs
    )
    assert abs(raw - 0.20) > 0.05, "the two scales genuinely differ here"


def test_the_slope_matches_a_hand_computed_regression() -> None:
    r = depth_response([(1, 0.0), (2, 1.0), (4, 2.0), (8, 3.0)], reader="x", b=1.0, scale="S")
    assert r.slope == pytest.approx(1.0)


def test_k_zero_is_excluded_rather_than_logging_it() -> None:
    r = depth_response([(0, 0.0), (1, 0.2), (2, 0.4), (4, 0.6)], reader="x", b=1.0, scale="S")
    assert 0 not in r.ks
    assert r.ks == (1, 2, 4)


def test_an_all_one_k_curve_is_flat_not_a_division_error() -> None:
    r = depth_response([(4, 0.2), (4, 0.9), (4, 0.5)], reader="x", b=1.0, scale="S", min_points=2)
    assert r.slope is None
    assert r.sign == FLAT


def test_math_log2_agrees_with_the_fixture_grid() -> None:
    for k in (1, 2, 4, 8, 16, 32):
        assert math.log2(k) == float(k.bit_length() - 1)


def test_a_reader_that_cannot_reach_the_deep_cells_blocks_the_comparison() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    with pytest.raises(UnsupportedComparisonError, match="unequal K supports"):
        compare_readers(responses_from_conditions(cells))


def test_skipping_still_records_what_was_skipped() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    assert len(rep.unsupported) > 0
    assert rep.unsupported_rate > 0
    assert all(c.support != "common" for c in rep.unsupported)


def test_count_and_skip_agree_on_the_bookkeeping() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4))
    cells += _panel({"qwen": -0.05}, ks=(1, 2))
    a = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    b = compare_readers(responses_from_conditions(cells), on_unsupported="count")
    assert len(a.unsupported) == len(b.unsupported)
    assert a.unsupported_rate == b.unsupported_rate


def test_the_shared_support_is_named_which_reader_was_narrower() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    kinds = {c.support for c in rep.unsupported}
    assert kinds <= {"truncated_a", "truncated_b", "truncated_both", "partial_overlap"}


def test_the_unsupported_rate_reaches_the_serialised_report() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    d = compare_readers(responses_from_conditions(cells), on_unsupported="skip").as_dict()
    assert d["n_unsupported"] > 0
    assert d["unsupported_rate"] > 0
    assert d["unsupported"], "the individual skipped comparisons are listed"


def test_the_headline_mentions_the_truncation() -> None:
    cells = _panel({"gemma": +0.05, "phi": +0.04}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    assert rep.n_pairs > 0, "gemma|phi is comparable"
    assert rep.unsupported, "gemma|qwen and phi|qwen are not"
    assert "refused" in rep.headline()
    assert "narrower depth range" in rep.headline()


def test_a_fully_uncomparable_panel_says_so_rather_than_reporting_zero() -> None:
    cells = _panel({"gemma": +0.05}, ks=(1, 2, 4, 8, 16, 32))
    cells += _panel({"qwen": -0.05}, ks=(1, 2, 4))
    rep = compare_readers(responses_from_conditions(cells), on_unsupported="skip")
    assert rep.n_pairs == 0
    assert rep.flip_rate is None
    assert "not attributable across readers" in rep.headline()


def test_an_unknown_support_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_unsupported"):
        compare_readers(
            responses_from_conditions(_panel({"gemma": 0.1, "qwen": -0.1})), on_unsupported="ignore"
        )


def test_the_2606_reference_is_carried_as_a_comparison() -> None:
    assert REFERENCE_FLIP_RATE == 0.31
    d = _report(_panel({"gemma": +0.05, "qwen": -0.05})).as_dict()
    assert d["reference"]["flip_rate"] == 0.31
    assert "not a threshold" in d["reference"]["note"]


def test_a_flip_rate_wildly_different_from_the_reference_is_still_reported() -> None:
    rep = _report(_panel({"gemma": +0.10, "qwen": -0.10}))
    d = rep.as_dict()
    assert d["flip_rate"] != REFERENCE_FLIP_RATE
    assert d["flip_rate"] is not None and d["flip_rate"] > 0


def test_the_stronger_reader_falling_is_the_claim_under_test() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}), strengths=SYNTHETIC_STRENGTHS)
    d = rep.by_strength_direction()
    assert d["stronger_falls"] == rep.n_flips
    assert d["stronger_rises"] == 0


def test_the_synthetic_panel_reproduces_that_pattern() -> None:
    rep = _report(synthetic_conditions(), strengths=SYNTHETIC_STRENGTHS)
    assert rep.n_flips > 0
    assert rep.by_strength_direction()["stronger_falls"] == rep.n_flips


def test_missing_strengths_land_in_unknown_not_in_equal() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}))
    d = rep.by_strength_direction()
    assert d["unknown"] == rep.n_flips
    assert d["equal_strength"] == 0


def test_a_partial_strength_table_does_not_silently_pair() -> None:
    rep = _report(_panel({"gemma": +0.05, "qwen": -0.05}), strengths={"gemma": 0.78})
    assert rep.by_strength_direction()["unknown"] == rep.n_flips


def test_the_strength_breakdown_counts_flips_only() -> None:
    rep = _report(
        _panel({"gemma": +0.05, "qwen": -0.05, "phi": +0.05}),
        strengths=SYNTHETIC_STRENGTHS,
    )
    total = sum(rep.by_strength_direction().values())
    assert total == rep.n_flips


def test_flips_are_kept_when_stratified() -> None:
    cells: list[ConditionScore] = []
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
    rep = _report(synthetic_conditions())
    assert sum(rep.by_scale().values()) == rep.n_pairs
    for scale, n in rep.by_scale().items():
        assert rep.flips_by_scale().get(scale, 0) <= n


def test_duplicate_rows_at_one_k_are_averaged_not_double_counted() -> None:
    cells = [
        ConditionScore(reader="gemma", k=1, b=128.0, scale="S", score=0.20),
        ConditionScore(reader="gemma", k=1, b=128.0, scale="S", score=0.40),
        ConditionScore(reader="gemma", k=2, b=128.0, scale="S", score=0.50),
        ConditionScore(reader="gemma", k=4, b=128.0, scale="S", score=0.70),
    ]
    out = responses_from_conditions(cells)
    assert len(out) == 1
    assert out[0].n_points == 3, "three distinct K, not four rows"


def test_responses_are_grouped_by_reader_budget_and_scale() -> None:
    cells: list[ConditionScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0):
            cells += _panel({"gemma": 0.05, "qwen": 0.05}, b=b, scale=scale)
    out = responses_from_conditions(cells)
    assert len(out) == 2 * 2 * 2
    assert {(r.b, r.scale, r.reader) for r in out} == {
        (b, scale, reader)
        for scale in ("S", "M")
        for b in (128.0, 512.0)
        for reader in ("gemma", "qwen")
    }


def test_arms_are_kept_apart() -> None:
    cells = _panel({"gemma": +0.05, "qwen": -0.05})
    cells += [
        ConditionScore(reader=c.reader, k=c.k, b=c.b, scale=c.scale, score=c.score, arm="B")
        for c in _panel({"gemma": -0.05, "qwen": +0.05})
    ]
    out = responses_from_conditions(cells)
    assert {(r.reader, r.arm) for r in out} == {
        (reader, arm) for reader in ("gemma", "qwen") for arm in ("A", "B")
    }


def test_a_response_serialises_with_its_ks() -> None:
    r = depth_response([(1, 0.2), (2, 0.4), (4, 0.6)], reader="gemma", b=128.0, scale="S", arm="A")
    d = r.as_dict()
    assert d["ks"] == [1, 2, 4]
    assert d["n_points"] == 3
    assert d["sign_name"] == "rises"


def test_the_report_is_json_serialisable() -> None:
    json.dumps(_report(synthetic_conditions(), strengths=SYNTHETIC_STRENGTHS).as_dict())


def test_the_report_is_deterministic() -> None:
    a = _report(synthetic_conditions(), strengths=SYNTHETIC_STRENGTHS).as_dict()
    b = _report(list(reversed(synthetic_conditions())), strengths=SYNTHETIC_STRENGTHS).as_dict()
    assert a == b


def test_the_three_signs_are_the_three_named_states() -> None:
    assert (FALLS, FLAT, RISES) == (-1, 0, 1)


def test_higher_is_better_is_stated_in_the_sign_names() -> None:
    up = depth_response([(1, 0.2), (2, 0.4), (4, 0.6)], reader="x", b=1.0, scale="S")
    assert up.sign == RISES and up.sign_name == "rises"
    down = depth_response([(1, 0.6), (2, 0.4), (4, 0.2)], reader="x", b=1.0, scale="S")
    assert down.sign == FALLS and down.sign_name == "falls"


def test_a_depth_response_can_be_built_directly_for_a_test() -> None:
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
