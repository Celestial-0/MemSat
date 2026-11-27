from __future__ import annotations

from collections.abc import Sequence

import pytest

from msat.judges.kappa import (
    CAUTIONARY_KAPPA,
    agreement_matrix_table,
    base_rates,
    cohen_kappa,
    kappa_verdict,
    panel_agreement,
)
from msat.judges.verdicts import UnscorableError, Verdict, as_verdict

K = ("A|1|512", "q1")


def _v(items: Sequence[str]) -> dict:
    bad = [c for c in items if c not in {"c", "i"}]
    if bad:
        raise ValueError(f"verdict fixtures use 'c'/'i' only; got {bad}")
    return {
        (K[0], f"q{i}"): (Verdict.CORRECT if c == "c" else Verdict.INCORRECT)
        for i, c in enumerate(items)
    }


def _cells(n_cc: int, n_ci: int, n_ic: int, n_ii: int) -> dict:
    items: dict[tuple[str, str], Verdict] = {}
    i = 0
    for _a, _b, n in (
        (Verdict.CORRECT, Verdict.CORRECT, n_cc),
        (Verdict.CORRECT, Verdict.INCORRECT, n_ci),
        (Verdict.INCORRECT, Verdict.CORRECT, n_ic),
        (Verdict.INCORRECT, Verdict.INCORRECT, n_ii),
    ):
        for _ in range(n):
            items[(K[0], f"q{i}")] = _a
            i += 1
    return items


def _vs(text: str) -> dict:
    return _v(list(text))


def _judge_b(n_cc: int, n_ci: int, n_ic: int, n_ii: int, a: dict) -> dict:
    keys = sorted(a, key=lambda k: int(k[1][1:]))
    out = dict(a)
    seq = (
        [Verdict.CORRECT] * n_cc
        + [Verdict.INCORRECT] * n_ci
        + [Verdict.CORRECT] * n_ic
        + [Verdict.INCORRECT] * n_ii
    )
    for k, v in zip(keys, seq, strict=True):
        out[k] = v
    return out


def test_perfect_agreement_with_a_balanced_base_rate_gives_kappa_one() -> None:
    r = cohen_kappa(_vs("cici"), _vs("cici"))
    assert r.p_observed == pytest.approx(1.0)
    assert r.p_expected == pytest.approx(0.5)
    assert r.kappa == pytest.approx(1.0)


def test_the_well_known_textbook_table() -> None:
    a = _cells(8, 2, 8, 2)
    b = _judge_b(8, 2, 8, 2, a)
    r = cohen_kappa(a, b)
    assert r.n == 20
    assert r.matrix[(Verdict.CORRECT, Verdict.CORRECT)] == 8
    assert r.matrix[(Verdict.CORRECT, Verdict.INCORRECT)] == 2
    assert r.matrix[(Verdict.INCORRECT, Verdict.CORRECT)] == 8
    assert r.matrix[(Verdict.INCORRECT, Verdict.INCORRECT)] == 2
    assert r.p_observed == pytest.approx(0.5)
    assert r.p_expected == pytest.approx(0.5)
    assert r.kappa == pytest.approx(0.0, abs=1e-9)


def test_high_agreement_on_a_skewed_base_rate_is_corrected_down() -> None:
    a = _v(["c"] + ["i"] * 9)
    b = _v(["c", "c"] + ["i"] * 8)
    r = cohen_kappa(a, b)
    assert r.p_observed == pytest.approx(0.9)
    assert r.p_expected == pytest.approx(0.74)
    assert r.kappa == pytest.approx(0.16 / 0.26)


def test_total_disagreement_gives_negative_one() -> None:
    r = cohen_kappa(_vs("cici"), _vs("icic"))
    assert r.kappa == pytest.approx(-1.0)


def test_a_perfectly_skewed_matrix_leaves_kappa_undefined() -> None:
    r = cohen_kappa(_vs("iiii"), _vs("iiii"))
    assert r.kappa is None
    assert r.p_observed == pytest.approx(1.0)
    assert "does not exist" in r.undefined_reason


def test_both_judges_correct_on_everything_is_also_undefined() -> None:
    r = cohen_kappa(_vs("cccc"), _vs("cccc"))
    assert r.kappa is None
    assert "does not exist" in r.undefined_reason


def test_no_overlap_is_undefined_for_a_different_reason() -> None:
    r = cohen_kappa({}, {})
    assert r.kappa is None
    assert "no items" in r.undefined_reason


def test_the_agreement_matrix_is_returned_with_the_statistic() -> None:
    r = cohen_kappa(_v("cici"), _v("cici"))
    cells = r.cells()
    assert cells["correct|correct"] == 2
    assert cells["incorrect|incorrect"] == 2
    assert cells["correct|incorrect"] == 0


def test_the_matrix_is_asymmetric_where_the_judges_are() -> None:
    a = _cells(3, 0, 0, 3)
    b = _judge_b(1, 2, 0, 3, a)
    r = cohen_kappa(a, b)
    assert r.matrix[(Verdict.CORRECT, Verdict.INCORRECT)] == 2
    assert r.matrix[(Verdict.INCORRECT, Verdict.CORRECT)] == 0
    assert r.matrix[(Verdict.INCORRECT, Verdict.INCORRECT)] == 3


def test_items_scored_by_one_judge_only_are_excluded_and_counted() -> None:
    a = _vs("cici")
    b = dict(a)
    b[("A|1|512", "q9")] = Verdict.CORRECT
    r = cohen_kappa(a, b)
    assert r.n == 4
    assert r.n_excluded == 1
    assert r.excluded_reasons["scored by only one judge"] == 1


def test_explicitly_excluded_items_are_counted_with_their_reason() -> None:
    a = _vs("cici")
    b = _vs("cici")
    r = cohen_kappa(a, b, excluded={("A|1|512", "q2"): "judge declined"})
    assert r.n == 3
    assert r.excluded_reasons["judge declined"] == 1
    assert r.fraction_excluded == pytest.approx(0.25)


def test_dropping_items_changes_kappa_and_the_change_is_visible() -> None:
    full = cohen_kappa(_vs("cici"), _vs("ccic"))
    part = cohen_kappa(_vs("cici"), _vs("ccic"), excluded={("A|1|512", "q0"): "refused"})
    assert part.kappa != full.kappa
    assert part.n_excluded == 1


def test_the_bar_is_a_number_in_the_module_not_a_comment() -> None:
    assert CAUTIONARY_KAPPA == 0.118


def test_a_kappa_below_the_bar_says_so() -> None:
    r = cohen_kappa(_vs("cici"), _vs("iiii"))
    v = kappa_verdict(r)
    assert v["verdict"].startswith("BELOW")


def test_a_kappa_clearing_the_bar_says_so() -> None:
    r = cohen_kappa(_vs("cici"), _vs("cici"))
    assert kappa_verdict(r)["verdict"].endswith("clears the 0.118 bar.")


def test_an_undefined_kappa_never_reads_as_a_pass() -> None:
    v = kappa_verdict(cohen_kappa(_vs("iiii"), _vs("iiii")))
    assert v["kappa"] is None
    assert v["verdict"].startswith("UNDEFINED")


def test_a_heavy_exclusion_warns_even_when_kappa_clears() -> None:
    r = cohen_kappa(_vs("cici"), _vs("cici"), excluded={("A|1|512", "q0"): "refused"})
    assert "minority" in kappa_verdict(r)["verdict"]


def test_base_rates_are_reported_because_kappa_alone_is_ambiguous() -> None:
    rates = base_rates({"aligned_wrong": _vs("iii"), "balanced": _vs("ici")})
    assert rates["aligned_wrong"] == pytest.approx(0.0)
    assert rates["balanced"] == pytest.approx(1 / 3)


def test_two_judges_give_one_pair() -> None:
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici")})
    assert out["n_pairs"] == 1
    assert out["kappa_mean"] == pytest.approx(1.0)
    assert out["panel_usable"] is True


def test_three_judges_give_three_pairs_and_a_spread() -> None:
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici"), "c": _vs("iiii")})
    assert out["n_pairs"] == 3
    assert out["kappa_min"] < out["kappa_max"]
    assert out["spread"] > 0
    assert "outlier" in out["spread_note"]


def test_the_worst_pair_decides_whether_the_panel_is_usable() -> None:
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici"), "c": _vs("iiii")})
    assert out["panel_usable"] is False


def test_the_n_by_n_matrix_is_counts_not_rates() -> None:
    m = agreement_matrix_table({"a": _vs("cici"), "b": _vs("cici")})
    assert m["a"]["a"]["correct|correct"] == 2
    assert m["a"]["b"]["incorrect|incorrect"] == 2


@pytest.mark.parametrize("raw", ["correct", "CORRECT", " true ", True, 1, "yes"])
def test_clear_verdicts_are_accepted(raw) -> None:
    assert as_verdict(raw) is Verdict.CORRECT


@pytest.mark.parametrize("raw", ["incorrect", "false", "no", False, 0, "  INCORRECT "])
def test_clear_negative_verdicts_are_accepted(raw) -> None:
    assert as_verdict(raw) is Verdict.INCORRECT


@pytest.mark.parametrize(
    "raw",
    [
        "the answer looks correct",
        "probably right",
        "correct-ish",
        "1 correct, 1 incorrect",
        "",
        None,
        0.5,
    ],
)
def test_hedged_output_is_refused_not_coerced(raw) -> None:
    with pytest.raises(UnscorableError, match="Hedged output is not a verdict"):
        as_verdict(raw)
