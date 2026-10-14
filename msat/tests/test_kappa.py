"""Cohen's kappa against HAND-COMPUTED tables (PRD §R8.2).

Every fixture here is small enough to work the arithmetic out on paper. A kappa
implementation tested with another kappa implementation proves only that the two
agree with each other; the bar in 2511.02246 is a number someone will check.
"""

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
    """Verdicts from 'c'/'i' characters. Index i becomes qid f"q{i}".

    A validated sequence rather than a bare string, because `"iiii c"` has eleven
    characters and reads as ten items — a space silently became an INCORRECT and
    every arithmetic fixture built on it was wrong.
    """
    bad = [c for c in items if c not in {"c", "i"}]
    if bad:
        raise ValueError(f"verdict fixtures use 'c'/'i' only; got {bad}")
    return {
        (K[0], f"q{i}"): (Verdict.CORRECT if c == "c" else Verdict.INCORRECT)
        for i, c in enumerate(items)
    }


def _cells(n_cc: int, n_ci: int, n_ic: int, n_ii: int) -> dict:
    """A verdict map defined by its AGREEMENT CELLS, not by a string.

    Writing the cells out removes the whole class of arithmetic slip in the
    fixtures: `ccciii` vs `ciiiii` has to be counted by eye, while
    `(8, 0, 0, 2)` states the table directly and can be checked against the
    expected marginals without counting characters.
    """
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
    """A balanced fixture from a space-free 'c'/'i' string.

    Use this only where the marginals are obviously equal, so the arithmetic is
    visible without counting. Anything with an asymmetric base rate should be built
    with `_cells`, where the table can be checked directly.
    """
    return _v(list(text))


def _judge_b(n_cc: int, n_ci: int, n_ic: int, n_ii: int, a: dict) -> dict:
    """Judge B's verdicts, aligned positionally with a fixture built by `_cells`."""
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


# ── the arithmetic, by hand ────────────────────────────────────────────────────


def test_perfect_agreement_with_a_balanced_base_rate_gives_kappa_one() -> None:
    """A: c i c i / B: c i c i

    p_o = 4/4 = 1.0.  Marginals are (2,2) both ways, so
    p_e = (2/4)(2/4) + (2/4)(2/4) = 0.25 + 0.25 = 0.5.
    kappa = (1.0 - 0.5) / (1 - 0.5) = 1.0
    """
    r = cohen_kappa(_vs("cici"), _vs("cici"))
    assert r.p_observed == pytest.approx(1.0)
    assert r.p_expected == pytest.approx(0.5)
    assert r.kappa == pytest.approx(1.0)


def test_the_well_known_textbook_table() -> None:
    """A 2x2 table read straight off the marginals.

        A correct  A wrong
    B correct      8        8      -> B correct = 16
    B wrong        2        2      -> B wrong    =  4
    A correct = 10, A wrong = 10.

    p_o = (8 + 2) / 20 = 0.5
    p_e = (10/20)(16/20) + (10/20)(4/20) = 0.4 + 0.1 = 0.5
    kappa = 0 / 0.5 = 0
    """
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
    """The whole reason kappa exists.

    A: 1 correct, 9 incorrect.  B: 2 correct, 8 incorrect.  They agree on 9 of 10.
    p_o = 9/10 = 0.9
    p_e = (1/10)(2/10) + (9/10)(8/10) = 0.02 + 0.72 = 0.74
    kappa = (0.9 - 0.74) / (1 - 0.74) = 0.16 / 0.26 = 0.615

    Percent agreement 0.90, kappa 0.615. The correction IS the point: 90% raw
    agreement looks excellent and mostly reflects the base rate.
    """
    a = _v(["c"] + ["i"] * 9)
    b = _v(["c", "c"] + ["i"] * 8)
    r = cohen_kappa(a, b)
    assert r.p_observed == pytest.approx(0.9)
    assert r.p_expected == pytest.approx(0.74)
    assert r.kappa == pytest.approx(0.16 / 0.26)


def test_total_disagreement_gives_negative_one() -> None:
    """Every verdict swapped. p_o = 0 and p_e = 0.5, so kappa = -1."""
    r = cohen_kappa(_vs("cici"), _vs("icic"))
    assert r.kappa == pytest.approx(-1.0)


def test_a_perfectly_skewed_matrix_leaves_kappa_undefined() -> None:
    """Both judges say INCORRECT on everything: p_e = 1, denominator is zero.

    The statistic does not exist. Returning 0.0 or 1.0 misreports it, and crashing
    means the panel cannot be reported at all.
    """
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


# ── the matrix is stored, not just the statistic ───────────────────────────────


def test_the_agreement_matrix_is_returned_with_the_statistic() -> None:
    """kappa is a symptom; the matrix is the diagnosis. A reader cannot tell
    'misaligned on the rare class' from 'random' with kappa alone."""
    r = cohen_kappa(_v("cici"), _v("cici"))
    cells = r.cells()
    assert cells["correct|correct"] == 2
    assert cells["incorrect|incorrect"] == 2
    assert cells["correct|incorrect"] == 0


def test_the_matrix_is_asymmetric_where_the_judges_are() -> None:
    """correct->incorrect and incorrect->correct are different cells with different
    meanings, and collapsing them is how a directional bias disappears.

    6 items: judge A says correct on q0-q2; judge B says correct on q0 only.
    So 2 rows go correct->incorrect and none go the other way.
    """
    a = _cells(3, 0, 0, 3)
    b = _judge_b(1, 2, 0, 3, a)
    r = cohen_kappa(a, b)
    assert r.matrix[(Verdict.CORRECT, Verdict.INCORRECT)] == 2
    assert r.matrix[(Verdict.INCORRECT, Verdict.CORRECT)] == 0
    assert r.matrix[(Verdict.INCORRECT, Verdict.INCORRECT)] == 3


# ── exclusion, and the count of what was dropped ───────────────────────────────


def test_items_scored_by_one_judge_only_are_excluded_and_counted() -> None:
    """Silently ignoring them would let a judge improve kappa by scoring fewer items."""
    a = _vs("cici")
    b = dict(a)
    b[("A|1|512", "q9")] = Verdict.CORRECT  # only judge B scored it
    r = cohen_kappa(a, b)
    assert r.n == 4
    assert r.n_excluded == 1
    assert r.excluded_reasons["scored by only one judge"] == 1


def test_explicitly_excluded_items_are_counted_with_their_reason() -> None:
    """Dropping refusals silently makes kappa look better as refusals rise."""
    a = _vs("cici")
    b = _vs("cici")
    r = cohen_kappa(a, b, excluded={("A|1|512", "q2"): "judge declined"})
    assert r.n == 3
    assert r.excluded_reasons["judge declined"] == 1
    assert r.fraction_excluded == pytest.approx(0.25)


def test_dropping_items_changes_kappa_and_the_change_is_visible() -> None:
    """The reason the exclusion count is reported beside every kappa."""
    full = cohen_kappa(_vs("cici"), _vs("ccic"))
    part = cohen_kappa(_vs("cici"), _vs("ccic"), excluded={("A|1|512", "q0"): "refused"})
    assert part.kappa != full.kappa
    assert part.n_excluded == 1


# ── the verdict against 2511.02246 ─────────────────────────────────────────────


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
    """A panel that is aligned by being uniformly wrong must not be reported as
    agreeing."""
    v = kappa_verdict(cohen_kappa(_vs("iiii"), _vs("iiii")))
    assert v["kappa"] is None
    assert v["verdict"].startswith("UNDEFINED")


def test_a_heavy_exclusion_warns_even_when_kappa_clears() -> None:
    r = cohen_kappa(_vs("cici"), _vs("cici"), excluded={("A|1|512", "q0"): "refused"})
    assert "minority" in kappa_verdict(r)["verdict"]


# ── the panel ──────────────────────────────────────────────────────────────────


def test_base_rates_are_reported_because_kappa_alone_is_ambiguous() -> None:
    """Two panels can share a kappa with opposite failure modes."""
    rates = base_rates({"aligned_wrong": _vs("iii"), "balanced": _vs("ici")})
    assert rates["aligned_wrong"] == pytest.approx(0.0)
    assert rates["balanced"] == pytest.approx(1 / 3)


def test_two_judges_give_one_pair() -> None:
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici")})
    assert out["n_pairs"] == 1
    assert out["kappa_mean"] == pytest.approx(1.0)
    assert out["panel_usable"] is True


def test_three_judges_give_three_pairs_and_a_spread() -> None:
    """A wide spread means one judge is an outlier, and averaging it away would hide
    the panel that needs re-thinking."""
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici"), "c": _vs("iiii")})
    assert out["n_pairs"] == 3
    assert out["kappa_min"] < out["kappa_max"]
    assert out["spread"] > 0
    assert "outlier" in out["spread_note"]


def test_the_worst_pair_decides_whether_the_panel_is_usable() -> None:
    out = panel_agreement({"a": _vs("cici"), "b": _vs("cici"), "c": _vs("iiii")})
    assert out["panel_usable"] is False


def test_the_n_by_n_matrix_is_counts_not_rates() -> None:
    """A rate over a different number of scored items per judge is not comparable,
    and the counts make the denominators visible."""
    m = agreement_matrix_table({"a": _vs("cici"), "b": _vs("cici")})
    assert m["a"]["a"]["correct|correct"] == 2
    assert m["a"]["b"]["incorrect|incorrect"] == 2


# ── verdict coercion is deliberately narrow ────────────────────────────────────


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
    """A judge that hedges is a judge whose reliability we are trying to measure.
    Reading its hedges as verdicts manufactures agreement and inflates kappa."""
    with pytest.raises(UnscorableError, match="Hedged output is not a verdict"):
        as_verdict(raw)
