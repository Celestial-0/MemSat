"""The P5 GO / NO-GO gate, all four branches.

Architecture §P5: proceed if ANY of (i)-(iv) holds; NONE is a bounded negative and
a pre-agreed pivot.

The tests concentrate on the ways a gate can be made to pass:

  * branch (iv) is satisfied by an EMPTY degenerate list when the check never ran
  * branch (i) passes on judge noise that was assumed to be zero
  * a missing measurement is treated as a pass
  * cancelling noise in a flat grid is reported as "nothing found" rather than
    "below threshold"
"""

from __future__ import annotations

import json

import pytest

from msat.gate import ACCURACY_THRESHOLD_PP, RECALL_THRESHOLD_PP, GateOutcome, gate

CLEAN = {"accuracy_by_cell": {}, "recall_by_k": {}, "stratified": None, "degenerate_cells": []}


def _gate(**kw):
    base = dict(CLEAN)
    base.update(kw)
    return gate(**base)


# ── the four branches exist and are reported ───────────────────────────────────


def test_all_four_branches_are_always_reported() -> None:
    """A branch that is only computed when it passes cannot be shown to have failed."""
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6}, recall_by_k={1: 0.3, 2: 0.5})
    assert len(v.branches) == 4
    assert {str(b.branch) for b in v.branches} == {
        "(i) accuracy varies >= 5pp",
        "(ii) recall@K varies >= 10pp",
        "(iii) stratified structure",
        "(iv) K x B interaction estimable",
    }


def test_the_verdict_is_json_serialisable() -> None:
    """`go_no_go.json` is the deliverable. A gate that only prints lives in a
    scrollback buffer."""
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6})
    json.loads(json.dumps(v.as_dict()))


def test_the_pivot_list_is_carried_with_the_verdict() -> None:
    """A bounded negative is a result, and the pivot is decided IN ADVANCE."""
    d = _gate().as_dict()
    assert d["outcome"] == "BOUNDED_NEGATIVE"
    assert d["n_interaction_cells"] == 0
    assert len(d["pivot_if_negative"]) == 4
    assert "IN ADVANCE" in d["note"]


def test_branch_iv_alone_is_enough_because_it_is_a_design_property() -> None:
    """Architecturally §P5 says proceed if ANY branch holds, and (iv) is decided
    BEFORE the run - it is a property of the design, not of its results.

    So a fully estimable grid GOes on (iv) alone with no measurements at all. That
    is the spec, and the reason `n_interaction_cells` exists: the same gate with
    the estimability check UNRUN must not also GO.
    """
    assert _gate(n_interaction_cells=108).outcome is GateOutcome.GO
    assert _gate(n_interaction_cells=108).passed_branches == ["(iv) K x B interaction estimable"]
    assert _gate().outcome is GateOutcome.BOUNDED_NEGATIVE


# ── branch (iv), the headline, and it is checked first ─────────────────────────


def test_a_single_degenerate_cell_fails_branch_iv() -> None:
    v = _gate(degenerate_cells=[{"K": 64, "B": 128, "scale": "S"}])
    b = next(b for b in v.branches if str(b.branch).startswith("(iv)"))
    assert b.passed is False
    assert v.interaction_estimable is False


def test_branch_iv_failure_is_called_out_even_when_another_branch_passes() -> None:
    """An interaction estimated over a grid with holes in it is not the interaction
    this design declared."""
    v = _gate(accuracy_by_cell={"a": 0.2, "b": 0.9}, degenerate_cells=[{"K": 64}])
    assert v.proceed is True, "branch (i) genuinely passed"
    assert v.interaction_estimable is False
    assert any("Branch (iv) did not pass" in n for n in v.notes)


def test_a_fully_checked_clean_grid_passes_branch_iv() -> None:
    assert _gate(n_interaction_cells=108).interaction_estimable is True


# ── branch (i) is measured against judge noise, not against zero ───────────────


def test_accuracy_spread_below_five_pp_fails() -> None:
    v = _gate(accuracy_by_cell={"a": 0.500, "b": 0.540})
    b = next(b for b in v.branches if str(b.branch).startswith("(i)"))
    assert b.passed is False
    assert b.observed["spread_pp"] == pytest.approx(4.0)


def test_accuracy_spread_at_or_above_five_pp_passes() -> None:
    v = _gate(accuracy_by_cell={"a": 0.500, "b": 0.550})
    assert next(b for b in v.branches if str(b.branch).startswith("(i)")).passed


def test_judge_noise_raises_the_threshold() -> None:
    """6pp of variation means nothing if two judges disagree by 8pp on identical
    outputs. Assuming judge noise is zero is how a flat grid is reported as a finding."""
    cells = {"a": 0.500, "b": 0.560}  # 6pp of spread
    assert next(
        b for b in _gate(accuracy_by_cell=cells).branches if str(b.branch).startswith("(i)")
    ).passed
    noisy = _gate(accuracy_by_cell=cells, judge_floor_pp=8.0)
    b = next(b for b in noisy.branches if str(b.branch).startswith("(i)"))
    assert b.passed is False
    assert b.threshold["judge_floor_pp"] == 8.0
    assert pytest.approx(13.0) == ACCURACY_THRESHOLD_PP + 8.0


# ── branch (ii) ────────────────────────────────────────────────────────────────


def test_recall_spread_uses_a_ten_pp_threshold() -> None:
    assert (
        next(
            b
            for b in _gate(recall_by_k={1: 0.30, 2: 0.39}).branches
            if str(b.branch).startswith("(ii)")
        ).passed
        is False
    )
    assert (
        next(
            b
            for b in _gate(recall_by_k={1: 0.30, 2: 0.42}).branches
            if str(b.branch).startswith("(ii)")
        ).passed
        is True
    )
    assert RECALL_THRESHOLD_PP == 10.0
    assert ACCURACY_THRESHOLD_PP == 5.0


# ── branch (iii) ───────────────────────────────────────────────────────────────


def test_stratified_structure_passes_only_on_the_declared_rule() -> None:
    """multi-session rising while single-session is flat or declining."""
    up = {"multi_session_slope": 0.08, "single_session_slope": 0.00, "structure_detected": True}
    both_up = {
        "multi_session_slope": 0.08,
        "single_session_slope": 0.07,
        "structure_detected": False,
    }
    assert next(
        b for b in _gate(stratified=up).branches if str(b.branch).startswith("(iii)")
    ).passed
    assert not next(
        b for b in _gate(stratified=both_up).branches if str(b.branch).startswith("(iii)")
    ).passed


def test_a_stratification_that_was_not_run_cannot_pass() -> None:
    v = _gate()
    b = next(x for x in v.branches if str(x.branch).startswith("(iii)"))
    assert b.passed is False
    assert "not computed" in b.detail


# ── absent data is not a pass ──────────────────────────────────────────────────


def test_missing_accuracy_fails_branch_one_and_says_why() -> None:
    v = _gate()
    b = next(x for x in v.branches if str(x.branch).startswith("(i)"))
    assert b.passed is False
    assert b.observed is None
    assert any("missing measurement is not a pass" in n for n in v.notes)


def test_missing_recall_fails_branch_two_and_says_why() -> None:
    v = _gate()
    assert any("no recall data" in n for n in v.notes)


def test_an_unbalanced_grid_is_a_pass_and_a_degenerate_cell_is_a_fail() -> None:
    """Both need `n_interaction_cells > 0` to mean anything at all."""
    assert _gate(n_interaction_cells=108).interaction_estimable is True
    assert _gate(n_interaction_cells=108, degenerate_cells=[{"K": 64}]).interaction_estimable is (
        False
    )


def test_an_uncheckable_grid_is_not_an_empty_degenerate_list() -> None:
    """`degenerate_cells == []` on a grid nobody examined is not estimability.

    Without `n_interaction_cells`, a gate called with no data passes branch (iv) -
    the headline branch - and returns GO. That is the single worst failure this
    module could have.
    """
    v = _gate()  # no data of any kind
    b = next(x for x in v.branches if str(x.branch).startswith("(iv)"))
    assert b.passed is False
    assert "NOT EVALUATED" in b.detail
    assert v.interaction_estimable is False
    assert v.outcome is GateOutcome.BOUNDED_NEGATIVE
    assert any("Branch (iv) did not pass" in n for n in v.notes)


def test_a_gate_with_no_data_at_all_is_a_bounded_negative() -> None:
    """Not a crash, and not a GO. A gate that crashes on missing input cannot be
    used to report that the input is missing."""
    v = _gate()
    assert v.outcome is GateOutcome.BOUNDED_NEGATIVE
    assert v.proceed is False


# ── the aggregate verdict ──────────────────────────────────────────────────────


def test_any_passing_branch_is_enough() -> None:
    v = _gate(recall_by_k={1: 0.30, 2: 0.50})
    assert v.proceed is True
    assert len(v.passed_branches) == 1


def test_all_four_passing_reports_all_four() -> None:
    v = _gate(
        accuracy_by_cell={"a": 0.5, "b": 0.9},
        recall_by_k={1: 0.1, 2: 0.9},
        stratified={
            "multi_session_slope": 0.1,
            "single_session_slope": 0.0,
            "structure_detected": True,
        },
        n_interaction_cells=108,
    )
    assert len(v.passed_branches) == 4
    assert v.proceed is True


def test_the_verdict_records_the_scale_it_was_decided_on() -> None:
    """n_cells and n_questions, so a verdict can be matched to the run it belongs to."""
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6}, n_questions=500)
    assert v.as_dict()["n_cells"] == 2
    assert v.as_dict()["n_questions"] == 500


def test_the_cell_count_is_the_number_of_cells_not_the_number_of_calls() -> None:
    """Three replicates per cell are not three cells. Counting calls would make a
    12-cell grid look like 36 cells and inflate the apparent evidence."""
    v = _gate(accuracy_by_cell={f"c{i}": 0.5 for i in range(12)})
    assert v.n_cells == 12
    assert v.as_dict()["n_cells"] == 12
