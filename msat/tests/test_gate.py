from __future__ import annotations

import json

import pytest

from msat.gate import ACCURACY_THRESHOLD_PP, RECALL_THRESHOLD_PP, GateOutcome, gate

CLEAN = {"accuracy_by_cell": {}, "recall_by_k": {}, "stratified": None, "degenerate_cells": []}


def _gate(**kw):
    base = dict(CLEAN)
    base.update(kw)
    return gate(**base)


def test_all_four_branches_are_always_reported() -> None:
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6}, recall_by_k={1: 0.3, 2: 0.5})
    assert len(v.branches) == 4
    assert {str(b.branch) for b in v.branches} == {
        "(i) accuracy varies >= 5pp",
        "(ii) recall@K varies >= 10pp",
        "(iii) stratified structure",
        "(iv) K x B interaction estimable",
    }


def test_the_verdict_is_json_serialisable() -> None:
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6})
    json.loads(json.dumps(v.as_dict()))


def test_the_pivot_list_is_carried_with_the_verdict() -> None:
    d = _gate().as_dict()
    assert d["outcome"] == "BOUNDED_NEGATIVE"
    assert d["n_interaction_conditions"] == 0
    assert len(d["pivot_if_negative"]) == 4
    assert "IN ADVANCE" in d["note"]


def test_branch_iv_alone_is_enough_because_it_is_a_design_property() -> None:
    assert _gate(n_interaction_cells=108).outcome is GateOutcome.GO
    assert _gate(n_interaction_cells=108).passed_branches == ["(iv) K x B interaction estimable"]
    assert _gate().outcome is GateOutcome.BOUNDED_NEGATIVE


def test_a_single_degenerate_cell_fails_branch_iv() -> None:
    v = _gate(degenerate_cells=[{"K": 64, "B": 128, "scale": "S"}])
    b = next(b for b in v.branches if str(b.branch).startswith("(iv)"))
    assert b.passed is False
    assert v.interaction_estimable is False


def test_branch_iv_failure_is_called_out_even_when_another_branch_passes() -> None:
    v = _gate(accuracy_by_cell={"a": 0.2, "b": 0.9}, degenerate_cells=[{"K": 64}])
    assert v.proceed is True, "branch (i) genuinely passed"
    assert v.interaction_estimable is False
    assert any("Branch (iv) did not pass" in n for n in v.notes)


def test_a_fully_checked_clean_grid_passes_branch_iv() -> None:
    assert _gate(n_interaction_cells=108).interaction_estimable is True


def test_accuracy_spread_below_five_pp_fails() -> None:
    v = _gate(accuracy_by_cell={"a": 0.500, "b": 0.540})
    b = next(b for b in v.branches if str(b.branch).startswith("(i)"))
    assert b.passed is False
    assert b.observed["spread_pp"] == pytest.approx(4.0)


def test_accuracy_spread_at_or_above_five_pp_passes() -> None:
    v = _gate(accuracy_by_cell={"a": 0.500, "b": 0.550})
    assert next(b for b in v.branches if str(b.branch).startswith("(i)")).passed


def test_judge_noise_raises_the_threshold() -> None:
    cells = {"a": 0.500, "b": 0.560}
    assert next(
        b for b in _gate(accuracy_by_cell=cells).branches if str(b.branch).startswith("(i)")
    ).passed
    noisy = _gate(accuracy_by_cell=cells, judge_floor_pp=8.0)
    b = next(b for b in noisy.branches if str(b.branch).startswith("(i)"))
    assert b.passed is False
    assert b.threshold["judge_floor_pp"] == 8.0
    assert pytest.approx(13.0) == ACCURACY_THRESHOLD_PP + 8.0


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


def test_stratified_structure_passes_only_on_the_declared_rule() -> None:
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
    assert _gate(n_interaction_cells=108).interaction_estimable is True
    assert _gate(n_interaction_cells=108, degenerate_cells=[{"K": 64}]).interaction_estimable is (
        False
    )


def test_an_uncheckable_grid_is_not_an_empty_degenerate_list() -> None:
    v = _gate()
    b = next(x for x in v.branches if str(x.branch).startswith("(iv)"))
    assert b.passed is False
    assert "NOT EVALUATED" in b.detail
    assert v.interaction_estimable is False
    assert v.outcome is GateOutcome.BOUNDED_NEGATIVE
    assert any("Branch (iv) did not pass" in n for n in v.notes)


def test_a_gate_with_no_data_at_all_is_a_bounded_negative() -> None:
    v = _gate()
    assert v.outcome is GateOutcome.BOUNDED_NEGATIVE
    assert v.proceed is False


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
    v = _gate(accuracy_by_cell={"a": 0.5, "b": 0.6}, n_questions=500)
    assert v.as_dict()["n_conditions"] == 2
    assert v.as_dict()["n_questions"] == 500


def test_the_cell_count_is_the_number_of_cells_not_the_number_of_calls() -> None:
    v = _gate(accuracy_by_cell={f"c{i}": 0.5 for i in range(12)})
    assert v.n_conditions == 12
    assert v.as_dict()["n_conditions"] == 12


def test_gate_accepts_canonical_condition_parameters() -> None:
    v = gate(
        accuracy_by_condition={"c1": 0.2, "c2": 0.8},
        recall_by_k={1: 0.2, 2: 0.5},
        stratified=None,
        degenerate_conditions=[],
        n_interaction_conditions=108,
    )
    assert v.n_conditions == 2
    assert v.n_interaction_conditions == 108
    assert v.interaction_estimable is True
    assert v.as_dict()["n_conditions"] == 2
    assert v.as_dict()["n_interaction_conditions"] == 108
