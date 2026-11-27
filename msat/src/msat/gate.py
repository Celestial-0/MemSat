"""The GO / NO-GO gate and its four branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GateOutcome(StrEnum):
    GO = "GO"
    BOUNDED_NEGATIVE = "BOUNDED_NEGATIVE"


class Branch(StrEnum):
    ACCURACY = "(i) accuracy varies >= 5pp"
    RECALL = "(ii) recall@K varies >= 10pp"
    STRATIFIED = "(iii) stratified structure"
    INTERACTION = "(iv) K x B interaction estimable"


ACCURACY_THRESHOLD_PP = 5.0
RECALL_THRESHOLD_PP = 10.0


@dataclass(frozen=True, slots=True)
class BranchResult:
    branch: Branch
    passed: bool
    observed: Any
    threshold: Any
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": str(self.branch),
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class GateVerdict:
    outcome: GateOutcome
    branches: list[BranchResult] = field(default_factory=list)
    degenerate_conditions: list[dict[str, Any]] = field(default_factory=list)
    judge_floor_pp: float = 0.0
    n_conditions: int = 0
    n_questions: int = 0
    n_interaction_conditions: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def proceed(self) -> bool:
        return self.outcome is GateOutcome.GO

    @property
    def interaction_estimable(self) -> bool:
        return self.n_interaction_conditions > 0 and not self.degenerate_conditions

    @property
    def passed_branches(self) -> list[str]:
        return [str(b.branch) for b in self.branches if b.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "proceed": self.proceed,
            "branches": [b.as_dict() for b in self.branches],
            "passed_branches": self.passed_branches,
            "interaction_estimable": self.interaction_estimable,
            "degenerate_conditions": self.degenerate_conditions,
            "degenerate_cells": self.degenerate_conditions,
            "judge_floor_pp": self.judge_floor_pp,
            "n_conditions": self.n_conditions,
            "n_cells": self.n_conditions,
            "n_questions": self.n_questions,
            "n_interaction_conditions": self.n_interaction_conditions,
            "n_interaction_cells": self.n_interaction_conditions,
            "pivot_if_negative": [
                "ceiling analysis (already built)",
                "stratification by question type (already built)",
                "judge audit (P6)",
                "cost model (P2)",
            ],
            "note": (
                "A BOUNDED_NEGATIVE is a result, not a failure, and it is decided with "
                "the Guide IN ADVANCE (IMPLEMENTATION_PLAN.md §6) rather than "
                "discovered after the budget is spent."
            ),
            "notes": self.notes,
        }


def gate(
    *,
    accuracy_by_condition: dict[str, float] | None = None,
    accuracy_by_cell: dict[str, float] | None = None,
    recall_by_k: dict[int, float] | None = None,
    stratified: dict[str, Any] | None = None,
    degenerate_conditions: list[dict[str, Any]] | None = None,
    degenerate_cells: list[dict[str, Any]] | None = None,
    n_interaction_conditions: int | None = None,
    n_interaction_cells: int | None = None,
    judge_floor_pp: float = 0.0,
    n_questions: int = 0,
) -> GateVerdict:
    notes: list[str] = []
    acc = accuracy_by_condition if accuracy_by_condition is not None else (accuracy_by_cell or {})
    degen = degenerate_conditions if degenerate_conditions is not None else (degenerate_cells or [])
    n_ic = (
        n_interaction_conditions
        if n_interaction_conditions is not None
        else (n_interaction_cells or 0)
    )

    iv_ran = n_ic > 0
    b4_passed = iv_ran and not degen
    if not iv_ran:
        b4_detail = (
            "NOT EVALUATED: no (K, B) conditions were checked (n_interaction_conditions=0). An "
            "empty degenerate list here means the estimability check never ran, not "
            "that it passed."
        )
    elif degen:
        b4_detail = (
            f"{len(degen)} of {n_ic} condition(s) are degenerate, "
            f"so the K x B interaction cannot be estimated over the full grid"
        )
    else:
        b4_detail = f"all {n_ic} (K, B) conditions are estimable"

    b4 = BranchResult(
        branch=Branch.INTERACTION,
        passed=b4_passed,
        observed={
            "n_conditions_checked": n_ic,
            "n_cells_checked": n_ic,
            "n_degenerate_conditions": len(degen),
            "n_degenerate_cells": len(degen),
        },
        threshold={
            "max_degenerate_conditions": 0,
            "max_degenerate_cells": 0,
            "min_conditions_checked": 1,
            "min_cells_checked": 1,
        },
        detail=b4_detail,
    )
    branches: list[BranchResult] = [b4]

    if not b4_passed:
        notes.append(
            "Branch (iv) did not pass. Branches (i)-(iii) below are reported for the "
            "record but cannot support a headline claim: an interaction estimated over "
            "a grid with holes in it - or not checked at all - is not the interaction "
            "this design declared."
        )

    if acc:
        accs = list(acc.values())
        spread_pp = (max(accs) - min(accs)) * 100.0
        effective = ACCURACY_THRESHOLD_PP + judge_floor_pp
        b1 = BranchResult(
            branch=Branch.ACCURACY,
            passed=spread_pp >= effective,
            observed={
                "spread_pp": round(spread_pp, 4),
                "n_conditions": len(accs),
                "n_cells": len(accs),
            },
            threshold={"pp": ACCURACY_THRESHOLD_PP, "judge_floor_pp": judge_floor_pp},
            detail=(
                f"{spread_pp:.2f}pp of accuracy spread across {len(accs)} conditions vs a "
                f"{effective:.2f}pp threshold (5pp + {judge_floor_pp:.2f}pp judge floor)"
            ),
        )
    else:
        b1 = BranchResult(
            branch=Branch.ACCURACY,
            passed=False,
            observed=None,
            threshold={"pp": ACCURACY_THRESHOLD_PP},
            detail="no accuracy data; branch cannot pass on absent input",
        )
        notes.append("Branch (i) had no accuracy data. A missing measurement is not a pass.")
    branches.append(b1)

    if recall_by_k:
        recalls = list(recall_by_k.values())
        rspread_pp = (max(recalls) - min(recalls)) * 100.0
        b2 = BranchResult(
            branch=Branch.RECALL,
            passed=rspread_pp >= RECALL_THRESHOLD_PP,
            observed={"spread_pp": round(rspread_pp, 4), "n_k": len(recalls)},
            threshold={"pp": RECALL_THRESHOLD_PP},
            detail=(
                f"{rspread_pp:.2f}pp of recall spread over {len(recalls)} K values vs a "
                f"{RECALL_THRESHOLD_PP:.0f}pp threshold"
            ),
        )
    else:
        b2 = BranchResult(
            branch=Branch.RECALL,
            passed=False,
            observed=None,
            threshold={"pp": RECALL_THRESHOLD_PP},
            detail="no recall data; branch cannot pass on absent input",
        )
        notes.append("Branch (ii) had no recall data.")
    branches.append(b2)

    if stratified is None:
        b3 = BranchResult(
            branch=Branch.STRATIFIED,
            passed=False,
            observed=None,
            threshold={"rule": "multi-session rising while single-session flat/declining"},
            detail="not computed; a stratification that was not run cannot pass",
        )
    else:
        rising = float(stratified.get("multi_session_slope", 0.0))
        flat = float(stratified.get("single_session_slope", 0.0))
        b3 = BranchResult(
            branch=Branch.STRATIFIED,
            passed=bool(stratified.get("structure_detected", False)),
            observed={"multi_session_slope": rising, "single_session_slope": flat},
            threshold={"rule": "multi rises while single is flat or declining"},
            detail=stratified.get("detail", ""),
        )
    branches.append(b3)

    passed = any(b.passed for b in branches)
    return GateVerdict(
        outcome=GateOutcome.GO if passed else GateOutcome.BOUNDED_NEGATIVE,
        branches=branches,
        degenerate_conditions=list(degen),
        judge_floor_pp=judge_floor_pp,
        n_conditions=len(acc),
        n_questions=n_questions,
        n_interaction_conditions=n_ic,
        notes=notes,
    )
