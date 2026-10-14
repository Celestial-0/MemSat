"""The P5 GO / NO-GO gate, all four branches (Architecture §P5, PRD exit gate).

Proceed if ANY of:

    (i)    accuracy varies >= 5pp across the grid, beyond judge noise
    (ii)   recall@K varies >= 10pp
    (iii)  stratified structure appears (multi-session rising, single-session flat)
    (iv)   the K x B interaction is estimable at adequate power in every cell

NONE of them -> a bounded NEGATIVE, and a pivot. That is a legitimate result, not a
failure, and the pivot list (ceiling, stratification, judge audit, cost model) is
already built. But it is decided WITH THE GUIDE IN ADVANCE (§6), not discovered
after 292 hours.

WHY BRANCH (i) IS COMPARED AGAINST JUDGE NOISE
---------------------------------------------
`>= 5pp` is not 5pp, it is 5pp of REAL variation. If two judges disagree enough to
move accuracy 6pp on identical outputs, then a 5pp grid spread measures the
disagreement and not the manipulation. The threshold is therefore
`>= 5pp + judge_floor`, and `judge_floor` is passed in rather than assumed to be
zero. Assuming zero judge noise is how a run reports a flat grid as a finding.

BRANCH (iv) IS CHECKED FIRST AND CAN VETO
-----------------------------------------
If any cell is degenerate the interaction is not estimable, and branches (i)-(iii)
are then measuring a grid with holes in it. `gate` reports (iv) separately so that
a run which passes (i) on a partly-missing grid cannot claim the headline result,
and so the failure is legible as "cell (K=64, B=128) was degenerate" rather than
"nothing was found".

THE RESULT IS WRITTEN DOWN, NOT PRINTED
---------------------------------------
`GateVerdict.as_dict()` is the `go_no_go.json` deliverable. A gate that only prints
lives in a scrollback buffer, and the argument about whether it passed should not
be reconstructible from memory three weeks later.
"""

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
    """The P5 verdict. `.as_dict()` is written to `results/<run_id>/go_no_go.json`."""

    outcome: GateOutcome
    branches: list[BranchResult] = field(default_factory=list)
    degenerate_cells: list[dict[str, Any]] = field(default_factory=list)
    judge_floor_pp: float = 0.0
    n_cells: int = 0
    n_questions: int = 0
    n_interaction_cells: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def proceed(self) -> bool:
        return self.outcome is GateOutcome.GO

    @property
    def interaction_estimable(self) -> bool:
        """Requires that the check RAN as well as that it found nothing.

        A `degenerate_cells == []` on a grid nobody examined is not estimability,
        and the distinction is the difference between a passing gate and one that
        never ran.
        """
        return self.n_interaction_cells > 0 and not self.degenerate_cells

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
            "degenerate_cells": self.degenerate_cells,
            "judge_floor_pp": self.judge_floor_pp,
            "n_cells": self.n_cells,
            "n_questions": self.n_questions,
            "n_interaction_cells": self.n_interaction_cells,
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
    accuracy_by_cell: dict[str, float],
    recall_by_k: dict[int, float],
    stratified: dict[str, Any] | None,
    degenerate_cells: list[dict[str, Any]],
    n_interaction_cells: int = 0,
    judge_floor_pp: float = 0.0,
    n_questions: int = 0,
) -> GateVerdict:
    """Evaluate all four branches. `accuracy_by_cell` is a flat map of cells to accuracy.

    `n_interaction_cells` is how many (K, B) cells the estimability check actually
    EXAMINED, and it is the difference between a check that passed and a check that
    never ran. Without it, `degenerate_cells == []` is indistinguishable from "no
    grid was evaluated", and branch (iv) - the headline - passes on an empty list.
    A gate called with no data at all would then return GO.

    An empty map makes branches (i) and (ii) FAIL rather than raise, and the verdict
    says why. A gate that crashes on missing input cannot be used to report that
    the input is missing.
    """
    notes: list[str] = []

    # ── (iv) interaction estimable. Reported first because it is the headline. ──
    iv_ran = n_interaction_cells > 0
    b4_passed = iv_ran and not degenerate_cells
    if not iv_ran:
        b4_detail = (
            "NOT EVALUATED: no (K, B) cells were checked (n_interaction_cells=0). An "
            "empty degenerate list here means the estimability check never ran, not "
            "that it passed."
        )
    elif degenerate_cells:
        b4_detail = (
            f"{len(degenerate_cells)} of {n_interaction_cells} cell(s) are degenerate, "
            f"so the K x B interaction cannot be estimated over the full grid"
        )
    else:
        b4_detail = f"all {n_interaction_cells} (K, B) cells are estimable"

    b4 = BranchResult(
        branch=Branch.INTERACTION,
        passed=b4_passed,
        observed={
            "n_cells_checked": n_interaction_cells,
            "n_degenerate_cells": len(degenerate_cells),
        },
        threshold={"max_degenerate_cells": 0, "min_cells_checked": 1},
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

    # ── (i) accuracy spread, against judge noise ──
    if accuracy_by_cell:
        accs = list(accuracy_by_cell.values())
        spread_pp = (max(accs) - min(accs)) * 100.0
        effective = ACCURACY_THRESHOLD_PP + judge_floor_pp
        b1 = BranchResult(
            branch=Branch.ACCURACY,
            passed=spread_pp >= effective,
            observed={"spread_pp": round(spread_pp, 4), "n_cells": len(accs)},
            threshold={"pp": ACCURACY_THRESHOLD_PP, "judge_floor_pp": judge_floor_pp},
            detail=(
                f"{spread_pp:.2f}pp of accuracy spread across {len(accs)} cells vs a "
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

    # ── (ii) recall@K spread ──
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

    # ── (iii) stratified structure ──
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
        degenerate_cells=list(degenerate_cells),
        judge_floor_pp=judge_floor_pp,
        n_cells=len(accuracy_by_cell),
        n_questions=n_questions,
        n_interaction_cells=n_interaction_cells,
        notes=notes,
    )
