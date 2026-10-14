"""The executable plan (PRD §R4.4) and the crossed-grid verification.

From the manifest's ACCEPTED cells to a schedule the executor can run.

WHY THIS IS A SEPARATE STEP FROM THE MANIFEST
---------------------------------------------
The manifest answers "which conditions are valid". The plan answers "what will
actually be executed, in what order". Keeping them apart means a plan can be built
and inspected without re-running the guards, and the guards can be re-run without
invalidating a plan that has not been started.

THE REQUIREMENT THIS ENFORCES
-----------------------------
PRD §R4.4: arms A and G are ONE CROSSED GRID. Every `(K, B)` cell in
`K_grid x B_grid` is a distinct condition. A run that varies K at fixed B and then
B at fixed K is two MARGINAL SWEEPS — it does not satisfy R4.4 and it cannot
produce the interaction term, which is the paper's headline claim.

Marginal sweeps are the failure mode here because they are what the literature
actually does, and four papers holding depth fixed is the implicit baseline this
design departs from. It is very easy to build the thing everyone else built.

TWO WAYS A GRID STOPS BEING CROSSED, BOTH CHECKED
------------------------------------------------
1. A cell is missing — usually because the guard rejected it on one scale and the
   design never noticed.
2. A cell is present for SOME readers and not others. Then reader is confounded
   with the grid: the K curve measured on gemma is not the K curve measured on
   qwen, and averaging them produces a curve that belongs to no model.

The second is the quieter of the two, and it is the one a per-reader K cap creates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msat.runner.schedule import ScheduledUnit, build_schedule
from msat.store.responses import ConditionKey

PLAN_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class PlanCell:
    condition: ConditionKey
    n_questions: int
    degeneracy_fraction: float

    @property
    def k(self) -> int:
        return self.condition.k

    @property
    def b(self) -> float:
        return self.condition.b


@dataclass
class ExecutablePlan:
    """What will run, and what was left out."""

    cells: tuple[PlanCell, ...]
    rejected: tuple[PlanCell, ...]
    schedule: tuple[ScheduledUnit, ...]
    config_hash: str
    seed: int
    runs_per_cell: int

    @property
    def n_units(self) -> int:
        return len(self.schedule)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "runs_per_cell": self.runs_per_cell,
            "n_cells": len(self.cells),
            "n_rejected": len(self.rejected),
            "n_units": self.n_units,
            "cells": [
                {
                    **c.condition.as_dict(),
                    "n_questions": c.n_questions,
                    "degeneracy_fraction": c.degeneracy_fraction,
                }
                for c in self.cells
            ],
            "rejected": [
                {**c.condition.as_dict(), "degeneracy_fraction": c.degeneracy_fraction}
                for c in self.rejected
            ],
        }


def plan_from_manifest(
    manifest_path: Path,
    qids_by_scale: Mapping[str, Sequence[str]],
    *,
    seed: int,
    runs_per_cell: int,
) -> ExecutablePlan:
    """Build the executable plan from a `manifest.json` on disk.

    Only ACCEPTED cells become work. A rejected cell is carried through for the
    record, because a plan that quietly omits cells is indistinguishable from one
    that was built from a smaller grid — and the difference is the whole claim.
    """
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    accepted: list[PlanCell] = []
    rejected: list[PlanCell] = []
    for cell in raw["cells"]:
        entry = PlanCell(
            condition=ConditionKey.from_dict(cell),
            n_questions=int(cell["n_questions"]),
            degeneracy_fraction=float(cell["degeneracy_fraction"]),
        )
        (accepted if cell["status"] == "accepted" else rejected).append(entry)

    schedule: list[ScheduledUnit] = []
    for cell in accepted:
        qids = list(qids_by_scale.get(cell.condition.scale, []))
        schedule.extend(build_schedule([cell.condition], qids, runs_per_cell, seed=seed))
    return ExecutablePlan(
        cells=tuple(accepted),
        rejected=tuple(rejected),
        schedule=tuple(schedule),
        config_hash=str(raw["config_hash"]),
        seed=seed,
        runs_per_cell=runs_per_cell,
    )


# ── R4.4 verification ─────────────────────────────────────────────────────────


def verify_crossed_grid(
    plan: ExecutablePlan,
    *,
    k_grid: Sequence[int],
    b_grid: Sequence[float],
    scales: Sequence[str],
    readers: Sequence[str],
) -> dict[str, Any]:
    """Check the crossing is complete, per scale AND per reader.

    Returns a report rather than raising. The caller — `msat build-plan` — decides
    what an incomplete crossing means for the run, and the numbers here are what
    make that decision informed rather than a guess.
    """
    expected_kb = {(k, round(float(b), 6)) for k in k_grid for b in b_grid}
    problems: list[str] = []

    per_scale: dict[str, Any] = {}
    for scale in scales:
        for reader in readers:
            got = {
                (c.k, round(c.b, 6))
                for c in plan.cells
                if c.condition.scale == scale and c.condition.reader == reader
            }
            missing = sorted(expected_kb - got)
            extra = sorted(got - expected_kb)
            per_scale[f"{scale}/{reader}"] = {
                "n_cells": len(got),
                "n_expected": len(expected_kb),
                "missing": [list(m) for m in missing],
                "extra": [list(e) for e in extra],
                "complete": not missing and not extra,
            }
            if missing:
                problems.append(
                    f"{scale}/{reader}: {len(missing)} of {len(expected_kb)} (K,B) cells "
                    f"missing, e.g. {missing[:3]}. A marginal sweep cannot produce the "
                    f"interaction term."
                )

    # Reader-confounding: does every reader see the same grid?
    grids = {
        r: {(c.k, round(c.b, 6)) for c in plan.cells if c.condition.reader == r} for r in readers
    }
    distinct = {tuple(sorted(g)) for g in grids.values()}
    reader_balanced = len(distinct) == 1

    return {
        "requirement": "PRD §R4.4 - arms A and G are ONE crossed grid",
        "n_expected_cells": len(expected_kb) * len(scales) * len(readers),
        "n_planned_cells": len(plan.cells),
        "n_rejected_cells": len(plan.rejected),
        "crossed": not problems,
        "reader_balanced": reader_balanced,
        "per_scale_reader": per_scale,
        "problems": problems,
        "n_units": plan.n_units,
    }


def write_plan(plan: ExecutablePlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
