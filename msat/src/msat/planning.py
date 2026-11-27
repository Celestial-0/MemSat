"""The executable plan and the crossed-grid verification."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from msat.store import Condition

if TYPE_CHECKING:
    from msat.runner.schedule import ScheduledUnit

PLAN_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class PlanCondition:
    condition: Condition
    n_questions: int
    degeneracy_fraction: float


@dataclass
class ExecutablePlan:
    conditions: tuple[PlanCondition, ...] = ()
    rejected: tuple[PlanCondition, ...] = ()
    schedule: tuple[ScheduledUnit, ...] = ()
    config_hash: str = ""
    seed: int = 0
    repetition: int = 1
    canonical_config_hash: str | None = None
    effective_config_hash: str | None = None
    mode: str = "full"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.schedule)

    @property
    def n_observations(self) -> int:
        return len(self.schedule)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": PLAN_SCHEMA,
            "config_hash": self.config_hash,
            "canonical_config_hash": self.canonical_config_hash or self.config_hash,
            "effective_config_hash": self.effective_config_hash or self.config_hash,
            "mode": self.mode,
            "seed": self.seed,
            "repetition": self.repetition,
            "n_conditions": len(self.conditions),
            "n_rejected": len(self.rejected),
            "n_units": self.n_units,
            "conditions": [
                {
                    **c.condition.as_dict(),
                    "n_questions": c.n_questions,
                    "degeneracy_fraction": c.degeneracy_fraction,
                }
                for c in self.conditions
            ],
            "cells": [
                {
                    **c.condition.as_dict(),
                    "n_questions": c.n_questions,
                    "degeneracy_fraction": c.degeneracy_fraction,
                }
                for c in self.conditions
            ],
            "runs_per_cell": self.repetition,
            "rejected": [
                {**c.condition.as_dict(), "degeneracy_fraction": c.degeneracy_fraction}
                for c in self.rejected
            ],
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def plan_from_manifest(
    manifest_path: Path,
    qids_by_scale: Mapping[str, Sequence[str]],
    *,
    seed: int,
    runs_per_cell: int | None = None,
    repetition: int | None = None,
    bundle: Any = None,
    strict: bool = False,
    expected_mode: str | None = None,
) -> ExecutablePlan:
    n_repetitions = (
        repetition
        if repetition is not None
        else (runs_per_cell if runs_per_cell is not None else 1)
    )
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    raw_hash = str(raw.get("config_hash", ""))
    canonical_hash = str(raw.get("canonical_config_hash", raw_hash))
    effective_hash = str(raw.get("effective_config_hash", raw_hash))
    manifest_mode = str(raw.get("mode", "full"))
    metadata = dict(raw.get("metadata", {}))

    if expected_mode is not None and manifest_mode != expected_mode:
        raise ValueError(
            f"Manifest mode {manifest_mode!r} does not match expected mode {expected_mode!r}"
        )

    if bundle is not None and strict:
        from msat.experiment import validate_manifest_for_execution

        validate_manifest_for_execution(
            raw,
            bundle,
            qids_by_scale=qids_by_scale,
            seed=seed,
            runs_per_cell=n_repetitions,
            repetition=n_repetitions,
            strict_questions=True,
        )

    accepted: list[PlanCondition] = []
    rejected: list[PlanCondition] = []
    manifest_conditions = raw.get("conditions") or raw.get("cells", [])
    for cell in manifest_conditions:
        entry = PlanCondition(
            condition=Condition.from_dict(cell),
            n_questions=int(cell.get("n_questions", 1)),
            degeneracy_fraction=float(cell.get("degeneracy_fraction", 0.0)),
        )
        if cell.get("status", "accepted") in ("accepted", "supported"):
            accepted.append(entry)
        else:
            rejected.append(entry)
    for raw_rej in raw.get("rejected", []):
        entry = PlanCondition(
            condition=Condition.from_dict(raw_rej),
            n_questions=int(raw_rej.get("n_questions", 0)),
            degeneracy_fraction=float(raw_rej.get("degeneracy_fraction", 0.0)),
        )
        rejected.append(entry)

    schedule: list[ScheduledUnit] = []
    from msat.runner.schedule import build_schedule

    for cell in accepted:
        qids = sorted(qids_by_scale.get(cell.condition.scale, []))
        if strict and not qids:
            raise ValueError(
                f"Strict runtime planning requires questions for scale {cell.condition.scale!r}, but none were provided"
            )
        if strict and len(qids) != len(set(qids)):
            raise ValueError("Runtime question IDs must be unique within each scale.")
        schedule.extend(
            build_schedule(
                conditions=[cell.condition], qids=qids, repetition=n_repetitions, seed=seed
            )
        )
    random.Random(seed).shuffle(schedule)

    return ExecutablePlan(
        conditions=tuple(accepted),
        rejected=tuple(rejected),
        schedule=tuple(schedule),
        config_hash=raw_hash,
        seed=seed,
        repetition=n_repetitions,
        canonical_config_hash=canonical_hash,
        effective_config_hash=effective_hash,
        mode=manifest_mode,
        metadata=metadata,
    )


def verify_crossed_grid(
    plan: ExecutablePlan,
    *,
    k_grid: Sequence[int],
    b_grid: Sequence[float],
    scales: Sequence[str],
    readers: Sequence[str],
) -> dict[str, Any]:
    expected_kb = {(k, round(float(b), 6)) for k in k_grid for b in b_grid}
    expected_scales = set(scales)
    expected_readers = set(readers)
    problems: list[str] = []

    plan_cells = plan.conditions

    # 1. Reject duplicate cells in the plan
    seen_conditions: set[str] = set()
    duplicates: list[str] = []
    for c in plan_cells:
        ck = str(getattr(getattr(c, "condition", c), "key", c))
        if ck in seen_conditions:
            duplicates.append(ck)
        else:
            seen_conditions.add(ck)
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate condition(s) in plan: {duplicates[:3]}")

    # 2. Reject unexpected scales
    def _condition_attr(c: Any, name: str) -> str:
        return str(getattr(getattr(c, "condition", c), name, "") or "")

    unexpected_scales = sorted(
        {
            _condition_attr(c, "scale")
            for c in plan_cells
            if _condition_attr(c, "scale") not in expected_scales
        }
    )
    if unexpected_scales:
        problems.append(
            f"Unexpected scale(s) in plan: {unexpected_scales} (declared scales are {sorted(scales)})"
        )

    # 3. Reject unexpected readers
    unexpected_readers = sorted(
        {
            _condition_attr(c, "reader")
            for c in plan_cells
            if _condition_attr(c, "reader") not in expected_readers
        }
    )
    if unexpected_readers:
        problems.append(
            f"Unexpected reader(s) in plan: {unexpected_readers} (declared readers are {sorted(readers)})"
        )

    # 4. Check each (scale, reader) block for missing and extra (K, B) cells
    per_scale: dict[str, Any] = {}
    for scale in scales:
        for reader in readers:
            got = {
                (
                    getattr(c, "k", getattr(getattr(c, "condition", c), "k", None)),
                    round(getattr(c, "b", getattr(getattr(c, "condition", c), "b", 0.0)), 6),
                )
                for c in plan_cells
                if getattr(getattr(c, "condition", c), "scale", getattr(c, "scale", None)) == scale
                and getattr(getattr(c, "condition", c), "reader", getattr(c, "reader", None))
                == reader
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
            if extra:
                problems.append(
                    f"{scale}/{reader}: {len(extra)} extra (K,B) cells not in declared grid, e.g. {extra[:3]}."
                )

    grids = {
        r: {
            (
                getattr(c, "k", getattr(getattr(c, "condition", c), "k", None)),
                round(getattr(c, "b", getattr(getattr(c, "condition", c), "b", 0.0)), 6),
            )
            for c in plan_cells
            if getattr(getattr(c, "condition", c), "reader", getattr(c, "reader", None)) == r
        }
        for r in readers
    }
    distinct = {tuple(sorted(g)) for g in grids.values()}
    reader_balanced = len(distinct) == 1

    return {
        "requirement": "PRD §R4.4 - arms A and G are ONE crossed grid",
        "n_expected_cells": len(expected_kb) * len(scales) * len(readers),
        "n_planned_cells": len(plan_cells),
        "n_rejected_cells": len(plan.rejected),
        "crossed": not problems,
        "reader_balanced": reader_balanced,
        "per_scale_reader": per_scale,
        "problems": problems,
        "n_units": plan.n_units,
    }


def write_plan(plan: ExecutablePlan, path: Path, *, overwrite: bool = True) -> Path:
    p = Path(path)
    if not overwrite and p.exists():
        raise FileExistsError(f"Refusing to overwrite existing plan at {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
