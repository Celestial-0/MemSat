"""Randomised, interleaved schedule."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.store import Condition

SCHEDULE_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class ScheduledUnit:
    condition: Condition
    qid: str
    repetition: int

    @property
    def cell(self) -> tuple:
        return self.condition.key


def _stable_shuffle(items: list[Any], seed: int, salt: str) -> list[Any]:
    rng = random.Random(f"{seed}:{salt}")
    rng.shuffle(items)
    return items


def build_schedule(
    conditions: Sequence[Condition] | None = None,
    qids: Sequence[str] = (),
    repetition: int | None = None,
    *,
    seed: int,
    group_by: str = "reader",
    cells: Sequence[Condition] | None = None,
    runs_per_cell: int | None = None,
) -> tuple[ScheduledUnit, ...]:
    target_conditions = conditions if conditions is not None else (cells or ())
    n_repetitions = (
        repetition
        if repetition is not None
        else (runs_per_cell if runs_per_cell is not None else 1)
    )
    groups: dict[str, list[tuple]] = {}
    for condition in target_conditions:
        groups.setdefault(getattr(condition, group_by), []).append((condition,))

    ordered_groups = _stable_shuffle(sorted(groups.items()), seed, f"group:{group_by}")

    schedule: list[ScheduledUnit] = []
    for reader, members in ordered_groups:
        expanded = [
            (cond, qid, rep) for (cond,) in members for qid in qids for rep in range(n_repetitions)
        ]
        _stable_shuffle(expanded, seed, f"within:{reader}")
        schedule.extend(ScheduledUnit(c, q, r) for c, q, r in expanded)
    return tuple(schedule)


def schedule_digest(schedule: Sequence[ScheduledUnit]) -> str:
    h = hashlib.sha256()
    for unit in schedule:
        h.update(str(unit.condition.key).encode("utf-8"))
        h.update(b"\x1f")
        h.update(unit.qid.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(unit.repetition).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def verify_not_sequential_by_k(
    schedule: Sequence[ScheduledUnit],
    *,
    max_run_fraction: float = 0.05,
) -> dict[str, Any]:
    worst = 1
    current = 1
    previous = None
    ks: list[int] = []
    for unit in schedule:
        k = unit.condition.k
        ks.append(k)
        if k == previous:
            current += 1
            worst = max(worst, current)
        else:
            current = 1
            previous = k

    n = len(schedule)
    sorted_by_k = ks == sorted(ks)
    allowed = max(2, int(n * max_run_fraction))
    return {
        "n_units": n,
        "longest_same_k_run": worst,
        "allowed_run": allowed,
        "sorted_by_k": sorted_by_k,
        "sequential_by_k": sorted_by_k or worst > allowed,
        "distinct_k_levels": len(set(ks)),
    }
