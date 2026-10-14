"""Randomised, interleaved schedule (PRD §R5.1).

*"Never sequential by K — provider drift must not be able to masquerade as a
curve."*

The threat is specific and quiet. If the executor runs K=1 for all 500 items, then
K=2, then K=4, then anything that drifts monotonically over the run — thermal
throttling, a background job, a provider silently changing weights behind a tag —
lands on one end of the K axis. The curve then shows a trend that is time, not
depth, and nothing in the data distinguishes the two.

INTERLEAVING ALONE IS NOT ENOUGH
--------------------------------
Shuffling is necessary but not sufficient. The plan has a hard constraint that
shuffling fights: the engine loads one model's weights at a time, so grouping by
model is what keeps weight loads down from ~189,000 to ~5.

So the schedule is a **seeded stratified interleave**: cells are ordered by a
seeded shuffle WITHIN each model group, and the groups themselves are ordered by a
seeded shuffle. Randomisation is preserved *inside* the group where drift would
correlate with K, and the group order is randomised too, so drift cannot
systematically favour one reader over another.

Either choice is defensible on its own. Doing both, and stating which property each
one buys, is better than picking one and hoping.

THE SCHEDULE IS WRITTEN TO THE MANIFEST
---------------------------------------
So the exact order is reproducible and so a re-run can prove it used the same one.
A seed that is not recorded is a seed nobody can replay.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.store.responses import ConditionKey

SCHEDULE_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class ScheduledUnit:
    """One unit of work: a condition, a question, and a replicate."""

    condition: ConditionKey
    qid: str
    replicate: int

    @property
    def cell(self) -> tuple:
        return self.condition.key


def _stable_shuffle(items: list[Any], seed: int, salt: str) -> list[Any]:
    """Seeded shuffle whose result depends on `salt`, not on list length alone.

    The salt matters: shuffling the flat list and shuffling each model group with
    the same seed produce correlated orders, and correlated order is exactly the
    thing R5.1 exists to prevent.
    """
    rng = random.Random(f"{seed}:{salt}")
    rng.shuffle(items)
    return items


def build_schedule(
    cells: Sequence[ConditionKey],
    qids: Sequence[str],
    runs_per_cell: int,
    *,
    seed: int,
    group_by: str = "reader",
) -> tuple[ScheduledUnit, ...]:
    """The full work list, in the order it will be executed.

    Deterministic: the same inputs and seed give the same order, always. That is
    what lets the schedule be written to the manifest and later verified.
    """
    groups: dict[str, list[tuple]] = {}
    for cell in cells:
        groups.setdefault(getattr(cell, group_by), []).append((cell,))

    ordered_groups = _stable_shuffle(sorted(groups.items()), seed, f"group:{group_by}")

    schedule: list[ScheduledUnit] = []
    for reader, members in ordered_groups:
        expanded = [
            (cell, qid, rep) for (cell,) in members for qid in qids for rep in range(runs_per_cell)
        ]
        _stable_shuffle(expanded, seed, f"within:{reader}")
        schedule.extend(ScheduledUnit(c, q, r) for c, q, r in expanded)
    return tuple(schedule)


def schedule_digest(schedule: Sequence[ScheduledUnit]) -> str:
    """Hash of the exact order, recorded in the manifest.

    Order is part of the result: two runs with the same work but a different order
    are not comparable, and the digest is what makes that checkable in one
    comparison rather than by re-deriving the schedule.
    """
    h = hashlib.sha256()
    for unit in schedule:
        h.update(str(unit.condition.key).encode("utf-8"))
        h.update(b"\x1f")
        h.update(unit.qid.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(unit.replicate).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def verify_not_sequential_by_k(
    schedule: Sequence[ScheduledUnit],
    *,
    max_run_fraction: float = 0.05,
) -> dict[str, Any]:
    """Assert the schedule is not effectively SORTED by K.

    The criterion is a FRACTION, not an absolute run length, and that matters. A
    uniform shuffle over 4 K levels and ~200 units will naturally produce a run of
    4 or 5 somewhere — that is what randomness looks like, and a test demanding no
    repeats at all would fail a correctly shuffled schedule.

    What would actually be wrong is the schedule arriving in K order, or in long
    blocks of one K. So: measure the longest same-K run and compare it to the
    schedule length.
    """
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
