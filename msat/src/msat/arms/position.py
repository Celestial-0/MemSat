"""Arm C — position, with the retrieved SET held identical (PRD §R4, arms.yaml).

`isolates: position`. Four orderings, balanced seeds. The retrieved set is
IDENTICAL across all four, which is the entire reason the arm is clean: any
accuracy difference is caused by order and by nothing else.

THE VARIANTS, AND WHY EACH IS HERE
-----------------------------------
    natural        rank order from the retriever - the baseline ordering
    reversed       worst-ranked first
    shuffled       seeded permutation; BALANCED so no variant owns one ordering
    gold_first     gold sessions hoisted to the front

`gold_first` is the one that can be mistaken for a win. Hoisting evidence to the
front raises accuracy on most models, and reading that as "retrieval is the
bottleneck" inverts the logic: the arm changed position AND destroyed any
positional information the retriever had. It is reported as a POSITION VARIANT,
never as a retrieval result, and it is never pooled with the others.

BALANCED SEEDS ARE A REQUIREMENT, NOT A TIDYNESS
------------------------------------------------
`shuffled` with one seed gives one ordering, and a single ordering is a
confounded condition wearing the name of a randomised one. Seeds are assigned by
`(variant, replicate)` so that across the full replicate set every variant sees
every seed, which is what makes the arm's variance an estimate of ordering
variance rather than of one arbitrary permutation.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from msat.arms.render import Memory, RenderedPrompt, format_memories, render_template
from msat.store.responses import ConditionKey


class PositionVariant(StrEnum):
    """The four orderings. Names are the `position_variant` values on disk."""

    NATURAL = "natural"
    REVERSED = "reversed"
    SHUFFLED = "shuffled"
    GOLD_FIRST = "gold_first"


VARIANTS: tuple[PositionVariant, ...] = tuple(PositionVariant)


@dataclass(frozen=True, slots=True)
class PositionPlan:
    """Which seed each (variant, replicate) gets, and whether the set is balanced.

    `balanced` is ASSERTED by `msat run` before any cell is scheduled. An
    unbalanced arm is not a weaker arm, it is a different and unstated
    experiment, and it looks identical in the ledger.
    """

    n_replicates: int
    seed: int
    assignments: dict[PositionVariant, tuple[int, ...]]

    @property
    def balanced(self) -> bool:
        """Every variant sees the same SET of seeds.

        Compare sets, not sequences: every variant must cover the same seeds, and
        comparing in order would flag a design that is balanced but staggered -
        which is correct and deliberate.
        """
        if not self.assignments:
            return False
        seed_sets = {tuple(sorted(s)) for s in self.assignments.values()}
        return len(seed_sets) == 1

    def seed_for(self, variant: PositionVariant, replicate: int) -> int:
        seeds = self.assignments.get(variant)
        if not seeds:
            raise KeyError(f"no seeds assigned to position variant {variant}")
        return seeds[replicate % len(seeds)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "balanced": self.balanced,
            "assignments": {str(k): list(v) for k, v in self.assignments.items()},
        }


def build_position_plan(
    *, n_replicates: int, seed: int, variants: Sequence[PositionVariant] = VARIANTS
) -> PositionPlan:
    """Seeds are derived from `(seed, variant)` and shared across variants.

    Sharing the SEQUENCE across variants is what makes the set balanced. Deriving
    it per variant (`seed + hash(variant)`) would give every variant a different,
    non-overlapping set and silently unbalance the arm - which is the specific bug
    this function exists to prevent.
    """
    return PositionPlan(
        n_replicates=n_replicates,
        seed=seed,
        assignments={v: tuple(seed + i for i in range(n_replicates)) for v in variants},
    )


def reorder(
    memories: Sequence[Memory],
    variant: PositionVariant,
    *,
    gold_sessions: Sequence[str] = (),
    shuffle_seed: int | None = None,
) -> list[Memory]:
    """Apply one ordering. Returns a new list; never mutates the input."""
    out = list(memories)
    if variant is PositionVariant.NATURAL:
        return out
    if variant is PositionVariant.REVERSED:
        return list(reversed(out))
    if variant is PositionVariant.SHUFFLED:
        if shuffle_seed is None:
            raise ValueError(
                "the shuffled variant needs a seed. Without one it is a single "
                "arbitrary ordering pretending to be a randomised condition."
            )
        rng = random.Random(shuffle_seed)
        rng.shuffle(out)
        return out
    if variant is PositionVariant.GOLD_FIRST:
        gold = set(gold_sessions)
        if not gold:
            # No gold to hoist: silently falling back to natural would make this
            # variant a DUPLICATE of natural on exactly the items where position
            # was the thing under test.
            raise ValueError(
                "`gold_first` was requested with no gold sessions. It would render "
                "identically to `natural` and the arm would report a position effect "
                "that is really a duplicate-condition artifact."
            )
        head = [m for m in out if m.session_id in gold]
        tail = [m for m in out if m.session_id not in gold]
        return head + tail
    raise ValueError(f"unknown position variant {variant!r}")


@dataclass
class PositionArm:
    """Arm C. Identical set, varied order."""

    id: str = "C"
    isolates: str = "position"
    template: str = ""
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        *,
        variant: PositionVariant = PositionVariant.NATURAL,
        shuffle_seed: int | None = None,
        corpus_fraction: float = 0.0,
        budget_tokens: int | None = None,
        **kwargs: Any,
    ) -> RenderedPrompt:
        if not self.template:
            raise ValueError("arm C has no template")
        ordered = reorder(
            memories,
            variant,
            gold_sessions=list(kwargs.get("gold_sessions") or ()),
            shuffle_seed=shuffle_seed,
        )
        system, user = render_template(
            self.template,
            {
                "memories": format_memories(ordered, budget_tokens),
                "question": question,
                "K": len(ordered),
                "B": condition.b,
                "arm": self.id,
            },
        )
        return RenderedPrompt(
            arm=self.id,
            condition=condition,
            system=system,
            user=user,
            memory_ids=tuple(m.memory_id for m in ordered),
            session_ids=tuple(m.session_id for m in ordered),
            ranks=tuple(range(1, len(ordered) + 1)),
            k=len(ordered),
            b=condition.b,
            b_mode=condition.b_mode,
            corpus_fraction=corpus_fraction,
            position_variant=f"position/{variant}",
            qid=str(kwargs.get("qid") or ""),
        )


def verify_same_set(variants: Mapping[PositionVariant, Sequence[Memory]]) -> dict[str, Any]:
    """Assert every ordering carries the SAME set, just reordered.

    The defining property of Arm C and the one a refactor breaks quietly: sort one
    arm's memories in place, or drop one in a variant, and the arm still runs, still
    scores, and is now measuring two different retrieval outcomes.
    """
    sets = {v: frozenset(m.memory_id for m in mems) for v, mems in variants.items()}
    distinct = set(sets.values())
    union = set().union(*distinct) if distinct else set()
    sizes = {v: len(s) for v, s in sets.items()}
    return {
        "identical": len(distinct) <= 1,
        "n_variants": len(sets),
        "sizes": {str(v): n for v, n in sizes.items()},
        "missing": {str(v): sorted(union - s) for v, s in sets.items() if union - s},
        "note": (
            "Arm C is clean only when every ordering holds the same retrieved set. "
            "`missing` must be empty for every variant."
        ),
    }
