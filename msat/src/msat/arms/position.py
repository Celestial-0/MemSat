"""Arm C, position with the retrieved set held identical."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from msat.arms.render import Memory, RenderedPrompt, format_memories, render_template
from msat.store import Condition


class PositionVariant(StrEnum):
    NATURAL = "natural"
    REVERSED = "reversed"
    SHUFFLED = "shuffled"
    GOLD_FIRST = "gold_first"


VARIANTS: tuple[PositionVariant, ...] = tuple(PositionVariant)


@dataclass(frozen=True, slots=True)
class PositionPlan:
    n_repetitions: int
    seed: int
    assignments: dict[PositionVariant, tuple[int, ...]]

    @property
    def balanced(self) -> bool:
        if not self.assignments:
            return False
        seed_sets = {tuple(sorted(s)) for s in self.assignments.values()}
        return len(seed_sets) == 1

    def seed_for(self, variant: PositionVariant, repetition: int) -> int:
        seeds = self.assignments.get(variant)
        if not seeds:
            raise KeyError(f"no seeds assigned to position variant {variant}")
        return seeds[repetition % len(seeds)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_repetitions": self.n_repetitions,
            "seed": self.seed,
            "balanced": self.balanced,
            "assignments": {str(k): list(v) for k, v in self.assignments.items()},
        }


def build_position_plan(
    *, n_repetitions: int, seed: int, variants: Sequence[PositionVariant] = VARIANTS
) -> PositionPlan:
    return PositionPlan(
        n_repetitions=n_repetitions,
        seed=seed,
        assignments={v: tuple(seed + i for i in range(n_repetitions)) for v in variants},
    )


def reorder(
    memories: Sequence[Memory],
    variant: PositionVariant,
    *,
    gold_sessions: Sequence[str] = (),
    shuffle_seed: int | None = None,
) -> list[Memory]:
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
    id: str = "C"
    isolates: str = "position"
    template: str = ""
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
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
