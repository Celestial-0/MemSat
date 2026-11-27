"""Arm G, the budget sweep that forms half of the crossed grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.arms.render import ArmRegistry, Memory, RenderedPrompt, SimpleArm
from msat.store import Condition

CROSSED_ARMS: tuple[str, str] = ("A", "G")


@dataclass(frozen=True, slots=True)
class BudgetArm(SimpleArm):
    id: str
    isolates: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
        *,
        budget_tokens: int | None = None,
        corpus_fraction: float = 0.0,
        qid: str = "",
        **_: Any,
    ) -> RenderedPrompt:
        base = SimpleArm.render(
            self,
            question,
            memories,
            condition,
            budget_tokens=None if budget_tokens is None else int(budget_tokens),
            corpus_fraction=corpus_fraction,
        )
        return RenderedPrompt(
            arm=self.id,
            condition=base.condition,
            system=base.system,
            user=base.user,
            memory_ids=base.memory_ids,
            session_ids=tuple(m.session_id for m in memories),
            ranks=base.ranks,
            k=base.k,
            b=base.b,
            b_mode=base.b_mode,
            corpus_fraction=corpus_fraction,
            position_variant="natural",
            qid=qid,
        )


def assert_crossed(registry: ArmRegistry, declared: Sequence[str]) -> dict[str, Any]:
    declared_set = tuple(sorted(declared))
    code_set = tuple(sorted(CROSSED_ARMS))
    missing = [a for a in declared_set if a not in registry.arms]
    return {
        "declared_in_config": list(declared_set),
        "hard_coded": list(code_set),
        "agree": declared_set == code_set,
        "missing_from_registry": missing,
        "note": (
            "A and G are ONE crossed grid. Varying K at fixed B and then B at fixed "
            "K is two marginal sweeps and yields no interaction term."
        ),
    }


def crossed_cells(k_grid: Sequence[int], b_grid: Sequence[float]) -> list[tuple[int, float]]:
    return [(k, float(b)) for k in k_grid for b in b_grid]
