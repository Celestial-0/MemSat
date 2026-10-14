"""Arm G — the budget sweep, which is HALF OF THE CROSSED GRID (PRD §R4.4).

Arm G is not a separate sweep. Its cells with Arm A's ARE the crossed grid, and
the `log2(K) * log2(B)` interaction between them is the paper's headline result
(R9.1). `arms.yaml` declares `crossed_grid: arms: [A, G]` precisely so the design
is DATA rather than an emergent property of calling two functions in a loop.

So the temptation this module exists to remove is writing G as "run Arm A at each
B" with its own loop, its own condition builder, and its own manifest - which is
exactly how a design silently degrades into two marginal sweeps while every
individual test still passes.

THE DIFFERENCE BETWEEN A AND G IS THE PROMPT AND NOTHING ELSE
------------------------------------------------------------
Both render natural top-K in rank order with the same per-memory budget. G differs
only in that its template names `B` explicitly, so the reader is told the budget it
is being given. That is the confound G isolates: an explicit budget statement is a
plausible way to move accuracy without changing a single retrieved memory.

Because the two share a renderer, a change to how memories are formatted moves both
arms together and cancels. Building G as a separate renderer would lose that
cancellation and make a formatting change look like a budget effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.arms.render import ArmRegistry, Memory, RenderedPrompt, SimpleArm
from msat.store.responses import ConditionKey

CROSSED_ARMS: tuple[str, str] = ("A", "G")
"""The two arms that form one grid. Duplicated from `arms.yaml` here ONLY so the
registry can assert the config agrees - a mismatch means the code and the data
disagree about the design, which is the failure the plan calls most dangerous."""


@dataclass(frozen=True, slots=True)
class BudgetArm(SimpleArm):
    """Arm G. Same renderer as A, different template.

    Frozen, because the parent is: a mutable subclass of a frozen dataclass is a
    silent way to make one arm hashable and the other not, and the cache key is
    computed from the renderer.

    `id` and `isolates` stay REQUIRED. Giving them defaults here would place a
    default-bearing attribute before `template`, which the dataclass machinery
    rejects - and the fix for that is to keep them required, not to reorder the
    parent's fields around a subclass.
    """

    id: str
    isolates: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        *,
        budget_tokens: int | None = None,
        corpus_fraction: float = 0.0,
        qid: str = "",
        **_: Any,
    ) -> RenderedPrompt:
        """Identical to `SimpleArm` except the manifest carries `session_ids` and
        `qid`, which `recall@K` needs and `SimpleArm` does not populate.

        The widening of `budget_tokens` from the base's `int | None` to
        `float | None` is deliberate and is the ONLY difference in behaviour: a B
        declared as `512.0` and one declared as `512` must render identically, and
        narrowing here would round `511.5` and put a different prompt in the cache
        than the condition key names.
        """
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
    """Check the code's crossed pair against the config's.

    `arms.yaml` is the single source of truth for the grid (PRD §R0.1). If code
    hard-codes a different pair, one of them is stale and the interaction term will
    be estimated over the wrong cells. Asserting beats a comment.
    """
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
    """Every `(K, B)` pair, for both arms. The full crossing, nothing derived.

    Exposed as a function so the manifest builder, the schedule and the test all
    enumerate the same set from the same place. A test that recomputes the expected
    crossing has its own copy of the grid, and a copy is where a trim to `K_grid`
    gets half-applied.
    """
    return [(k, float(b)) for k in k_grid for b in b_grid]
