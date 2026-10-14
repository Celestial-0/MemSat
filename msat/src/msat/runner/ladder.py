"""The smoke ladder: what runs before the real run is allowed to start.

WHY A LADDER AND NOT A SINGLE GATE
-----------------------------------
`make selftest` proves the machinery is correct on synthetic data. It cannot prove
the machinery is correct on the real corpus, because the real corpus is where the
corpus is too big, the filler file is missing, and the reader silently ignores
half the prompt. Those failures are only visible by running real generations, and
they are expensive to find late.

The ladder is ordered by HOW MUCH A FAILURE COSTS, not by how the arms relate to
each other:

    rung 1  arm E            ~1     does the pipeline run at all?
    rung 2  arms A/D/F       ~2     do the real arms render and score?
    rung 3  full grid        all    everything

THE ORDER IS AN INVARIANT, NOT A PREFERENCE
--------------------------------------------
Arm E is the cheapest arm and it is also the one that must fail LOUDLY. It feeds a
fabricated fact in a fabricated session, so a correct pipeline is confident and
wrong and an incorrect one is hesitant. Anything less than that confidence means
the reader is not reading the memories, the scoring is broken, or the store is
returning the wrong body — and all three of those silently improve Arm A, because
Arm A is measured against the same broken reader. A ladder that starts anywhere
else spends the full budget producing a confident number about nothing.

Reversing this is the natural mistake and it is expensive. Running the full grid
first means 189,000 calls before discovering that the filler's missing, which is a
292-hour way to learn something `ls` would have told you in a second.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

RungName = Literal["negative", "baselines", "full"]


@dataclass(frozen=True, slots=True)
class Rung:
    """One step of the ladder."""

    name: RungName
    n_questions: int
    arms: tuple[str, ...]
    scales: tuple[str, ...]
    requires_prior: RungName | None
    description: str

    def cells(self, grid: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"arm": arm, "scale": scale, **dict(g)}
            for arm in self.arms
            for scale in self.scales
            for g in grid
        ]


LADDER: tuple[Rung, ...] = (
    Rung(
        name="negative",
        n_questions=1,
        arms=("E",),
        scales=("S",),
        requires_prior=None,
        description="arm E on one question. Fails loudly by design.",
    ),
    Rung(
        name="baselines",
        n_questions=1,
        arms=("A", "D", "F"),
        scales=("S",),
        requires_prior="negative",
        description="the real arms render and score on one question each",
    ),
    Rung(
        name="full",
        n_questions=-1,  # all of them
        arms=("A", "B", "C", "D", "E", "F", "G"),
        scales=("S", "M"),
        requires_prior="baselines",
        description="the full grid",
    ),
)


class LadderOrderViolationError(RuntimeError):
    """Someone tried to enter the ladder above where they are.

    Not a warning. The whole value of a ladder is that it cannot be climbed from
    the top, and a warning is a thing people click through at 3am on day nine.
    """


@dataclass
class LadderState:
    """Which rungs have passed. Persisted, so it survives a restart."""

    passed: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)

    def first_unpassed(self) -> Rung | None:
        for rung in LADDER:
            if rung.name not in self.passed:
                return rung
        return None

    def may_enter(self, name: RungName) -> tuple[bool, str]:
        """Can this rung start? Returns the reason either way.

        The reason matters more than the boolean: when the ladder blocks a run at
        3am, "you skipped rung 1" is actionable and "validation failed" is not.
        """
        rung = _by_name(name)
        if rung.requires_prior is None:
            return True, "entry point"
        prior = _by_name(rung.requires_prior)
        if prior.name in self.passed:
            return True, f"after {prior.name}"
        blocking = self.first_unpassed()
        hint = f" (blocked at {blocking.name})" if blocking else ""
        return False, f"rung {prior.name} has not passed{hint}"

    def pass_rung(self, name: RungName, note: str = "") -> None:
        self.passed.add(name)
        if note:
            self.notes[name] = note

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": sorted(self.passed),
            "notes": dict(self.notes),
            "next": _next_name(self),
        }


def _next_name(state: LadderState) -> str | None:
    """mypy will not narrow through two calls to the same function, and a
    `first_unpassed()` that changes between them is a bug anyway."""
    nxt = state.first_unpassed()
    return nxt.name if nxt is not None else None


def _by_name(name: RungName) -> Rung:
    for r in LADDER:
        if r.name == name:
            return r
    raise KeyError(f"no rung named {name!r}; the ladder has {[r.name for r in LADDER]}")


def check_order(state: LadderState, name: RungName) -> None:
    """Raise rather than return, so a caller cannot forget to check."""
    ok, reason = state.may_enter(name)
    if not ok:
        raise LadderOrderViolationError(
            f"cannot enter rung {name!r}: {reason}. The ladder is ordered by what a "
            f"failure costs, not by what reads well, and it cannot be climbed from "
            f"the top."
        )
