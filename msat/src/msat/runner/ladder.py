"""The smoke ladder: what runs before the real run may start."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

RungName = Literal["negative", "baselines", "crossed"]


@dataclass(frozen=True, slots=True)
class Rung:
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
        name="crossed",
        n_questions=-1,
        arms=("A", "B", "C", "D", "E", "F", "G"),
        scales=("S", "M"),
        requires_prior="baselines",
        description="the crossed grid",
    ),
)


class LadderOrderViolationError(RuntimeError):
    """Raised when a rung is entered above the last completed one."""


@dataclass
class LadderState:
    passed: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)

    def first_unpassed(self) -> Rung | None:
        for rung in LADDER:
            if rung.name not in self.passed:
                return rung
        return None

    def may_enter(self, name: RungName) -> tuple[bool, str]:
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
    nxt = state.first_unpassed()
    return nxt.name if nxt is not None else None


def _by_name(name: RungName) -> Rung:
    for r in LADDER:
        if r.name == name:
            return r
    raise KeyError(f"no rung named {name!r}; the ladder has {[r.name for r in LADDER]}")


def check_order(state: LadderState, name: RungName) -> None:
    ok, reason = state.may_enter(name)
    if not ok:
        raise LadderOrderViolationError(
            f"cannot enter rung {name!r}: {reason}. The ladder is ordered by what a "
            f"failure costs, not by what reads well, and it cannot be climbed from "
            f"the top."
        )
