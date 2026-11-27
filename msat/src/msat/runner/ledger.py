"""`finish_reason` policy and the per-cell completion ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

EXCLUDED_FINISH_REASONS: Mapping[str, str] = {
    "length": (
        "output hit the token cap: the condition was too long for the reader to "
        "answer, so a miss would blame the reader for a design constraint"
    ),
    "refusal": "the reader declined; that is not the same as not knowing",
    "empty": "no output at all",
    "error": "the call failed after its retries",
}


class CellState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class CellLedger:
    n_planned: int = 0
    n_done: int = 0
    n_failed: int = 0
    n_excluded: int = 0
    by_finish_reason: dict[str, int] = field(default_factory=dict)

    def record(self, finish_reason: str) -> None:
        self.by_finish_reason[finish_reason] = self.by_finish_reason.get(finish_reason, 0) + 1

    @property
    def completion_rate(self) -> float:
        return self.n_done / self.n_planned if self.n_planned else 0.0

    @property
    def state(self) -> CellState:
        if self.n_done == 0:
            return CellState.FAILED
        if self.n_done < self.n_planned:
            return CellState.PARTIAL
        return CellState.COMPLETE

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "completion_rate": round(self.completion_rate, 6),
            "state": str(self.state),
            "excluded_breakdown": {
                k: v
                for k, v in sorted(self.by_finish_reason.items())
                if k in EXCLUDED_FINISH_REASONS
            },
        }


@dataclass
class RunLedger:
    cells: dict[tuple, CellLedger] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)

    def cell(self, key: tuple) -> CellLedger:
        return self.cells.setdefault(key, CellLedger())

    def plan(self, key: tuple) -> None:
        self.cell(key).n_planned += 1

    def complete(self, key: tuple, finish_reason: str) -> None:
        entry = self.cell(key)
        entry.record(finish_reason)
        if finish_reason in EXCLUDED_FINISH_REASONS:
            entry.n_excluded += 1
        else:
            entry.n_done += 1

    def fail(self, key: tuple, finish_reason: str, detail: str = "") -> None:
        entry = self.cell(key)
        entry.record(finish_reason)
        entry.n_failed += 1
        self.exclusions.append(
            {
                "cell": list(key),
                "finish_reason": finish_reason,
                "reason": EXCLUDED_FINISH_REASONS.get(finish_reason, "unclassified"),
                "detail": detail,
            }
        )

    @property
    def total_excluded(self) -> int:
        return sum(c.n_excluded for c in self.cells.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cells": len(self.cells),
            "n_planned": sum(c.n_planned for c in self.cells.values()),
            "n_done": sum(c.n_done for c in self.cells.values()),
            "n_failed": sum(c.n_failed for c in self.cells.values()),
            "n_excluded": self.total_excluded,
            "cells": [_cell_record(k, v) for k, v in sorted(self.cells.items())],
            "exclusions": self.exclusions,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> RunLedger:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls(exclusions=list(raw.get("exclusions", [])))
        for cell in raw.get("cells", []):
            ledger.cells[_typed_key(cell["key"])] = CellLedger(
                n_planned=int(cell["n_planned"]),
                n_done=int(cell["n_done"]),
                n_failed=int(cell["n_failed"]),
                n_excluded=int(cell["n_excluded"]),
                by_finish_reason=dict(cell.get("by_finish_reason", {})),
            )
        return ledger


def _cell_record(key: tuple, cell: CellLedger) -> dict[str, Any]:
    return {"key": list(key), **cell.as_dict()}


def _typed_key(parts: list[Any]) -> tuple:
    if len(parts) != 6:
        return tuple(parts)
    arm, k, b, b_mode, scale, reader = parts
    return (str(arm), int(k), float(b), str(b_mode), str(scale), str(reader))


def classify_finish(
    text: str, completion_tokens: int, max_tokens: int, *, stop: tuple[str, ...] = ()
) -> str:
    if not text.strip():
        return "empty"
    lowered = text.lower()
    if any(
        marker in lowered for marker in ("i cannot", "i can't assist", "i'm sorry", "i am sorry")
    ):
        return "refusal"
    if completion_tokens >= max_tokens:
        return "length"
    return "stop"


def is_excluded(finish_reason: str) -> bool:
    return finish_reason in EXCLUDED_FINISH_REASONS
