"""Write-ahead log and resume."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

WAL_SCHEMA = 1


class WalState(StrEnum):
    DONE = "done"
    FAILED = "failed"
    STARTED = "started"


@dataclass(frozen=True, slots=True)
class WalEntry:
    seq: int
    state: WalState
    cell: tuple
    qid: str
    repetition: int
    cache_key: str = ""
    attempts: int = 1
    detail: str = ""

    @property
    def unit_id(self) -> str:
        return f"{self.cell}|{self.qid}|{self.repetition}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "state": str(self.state),
            "cell": list(self.cell),
            "qid": self.qid,
            "repetition": self.repetition,
            "replicate": self.repetition,
            "cache_key": self.cache_key,
            "attempts": self.attempts,
            "detail": self.detail,
        }


@dataclass
class ResumeState:
    done: set[str] = field(default_factory=set)
    failed: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    n_entries: int = 0

    def attempt_count(self, unit_id: str) -> int:
        return self.attempts.get(unit_id, 0)

    def is_exhausted(self, unit_id: str, max_attempts: int) -> bool:
        return self.attempt_count(unit_id) >= max_attempts


def read_wal(path: Path) -> Iterator[WalEntry]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                yield WalEntry(
                    seq=int(raw["seq"]),
                    state=WalState(raw["state"]),
                    cell=tuple(raw["cell"]),  # type: ignore[arg-type]
                    qid=str(raw["qid"]),
                    repetition=int(raw.get("repetition", raw.get("replicate", 0))),
                    cache_key=str(raw.get("cache_key", "")),
                    attempts=int(raw.get("attempts", 1)),
                    detail=str(raw.get("detail", "")),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue


def build_resume_state(entries: Iterable[WalEntry]) -> ResumeState:
    state = ResumeState()
    for e in entries:
        state.n_entries += 1
        uid = e.unit_id
        state.order.append(uid)
        state.attempts[uid] = max(state.attempts.get(uid, 0), e.attempts)
        if e.state is WalState.DONE:
            state.done.add(uid)
            state.failed.pop(uid, None)
        elif e.state is WalState.FAILED:
            state.failed[uid] = e.attempts
    return state


class Wal:
    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = max((e.seq for e in read_wal(self.path)), default=-1) + 1 if resume else 0
        self._fh = self.path.open("a" if resume else "w", encoding="utf-8", newline="\n")

    def append(
        self,
        state: WalState,
        cell: tuple,
        qid: str,
        repetition: int,
        *,
        cache_key: str = "",
        attempts: int = 1,
        detail: str = "",
    ) -> WalEntry:
        entry = WalEntry(
            seq=self._seq,
            state=state,
            cell=cell,
            qid=qid,
            repetition=repetition,
            cache_key=cache_key,
            attempts=attempts,
            detail=detail,
        )
        self._fh.write(json.dumps(entry.as_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        self._fh.flush()
        # fsync per append, so a kill -9 mid-run cannot lose a completed call.
        os.fsync(self._fh.fileno())
        self._seq += 1
        return entry

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Wal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
