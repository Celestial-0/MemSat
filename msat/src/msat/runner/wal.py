"""Write-ahead log and resume (PRD §R5.2).

*"kill -9 mid-run → resume → IDENTICAL results."*

WHAT THE WAL IS ACTUALLY FOR
----------------------------
Not for recovering the responses — those are already durable in the content-addressed
store, committed by atomic rename before the WAL says anything about them. The WAL
records what was **attempted**, so that resume knows three things:

    1. which units are DONE (and can be verified against the store, not trusted)
    2. which FAILED and how many times (so retries are bounded)
    3. the order already executed

The store is the source of truth. The WAL is the journal. On resume the two are
RECONCILED rather than one being assumed to match the other — a WAL entry saying
`done` with no response in the store means that response was lost, and treating the
WAL's word for it would silently shrink the run.

APPEND-ONLY, AND WHY THAT IS NOT OPTIONAL
-----------------------------------------
Rewriting the WAL in place to compact it is the obvious optimisation and it breaks
the one property that matters: after a `kill -9` the file may end mid-line. An
append-only log can lose its LAST line and still be completely parseable. A
rewritten file can be left structurally inconsistent, and then there is no way to
tell which entries were real.

`fsync` ON EVERY APPEND
----------------------
Also not optional. Without it the OS may hold the write in a buffer when the process
dies, and the resumed run reads a WAL that is missing entries the executor believes
it wrote. That failure is intermittent and depends on machine state, which is the
worst kind.
"""

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
    replicate: int
    cache_key: str = ""
    attempts: int = 1
    detail: str = ""

    @property
    def unit_id(self) -> str:
        return f"{self.cell}|{self.qid}|{self.replicate}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "state": str(self.state),
            "cell": list(self.cell),
            "qid": self.qid,
            "replicate": self.replicate,
            "cache_key": self.cache_key,
            "attempts": self.attempts,
            "detail": self.detail,
        }


@dataclass
class ResumeState:
    """Reconciled view of a run, rebuilt from the WAL."""

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
    """Yield every parseable entry. A truncated final line is skipped, not fatal.

    That is the entire reason the log is append-only: after `kill -9` the file may
    end mid-write, and the damaged line is always the LAST one — so everything
    before it is trustworthy and the run resumes correctly.
    """
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
                    replicate=int(raw["replicate"]),
                    cache_key=str(raw.get("cache_key", "")),
                    attempts=int(raw.get("attempts", 1)),
                    detail=str(raw.get("detail", "")),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue


def build_resume_state(entries: Iterable[WalEntry]) -> ResumeState:
    """Fold the WAL into the set of units that need no work.

    `failed` is kept separately from `done`: a failed unit is retried up to
    `max_attempts` and then DROPPED and excluded from paired comparisons, never
    silently treated as complete.
    """
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
    """Append-only journal with an fsync per append."""

    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = max((e.seq for e in read_wal(self.path)), default=-1) + 1 if resume else 0
        # Truncate on a fresh run: resuming without `--resume` is a NEW run and
        # inheriting a stale journal would skip work it never did.
        self._fh = self.path.open("a" if resume else "w", encoding="utf-8", newline="\n")

    def append(
        self,
        state: WalState,
        cell: tuple,
        qid: str,
        replicate: int,
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
            replicate=replicate,
            cache_key=cache_key,
            attempts=attempts,
            detail=detail,
        )
        self._fh.write(json.dumps(entry.as_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        self._fh.flush()
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
