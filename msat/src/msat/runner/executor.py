"""The executor (PRD §R5.1-§R5.6).

Joins the four pieces built separately — schedule, WAL, store, ledger — into the
loop that actually spends money. They were built apart on purpose: the spine is
very hard to debug when three concerns live in one class, and each of the four has
its own failing modes.

THE LOOP, IN ORDER, AND WHY THE ORDER MATTERS
--------------------------------------------
For each scheduled unit:

    1. compute the cache key
    2. if the store already has it -> WARM HIT, no call, no spend
    3. else generate, classify finish_reason, store (atomic), THEN WAL
    4. ledger

Step 3's ordering is the whole design. The response is committed to the store
BEFORE the WAL says it was done, so the WAL can only ever *lag* the store, never
lead it. A WAL that claimed work the store does not have would, on resume, skip
units that were never actually done — and the run would report success having
silently shrunk.

RECONCILIATION, NOT TRUST
-------------------------
Resume re-reads the WAL and then checks every `done` entry against the store. A
WAL entry with no response is treated as NOT DONE. The journal is evidence, the
store is the record.

FAILURES
--------
Retry to `max_attempts`, then mark FAILED and record an exclusion. Never counted as
a miss: a failed call is missing data, and treating it as a wrong answer shrinks
the denominator in whichever direction the failures happen to fall.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from msat.runner.ledger import RunLedger, classify_finish
from msat.runner.schedule import ScheduledUnit, build_schedule, schedule_digest
from msat.runner.wal import Wal, WalState, build_resume_state, read_wal
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse, cache_key


class BudgetExceededError(RuntimeError):
    """The run's declared ceiling was reached mid-flight.

    Enforced LIVE, per call, not estimated up front. N4 requires a hard ceiling, and
    an estimate computed from assumed throughput is exactly the number that turns
    out to be wrong on the day.
    """


@runtime_checkable
class Generator(Protocol):
    """Produces one response. Injected, so the executor is testable with no GPU.

    Same reasoning as the embedder in P3: baking a model in makes the loop
    untestable off the execution box, which is where a bug in it would be found
    last.
    """

    def __call__(self, unit: ScheduledUnit, *, attempt: int) -> StoredResponse: ...


#: Renders a unit into the request body the cache key is computed from. Separate
#: from the generator on purpose: the KEY must be derivable BEFORE anything is
#: spent, so the store can answer "already have this?" without loading a model.
Render = Callable[[ScheduledUnit], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_dir: Path
    seed: int = 20261012
    runs_per_cell: int = 3
    max_attempts: int = 3
    checkpoint_every: int = 50
    """WAL fsync happens per append anyway; this is how often the LEDGER is
    rewritten. Frequent enough that a kill loses little, rare enough that the
    rewrite is not the bottleneck."""
    max_usd: float | None = None
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    concurrency: int = 1
    """Recorded because concurrency changes latency (R5.3). Queue wait is recorded
    separately from server time for exactly that reason."""

    @property
    def responses_dir(self) -> Path:
        return self.run_dir / "responses"

    @property
    def wal_path(self) -> Path:
        return self.run_dir / "WAL"

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / "ledger.json"


@dataclass
class RunResult:
    n_planned: int = 0
    n_warm_hits: int = 0
    n_generated: int = 0
    n_failed: int = 0
    n_excluded: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd_spent: float = 0.0
    wall_seconds: float = 0.0
    schedule_digest: str = ""
    resumed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_planned": self.n_planned,
            "n_warm_hits": self.n_warm_hits,
            "n_generated": self.n_generated,
            "n_failed": self.n_failed,
            "n_excluded": self.n_excluded,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd_spent": round(self.usd_spent, 6),
            "wall_seconds": round(self.wall_seconds, 3),
            "schedule_digest": self.schedule_digest,
            "resumed": self.resumed,
        }


def make_schedule(
    cells: Sequence[ConditionKey],
    qids: Sequence[str],
    config: RunConfig,
) -> tuple[ScheduledUnit, ...]:
    return build_schedule(cells, qids, config.runs_per_cell, seed=config.seed)


def reconcile(
    schedule: Sequence[ScheduledUnit],
    wal_path: Path,
    store: ResponseStore,
    max_attempts: int,
) -> tuple[list[ScheduledUnit], int]:
    """Work still to do, and how many units were skipped as already done.

    A WAL `done` whose key is NOT in the store is NOT treated as done. That is the
    store being the record rather than the journal, and it is the difference
    between resuming correctly and silently dropping units.
    """
    state = build_resume_state(read_wal(wal_path))
    remaining: list[ScheduledUnit] = []
    skipped = 0
    for unit in schedule:
        uid = f"{unit.condition.key}|{unit.qid}|{unit.replicate}"
        if uid in state.done and _durable(store, wal_path, unit):
            # THE STORE IS THE RECORD. A WAL entry alone is a claim, not evidence.
            # Trusting it is how a resume silently skips units that were never done
            # - and the run then reports success having shrunk.
            skipped += 1
            continue
        if state.is_exhausted(uid, max_attempts):
            continue  # already failed its retries; recorded in the ledger
        remaining.append(unit)
    return remaining, skipped


def _durable(store: ResponseStore, wal_path: Path, unit: ScheduledUnit) -> bool:
    """Is there a response behind this WAL entry?

    Checks the store by the key the WAL recorded, so it does not need to re-render
    the request — which is the whole reason the cache key is recorded per entry.
    """
    from msat.runner.wal import read_wal as _read

    for entry in _read(wal_path):
        if entry.qid != unit.qid or entry.replicate != unit.replicate:
            continue
        if entry.cell != unit.condition.key:
            continue
        if entry.cache_key and store.has(entry.cache_key):
            return True
    return False


class Executor:
    """Runs a schedule against a store, resumably, under a budget."""

    def __init__(
        self,
        config: RunConfig,
        generator: Generator,
        render: Render,
        *,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.generator = generator
        self.render = render
        self.store = ResponseStore(config.responses_dir)
        self.ledger = RunLedger.load(config.ledger_path)
        self._resume = resume
        self._usd = 0.0

    # ── budget (N4) ───────────────────────────────────────────────────────────
    def _charge(self, prompt_tokens: int, completion_tokens: int) -> float:
        c = self.config
        usd = (
            prompt_tokens * c.input_cost_per_mtok + completion_tokens * c.output_cost_per_mtok
        ) / 1_000_000
        if c.max_usd is not None and self._usd + usd > c.max_usd:
            raise BudgetExceededError(
                f"run would spend ${self._usd + usd:,.4f}, over the ${c.max_usd:,.2f} "
                f"ceiling. Stopping here rather than finishing over budget."
            )
        self._usd += usd
        return usd

    _usd: float = 0.0

    # ── the loop ──────────────────────────────────────────────────────────────
    def run(self, schedule: Sequence[ScheduledUnit]) -> RunResult:
        cfg = self.config
        started = time.perf_counter()
        remaining, skipped = reconcile(schedule, cfg.wal_path, self.store, cfg.max_attempts)

        result = RunResult(
            n_planned=len(schedule),
            n_warm_hits=skipped,
            schedule_digest=schedule_digest(schedule),
            resumed=skipped > 0,
        )
        self._usd = 0.0

        with Wal(cfg.wal_path, resume=self._resume or skipped > 0) as wal:
            for index, unit in enumerate(remaining, start=1):
                key = self._key_for(unit)
                self.ledger.plan(unit.condition.key)

                if self.store.has(key):
                    # Warm hit: no call, no spend. Counted so a fully-cached replay
                    # is visibly a replay and not an empty run.
                    result.n_warm_hits += 1
                    wal.append(
                        WalState.DONE, unit.condition.key, unit.qid, unit.replicate, cache_key=key
                    )
                    continue

                for attempt in range(1, cfg.max_attempts + 1):
                    try:
                        record = self.generator(unit, attempt=attempt)
                    except Exception as exc:  # noqa: BLE001 - policy is per-call
                        if attempt >= cfg.max_attempts:
                            wal.append(
                                WalState.FAILED,
                                unit.condition.key,
                                unit.qid,
                                unit.replicate,
                                attempts=attempt,
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            self.ledger.fail(
                                unit.condition.key, "error", f"{type(exc).__name__}: {exc}"
                            )
                            result.n_failed += 1
                            break
                        continue  # retry

                    reason = classify_finish(
                        record.text,
                        record.completion_tokens,
                        int(record.request.get("max_tokens", 512)),
                    )
                    record = replace(
                        record,
                        key=key,
                        condition=unit.condition.as_dict(),
                        request=self.render(unit),
                        finish_reason=reason,
                    )
                    # STORE FIRST. The WAL may only ever lag the store.
                    self.store.put(record)
                    wal.append(
                        WalState.DONE,
                        unit.condition.key,
                        unit.qid,
                        unit.replicate,
                        cache_key=key,
                        attempts=attempt,
                    )

                    result.n_generated += 1
                    result.prompt_tokens += record.prompt_tokens
                    result.completion_tokens += record.completion_tokens
                    self._charge(record.prompt_tokens, record.completion_tokens)
                    self.ledger.complete(unit.condition.key, reason)
                    if reason != "stop":
                        result.n_excluded += 1
                    break

                if index % cfg.checkpoint_every == 0:
                    self.ledger.write(cfg.ledger_path)

            self.ledger.write(cfg.ledger_path)

        result.n_excluded = self.ledger.total_excluded
        result.usd_spent = self._usd
        result.wall_seconds = time.perf_counter() - started
        return result

    def _key_for(self, unit: ScheduledUnit) -> str:
        """The cache key, derivable WITHOUT spending anything.

        That is why rendering is a separate callable rather than part of the
        generator: the store has to be able to answer "already have this?" before
        a model is loaded, or a fully-cached replay would load 5 checkpoints to
        discover it has nothing to do.
        """
        return cache_key(unit.condition, self.render(unit))


def status_report(config: RunConfig) -> dict[str, Any]:
    """Per-cell completion, from the ledger (PRD §R5.4)."""
    ledger = RunLedger.load(config.ledger_path)
    store = ResponseStore(config.responses_dir)
    return {
        "run_dir": str(config.run_dir),
        "n_stored_responses": len(store),
        "n_cells": len(ledger.cells),
        "n_planned": sum(c.n_planned for c in ledger.cells.values()),
        "n_done": sum(c.n_done for c in ledger.cells.values()),
        "n_failed": sum(c.n_failed for c in ledger.cells.values()),
        "n_excluded": ledger.total_excluded,
        "cells": ledger.as_dict()["cells"],
    }
