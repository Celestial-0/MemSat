"""The executor: the loop that spends money."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from msat.runner.ledger import RunLedger, classify_finish
from msat.runner.schedule import ScheduledUnit, build_schedule, schedule_digest
from msat.runner.wal import Wal, WalState, build_resume_state, read_wal
from msat.store import Condition, ResponseStore, StoredResponse, cache_key


class BudgetExceededError(RuntimeError):
    """The run's declared budget ceiling was reached mid-flight."""


@runtime_checkable
class Generator(Protocol):
    def __call__(self, unit: ScheduledUnit, *, attempt: int) -> StoredResponse: ...


Render = Callable[[ScheduledUnit], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_dir: Path
    seed: int = 20261012
    runs_per_cell: int = 3
    max_attempts: int = 3
    checkpoint_every: int = 50
    max_usd: float | None = None
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    concurrency: int = 1

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
    cells: Sequence[Condition],
    qids: Sequence[str],
    config: RunConfig,
) -> tuple[ScheduledUnit, ...]:
    return build_schedule(cells, qids, repetition=config.runs_per_cell, seed=config.seed)


def reconcile(
    schedule: Sequence[ScheduledUnit],
    wal_path: Path,
    store: ResponseStore,
    max_attempts: int,
    expected_keys: Mapping[str, str] | None = None,
) -> tuple[list[ScheduledUnit], int]:
    # Single-pass read of the WAL file: O(W)
    wal_entries = list(read_wal(wal_path))
    state = build_resume_state(wal_entries)

    # Collect candidate cache keys for DONE entries in the same pass
    done_cache_keys: dict[str, list[str]] = {}
    for entry in wal_entries:
        if entry.state is WalState.DONE and entry.cache_key:
            done_cache_keys.setdefault(entry.unit_id, []).append(entry.cache_key)

    # Verified cache of store lookups to avoid duplicate file reads
    checked_store_keys: dict[str, bool] = {}

    def _is_durable_key(ck: str) -> bool:
        if ck not in checked_store_keys:
            # Must exist and be valid uncorrupted JSON in store
            checked_store_keys[ck] = bool(store.has(ck) and store.get(ck) is not None)
        return checked_store_keys[ck]

    remaining: list[ScheduledUnit] = []
    skipped = 0
    # Single pass over schedule: O(N)
    for unit in schedule:
        uid = f"{unit.condition.key}|{unit.qid}|{unit.repetition}"
        if uid in state.done:
            keys = done_cache_keys.get(uid, [])
            if expected_keys is not None:
                expected = expected_keys[uid]
                if expected not in keys and any(_is_durable_key(key) for key in keys):
                    raise ValueError(
                        "Request bytes changed for completed work; resume requires the original inputs."
                    )
                keys = [key for key in keys if key == expected]
            if any(_is_durable_key(k) for k in keys):
                skipped += 1
                continue
        if state.is_exhausted(uid, max_attempts):
            continue
        remaining.append(unit)
    return remaining, skipped


class Executor:
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

    def run(self, schedule: Sequence[ScheduledUnit]) -> RunResult:
        cfg = self.config
        started = time.perf_counter()
        expected_keys = {
            f"{unit.condition.key}|{unit.qid}|{unit.repetition}": self._key_for(unit)
            for unit in schedule
        }
        remaining, skipped = reconcile(
            schedule, cfg.wal_path, self.store, cfg.max_attempts, expected_keys
        )
        remaining_set = set(remaining)

        result = RunResult(
            n_planned=len(schedule),
            n_warm_hits=skipped,
            schedule_digest=schedule_digest(schedule),
            resumed=skipped > 0,
        )
        self._usd = 0.0

        # For skipped units from reconcile, ensure repetition membership is merged and ledger counts match
        for unit in schedule:
            if unit not in remaining_set:
                key = self._key_for(unit)
                record = self.store.get(key)
                if record is not None:
                    if unit.repetition not in record.repetitions:
                        record = replace(
                            record,
                            repetitions=tuple(sorted(set(record.repetitions) | {unit.repetition})),
                        )
                        self.store.put(record)
                    cell_ledger = self.ledger.cells.get(unit.condition.key)
                    if cell_ledger is None or (
                        cell_ledger.n_done + cell_ledger.n_excluded + cell_ledger.n_failed
                        < cell_ledger.n_planned
                    ):
                        self.ledger.plan(unit.condition.key)
                        self.ledger.complete(unit.condition.key, record.finish_reason)
                        if record.finish_reason != "stop":
                            result.n_excluded += 1

        with Wal(cfg.wal_path, resume=self._resume or skipped > 0) as wal:
            for index, unit in enumerate(remaining, start=1):
                key = self._key_for(unit)
                self.ledger.plan(unit.condition.key)

                if self.store.has(key):
                    result.n_warm_hits += 1
                    record = self.store.get(key)
                    if record is not None:
                        if unit.repetition not in record.repetitions:
                            record = replace(
                                record,
                                repetitions=tuple(
                                    sorted(set(record.repetitions) | {unit.repetition})
                                ),
                            )
                            self.store.put(record)
                        self.ledger.complete(unit.condition.key, record.finish_reason)
                        if record.finish_reason != "stop":
                            result.n_excluded += 1
                    wal.append(
                        WalState.DONE, unit.condition.key, unit.qid, unit.repetition, cache_key=key
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
                                unit.repetition,
                                attempts=attempt,
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            self.ledger.fail(
                                unit.condition.key, "error", f"{type(exc).__name__}: {exc}"
                            )
                            result.n_failed += 1
                            break
                        continue

                    reason = (
                        record.finish_reason
                        if record.finish_reason not in ("", "stop")
                        else classify_finish(
                            record.text,
                            record.completion_tokens,
                            int(record.request.get("max_tokens", 512)),
                        )
                    )
                    existing = self.store.get(key)
                    existing_reps = set(existing.repetitions) if existing else set()
                    merged_reps = tuple(
                        sorted(existing_reps | set(record.repetitions) | {unit.repetition})
                    )
                    record = replace(
                        record,
                        key=key,
                        condition=unit.condition.as_dict(),
                        request=self.render(unit),
                        finish_reason=reason,
                        repetitions=merged_reps,
                    )
                    self.store.put(record)
                    wal.append(
                        WalState.DONE,
                        unit.condition.key,
                        unit.qid,
                        unit.repetition,
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
        return cache_key(unit.condition, self.render(unit))


def status_report(config: RunConfig) -> dict[str, Any]:
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
