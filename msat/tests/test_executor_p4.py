"""The executor loop, end to end, with no model.

Uses an injected generator that is deterministic and deliberately misbehaves in
controlled ways, because the failure paths are the parts worth testing: warm hits,
retries, exhaustion, exclusions, budget abort, and resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msat.runner.executor import (
    BudgetExceededError,
    Executor,
    RunConfig,
    make_schedule,
    reconcile,
    status_report,
)
from msat.runner.ledger import RunLedger
from msat.runner.wal import WalState, build_resume_state, read_wal
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse

CELLS = [
    ConditionKey("A+G", k, b, "per_memory_tokens", s, r)
    for k in (1, 4)
    for b in (128.0, 512.0)
    for s in ("S",)
    for r in ("gemma",)
]
QIDS = ["q1", "q2"]


def render(unit) -> dict[str, Any]:
    """Request body. Distinct per unit so keys do not collide."""
    return {
        "model": "local/reader",
        "messages": [{"role": "user", "content": f"{unit.qid}:{unit.replicate}"}],
        "max_tokens": 32,
        "temperature": 0.7,
        "seed": hash((unit.qid, unit.replicate)) % 100_000,
    }


def make_generator(text: str = "answer", calls: list | None = None):
    def gen(unit, *, attempt: int):
        if calls is not None:
            calls.append(unit)
        body = render(unit)
        return StoredResponse(
            key="",
            condition=unit.condition.as_dict(),
            request=body,
            text=text,
            finish_reason="stop",
            model_version="sha:test",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )

    return gen


def cfg(tmp_path: Path, **kw: Any) -> RunConfig:
    opts: dict[str, Any] = {"runs_per_cell": 1, "checkpoint_every": 1000}
    opts.update(kw)
    return RunConfig(run_dir=tmp_path, **opts)


# ── the happy path ────────────────────────────────────────────────────────────
def test_a_fresh_run_generates_everything(tmp_path: Path) -> None:
    ex = Executor(cfg(tmp_path), make_generator(), render)
    sched = make_schedule(CELLS, QIDS, ex.config)
    result = ex.run(sched)

    assert result.n_planned == len(CELLS) * len(QIDS)
    assert result.n_generated == result.n_planned
    assert result.n_warm_hits == 0
    assert result.n_failed == 0
    assert len(ResponseStore(tmp_path / "responses")) == result.n_generated


def test_the_store_is_committed_before_the_wal(tmp_path: Path) -> None:
    """Every WAL `done` must have a response behind it.

    The WAL may only ever LAG the store. A WAL entry with no response would, on
    resume, skip a unit that was never actually done.
    """
    ex = Executor(cfg(tmp_path), make_generator(), render)
    ex.run(make_schedule(CELLS, QIDS, ex.config))

    store = ResponseStore(tmp_path / "responses")
    state = build_resume_state(read_wal(tmp_path / "WAL"))
    assert len(state.done) == len(store)


def test_a_second_run_is_entirely_warm_and_spends_nothing(tmp_path: Path) -> None:
    sched = make_schedule(CELLS, QIDS, cfg(tmp_path))
    Executor(cfg(tmp_path), make_generator(), render).run(sched)

    calls: list = []
    ex = Executor(cfg(tmp_path), make_generator(calls=calls), render, resume=True)
    result = ex.run(sched)

    assert calls == [], "a warm run must not call the generator at all"
    assert result.n_generated == 0
    assert result.n_warm_hits == result.n_planned
    assert result.usd_spent == 0.0


# ── resume ────────────────────────────────────────────────────────────────────
def test_resume_after_a_partial_wal_does_only_the_remainder(tmp_path: Path) -> None:
    """A WAL that claims more than the store holds must NOT be trusted."""
    sched = make_schedule(CELLS, QIDS, cfg(tmp_path))

    # Fabricate a WAL entry with no response behind it.
    with __import__("msat.runner.wal", fromlist=["Wal"]).Wal(tmp_path / "WAL") as wal:
        wal.append(WalState.DONE, CELLS[0].key, "q1", 0, cache_key="deadbeef")

    remaining, skipped = reconcile(
        sched, tmp_path / "WAL", ResponseStore(tmp_path / "responses"), 3
    )
    assert skipped == 0, "a WAL entry with no response is not evidence of work done"
    assert len(remaining) == len(sched)


def test_resume_after_a_real_partial_run_skips_what_is_stored(tmp_path: Path) -> None:
    ex = Executor(cfg(tmp_path), make_generator(), render)
    sched = make_schedule(CELLS, QIDS, ex.config)
    ex.run(sched[: len(sched) // 2])

    ex2 = Executor(cfg(tmp_path), make_generator(), render, resume=True)
    result = ex2.run(sched)
    assert result.n_warm_hits == len(sched) // 2
    assert result.n_generated == len(sched) - len(sched) // 2


# ── failures ──────────────────────────────────────────────────────────────────
def test_a_transient_failure_is_retried_then_succeeds(tmp_path: Path) -> None:
    attempts: list[int] = []

    def flaky(unit, *, attempt: int):
        attempts.append(attempt)
        if attempt < 2:
            raise ConnectionError("transient")
        return make_generator()(unit, attempt=attempt)

    ex = Executor(cfg(tmp_path), flaky, render)
    result = ex.run(make_schedule(CELLS[:1], ["q1"], ex.config))
    assert attempts == [1, 2]
    assert result.n_failed == 0
    assert result.n_generated == 1


def test_an_exhausted_unit_is_failed_not_counted_as_a_miss(tmp_path: Path) -> None:
    """A failed call is MISSING DATA. Treating it as a wrong answer shrinks the
    denominator in whichever direction the failures happen to fall."""

    def always_fails(unit, *, attempt: int):
        raise TimeoutError("nope")

    ex = Executor(cfg(tmp_path), always_fails, render)
    result = ex.run(make_schedule(CELLS[:1], ["q1"], ex.config))
    assert result.n_failed == 1
    assert result.n_generated == 0

    ledger = RunLedger.load(tmp_path / "ledger.json")
    cell = ledger.cell(CELLS[0].key)
    assert cell.n_done == 0, "a failure must never be recorded as a completed answer"
    assert ledger.exclusions[0]["finish_reason"] == "error"


def test_a_failed_unit_is_not_retried_past_max_attempts_on_resume(tmp_path: Path) -> None:
    def always_fails(unit, *, attempt: int):
        raise TimeoutError("nope")

    ex = Executor(cfg(tmp_path), always_fails, render)
    sched = make_schedule(CELLS[:1], ["q1"], ex.config)
    ex.run(sched)

    ex2 = Executor(cfg(tmp_path), always_fails, render, resume=True)
    result = ex2.run(sched)
    assert result.n_generated == 0
    assert result.n_warm_hits == 0


# ── finish_reason policy in the loop ──────────────────────────────────────────
def test_a_truncated_answer_is_excluded_not_scored_as_a_miss(tmp_path: Path) -> None:
    """Pooling a truncation with ordinary misses biases the curve in a direction
    nobody can detect from the results."""

    def truncating(unit, *, attempt: int):
        rec = make_generator()(unit, attempt=attempt)
        return type(rec)(
            **{
                **{f: getattr(rec, f) for f in rec.__dataclass_fields__},
                "text": "cut off",
                "completion_tokens": 32,
            }
        )

    ex = Executor(cfg(tmp_path), truncating, render)
    result = ex.run(make_schedule(CELLS[:1], ["q1"], ex.config))
    assert result.n_generated == 1
    assert result.n_excluded == 1

    ledger = RunLedger.load(tmp_path / "ledger.json")
    assert ledger.cell(CELLS[0].key).n_done == 0
    assert ledger.cell(CELLS[0].key).by_finish_reason["length"] == 1


# ── the budget (N4) ───────────────────────────────────────────────────────────
def test_the_budget_aborts_live_rather_than_finishing_over(tmp_path: Path) -> None:
    """An up-front estimate from assumed throughput is exactly the number that is
    wrong on the day."""
    ex = Executor(
        cfg(tmp_path, max_usd=0.000_05, input_cost_per_mtok=1.0, output_cost_per_mtok=1.0),
        make_generator(),
        render,
    )
    with pytest.raises(BudgetExceededError, match="over the"):
        ex.run(make_schedule(CELLS, QIDS, ex.config))


def test_cost_accumulates_when_under_budget(tmp_path: Path) -> None:
    ex = Executor(
        cfg(tmp_path, max_usd=100.0, input_cost_per_mtok=1.0, output_cost_per_mtok=1.0),
        make_generator(),
        render,
    )
    result = ex.run(make_schedule(CELLS[:1], ["q1"], ex.config))
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.usd_spent == pytest.approx(15 / 1_000_000)


# ── status ────────────────────────────────────────────────────────────────────
def test_status_reports_per_cell_completion(tmp_path: Path) -> None:
    ex = Executor(cfg(tmp_path), make_generator(), render)
    ex.run(make_schedule(CELLS, QIDS, ex.config))

    report = status_report(cfg(tmp_path))
    assert report["n_stored_responses"] == len(CELLS) * len(QIDS)
    assert report["n_planned"] == len(CELLS) * len(QIDS)
    assert report["n_done"] == len(CELLS) * len(QIDS)
    assert report["n_failed"] == 0
    assert len(report["cells"]) == len(CELLS)


def test_the_ledger_is_checkpointed_during_the_run(tmp_path: Path) -> None:
    """A kill must not lose the whole run's bookkeeping."""
    ex = Executor(cfg(tmp_path, checkpoint_every=1), make_generator(), render)
    ex.run(make_schedule(CELLS, QIDS, ex.config))
    assert (tmp_path / "ledger.json").exists()
    assert json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["n_done"] > 0


def test_the_run_is_deterministic(tmp_path: Path) -> None:
    """Same schedule, same seed, same work order — replay depends on it."""
    a = make_schedule(CELLS, QIDS, cfg(tmp_path, seed=5))
    b = make_schedule(CELLS, QIDS, cfg(tmp_path, seed=5))
    assert [u.condition.key for u in a] == [u.condition.key for u in b]
    assert [u.qid for u in a] == [u.qid for u in b]
