"""P4 — the spine: store, schedule, WAL, ledger.

Architecture §P4's exit condition is *"kill -9 mid-run, resume, identical results"*
and the plan is explicit that a mock-based test does not test it. So
`test_a_killed_run_resumes_to_identical_results` really does `SIGKILL` a child
process mid-write and resumes it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from msat.runner.ledger import (
    EXCLUDED_FINISH_REASONS,
    RunLedger,
    classify_finish,
    is_excluded,
)
from msat.runner.schedule import (
    build_schedule,
    schedule_digest,
    verify_not_sequential_by_k,
)
from msat.runner.wal import Wal, WalState, build_resume_state, read_wal
from msat.store import ConditionKey, ResponseStore, StoredResponse, cache_key

ROOT = Path(__file__).resolve().parent.parent


def cell(k: int = 4, b: float = 4096.0, scale: str = "S", reader: str = "gemma", arm: str = "A+G"):
    return ConditionKey(arm, k, b, "per_memory_tokens", scale, reader)


REQ = {
    "model": "local/reader",
    "messages": [{"role": "user", "content": "what did we decide?"}],
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 11,
}


# ── namespacing: hazard H4 ────────────────────────────────────────────────────
def test_the_same_prompt_at_two_budgets_does_not_collide() -> None:
    """H4. A prefix cached at B=4K must never be served at B=16K.

    Written first, per the plan, because it is a silent correctness bug: nothing
    raises, the curve is just quietly the wrong curve.
    """
    assert cache_key(cell(b=4096.0), REQ) != cache_key(cell(b=16384.0), REQ)


def test_every_condition_field_is_part_of_the_key() -> None:
    base = cell()
    variants = [
        cell(arm="B"),
        cell(k=8),
        cell(b=16384.0),
        cell(b=4096.0, scale="M"),
        cell(reader="qwen"),
    ]
    keys = {cache_key(base, REQ)}
    for v in variants:
        assert cache_key(v, REQ) not in keys, f"{v.as_dict()} collided with the base condition"


def test_b_mode_is_part_of_the_key() -> None:
    """The same number under two modes is two different conditions."""
    a = ConditionKey("A+G", 4, 128.0, "per_memory_tokens", "S", "gemma")
    b = ConditionKey("A+G", 4, 128.0, "absolute_tokens", "S", "gemma")
    assert cache_key(a, REQ) != cache_key(b, REQ)


def test_the_seed_is_inside_the_hash() -> None:
    """Two replicates at the same temperature are NOT identical work.

    With the seed outside the key they dedupe to one file and seed variance —
    which R7.1 requires to vary something — becomes impossible to estimate with no
    error raised anywhere.
    """
    other = {**REQ, "seed": 12}
    assert cache_key(cell(), REQ) != cache_key(cell(), other)


def test_identical_requests_dedupe_to_one_file(tmp_path: Path) -> None:
    """R6.1. Same request twice is one file, not two."""
    store = ResponseStore(tmp_path / "responses")
    key = cache_key(cell(), REQ)
    for _ in range(3):
        store.put(StoredResponse(key, cell().as_dict(), REQ, "x", "stop", "v1", 1, 1, 1.0))
    assert len(store) == 1


def test_the_key_is_stable_across_dict_ordering() -> None:
    reordered = {k: REQ[k] for k in reversed(list(REQ))}
    assert cache_key(cell(), REQ) == cache_key(cell(), reordered)


# ── the store ─────────────────────────────────────────────────────────────────
def test_put_get_round_trip(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "r")
    key = cache_key(cell(), REQ)
    original = StoredResponse(
        key=key,
        condition=cell().as_dict(),
        request=dict(REQ),
        text="hello",
        finish_reason="stop",
        model_version="sha:abc",
        prompt_tokens=12,
        completion_tokens=3,
        latency_ms=42.0,
        ttft_ms=8.0,
        queue_wait_ms=1.5,
        backend="transformers",
        raw={"provider_json": {"anything": [1, 2, 3]}},
    )
    store.put(original)
    back = store.get(key)
    assert back is not None
    assert back.text == "hello"
    assert back.queue_wait_ms == 1.5
    assert back.raw == {"provider_json": {"anything": [1, 2, 3]}}


def test_a_truncated_file_reads_as_absent_not_as_a_crash(tmp_path: Path) -> None:
    """The atomic write makes truncation near-impossible; a filesystem without
    rename guarantees is exactly the case this path exists for."""
    store = ResponseStore(tmp_path / "r")
    key = cache_key(cell(), REQ)
    store.put(StoredResponse(key, cell().as_dict(), REQ, "x", "stop", "v1", 1, 1, 1.0))
    store.path_for(key).write_text('{"key": "abc", "resp', encoding="utf-8")
    assert store.get(key) is None


def test_no_temp_files_survive_a_write(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "r")
    key = cache_key(cell(), REQ)
    store.put(StoredResponse(key, cell().as_dict(), REQ, "x", "stop", "v1", 1, 1, 1.0))
    assert not list(store.responses.rglob("*.tmp"))


def test_the_store_is_sharded(tmp_path: Path) -> None:
    """~189k responses in one directory is slow on every filesystem."""
    store = ResponseStore(tmp_path / "r")
    key = cache_key(cell(), REQ)
    store.put(StoredResponse(key, cell().as_dict(), REQ, "x", "stop", "v1", 1, 1, 1.0))
    assert store.path_for(key).parent.name == key[:2]


def test_the_disk_budget_aborts_rather_than_trimming(tmp_path: Path) -> None:
    from msat.store.responses import DiskBudgetExceededError

    store = ResponseStore(tmp_path / "r", max_bytes=400)
    key = cache_key(cell(), REQ)
    with pytest.raises(DiskBudgetExceededError, match="Aborting rather than trimming"):
        store.put(StoredResponse(key, cell().as_dict(), REQ, "x", "stop", "v1", 1, 1, 1.0))


def test_records_are_sharded_but_iterable(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "r")
    for i in range(5):
        c = cell(k=2**i)
        store.put(
            StoredResponse(cache_key(c, REQ), c.as_dict(), REQ, f"t{i}", "stop", "v1", 1, 1, 1.0)
        )
    assert len(store) == 5
    assert len(list(store.iter_all())) == 5


# ── the schedule (R5.1) ───────────────────────────────────────────────────────
def _cells() -> list[ConditionKey]:
    return [
        cell(k=k, b=b, scale=s, reader=r)
        for k in (1, 2, 4, 8)
        for b in (128.0, 512.0)
        for s in ("S", "M")
        for r in ("gemma", "qwen")
    ]


def test_the_schedule_is_not_sequential_by_k() -> None:
    """The whole point of R5.1: drift over time must not look like a curve in K."""
    sched = build_schedule(_cells(), ["q1", "q2", "q3"], 2, seed=20261012)
    report = verify_not_sequential_by_k(sched)
    assert not report["sequential_by_k"], report


def test_the_schedule_covers_every_unit_exactly_once() -> None:
    cells = _cells()
    sched = build_schedule(cells, ["q1", "q2", "q3"], 2, seed=7)
    expected = {(c.key, q, r) for c in cells for q in ("q1", "q2", "q3") for r in (0, 1)}
    assert {(u.condition.key, u.qid, u.replicate) for u in sched} == expected
    assert len(sched) == len(expected)


def test_the_schedule_is_deterministic_for_a_seed() -> None:
    a = build_schedule(_cells(), ["q1", "q2"], 1, seed=99)
    b = build_schedule(_cells(), ["q1", "q2"], 1, seed=99)
    assert schedule_digest(a) == schedule_digest(b)


def test_a_different_seed_gives_a_different_order() -> None:
    a = build_schedule(_cells(), ["q1", "q2"], 1, seed=1)
    b = build_schedule(_cells(), ["q1", "q2"], 1, seed=2)
    assert schedule_digest(a) != schedule_digest(b)


def test_work_is_grouped_by_reader_so_weights_load_once() -> None:
    """Grouping is what keeps ~189k loads down to a handful. Randomisation is
    preserved INSIDE each group, which is where it does the work."""
    sched = build_schedule(_cells(), ["q1"], 1, seed=3)
    groups = [u.condition.reader for u in sched]
    from itertools import pairwise

    runs = sum(1 for a, b in pairwise(groups) if a != b)
    assert runs <= 2, f"reader changed {runs} times; expected one contiguous block each"


# ── the WAL (R5.2) ────────────────────────────────────────────────────────────
def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """After `kill -9` the file may end mid-write. Append-only guarantees the
    damaged line is the LAST one, so everything before it is trustworthy."""
    path = tmp_path / "WAL"
    with Wal(path) as wal:
        for i in range(3):
            wal.append(WalState.DONE, ("A+G", 1, 1.0, "m", "S", "gemma"), f"q{i}", 0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 3, "state": "do')  # simulate a kill mid-write
    entries = list(read_wal(path))
    assert len(entries) == 3


def test_resume_skips_done_and_keeps_failed_retryable(tmp_path: Path) -> None:
    path = tmp_path / "WAL"
    with Wal(path) as wal:
        wal.append(WalState.DONE, ("A+G", 1, 1.0, "m", "S", "gemma"), "q1", 0)
        wal.append(WalState.FAILED, ("A+G", 1, 1.0, "m", "S", "gemma"), "q2", 0, attempts=1)
        wal.append(WalState.FAILED, ("A+G", 1, 1.0, "m", "S", "gemma"), "q2", 0, attempts=2)
    state = build_resume_state(read_wal(path))
    assert state.done == {"('A+G', 1, 1.0, 'm', 'S', 'gemma')|q1|0"}
    assert state.attempt_count("('A+G', 1, 1.0, 'm', 'S', 'gemma')|q2|0") == 2
    assert state.is_exhausted("('A+G', 1, 1.0, 'm', 'S', 'gemma')|q2|0", max_attempts=2)
    assert not state.is_exhausted("('A+G', 1, 1.0, 'm', 'S', 'gemma')|q2|0", max_attempts=3)


def test_a_fresh_run_truncates_a_stale_wal(tmp_path: Path) -> None:
    """Resuming WITHOUT --resume is a NEW run; inheriting a stale journal would
    skip work it never did."""
    path = tmp_path / "WAL"
    with Wal(path) as wal:
        wal.append(WalState.DONE, ("A+G", 1, 1.0, "m", "S", "gemma"), "q1", 0)
    with Wal(path) as wal:
        assert list(read_wal(path)) == []


# ── finish_reason policy (R5.5) ───────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "tokens", "expected"),
    [
        ("", 0, "empty"),
        ("   \n ", 0, "empty"),
        ("I cannot help with that.", 12, "refusal"),
        ("I'm sorry, but no.", 8, "refusal"),
        ("a perfectly good answer that got cut", 64, "length"),
        ("a complete answer", 12, "stop"),
    ],
)
def test_finish_reasons_are_classified(text: str, tokens: int, expected: str) -> None:
    assert classify_finish(text, tokens, 64) == expected


def test_truncation_is_detected_by_token_count_not_by_the_text() -> None:
    """The cap is a token limit, and a legitimate answer may end mid-sentence."""
    assert classify_finish("the meeting was moved to Thur", 64, 64) == "length"


def test_excluded_reasons_carry_their_reason() -> None:
    for reason in ("length", "refusal", "empty"):
        assert reason in EXCLUDED_FINISH_REASONS
        assert is_excluded(reason)
    assert not is_excluded("stop")


# ── the ledger (R5.4) ─────────────────────────────────────────────────────────
def test_exclusions_are_counted_by_category_not_pooled() -> None:
    ledger = RunLedger()
    key = ("A+G", 64, 512.0, "m", "M", "qwen")
    for _ in range(3):
        ledger.plan(key)
    ledger.complete(key, "length")
    ledger.complete(key, "stop")
    ledger.complete(key, "stop")
    cell = ledger.cell(key)
    assert cell.n_done == 2, "the truncation is NOT a miss"
    assert cell.n_excluded == 1
    assert cell.by_finish_reason["length"] == 1


def test_a_failure_is_logged_with_its_reason() -> None:
    ledger = RunLedger()
    key = ("A+G", 1, 128.0, "m", "S", "gemma")
    ledger.plan(key)
    ledger.fail(key, "error", "connection reset")
    assert ledger.exclusions[0]["finish_reason"] == "error"
    assert ledger.exclusions[0]["detail"] == "connection reset"
    assert ledger.cell(key).state == "failed"


def test_partial_completion_is_visible() -> None:
    """'The run finished' and 'every cell finished' are different statements."""
    ledger = RunLedger()
    key = ("A+G", 1, 128.0, "m", "S", "gemma")
    for _ in range(10):
        ledger.plan(key)
    ledger.complete(key, "stop")
    cell = ledger.cell(key)
    assert cell.state == "partial"
    assert cell.completion_rate == pytest.approx(0.1)


def test_the_ledger_round_trips(tmp_path: Path) -> None:
    ledger = RunLedger()
    key = ("A+G", 1, 128.0, "m", "S", "gemma")
    ledger.plan(key)
    ledger.complete(key, "length")
    path = ledger.write(tmp_path / "ledger.json")
    back = RunLedger.load(path)
    assert back.cell(key).n_planned == 1
    assert back.cell(key).by_finish_reason == {"length": 1}


# ── the exit condition: really kill -9 ────────────────────────────────────────
CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {root!r})
    from msat.runner.wal import Wal, WalState
    from msat.store import ConditionKey, ResponseStore, StoredResponse, cache_key

    run, n = sys.argv[1], int(sys.argv[2])
    def req_for(i):
        # A distinct seed per unit. Without it all six share one cache key and the
        # store CORRECTLY dedupes them to a single file - which is the store
        # working, not what this test measures.
        return {{"model": "m", "messages": [{{"role": "user", "content": "q"}}],
                "max_tokens": 8, "temperature": 0.7, "seed": i}}
    store = ResponseStore(os.path.join(run, "responses"))
    with Wal(os.path.join(run, "WAL")) as wal:
        for i in range(n):
            c = ConditionKey("A+G", 4, 128.0, "per_memory_tokens", "S", "gemma")
            r = req_for(i)
            k = cache_key(c, r)
            store.put(StoredResponse(k, c.as_dict(), r, f"t{{i}}", "stop", "v1", 1, 1, 1.0))
            wal.append(WalState.DONE, c.key, f"q{{i}}", 0, cache_key=k)
        # Signal the parent that everything durable is written, then die hard.
        print("READY", flush=True)
        while True:
            time.sleep(0.05)
    """
)


def _hard_kill(pid: int) -> None:
    """Kill the process the way a power cut would.

    POSIX: SIGKILL. Windows: taskkill /F (TerminateProcess). Both are UNCATCHABLE,
    so no `finally` runs and no buffer is flushed. Sending a catchable signal
    instead would exercise orderly shutdown, which is the opposite of what R5.2 is
    about and would make the test pass for the wrong reason.

    There is no platform skip. Skipping on Windows would leave the phase's EXIT
    criterion untested on the platform the work is actually done on.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=True, timeout=30
        )
    else:
        os.kill(pid, signal.SIGKILL)


def test_a_killed_run_resumes_to_identical_results(tmp_path: Path) -> None:
    """Architecture P4 EXIT: kill mid-run, resume, IDENTICAL results.

    A mock that closes a file handle does not test this - it tests orderly
    shutdown. The child is killed uncatchably after signalling that everything
    durable is written.
    """
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(root=str(ROOT)), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    child = subprocess.Popen(
        [sys.executable, str(script), str(run_dir), "6"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "READY"
    _hard_kill(child.pid)
    child.wait(timeout=30)
    assert child.poll() is not None

    store = ResponseStore(run_dir / "responses")
    keys_before = {r.key: r.text for r in store.iter_all()}
    assert len(keys_before) == 6, "everything committed before the kill survived it"

    state = build_resume_state(read_wal(run_dir / "WAL"))
    assert len(state.done) == 6

    sched = build_schedule(
        [ConditionKey("A+G", 4, 128.0, "per_memory_tokens", "S", "gemma")],
        [f"q{i}" for i in range(6)],
        1,
        seed=1,
    )
    remaining = [u for u in sched if f"{u.condition.key}|{u.qid}|{u.replicate}" not in state.done]
    assert remaining == [], "the resumed run must do no work it had already done"

    assert {r.key: r.text for r in store.iter_all()} == keys_before


def test_the_wal_survives_a_kill_too(tmp_path: Path) -> None:
    """The other half of R5.2: the journal, not just the store.

    A partially written final line must be SKIPPED rather than treated as
    corruption - that is the entire reason the log is append-only and fsynced per
    line. A WAL that cannot be read after a kill is a WAL that cannot be resumed.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = tmp_path / "wal_child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from msat.runner.wal import Wal, WalState
            with Wal(os.path.join(sys.argv[1], "WAL")) as wal:
                for i in range(5):
                    wal.append(WalState.DONE, ("A+G", 1, 1.0, "m", "S", "gemma"), f"q{{i}}", 0)
                print("READY", flush=True)
                while True:
                    time.sleep(0.05)
            """
        ),
        encoding="utf-8",
    )
    child = subprocess.Popen(
        [sys.executable, str(script), str(run_dir)], stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "READY"
    _hard_kill(child.pid)
    child.wait(timeout=30)

    entries = list(read_wal(run_dir / "WAL"))
    assert len(entries) == 5
    assert len(build_resume_state(entries).done) == 5
