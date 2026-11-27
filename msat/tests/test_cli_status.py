from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from msat.cli import StatusResult, app, get_last_result
from msat.runner.executor import RunConfig
from msat.runner.ledger import RunLedger
from msat.store import Condition, ResponseStore, StoredResponse, cache_key

RUNNER = CliRunner()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results" / "pilot-01"
    d.mkdir(parents=True)
    store = ResponseStore(d / "responses")

    k_s = Condition(arm="A", k=1, b=128, b_mode="budget", scale="S", reader="gemma")
    k_m = Condition(arm="A", k=32, b=2048, b_mode="budget", scale="M", reader="phi")

    ledger = RunLedger()
    for _ in range(11):
        ledger.plan(k_s.key)
    for _ in range(10):
        ledger.complete(k_s.key, "stop")
    for _ in range(5):
        ledger.plan(k_m.key)
    for _ in range(3):
        ledger.complete(k_m.key, "stop")
    ledger.fail(k_m.key, "refusal", "the reader declined")
    ledger.complete(k_m.key, "length")
    ledger.write(d / "ledger.json")

    for i in range(13):
        req = {
            "model": "test/reader-1",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "temperature": 0.0,
        }
        store.put(
            StoredResponse(
                key=cache_key(k_s if i < 10 else k_m, req),
                condition=k_s.as_dict() if i < 10 else k_m.as_dict(),
                request=req,
                text="answer",
                finish_reason="stop",
                model_version="test",
                prompt_tokens=100,
                completion_tokens=2,
                latency_ms=1.0,
            )
        )
    return d


def _status(run_dir: Path, *args: str):
    return RUNNER.invoke(app, ["status", run_dir.name, "--results", str(run_dir.parent), *args])


def test_status_reports_a_run(run_dir: Path) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.run_id == "pilot-01"
    assert res.n_done == 13
    assert res.n_planned == 16
    assert res.n_cells == 2


def test_status_surfaces_failures_and_exclusions(run_dir: Path) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.n_failed == 1
    assert res.n_excluded == 1


def test_status_names_the_run_directory_when_it_is_missing(tmp_path: Path) -> None:
    r = RUNNER.invoke(app, ["status", "nope", "--results", str(tmp_path)])
    assert r.exit_code == 1
    assert "no run directory" in r.output


def test_status_requires_a_run_id() -> None:
    assert RUNNER.invoke(app, ["status"]).exit_code == 1


def test_status_accepts_the_run_alias(run_dir: Path) -> None:
    r = RUNNER.invoke(app, ["status", "--run", run_dir.name, "--results", str(run_dir.parent)])
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.run_id == run_dir.name


def test_status_rejects_run_id_given_twice(run_dir: Path) -> None:
    r = RUNNER.invoke(app, ["status", "a", "--run", "b", "--results", str(run_dir.parent)])
    assert r.exit_code == 1
    assert "twice" in r.output


def test_the_fixture_itself_is_consistent(run_dir: Path) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "INCONSISTENT" not in r.output


def test_a_ledger_ahead_of_the_store_is_an_assertion_failure(run_dir: Path) -> None:
    assert _status(run_dir).exit_code == 0
    ledger = RunLedger.load(run_dir / "ledger.json")
    victim = ledger.cells[
        Condition(arm="A", k=1, b=128, b_mode="budget", scale="S", reader="gemma").key
    ]
    victim.n_done -= 1
    victim.n_planned -= 1
    ledger.write(run_dir / "ledger.json")
    assert _status(run_dir).exit_code == 0

    victim.n_done += 2
    victim.n_planned += 2
    ledger.write(run_dir / "ledger.json")

    r = _status(run_dir)
    assert r.exit_code == 8, r.output
    assert "INCONSISTENT" in r.output
    assert "ahead of the record" in r.output


def test_a_wal_ahead_of_the_store_is_not_flagged(run_dir: Path) -> None:
    ledger = RunLedger.load(run_dir / "ledger.json")
    for cell in ledger.cells.values():
        cell.n_done = 0
    ledger.write(run_dir / "ledger.json")

    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "INCONSISTENT" not in r.output


def test_status_shows_a_fresh_ladder_as_all_todo(run_dir: Path) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.next_rung == "negative"


def test_status_shows_a_climbed_ladder(run_dir: Path) -> None:
    (run_dir / "ladder.json").write_text(
        json.dumps({"passed": ["negative"], "notes": {"negative": "arm E accuracy 0.19"}}),
        encoding="utf-8",
    )
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.next_rung == "baselines"


def test_an_unreadable_ladder_is_refused_not_ignored(run_dir: Path) -> None:
    (run_dir / "ladder.json").write_text("{not json", encoding="utf-8")
    r = _status(run_dir)
    assert r.exit_code == 1
    assert "unreadable" in r.output


def test_the_ladder_can_be_suppressed(run_dir: Path) -> None:
    r = _status(run_dir, "--no-ladder")
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.ladder_state is None
    assert res.next_rung is None
    assert res.n_done == 13


def test_status_touches_nothing(run_dir: Path) -> None:
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    _status(run_dir)
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before


def test_status_reaches_zero_with_every_socket_blocked(run_dir: Path, no_network: None) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.n_done == 13


def test_status_report_agrees_with_the_run_config(run_dir: Path) -> None:
    from msat.runner.executor import status_report

    r = status_report(RunConfig(run_dir=run_dir))
    assert r["n_cells"] == 2
    assert r["n_planned"] == 16
    assert r["n_done"] == 13
    assert r["n_excluded"] == 1
    assert r["n_stored_responses"] == 13
