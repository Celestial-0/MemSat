"""`msat status` — the thing you run at 3am on day nine.

Two things are being tested here, and the second one is the reason this file
exists.

FIRST, that it reports. A status command that cannot run is worse than no status
command, because it teaches you not to check.

SECOND, that it refuses to report a healthy run when the ledger is ahead of the
store. That direction of disagreement is the dangerous one. The other direction
(WAL ahead of store) is a crash mid-commit and is safe to resume. A ledger that
credits work the record does not contain is a run reporting numbers that no
stored response supports, and nothing downstream would ever notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from msat.cli import app
from msat.runner.executor import RunConfig
from msat.runner.ledger import RunLedger
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse, cache_key

RUNNER = CliRunner()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run directory with a store and a ledger.

    Built with the real `RunLedger` API rather than a hand-rolled dict. A
    plausible-looking hand-written ledger would only prove that status agrees
    with my own guess at the schema — and the schema is the thing most likely
    to have drifted.
    """
    d = tmp_path / "results" / "pilot-01"
    d.mkdir(parents=True)
    store = ResponseStore(d / "responses")

    k_s = ConditionKey(arm="A", k=1, b=128, b_mode="budget", scale="S", reader="gemma")
    k_m = ConditionKey(arm="A", k=32, b=2048, b_mode="budget", scale="M", reader="phi")

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
    ledger.complete(k_m.key, "length")  # an EXCLUDED finish reason
    ledger.write(d / "ledger.json")

    # The store MUST hold one response per credited unit. A fixture where the
    # ledger is already ahead of the record would make the consistency check
    # fire on every test and prove nothing about the deliberate case.
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


# ── it reports ─────────────────────────────────────────────────────────────────


def test_status_reports_a_run(run_dir: Path) -> None:
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "pilot-01" in r.output
    assert "13 / 16 done" in r.output
    assert "2 cells" in r.output
    assert "K=1 B=128.0" in r.output, "the cell key must be unpacked, not blanked"


def test_status_surfaces_failures_and_exclusions(run_dir: Path) -> None:
    out = _status(run_dir).output
    assert "failed    1" in out
    assert "excluded  1" in out
    # The reason an exclusion exists has to survive to the surface: an
    # unexplained drop in the denominator is how a broken cell flatters itself.
    assert "paired comparisons" in out


def test_status_names_the_run_directory_when_it_is_missing(tmp_path: Path) -> None:
    r = RUNNER.invoke(app, ["status", "nope", "--results", str(tmp_path)])
    assert r.exit_code == 1
    assert "no run directory" in r.output


def test_status_requires_a_run_id() -> None:
    assert RUNNER.invoke(app, ["status"]).exit_code == 1


def test_status_accepts_the_run_alias(run_dir: Path) -> None:
    r = RUNNER.invoke(app, ["status", "--run", run_dir.name, "--results", str(run_dir.parent)])
    assert r.exit_code == 0, r.output


def test_status_rejects_run_id_given_twice(run_dir: Path) -> None:
    r = RUNNER.invoke(app, ["status", "a", "--run", "b", "--results", str(run_dir.parent)])
    assert r.exit_code == 1
    assert "twice" in r.output


# ── it refuses to call a lying run healthy ─────────────────────────────────────


def test_the_fixture_itself_is_consistent(run_dir: Path) -> None:
    """Guards the guards. If this drifts, the INCONSISTENT tests below stop
    testing anything."""
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "INCONSISTENT" not in r.output


def test_a_ledger_ahead_of_the_store_is_an_assertion_failure(run_dir: Path) -> None:
    """The dangerous direction, produced deliberately.

    The ledger credits 13 units and the store holds 13, then one is removed —
    the shape a bad rebuild leaves behind. Nothing downstream looks at the store
    again, so without this check the run reports numbers no stored response
    supports.
    """
    assert _status(run_dir).exit_code == 0
    ledger = RunLedger.load(run_dir / "ledger.json")
    victim = ledger.cells[
        ConditionKey(arm="A", k=1, b=128, b_mode="budget", scale="S", reader="gemma").key
    ]
    victim.n_done -= 1
    victim.n_planned -= 1
    ledger.write(run_dir / "ledger.json")  # now 12 credited, 13 stored -- safe
    assert _status(run_dir).exit_code == 0

    victim.n_done += 2
    victim.n_planned += 2
    ledger.write(run_dir / "ledger.json")  # now 14 credited, 13 stored -- flagged

    r = _status(run_dir)
    assert r.exit_code == 8, r.output
    assert "INCONSISTENT" in r.output
    assert "ahead of the record" in r.output
    assert "msat replay" in r.output


def test_a_wal_ahead_of_the_store_is_not_flagged(run_dir: Path) -> None:
    """The SAFE direction. A crash between append and commit leaves the WAL
    ahead of the store; that is expected, and resume fixes it. It must not raise
    an alarm, or the alarm fires on every kill and gets ignored."""
    ledger = RunLedger.load(run_dir / "ledger.json")
    for cell in ledger.cells.values():
        cell.n_done = 0
    ledger.write(run_dir / "ledger.json")  # store ahead of the ledger by 13

    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "INCONSISTENT" not in r.output


# ── the ladder, which is the other reason status exists ────────────────────────


def test_status_shows_a_fresh_ladder_as_all_todo(run_dir: Path) -> None:
    out = _status(run_dir).output
    assert "ladder" in out
    assert "next: negative" in out
    assert "blocked until it passes" in out


def test_status_shows_a_climbed_ladder(run_dir: Path) -> None:
    (run_dir / "ladder.json").write_text(
        json.dumps({"passed": ["negative"], "notes": {"negative": "arm E accuracy 0.19"}}),
        encoding="utf-8",
    )
    out = _status(run_dir).output
    assert "next: baselines" in out
    assert "negative" not in out.split("ladder")[1].split("next:")[0].replace("arms", "")


def test_an_unreadable_ladder_is_refused_not_ignored(run_dir: Path) -> None:
    """Reporting "no ladder state" when the file exists but is corrupt would be a
    silent downgrade of the gate to decorative."""
    (run_dir / "ladder.json").write_text("{not json", encoding="utf-8")
    r = _status(run_dir)
    assert r.exit_code == 1
    assert "unreadable" in r.output


def test_the_ladder_can_be_suppressed(run_dir: Path) -> None:
    """Matched on the marker, not the word `ladder` — the test's own name and the
    run path both contain it, and an assertion that passes for the wrong reason
    is worse than no assertion."""
    out = _status(run_dir, "--no-ladder").output
    assert "next:" not in out
    assert "blocked until it passes" not in out
    assert "13 / 16 done" in out  # the report itself still printed


# ── status is a read-only offline command ──────────────────────────────────────


def test_status_touches_nothing(run_dir: Path) -> None:
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    _status(run_dir)
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before


def test_status_reaches_zero_with_every_socket_blocked(run_dir: Path, no_network: None) -> None:
    """The strong form of PRD §N3 for this command.

    `test_cli.py` allows `status` to reach a stub or a refusal here, because the
    point is that nothing before it touched a socket. This one requires the whole
    command - ledger, store, ladder - to complete under a `socket.socket` that
    raises on construction. A status command that quietly phoned home to report
    spend would pass the weaker test and fail this one.
    """
    r = _status(run_dir)
    assert r.exit_code == 0, r.output
    assert "13 / 16 done" in r.output


def test_status_report_agrees_with_the_run_config(run_dir: Path) -> None:
    from msat.runner.executor import status_report

    r = status_report(RunConfig(run_dir=run_dir))
    assert r["n_cells"] == 2
    assert r["n_planned"] == 16
    assert r["n_done"] == 13
    assert r["n_excluded"] == 1
    assert r["n_stored_responses"] == 13
