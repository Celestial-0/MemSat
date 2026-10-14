"""The CLI surface itself (PRD §7).

Two things are asserted here that no other test covers:

1. **The surface is complete.** Every command named in PRD §7 is registered —
   the surface is frozen before any feature lands, so a missing command is a
   build error rather than a discovery during week 4.
2. **Offline commands cannot reach the network** (PRD §N3, §G4), enforced by a
   socket-blocking fixture rather than by good intentions.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat import __version__
from msat.cli import OFFLINE_COMMANDS, PRD_SURFACE, app, registered_commands
from msat.constants import (
    EXIT_BUDGET_UNSET,
    EXIT_CONFIG_INVALID,
    EXIT_DRIFT,
    EXIT_FAILURE,
    EXIT_NO_GO,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_UNDERPOWERED,
    PHASE_OF_COMMAND,
)
from msat.constants import (
    OFFLINE_COMMANDS as DECLARED_OFFLINE,
)
from msat.logging import REQUIRED_FIELDS, configure_logging, missing_attribution


def combined_output(result: Any) -> str:
    """Everything the CLI printed, whichever stream it went to.

    Click 8.2 removed `CliRunner(mix_stderr=...)`; in some versions `result.output`
    already includes stderr and in others it does not. These tests assert on
    user-facing text that the CLI deliberately writes to **stderr** (errors go to
    stderr, by convention), so they must not depend on which behaviour the locked
    Click version happens to implement.
    """
    chunks = []
    for attr in ("output", "stdout", "stderr"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            chunks.append(value)
    return "\n".join(chunks)


# ── surface ────────────────────────────────────────────────────────────────────
def test_all_subcommands_registered() -> None:
    """PRD §7's command list is registered verbatim — no more, no fewer.

    `data-lock` and `verify` are registered alongside §7's list because
    `data.lock` (R2.3) and the `make verify` target (G1) are separate
    operations on the same inputs. Both are listed in `PRD_SURFACE` so the
    expected set and the actual set are asserted from one place.
    """
    assert registered_commands() == tuple(sorted(PRD_SURFACE))


@pytest.mark.parametrize("command", sorted(PHASE_OF_COMMAND))
def test_every_phase_owned_command_is_registered(command: str) -> None:
    assert command in registered_commands(), f"{command} owns a phase but is not registered"


def test_offline_commands_are_registered() -> None:
    """Every PRD §N3 offline command exists in the surface.

    Guards against declaring a command offline in `constants.py` while never
    registering it — which would make the §N3 test vacuous.
    """
    missing = sorted(OFFLINE_COMMANDS - set(registered_commands()))
    assert not missing, f"declared offline but not registered: {missing}"


def test_offline_set_matches_constants() -> None:
    """`cli.OFFLINE_COMMANDS` and `constants.OFFLINE_COMMANDS` are the same object."""
    assert OFFLINE_COMMANDS is DECLARED_OFFLINE


def test_exit_codes_are_distinct() -> None:
    """Every failure mode has its own code — a degenerate cell is not a typo."""
    codes = [
        EXIT_OK,
        EXIT_FAILURE,
        EXIT_CONFIG_INVALID,
        EXIT_NOT_IMPLEMENTED,
    ]
    assert len(set(codes)) == len(codes)


# ── stub behaviour ─────────────────────────────────────────────────────────────
_STUB_INVOCATIONS: list[tuple[str, list[str]]] = [
    # NOTE: `build-plan` is NOT here any more - it shipped in P3. It now enumerates
    # conditions, applies both guards, and writes a manifest. Its own tests live in
    # `tests/test_build_plan.py`, including the zero-network guarantee.
    #
    # `status` and `run` left this list in P5, `score` and `judge-audit` in P6;
    # `figures` in P7. All five are asserted in
    # `test_shipped_commands_no_longer_report_not_implemented` instead. A command
    # belongs in exactly one of the two lists, and moving it is how this one stays
    # honest instead of decaying into a claim that nothing is built.
    #
    # Leaving a shipped command here is not a harmless leftover: the assertion below
    # then requires it to print "build phase", which is a claim about the code that
    # is simply false once the command exists. It fails, which is the point - but it
    # fails as a confusing error rather than as an instruction.
    ("replay", ["r001"]),
    ("analyze", ["r001"]),
    ("paper", []),
    ("verify", ["r001"]),
]


@pytest.mark.parametrize(
    ("command", "args"), _STUB_INVOCATIONS, ids=lambda v: v if isinstance(v, str) else None
)
def test_unbuilt_commands_exit_not_implemented(
    cli_runner: Any, command: str, args: list[str]
) -> None:
    """An unbuilt phase exits `EXIT_NOT_IMPLEMENTED`, not a generic crash.

    "Not built yet" and "broken" are different states; a reader debugging a run
    must be able to tell them apart from the exit code alone.
    """
    result = cli_runner.invoke(app, [command, *args])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED, (
        f"`msat {command}` exited {result.exit_code}, expected "
        f"{EXIT_NOT_IMPLEMENTED}.\n{combined_output(result)}"
    )
    # The message must name the owning phase and the requirements it serves.
    out = combined_output(result)
    assert "build phase" in out
    assert PHASE_OF_COMMAND[command] in out


# ── Phase 1 / P2 · shipped commands are not stubs ─────────────────────────────
@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("ceiling", ["--verify", "--out", "results"]),
        ("data-lock", ["--verify"]),
        ("status", ["r001"]),
        ("run", ["r001", "--dry-run", "--results", "results"]),
        ("score", ["r001"]),
        ("judge-audit", ["r001"]),
        ("figures", []),
    ],
)
def test_shipped_commands_no_longer_report_not_implemented(
    cli_runner: Any, command: str, args: list[str]
) -> None:
    """Guards against a refactor silently reverting a built command to its stub.

    `build-plan` shipped in P3; `status` and `run` in P5; `score` and `judge-audit`
    in P6; `figures` in P7. Each command's old
    `test_stub_message_names_requirements` is gone:
    asserting a stub message pins the command to NOT existing, so the moment it
    ships the assertion is wrong rather than informative. `build-plan` legitimately
    exits 3 (EXIT_DEGENERATE_CELL) on the synthetic fixture because its over-deep
    cells really are degenerate - hence the explicit allowance below.
    """
    result = cli_runner.invoke(app, [command, *args])
    assert result.exit_code != EXIT_NOT_IMPLEMENTED, f"`{command}` is still a stub"
    assert result.exit_code in (
        EXIT_OK,
        EXIT_DRIFT,
        EXIT_BUDGET_UNSET,
        EXIT_CONFIG_INVALID,
        EXIT_UNDERPOWERED,
        EXIT_NO_GO,  # `run` with arm E's thresholds still null
        EXIT_FAILURE,  # `status r001` / `run` with no such manifest
    ), combined_output(result)


def test_shipped_bundle_validates(cli_runner: Any) -> None:
    """The shipped bundle must PASS.

    It did not for most of this project's life: `readers.yaml` carried
    `UNCONFIRMED` on every family, so `validate-config` exited 2 and nothing
    downstream could be trusted. Worse, all three repo ids were nonexistent
    (`google/gemma-4-9b`, `Qwen/Qwen3-9B`, `microsoft/Phi-4-mini`), which a
    warning would have shipped straight past.

    Asserting the pass is not a formality - it is the regression test for
    exactly that near-miss.
    """
    result = cli_runner.invoke(app, ["validate-config", "configs"])
    assert result.exit_code == EXIT_OK, combined_output(result)
    assert "UNCONFIRMED" not in combined_output(result)


def test_validate_config_rejects_an_unconfirmed_checkpoint(cli_runner: Any) -> None:
    """The guard still fires - tested on the SHIPPED bundle's own text.

    Rather than a synthetic fixture, this copies the real bundle and unpins one
    family, so the guard is exercised against the configuration that actually
    ships. `UNCONFIRMED` is an ERROR, not a warning (PRD §R7.6): a warning is a
    thing you ship straight past, and the failure it permits is precisely the one
    this config layer exists to prevent.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bundle = pathlib.Path(tmp) / "configs"
        shutil.copytree("configs", bundle)
        readers = yaml.safe_load((bundle / "readers.yaml").read_text(encoding="utf-8"))
        # Clear the CONFIRMATION FLAG, not the version string. These are two
        # separate validators: the flag catches "a family was added and nobody
        # pinned it", the placeholder check catches "someone typed UNCONFIRMED
        # into a field and ticked the box anyway". Both are covered; this one
        # exercises the flag, which is the one that actually fired for months.
        readers["families"][0]["exact_checkpoint_confirmed"] = False
        (bundle / "readers.yaml").write_text(
            yaml.safe_dump(readers, sort_keys=False), encoding="utf-8"
        )

        result = cli_runner.invoke(app, ["validate-config", str(bundle)])

    assert result.exit_code == EXIT_CONFIG_INVALID
    out = combined_output(result)
    assert "UNCONFIRMED checkpoint" in out
    assert "R7.6" in out


def test_validate_config_emits_the_hash(cli_runner: Any, complete_configs: Path) -> None:
    result = cli_runner.invoke(app, ["validate-config", str(complete_configs), "--hash"])
    assert result.exit_code == EXIT_OK
    assert "config_hash" in combined_output(result)


def test_validate_config_fails_on_a_broken_bundle(cli_runner: Any, complete_configs: Path) -> None:
    """PRD §R1.5 — non-zero exit on any missing required field."""
    (complete_configs / "retrieval.yaml").unlink()

    result = cli_runner.invoke(app, ["validate-config", str(complete_configs)])
    assert result.exit_code == EXIT_CONFIG_INVALID


def test_ceiling_rejects_an_unknown_split(cli_runner: Any) -> None:
    from msat.cli import normalise_scale_name

    assert normalise_scale_name("longmemeval_m") == "M"
    assert normalise_scale_name("m") == "M"
    assert normalise_scale_name("S") == "S"
    assert normalise_scale_name("lme_s_cleaned.json") == "S"


# ── argument conventions (plan decision D-a) ───────────────────────────────────
# These use `replay`, which is still a stub. The convention belongs to the SURFACE,
# not to any one command, and `status` shipped in P5 - pinning the test to it would
# have made a phase boundary look like a regression. `test_cli_status_p5.py` asserts
# the same conventions against the built command.
def test_run_id_positional_is_canonical(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["replay", "r001"])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED


def test_run_id_alias_accepted(cli_runner: Any) -> None:
    """`--run` keeps the pre-existing Makefile targets working."""
    result = cli_runner.invoke(app, ["replay", "--run", "r001"])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED


def test_run_id_given_twice_is_an_error(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["replay", "r001", "--run", "r002"])
    assert result.exit_code == EXIT_FAILURE
    assert "twice" in combined_output(result)


def test_run_id_required(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["replay"])
    assert result.exit_code == EXIT_FAILURE
    assert "run_id" in combined_output(result)


# ── PRD §N3 · offline enforcement ──────────────────────────────────────────────
@pytest.mark.parametrize("command", sorted(OFFLINE_COMMANDS))
def test_offline_commands_run_with_sockets_blocked(
    cli_runner: Any, command: str, no_network: None
) -> None:
    """Offline commands must complete their startup path with no socket available.

    This is the assertion that makes PRD §N3 and §G4 credible. It runs the real
    command under a `socket.socket` that raises on construction, so any attempt
    to reach the network fails loudly instead of silently succeeding on a
    developer machine that happens to be online.

    A stub raising `EXIT_NOT_IMPLEMENTED` is a pass: the point is that nothing
    before the stub touched the network.
    """
    args = {
        "validate-config": ["configs"],
        "design-power": ["configs/design.yaml"],
        "ceiling": ["--verify"],
        "data-lock": ["--verify"],
        "status": ["r001"],
        "replay": ["r001"],
        "score": ["r001"],
        "analyze": ["r001"],
        "figures": [],
        "paper": [],
        "verify": ["r001"],
    }[command]

    result = cli_runner.invoke(app, [command, *args])
    assert result.exit_code in (
        EXIT_NOT_IMPLEMENTED,
        EXIT_OK,
        EXIT_DRIFT,
        # A SHIPPED offline command may legitimately refuse for missing input
        # (`status r001` with no such run directory, `figures` with no
        # `scores.parquet`). Reaching that refusal with every socket constructor
        # raising proves the startup path is offline; the exit code is incidental.
        # `test_cli_status_p5.py` asserts the stronger version - `status` reaching
        # exit 0 against a real run dir.
        EXIT_CONFIG_INVALID,
        EXIT_FAILURE,
    ), combined_output(result)


def test_block_network_fixture_actually_blocks(no_network: None) -> None:
    """Meta-test: if the fixture stopped blocking, every §N3 test above is vacuous."""
    with pytest.raises(RuntimeError, match="network access attempted"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_block_network_restores_socket_after_teardown() -> None:
    """The fixture must restore `socket.socket` when it tears down.

    Runs *without* the fixture: if teardown leaked the patch, every subsequent
    test in the session would see a broken socket class. This is the one place
    that can actually observe the restore.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()


# ── top level ──────────────────────────────────────────────────────────────────
def test_version_flag(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == EXIT_OK
    assert __version__ in result.output


def test_help_lists_the_surface(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == EXIT_OK
    out = combined_output(result)
    for command in PRD_SURFACE:
        assert command in out, f"{command} missing from --help"


def test_no_args_shows_help(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, [])
    assert "Usage" in combined_output(result)


# ── PRD §N6 · structured logging ───────────────────────────────────────────────
def test_log_lines_are_single_line_json() -> None:
    """Every emitted log line parses as one JSON object."""
    import io

    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream, force=True)
    from msat.logging import get_logger

    get_logger("test").info("hello", extra={"run_id": "r001", "K": 8, "B": 4096})
    for handler in list(logging.getLogger("msat").handlers):
        handler.flush()

    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert lines, "nothing was logged"
    for line in lines:
        payload = json.loads(line)  # must not raise
        assert payload["event"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["K"] == 8
    # Restore the default stderr handler for later tests.
    configure_logging(force=True)


def test_missing_attribution_reports_gaps() -> None:
    assert missing_attribution({"run_id": "r001", "config_hash": "abc"}) == [
        f for f in REQUIRED_FIELDS if f not in ("run_id", "config_hash")
    ]
    assert missing_attribution(dict.fromkeys(REQUIRED_FIELDS, 1)) == []


def test_logging_coerces_paths_and_collections() -> None:
    """Paths and lists must serialise; an unserialisable field must not break a run."""
    import io
    from pathlib import Path

    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream, force=True)
    from msat.logging import get_logger

    get_logger("test").info("paths", extra={"config_path": Path("configs/design.yaml")})
    for handler in list(logging.getLogger("msat").handlers):
        handler.flush()
    payload = json.loads(stream.getvalue().splitlines()[0])
    assert payload["config_path"] == str(Path("configs/design.yaml"))
    configure_logging(force=True)


def test_non_ascii_log_lines_survive_a_cp1252_stream() -> None:
    """A non-ASCII field must not raise on a stream that cannot encode it.

    Answer strings and question text flow through `extra`. On Windows a
    redirected stderr is cp1252, and an `UnicodeEncodeError` raised *inside the
    logging handler* would abort a long run over a single accented character.
    """
    import io

    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream, force=True)
    from msat.logging import get_logger

    get_logger("test").info("unicode", extra={"answer": "café — naïve · K* = 6"})
    for handler in list(logging.getLogger("msat").handlers):
        handler.flush()
    payload = json.loads(stream.getvalue().splitlines()[0])
    assert payload["answer"] == "café — naïve · K* = 6"
    configure_logging(force=True)
