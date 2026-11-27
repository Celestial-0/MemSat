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
from msat.cli import (
    OFFLINE_COMMANDS,
    PRD_SURFACE,
    PaperResult,
    StatusResult,
    ValidateConfigResult,
    app,
    get_last_result,
    registered_commands,
)
from msat.constants import (
    EXIT_BUDGET_UNSET,
    EXIT_CONFIG_INVALID,
    EXIT_DEGENERATE_CELL,
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
    chunks = []
    for attr in ("output", "stdout", "stderr"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            chunks.append(value)
    return "\n".join(chunks)


def test_all_subcommands_registered() -> None:
    assert registered_commands() == tuple(sorted(PRD_SURFACE))


@pytest.mark.parametrize("command", sorted(PHASE_OF_COMMAND))
def test_every_phase_owned_command_is_registered(command: str) -> None:
    assert command in registered_commands(), f"{command} owns a phase but is not registered"


def test_offline_commands_are_registered() -> None:
    missing = sorted(OFFLINE_COMMANDS - set(registered_commands()))
    assert not missing, f"declared offline but not registered: {missing}"


def test_offline_set_matches_constants() -> None:
    assert OFFLINE_COMMANDS is DECLARED_OFFLINE


def test_exit_codes_are_distinct() -> None:
    codes = [
        EXIT_OK,
        EXIT_FAILURE,
        EXIT_CONFIG_INVALID,
        EXIT_NOT_IMPLEMENTED,
    ]
    assert len(set(codes)) == len(codes)


def test_stubs_are_deleted_and_no_command_returns_not_implemented(cli_runner: Any) -> None:
    assert "verify" not in registered_commands()
    assert "replay" not in registered_commands()
    for command in registered_commands():
        assert command not in ("verify", "replay")


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("ceiling", ["--verify", "--out", "results"]),
        ("data-lock", ["--verify"]),
        ("status", ["r001"]),
        ("run", ["r001", "--dry-run", "--results", "results"]),
        ("score", ["r001", "--gold", "results/r001/gold.json"]),
        ("analyze", ["r001"]),
        ("judge-audit", ["r001", "--verdicts", "results/r001/verdicts.json"]),
        ("figures", []),
        ("paper", []),
    ],
)
def test_shipped_commands_no_longer_report_not_implemented(
    cli_runner: Any, command: str, args: list[str]
) -> None:
    result = cli_runner.invoke(app, [command, *args])
    assert result.exit_code != EXIT_NOT_IMPLEMENTED, f"`{command}` is still a stub"
    assert result.exit_code in (
        EXIT_OK,
        EXIT_DRIFT,
        EXIT_BUDGET_UNSET,
        EXIT_CONFIG_INVALID,
        EXIT_UNDERPOWERED,
        EXIT_NO_GO,
        EXIT_FAILURE,
    ), combined_output(result)


def test_shipped_bundle_validates(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["validate-config", "configs"])
    assert result.exit_code == EXIT_OK, combined_output(result)
    res = get_last_result()
    assert isinstance(res, ValidateConfigResult)
    assert not res.errors


def test_validate_config_rejects_an_unconfirmed_checkpoint(cli_runner: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = pathlib.Path(tmp) / "configs"
        shutil.copytree("configs", bundle)
        readers = yaml.safe_load((bundle / "readers.yaml").read_text(encoding="utf-8"))
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
    res = get_last_result()
    assert isinstance(res, ValidateConfigResult)
    assert res.config_hash is not None


def test_validate_config_fails_on_a_broken_bundle(cli_runner: Any, complete_configs: Path) -> None:
    (complete_configs / "retrieval.yaml").unlink()

    result = cli_runner.invoke(app, ["validate-config", str(complete_configs)])
    assert result.exit_code == EXIT_CONFIG_INVALID


def test_ceiling_rejects_an_unknown_split(cli_runner: Any) -> None:
    from msat.cli import normalise_scale_name

    assert normalise_scale_name("longmemeval_m") == "M"
    assert normalise_scale_name("m") == "M"
    assert normalise_scale_name("S") == "S"
    assert normalise_scale_name("lme_s_cleaned.json") == "S"


def test_run_id_positional_is_canonical(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["status", "r001"])
    assert result.exit_code == EXIT_OK
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.run_id == "r001"


def test_run_id_alias_accepted(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["status", "--run", "r001"])
    assert result.exit_code == EXIT_OK
    res = get_last_result()
    assert isinstance(res, StatusResult)
    assert res.run_id == "r001"


def test_run_id_given_twice_is_an_error(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["status", "r001", "--run", "r002"])
    assert result.exit_code == EXIT_FAILURE
    assert "twice" in combined_output(result)


def test_run_id_required(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["status"])
    assert result.exit_code == EXIT_FAILURE
    assert "run_id" in combined_output(result)


def test_data_lock_out_option_deleted(cli_runner: Any) -> None:
    result = cli_runner.invoke(app, ["data-lock", "--out", "results"])
    assert result.exit_code == 2
    assert "no such option" in combined_output(result).lower()


def test_profile_accepted_and_mode_rejected(cli_runner: Any) -> None:
    res_err = cli_runner.invoke(app, ["build-manifest", "--mode", "fast"])
    assert res_err.exit_code == 2
    assert "no such option" in combined_output(res_err).lower()

    res_ok = cli_runner.invoke(app, ["build-manifest", "--profile", "fast", "--help"])
    assert res_ok.exit_code == 0
    assert "--profile" in combined_output(res_ok)


def test_rung_crossed_accepted_and_full_rejected(cli_runner: Any) -> None:
    res_err = cli_runner.invoke(app, ["run", "r001", "--rung", "full"])
    assert res_err.exit_code != 0
    assert "unknown rung 'full'" in combined_output(res_err)

    res_ok = cli_runner.invoke(app, ["run", "r001", "--rung", "crossed", "--dry-run"])
    assert "unknown rung" not in combined_output(res_ok)
    assert "entering rung 'crossed'" in combined_output(res_ok)


def test_structured_results_rendered(cli_runner: Any) -> None:
    cli_runner.invoke(app, ["validate-config", "configs"])
    val_res = get_last_result()
    assert isinstance(val_res, ValidateConfigResult)

    cli_runner.invoke(app, ["status", "r001"])
    status_res = get_last_result()
    assert isinstance(status_res, StatusResult)
    assert status_res.run_id == "r001"

    cli_runner.invoke(app, ["paper"])
    paper_res = get_last_result()
    assert isinstance(paper_res, PaperResult)
    assert paper_res.built


@pytest.mark.parametrize("command", sorted(OFFLINE_COMMANDS))
def test_offline_commands_run_with_sockets_blocked(
    cli_runner: Any, command: str, no_network: None
) -> None:
    args = {
        "validate-config": ["configs"],
        "estimate-power": ["configs/design.yaml"],
        "ceiling": ["--verify"],
        "data-lock": ["--verify"],
        "build-manifest": ["--corpus", "synthetic"],
        "status": ["r001"],
        "score": ["r001"],
        "analyze": ["r001"],
        "judge-audit": ["r001", "--verdicts", "results/r001/verdicts.json"],
        "figures": [],
        "paper": [],
        "report-reader-capacity": [],
        "hardware-delta": ["r001"],
        "vram": ["configs"],
    }[command]

    result = cli_runner.invoke(app, [command, *args])
    assert result.exit_code in (
        EXIT_NOT_IMPLEMENTED,
        EXIT_OK,
        EXIT_DRIFT,
        EXIT_CONFIG_INVALID,
        EXIT_FAILURE,
        EXIT_BUDGET_UNSET,
        EXIT_DEGENERATE_CELL,
        EXIT_UNDERPOWERED,
        EXIT_NO_GO,
    ), combined_output(result)


def test_block_network_fixture_actually_blocks(no_network: None) -> None:
    with pytest.raises(RuntimeError, match="network access attempted"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_block_network_restores_socket_after_teardown() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()


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


def test_log_lines_are_single_line_json() -> None:
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
        payload = json.loads(line)
        assert payload["event"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["K"] == 8
    configure_logging(force=True)


def test_missing_attribution_reports_gaps() -> None:
    assert missing_attribution({"run_id": "r001", "config_hash": "abc"}) == [
        f for f in REQUIRED_FIELDS if f not in ("run_id", "config_hash")
    ]
    assert missing_attribution(dict.fromkeys(REQUIRED_FIELDS, 1)) == []


def test_logging_coerces_paths_and_collections() -> None:
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
