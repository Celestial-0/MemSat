"""Shared pytest fixtures for the `msat` test suite.

The important one is `no_network`. PRD §N3 requires that `verify`, `analyze`,
`figures`, and `paper` run with the network disabled, and PRD §G4 requires that
analysis never calls a model. Both are only credible if the *tests* cannot
reach the network either — otherwise a command that quietly opens a socket still
"passes" on a developer machine and fails in CI.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from msat.datasets import raw_path, resolve_data_dir


@contextmanager
def block_network() -> Iterator[None]:
    """Raise on any attempt to construct a socket, for the duration of the block."""
    real_socket = socket.socket

    class _Blocked(socket.socket):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("network access attempted; this command must run offline (PRD §N3)")

    socket.socket = _Blocked  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        socket.socket = real_socket  # type: ignore[misc]


@pytest.fixture
def no_network() -> Iterator[None]:
    """Fixture form of `block_network()` for use inside a single test."""
    with block_network():
        yield


@pytest.fixture
def cli_runner() -> Any:
    """A Typer `CliRunner` that does not swallow `SystemExit` into `result.exception`.

    Typer's default `CliRunner` wraps `typer.Exit` in `result.exception`; tests
    here read the real exit code from that exception, so we keep the standard
    behaviour and assert on `result.exit_code`.
    """
    from typer.testing import CliRunner

    return CliRunner()


# ── raw-data availability ──────────────────────────────────────────────────────
# The benchmarks are large and separately licensed, so they are absent from most
# checkouts. Tests that need them skip loudly rather than failing for the wrong
# reason.
DATA_DIR: Path = resolve_data_dir()

requires_s = pytest.mark.skipif(
    not raw_path("S", DATA_DIR).is_file(),
    reason=f"LongMemEval S not present under {DATA_DIR} (set MSAT_DATA_DIR)",
)
requires_m = pytest.mark.skipif(
    not raw_path("M", DATA_DIR).is_file(),
    reason=f"LongMemEval M (2.6 GB) not present under {DATA_DIR}",
)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def path_s() -> Path:
    return raw_path("S", DATA_DIR)


@pytest.fixture(scope="session")
def path_m() -> Path:
    return raw_path("M", DATA_DIR)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ── a loadable copy of the config bundle ───────────────────────────────────────
# The SHIPPED bundle is deliberately incomplete: the reader checkpoints are
# `UNCONFIRMED` until someone pastes real HuggingFace repo ids and revision
# commits, and PRD §R7.6 makes that a hard error rather than a warning. Tests
# that need to LOAD a bundle must therefore work on a completed copy — otherwise
# every validator test would fail for a reason that has nothing to do with the
# validator.
#
# `_CONFIRMED_IDS` supply the revision commit and release date only. The `model`
# string is left alone on purpose: it is the key `configs/pricing/latest.yaml`
# prices against, so renaming it here would make every cell look unpriced — which
# would be a test bug masquerading as a cost finding.
_CONFIRMED_IDS = {
    "gemma": ("0" * 40, "2026-01-01"),
    "qwen": ("1" * 40, "2026-01-01"),
    "phi": ("2" * 40, "2026-01-01"),
}


@pytest.fixture
def complete_configs(tmp_path: Path) -> Path:
    """A copy of `configs/` with every reader checkpoint confirmed.

    FUNCTION-scoped on purpose. Several tests deliberately corrupt this bundle to
    prove the validator catches the defect, and a session-scoped fixture means
    the first such test poisons every test that runs after it — which presents as
    a baffling pile of unrelated failures. The copy is 25 small text files; the
    cost of a fresh one is far below the cost of debugging shared state.
    """
    import shutil

    import yaml

    src = project_root_path() / "configs"
    dst = tmp_path / "configs"
    shutil.copytree(src, dst)

    path = dst / "readers.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for family in data["families"]:
        revision, released = _CONFIRMED_IDS[family["id"]]
        family["version"] = revision
        family["released"] = released
        family["exact_checkpoint_confirmed"] = True
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dst


def project_root_path() -> Path:
    return Path(__file__).resolve().parent.parent
