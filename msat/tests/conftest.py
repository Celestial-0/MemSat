"""Shared fixtures; `no_network` enforces the offline contract."""

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
    with block_network():
        yield


@pytest.fixture
def cli_runner() -> Any:
    from typer.testing import CliRunner

    return CliRunner()


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


_CONFIRMED_IDS = {
    "gemma": ("0" * 40, "2026-01-01"),
    "qwen": ("1" * 40, "2026-01-01"),
    "phi": ("2" * 40, "2026-01-01"),
}


@pytest.fixture
def complete_configs(tmp_path: Path) -> Path:
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
