"""MemSat experimental harness: retrieval depth K against context budget B."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main(argv: list[str] | None = None) -> Any:
    from msat.cli import main as _main

    return _main(argv)
