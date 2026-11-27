"""CLI adapter to write a tiny synthetic corpus so the plan phase is testable offline."""

from __future__ import annotations

import sys
from pathlib import Path

from msat._fixture import GOLD, LAYOUT, build, write_fixture

__all__ = ["GOLD", "LAYOUT", "build", "write_fixture"]

if __name__ == "__main__":  # pragma: no cover - manual regeneration
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/synthetic")
    print(write_fixture(target))
