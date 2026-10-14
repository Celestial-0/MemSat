"""Loader for the synthetic P3 fixture.

The fixture is GENERATED, not checked in as data — see
`scripts/make_synthetic_fixture.py` for why. This module is the only place that
knows how to load it, so `selftest` and the tests cannot drift apart on what the
fixture is.

Generated rather than stored because a JSON blob nobody can read is a blob nobody
can review, and the SHAPE of the data is exactly what the guards are about: three
questions at three corpus sizes, each session carrying a token unique to it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_generator() -> Any:
    path = _ROOT / "scripts" / "make_synthetic_fixture.py"
    spec = importlib.util.spec_from_file_location("_msat_fixture_gen", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise ImportError(f"cannot load the fixture generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_synthetic() -> list[dict[str, Any]]:
    """Every fixture question, freshly generated.

    Generated on each call rather than read from disk so there is no second copy
    that can go stale — a checked-in fixture and a generator disagreeing is the
    same class of bug as a constant drifting from its grid.
    """
    questions: list[dict[str, Any]] = _load_generator().build()["questions"]
    return questions
