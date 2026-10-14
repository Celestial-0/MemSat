"""msat — the MemSat experimental harness.

Sweeps retrieval depth K against context budget B in an LLM agent's long-term
memory, on a benchmark where the analytical recall ceiling is known to bind at
K = 6.

    A(K, B) = R(K; E, X, M) * U(gold | K, B, X, M, P, order)

`K` and `B` are swept as a crossed factorial. `E` (embedder), `X` (chunking),
`M` (similarity), `P` (prompt), and `order` (gold position) are pinned, named,
and hashed into every artifact — because an unnamed constant is a confound.

The primary result is the estimated `log2(K) * log2(B)` interaction term,
tested against a pre-registered sign prediction.

The full requirement contract is in ``PRD.md``; the build order is in
``IMPLEMENTATION_PLAN.md``.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main(argv: list[str] | None = None) -> Any:
    """Console-script entry point (``[project.scripts] msat = "msat:main"``).

    Imported lazily so that ``import msat`` stays cheap — `msat.constants` is
    imported by the test suite without pulling in Typer.
    """
    from msat.cli import main as _main

    return _main(argv)
