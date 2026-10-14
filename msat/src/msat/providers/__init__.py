"""Local inference backends.

The backend is a CONFIG value. See `base.py` for why that matters: the best
engine for throughput is not available on every platform, and hard-binding one
would mean rewriting the executor the moment a benchmark says to switch.

`transformers` is always available and is the default.
"""

from __future__ import annotations

from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    LocalBackend,
    backend_availability,
    select_backend,
)

__all__ = [
    "BackendUnavailableError",
    "GenerationRequest",
    "GenerationResult",
    "LocalBackend",
    "backend_availability",
    "select_backend",
]
