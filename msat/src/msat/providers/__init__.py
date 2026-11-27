"""Local inference backends."""

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
