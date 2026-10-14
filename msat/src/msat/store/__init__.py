"""Storage layer.

`responses` is the content-addressed store (PRD §R6.1). It is the only way a
response can enter a run, which is what makes "evaluation depends on storage,
never on the live model" a structural property rather than a convention.
"""

from __future__ import annotations

from msat.store.responses import (
    CONDITION_FIELDS,
    ConditionKey,
    DiskBudgetExceededError,
    ResponseStore,
    StoredResponse,
    cache_key,
    canonical_body,
    canonical_json,
)

__all__ = [
    "CONDITION_FIELDS",
    "ConditionKey",
    "DiskBudgetExceededError",
    "ResponseStore",
    "StoredResponse",
    "cache_key",
    "canonical_body",
    "canonical_json",
]
