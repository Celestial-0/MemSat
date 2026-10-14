"""Metrics layer.

`normalisation` executes PRD §R1.2's declared answer-normalisation and the §R8.1
string metrics built on it. It is driven by `configs/prompts/normalisation.yaml`
so the rules are auditable, reviewable, and inside the `config_hash` — a
normalisation buried in code is a normalisation nobody reviewed.
"""

from __future__ import annotations

from msat.metrics.normalisation import (
    Normaliser,
    Step,
    UnknownNormalisationError,
    exact_match,
    score_answer,
    token_f1,
)

__all__ = [
    "Normaliser",
    "Step",
    "UnknownNormalisationError",
    "exact_match",
    "score_answer",
    "token_f1",
]
