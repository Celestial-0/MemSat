"""`build-manifest`: enumerate conditions, apply the guards, emit a manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.guards import (
    QuestionStats,
    Unit,
    condition_verdict,
    corpus_summary,
)
from msat.memory import write as memory_write
from msat.store import Condition

MANIFEST_SCHEMA = 1

__all__ = [
    "Condition",
    "ConditionPlan",
    "Manifest",
    "build_manifest",
    "stats_from_corpus",
    "write_manifest",
]


@dataclass(frozen=True, slots=True)
class ConditionPlan:
    condition: Condition
    n_questions: int
    n_degenerate: int
    degeneracy_fraction: float
    status: str
    reason: str
    example_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.condition.as_dict(),
            "n_questions": self.n_questions,
            "n_degenerate": self.n_degenerate,
            "degeneracy_fraction": round(self.degeneracy_fraction, 6),
            "status": self.status,
            "reason": self.reason,
            "example_reason": self.example_reason,
        }


@dataclass
class Manifest:
    schema_version: int
    config_hash: str
    grid: dict[str, Any]
    conditions: tuple[ConditionPlan, ...] = ()
    corpus: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)
    canonical_config_hash: str | None = None
    effective_config_hash: str | None = None
    mode: str = "full"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> tuple[ConditionPlan, ...]:
        return tuple(c for c in self.conditions if c.status in ("accepted", "supported"))

    @property
    def supported(self) -> tuple[ConditionPlan, ...]:
        return tuple(c for c in self.conditions if c.status in ("accepted", "supported"))

    @property
    def rejected(self) -> tuple[ConditionPlan, ...]:
        return tuple(c for c in self.conditions if c.status in ("rejected", "unsupported"))

    @property
    def unsupported(self) -> tuple[ConditionPlan, ...]:
        return tuple(c for c in self.conditions if c.status in ("rejected", "unsupported"))

    @property
    def fully_degenerate(self) -> tuple[ConditionPlan, ...]:
        return tuple(c for c in self.rejected if c.reason.startswith("degeneracy"))

    def as_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "canonical_config_hash": self.canonical_config_hash or self.config_hash,
            "effective_config_hash": self.effective_config_hash or self.config_hash,
            "mode": self.mode,
            "grid": self.grid,
            "summary": {
                "n_conditions": len(self.conditions),
                "n_accepted": len(self.accepted),
                "n_rejected": len(self.rejected),
                "n_rejected_degeneracy": len(self.fully_degenerate),
                "collisions": self.collisions,
            },
            "collisions": self.collisions,
            "corpus": self.corpus,
            "guards": self.guards,
            "notes": list(self.notes),
            "conditions": [c.as_dict() for c in self.conditions],
            "cells": [c.as_dict() for c in self.conditions],
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @property
    def collisions(self) -> list[list[str]]:
        seen: dict[tuple, Condition] = {}
        out: list[list[str]] = []
        for c in self.conditions:
            if c.condition.key in seen:
                out.append([str(seen[c.condition.key].key), str(c.condition.key)])
            else:
                seen[c.condition.key] = c.condition
        return out


def stats_from_corpus(
    question_id: str,
    n_sessions: int,
    n_chars: int,
    *,
    chars_per_token: float,
) -> QuestionStats:
    return QuestionStats(
        question_id=question_id,
        units={
            "session": n_sessions,
            "chunk": n_sessions * 3,
            "fact": n_sessions * 8,
        },
        corpus_tokens=int(n_chars / chars_per_token),
    )


def mean_unit_tokens_for(
    corpus: Sequence[QuestionStats],
    n_chars: Sequence[int],
    n_sessions: Sequence[int],
    chars_per_token: float,
) -> float:
    total_sessions = sum(n_sessions)
    if total_sessions == 0:
        return 0.0
    return sum(n_chars) / total_sessions / chars_per_token


def build_manifest(
    bundle: Any,
    corpus: Mapping[str, Sequence[QuestionStats]],
    *,
    chars_per_token: float,
    raw_counts: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    unit: Unit = "session",
    mode: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Manifest:
    eff_bundle = getattr(bundle, "bundle", bundle)
    eff_mode = getattr(bundle, "mode", mode or "full")
    canonical_hash = getattr(
        bundle, "canonical_config_hash", getattr(bundle, "config_hash", eff_bundle.config_hash)
    )
    effective_hash = getattr(bundle, "effective_config_hash", eff_bundle.config_hash)

    design = eff_bundle.design
    b_levels = [(b.value, b.mode) for b in design.B_grid]
    notes: list[str] = [
        f"chars_per_token = {chars_per_token} (ASSUMED from configs/pricing; "
        "every B-guard verdict depends on it)",
        f"unit = {unit!r}; K counts {unit}s (PRD §R1.6)",
    ]

    guards_summary: dict[str, Any] = {}
    for scale in design.scales:
        qs = corpus.get(scale, [])
        guards_summary[scale] = corpus_summary(qs, unit)

    cells: list[ConditionPlan] = []
    for scale in design.scales:
        qs = list(corpus.get(scale, []))
        if not qs:
            notes.append(f"scale {scale}: NO CORPUS STATS - every cell on it is rejected")

        raw = list(raw_counts.get(scale, [])) if raw_counts else []
        if raw and qs:
            mean_unit = mean_unit_tokens_for(
                qs,
                [c for _, c in raw],
                [s for s, _ in raw],
                chars_per_token,
            )
        else:
            mean_unit = 0.0

        for reader in design.readers:
            for k in design.K_grid:
                for b, mode in b_levels:
                    cells.append(_plan_cell(k, b, mode, scale, reader, qs, unit, mean_unit))

    return Manifest(
        schema_version=MANIFEST_SCHEMA,
        config_hash=effective_hash,
        canonical_config_hash=canonical_hash,
        effective_config_hash=effective_hash,
        mode=eff_mode,
        metadata=dict(metadata or {}),
        grid={
            "K": list(design.K_grid),
            "B": [{"value": b, "mode": m} for b, m in b_levels],
            "scales": list(design.scales),
            "readers": list(design.readers),
            "arms": ["A+G"],
            "runs_per_cell": design.runs_per_cell,
        },
        conditions=tuple(cells),
        corpus={
            scale: corpus_summary(list(corpus.get(scale, [])), unit) for scale in design.scales
        },
        guards={
            "policy": "reject_cell — never truncate silently (PRD §R3.2)",
            "partial_degeneracy": (
                "accepted and carried as degeneracy_fraction; degenerate questions are "
                "excluded from paired comparisons at P4.3"
            ),
            "per_scale": guards_summary,
        },
        notes=tuple(notes),
    )


def _plan_condition(
    k: int,
    b: float,
    mode: str,
    scale: str,
    reader: str,
    corpus: Sequence[QuestionStats],
    unit: Unit,
    mean_unit: float,
) -> ConditionPlan:
    condition = Condition("A+G", k, b, mode, scale, reader)

    if not corpus:
        return ConditionPlan(
            condition,
            0,
            0,
            1.0,
            "rejected",
            "degeneracy: no corpus stats for this scale",
        )

    verdicts = [
        condition_verdict(k, b, s, unit, b_mode=mode, mean_unit_tokens=mean_unit or None)
        for s in corpus
    ]
    degenerate = [v for v in verdicts if v.degenerate]
    frac = len(degenerate) / len(verdicts)

    if frac >= 1.0:
        return ConditionPlan(
            condition,
            len(verdicts),
            len(degenerate),
            frac,
            "rejected",
            f"degeneracy: rejected for every one of {len(verdicts)} questions",
            degenerate[0].reason if degenerate else "",
        )

    return ConditionPlan(
        condition,
        len(verdicts),
        len(degenerate),
        frac,
        "accepted",
        "ok" if not degenerate else f"{len(degenerate)} of {len(verdicts)} questions degenerate",
        degenerate[0].reason if degenerate else "",
    )


_plan_cell = _plan_condition


def write_manifest(manifest: Manifest, path: Path, *, overwrite: bool = True) -> Path:
    import json

    p = Path(path)
    if not overwrite and p.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest at {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def build_from_memory(
    question_id: str,
    sessions: Sequence[tuple[str, str]],
    *,
    unit: Unit = "session",
) -> tuple[Any, Any, QuestionStats]:
    index, record = memory_write(question_id, sessions, unit=unit)
    stats = QuestionStats(
        question_id=question_id,
        units=index.count_by_unit(),
        corpus_tokens=index.corpus_tokens(),
    )
    return index, record, stats
