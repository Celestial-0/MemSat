"""`build-plan` — enumerate conditions, apply the guards, emit a manifest
(PRD §R3.2, §R3.3, §R3.4, §R5.6).

This is the last thing that happens before a GPU is touched, and it REJECTS.

REJECTS, NOT WARNS
------------------
A degenerate cell is not a cell that runs badly. It is a condition that is already
satisfied before the manipulation is applied: `K > n_sessions` means "retrieve the
top K" returns the whole conversation whatever K says, so the cell measures nothing
about K. Emitting it as a warning would put it in the grid, put it in the figures,
and produce a curve that rises and then flattens for a reason that has nothing to
do with retrieval depth. The flatten is the artefact.

WHY THE GRID SHRANK TODAY
-------------------------
Running this against the real corpus rejects **K=64 on the S scale outright**:
S questions hold a mean of 47.7 sessions, so K=64 exceeds the corpus for 100% of
them. P1 measured this (`frac_k_gt_corpus = 1.0` at K=64) and it sat in
`corpus_fraction.json` while `design.yaml` still declared K=64 as a headline level.
This is the K=64 defect, caught at design time by the guard rather than in week 6.

WHY A PARTLY-DEGENERATE CELL IS KEPT
------------------------------------
A cell degenerate for 50% of questions is not a cell to throw away — it is a cell
whose effective sample size is halved. Rejecting it discards real signal; keeping
it silently overstates precision. So it is ACCEPTED, carrying `degeneracy_fraction`,
and the degenerate questions are excluded from paired comparisons at P4.3. The
fraction travels with the cell so the analysis can never forget it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.config import ConfigBundle
from msat.guards import (
    QuestionStats,
    Unit,
    cell_verdict,
    corpus_summary,
)
from msat.memory import write as memory_write

MANIFEST_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class ConditionKey:
    """The whole condition, and nothing less (PRD §R5.6).

    `key` is what goes into the content-addressed store, so two cells that render
    an identical prompt MUST NOT collide here — and must not silently differ
    either. `b_mode` is part of the key because the same number under two modes is
    two different conditions.
    """

    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str

    @property
    def key(self) -> tuple[str, int, float, str, str, str]:
        return (self.arm, self.k, round(self.b, 6), self.b_mode, self.scale, self.reader)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "scale": self.scale,
            "reader": self.reader,
        }


@dataclass(frozen=True, slots=True)
class CellPlan:
    """One condition's verdict across the whole corpus."""

    condition: ConditionKey
    n_questions: int
    n_degenerate: int
    degeneracy_fraction: float
    status: str
    """`accepted` or `rejected`. Never `warn`."""
    reason: str
    example_reason: str = ""
    """One real question's rejection text, so a reader can see what went wrong
    rather than only how often."""

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
    cells: tuple[CellPlan, ...]
    corpus: dict[str, Any]
    guards: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> tuple[CellPlan, ...]:
        return tuple(c for c in self.cells if c.status == "accepted")

    @property
    def rejected(self) -> tuple[CellPlan, ...]:
        return tuple(c for c in self.cells if c.status == "rejected")

    @property
    def fully_degenerate(self) -> tuple[CellPlan, ...]:
        """Rejected for degeneracy — distinct from a cell dropped for capacity."""
        return tuple(c for c in self.rejected if c.reason.startswith("degeneracy"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "grid": self.grid,
            "summary": {
                "n_cells": len(self.cells),
                "n_accepted": len(self.accepted),
                "n_rejected": len(self.rejected),
                "n_rejected_degeneracy": len(self.fully_degenerate),
                "collisions": self.collisions,
            },
            "collisions": self.collisions,
            "corpus": self.corpus,
            "guards": self.guards,
            "notes": list(self.notes),
            "cells": [c.as_dict() for c in self.cells],
        }

    @property
    def collisions(self) -> list[list[str]]:
        seen: dict[tuple, ConditionKey] = {}
        out: list[list[str]] = []
        for c in self.cells:
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
    """Build guard inputs from measured per-question counts.

    `chars_per_token` is the config's declared ratio, and it is an ASSUMPTION. It
    propagates straight into the B-guard, which is why the ratio is recorded in the
    manifest: a manifest whose numbers depend on an assumed ratio has to say so, or
    a reader will take them for measured.
    """
    return QuestionStats(
        question_id=question_id,
        units={
            "session": n_sessions,
            # A session splits into turns; a turn into facts. ASSUMED ratios, and
            # only reachable if a run declares `unit: fact` or `unit: chunk`. The
            # default `session` unit never touches them.
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
    """Tokens per memory at this scale, from measured counts.

    Corpus-wide by necessity: the per-memory budget B applies uniformly across
    memories, so it needs one number. Guarding each question against its OWN mean
    would model a per-question budget the renderer does not implement.
    """
    total_sessions = sum(n_sessions)
    if total_sessions == 0:
        return 0.0
    return sum(n_chars) / total_sessions / chars_per_token


def build_manifest(
    bundle: ConfigBundle,
    corpus: Mapping[str, Sequence[QuestionStats]],
    *,
    chars_per_token: float,
    raw_counts: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    unit: Unit = "session",
) -> Manifest:
    """Enumerate every condition and apply both guards to every question.

    `raw_counts[scale]` is the measured `(n_sessions, n_chars)` per question,
    needed to derive `mean_unit_tokens` for the B-guard. Optional: without it the
    B-guard cannot be evaluated and that is reported rather than assumed.
    """
    design = bundle.design
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

    cells: list[CellPlan] = []
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
        config_hash=bundle.config_hash,
        grid={
            "K": list(design.K_grid),
            "B": [{"value": b, "mode": m} for b, m in b_levels],
            "scales": list(design.scales),
            "readers": list(design.readers),
            "arms": ["A+G"],
            "runs_per_cell": design.runs_per_cell,
        },
        cells=tuple(cells),
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


def _plan_cell(
    k: int,
    b: float,
    mode: str,
    scale: str,
    reader: str,
    corpus: Sequence[QuestionStats],
    unit: Unit,
    mean_unit: float,
) -> CellPlan:
    condition = ConditionKey("A+G", k, b, mode, scale, reader)

    if not corpus:
        return CellPlan(
            condition,
            0,
            0,
            1.0,
            "rejected",
            "degeneracy: no corpus stats for this scale",
        )

    verdicts = [
        cell_verdict(k, b, s, unit, b_mode=mode, mean_unit_tokens=mean_unit or None) for s in corpus
    ]
    degenerate = [v for v in verdicts if v.degenerate]
    frac = len(degenerate) / len(verdicts)

    if frac >= 1.0:
        return CellPlan(
            condition,
            len(verdicts),
            len(degenerate),
            frac,
            "rejected",
            f"degeneracy: rejected for every one of {len(verdicts)} questions",
            degenerate[0].reason if degenerate else "",
        )

    return CellPlan(
        condition,
        len(verdicts),
        len(degenerate),
        frac,
        "accepted",
        "ok" if not degenerate else f"{len(degenerate)} of {len(verdicts)} questions degenerate",
        degenerate[0].reason if degenerate else "",
    )


def write_manifest(manifest: Manifest, path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def build_from_memory(
    question_id: str,
    sessions: Sequence[tuple[str, str]],
    *,
    unit: Unit = "session",
) -> tuple[Any, Any, QuestionStats]:
    """Write one synthetic question to memory and return its guard inputs.

    Used by `make selftest` so the whole path — write, guard, manifest — runs end to
    end on the fixture with no dataset and no GPU.
    """
    index, record = memory_write(question_id, sessions, unit=unit)
    stats = QuestionStats(
        question_id=question_id,
        units=index.count_by_unit(),
        corpus_tokens=index.corpus_tokens(),
    )
    return index, record, stats
