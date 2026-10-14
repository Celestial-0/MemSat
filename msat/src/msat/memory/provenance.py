"""Write-time provenance (Architecture §P3.2).

Every session that enters memory produces a record of **what was written, what
was dropped, and why**. Nothing is dropped silently.

WHY THIS EXISTS RATHER THAN A LOG LINE
-------------------------------------
EvalMem measured encoding failures at 7.7% and retrieval failures at 22.1%. Those
are properties of the memory system under test - so ours has to be measured too,
and reported as its own number. If our write path caps recall, that cap is a
K-independent constant sitting between the retriever and the reader, and it will
be mistaken for a property of K. The only defence is to state it.

A cap that is not reported is indistinguishable from a retrieval failure, and the
difference matters: one is fixed by making memory better, the other by making the
retriever better.

THE RECORD IS DATA, NOT PROSE
-----------------------------
`WriteRecord` is content-addressable and hashable so a manifest can pin it, and
so the same corpus written twice produces the same provenance. Provenance that
does not reproduce is not provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Reason = Literal[
    "written",  # the unit entered the index intact
    "empty_source",  # the session parsed to nothing
    "too_short",  # below the chunker's floor
    "embedding_failed",  # the embedder raised; retried, then given up on
    "zero_vector",  # embedding was all zeros - a real failure, not a no-op
    "duplicate",  # byte-identical to a unit already written
]


@dataclass(frozen=True, slots=True)
class UnitRecord:
    """One retrievable unit and what became of it."""

    unit_id: str
    unit_type: str
    """`session` | `chunk` | `fact` (PRD §R1.6). Reported, never implicit."""
    source_id: str
    """The session this came from."""
    n_tokens: int
    content_sha256: str
    status: Reason = "written"
    detail: str = ""

    @property
    def kept(self) -> bool:
        return self.status == "written"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WriteRecord:
    """The write path's audit trail for one question."""

    question_id: str
    unit: str
    units: tuple[UnitRecord, ...]
    embedder: str
    """Name and version of the embedder. Recorded because swapping it changes
    recall, and a recall number without its embedder is unattributable."""
    embed_dim: int
    seeded: bool
    """Whether the write path was deterministic. A non-deterministic write path
    makes every retrieval number irreproducible, so it is recorded rather than
    assumed."""
    write_hash: str
    """SHA-256 over the ordered unit records. Identical writes produce identical
    provenance; any difference shows up here."""

    # ── the encoding-ceiling numbers ──────────────────────────────────────────
    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_written(self) -> int:
        return sum(u.kept for u in self.units)

    @property
    def n_dropped(self) -> int:
        return self.n_units - self.n_written

    @property
    def encoding_ceiling(self) -> float:
        """Fraction of source units that reached the index.

        This is the number that must be reported next to recall. It is a ceiling
        on everything downstream: a unit that never entered memory can never be
        retrieved, however good the retriever is.
        """
        return self.n_written / self.n_units if self.n_units else 0.0

    @property
    def dropped_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for u in self.units:
            if not u.kept:
                counts[u.status] = counts.get(u.status, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "unit": self.unit,
            "embedder": self.embedder,
            "embed_dim": self.embed_dim,
            "seeded": self.seeded,
            "write_hash": self.write_hash,
            "n_units": self.n_units,
            "n_written": self.n_written,
            "n_dropped": self.n_dropped,
            "encoding_ceiling": round(self.encoding_ceiling, 6),
            "dropped_by_reason": self.dropped_by_reason,
            "units": [u.as_dict() for u in self.units],
        }


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_hash(records: tuple[UnitRecord, ...]) -> str:
    """Order-sensitive hash over unit records.

    ORDER-SENSITIVE on purpose: the same units in a different order is a different
    memory - retrieval returns rank order, so order is part of what was written.
    """
    h = hashlib.sha256()
    for r in records:
        h.update(r.unit_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(r.content_sha256.encode("utf-8"))
        h.update(b"\x00")
        h.update(r.status.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ProvenanceReport:
    """Corpus-level rollup across questions."""

    records: tuple[WriteRecord, ...] = field(default_factory=tuple)

    @property
    def encoding_ceiling(self) -> float:
        total = sum(r.n_units for r in self.records)
        return sum(r.n_written for r in self.records) / total if total else 0.0

    def by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            for reason, n in r.dropped_by_reason.items():
                out[reason] = out.get(reason, 0) + n
        return dict(sorted(out.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_questions": len(self.records),
            "encoding_ceiling": round(self.encoding_ceiling, 6),
            "dropped_by_reason": self.by_reason(),
            "questions": [r.as_dict() for r in self.records],
        }
