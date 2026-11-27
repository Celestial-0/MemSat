"""Write-time provenance: what was written, what was dropped, and why."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Reason = Literal[
    "written",
    "empty_source",
    "too_short",
    "embedding_failed",
    "zero_vector",
    "duplicate",
]


@dataclass(frozen=True, slots=True)
class UnitRecord:
    unit_id: str
    unit_type: str
    source_id: str
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
    question_id: str
    unit: str
    units: tuple[UnitRecord, ...]
    embedder: str
    embed_dim: int
    seeded: bool
    write_hash: str

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
