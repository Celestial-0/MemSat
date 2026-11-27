"""The write path: extract, embed, index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from msat.memory.chunk import Unit, to_units
from msat.memory.embed import Embedder, HashEmbedder
from msat.memory.provenance import (
    Reason,
    UnitRecord,
    WriteRecord,
    content_hash,
    record_hash,
)


@dataclass
class MemoryIndex:
    units: list[Unit] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    unit_index: dict[str, int] = field(default_factory=dict)

    def add(self, unit: Unit, vector: list[float]) -> None:
        self.unit_index[unit.unit_id] = len(self.units)
        self.units.append(unit)
        self.vectors.append(vector)

    def __len__(self) -> int:
        return len(self.units)

    def corpus_tokens(self) -> int:
        return sum(u.n_tokens for u in self.units)

    def count_by_unit(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for u in self.units:
            counts[u.unit_type] = counts.get(u.unit_type, 0) + 1
        return counts


def _record_units(
    sessions: Sequence[tuple[str, str]],
    unit: str,
    target_tokens: int,
    seen: dict[str, str],
) -> tuple[list[tuple[Unit, UnitRecord]], list[UnitRecord]]:
    pending: list[tuple[Unit, UnitRecord]] = []
    dropped: list[UnitRecord] = []

    for source_id, text in sessions:
        built = to_units([(source_id, text)], unit, target_tokens=target_tokens)
        if not built:
            dropped.append(
                UnitRecord(
                    f"{unit}:{source_id}",
                    unit,
                    source_id,
                    0,
                    content_hash(text),
                    "empty_source",
                    "source produced no unit",
                )
            )
            continue

        for u in built:
            digest = content_hash(u.text)
            if not u.text.strip():
                dropped.append(
                    UnitRecord(
                        u.unit_id,
                        u.unit_type,
                        u.source_id,
                        u.n_tokens,
                        digest,
                        "empty_source",
                        "unit text was blank",
                    )
                )
                continue
            if digest in seen:
                dropped.append(
                    UnitRecord(
                        u.unit_id,
                        u.unit_type,
                        u.source_id,
                        u.n_tokens,
                        digest,
                        "duplicate",
                        f"identical content already written as {seen[digest]}",
                    )
                )
                continue
            seen[digest] = u.unit_id
            pending.append((u, UnitRecord(u.unit_id, u.unit_type, u.source_id, u.n_tokens, digest)))

    return pending, dropped


def write(
    question_id: str,
    sessions: Sequence[tuple[str, str]],
    embedder: Embedder | None = None,
    *,
    unit: str = "session",
    target_tokens: int = 256,
) -> tuple[MemoryIndex, WriteRecord]:
    enc = embedder if embedder is not None else HashEmbedder()

    seen: dict[str, str] = {}
    pending, dropped = _record_units(sessions, unit, target_tokens, seen)

    index = MemoryIndex()
    records: list[UnitRecord] = list(dropped)

    if pending:
        vectors = enc.encode([u.text for u, _ in pending])
        for (u, rec), vec in zip(pending, vectors, strict=True):
            index.add(u, vec)
            if not any(abs(x) > 0.0 for x in vec):
                records.append(
                    UnitRecord(
                        u.unit_id,
                        u.unit_type,
                        u.source_id,
                        u.n_tokens,
                        rec.content_sha256,
                        "zero_vector",
                        "embedder returned an all-zero vector",
                    )
                )
                continue
            records.append(
                UnitRecord(
                    u.unit_id,
                    u.unit_type,
                    u.source_id,
                    u.n_tokens,
                    rec.content_sha256,
                    "written",
                )
            )

    ordered = tuple(sorted(records, key=lambda r: r.unit_id))
    return index, WriteRecord(
        question_id=question_id,
        unit=unit,
        units=ordered,
        embedder=getattr(enc, "name", type(embedder).__name__),
        embed_dim=int(getattr(enc, "dim", 0)),
        seeded=True,
        write_hash=record_hash(ordered),
    )


def corpus_stats(
    question_id: str,
    index: MemoryIndex,
    *,
    units_available: Mapping[str, int] | None = None,
) -> dict[str, object]:
    counts = index.count_by_unit()
    if units_available:
        counts.update(units_available)
    return {
        "question_id": question_id,
        "counts": counts,
        "corpus_tokens": index.corpus_tokens(),
    }


__all__ = ["MemoryIndex", "Reason", "corpus_stats", "write"]
