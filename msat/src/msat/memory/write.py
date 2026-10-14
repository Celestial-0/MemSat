"""The write path (Architecture §P3.1) — extract, embed, index.

We do not need a good memory system. We need OUR OWN, because the write-time
ceiling sits between the retriever and the reader and contaminates everything
downstream if it belongs to someone else and nobody measures it.

Every source produces a `WriteRecord`. Nothing enters or leaves memory without a
line in one. A retriever that cannot find a unit is ambiguous — the unit may be
absent, or the retriever is bad — and that is unresolvable after the fact. The
write log is what makes it resolvable.
"""

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
    """Units, their vectors, and the provenance that explains both.

    Vectors sit beside the units in one flat structure. At P3's scale (a few
    hundred questions x a few hundred sessions) that is well inside memory, and it
    makes the retriever a single dot-product loop with no database in the way —
    which is what "minimal and auditable" means concretely.
    """

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
    """Turn source text into unit records, classifying everything.

    Iterates the SOURCE, not the built units. `to_units` skips blank input, so
    walking the built units would omit a blank source entirely — and the encoding
    ceiling is computed from these records, which would make an unmeasured loss
    look like a clean one.
    """
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
    """Index one question's corpus and record exactly what happened.

    Returns the index AND its provenance together. Returning them separately
    invites a caller to keep the index and drop the log, and a log nobody kept is
    indistinguishable from no log.
    """
    # A separate local rather than reassigning the parameter: mypy cannot narrow
    # Embedder | None through x = x or default().
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
    """Counts and tokens the guards are evaluated against.

    Measured from what actually reached the index, not from the source corpus. The
    difference IS the encoding ceiling: if 8% of sessions failed to embed, the
    retrievable corpus is 8% smaller, and a K-guard measured against the SOURCE
    count would admit cells that are degenerate in practice.
    """
    counts = index.count_by_unit()
    if units_available:
        counts.update(units_available)
    return {
        "question_id": question_id,
        "counts": counts,
        "corpus_tokens": index.corpus_tokens(),
    }


__all__ = ["MemoryIndex", "Reason", "corpus_stats", "write"]
