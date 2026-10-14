"""The memory baseline (Architecture §P3.1).

Own the write path: extract -> embed -> retrieve, minimal and auditable.

The guiding constraint is from the plan: *a weak retriever is acceptable, a
CHANGING one is not.* Every component here is deterministic by construction, the
embedder is injected rather than imported, and the write path returns its
provenance alongside its index so the two cannot be separated.
"""

from __future__ import annotations

from msat.memory.chunk import Unit, approx_tokens, chunk_session, to_units
from msat.memory.embed import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from msat.memory.provenance import ProvenanceReport, UnitRecord, WriteRecord
from msat.memory.retrieve import Retrieved, gold_rank, retrieve
from msat.memory.write import MemoryIndex, corpus_stats, write

__all__ = [
    "Embedder",
    "HashEmbedder",
    "MemoryIndex",
    "ProvenanceReport",
    "Retrieved",
    "SentenceTransformerEmbedder",
    "Unit",
    "UnitRecord",
    "WriteRecord",
    "approx_tokens",
    "chunk_session",
    "corpus_stats",
    "cosine",
    "gold_rank",
    "retrieve",
    "to_units",
    "write",
]
