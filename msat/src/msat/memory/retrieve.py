"""Retrieval (Architecture §P3.1) — top-K in rank order, budget applied per unit.

The budget B is applied PER UNIT, matching `per_memory_tokens` in the design. That
is the mode chosen in P2 for a specific reason: with B as a truncation of the
concatenated list, the budget is a cutoff on K expressed in tokens, the two axes
are the same axis in different units, and 78 of 126 cells rendered a byte-identical
prompt to another cell. Here the budget changes DENSITY without changing how many
units were retrieved, which is what makes the K x B interaction estimable.

Both counts are always reported (PRD §R1.6): how many UNITS came back and how many
TOKENS they occupied. A result with only one of them is uninterpretable, because
the answer to "how much did the reader actually see" differs from the answer to
"how much did we retrieve".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from msat.memory.embed import cosine
from msat.memory.write import MemoryIndex


@dataclass(frozen=True, slots=True)
class Retrieved:
    """What a retriever returned, with its unit stated and unremovable.

    `unit` is a constructor argument rather than a default, so a caller has to say
    what it retrieved. R1.6 exists because "16" is ambiguous without it, and an
    optional label is one that eventually goes missing in a figure.
    """

    unit: str
    """`session` | `chunk` | `fact` - the thing K counts."""
    unit_ids: tuple[str, ...]
    n_tokens: int
    """Tokens of the rendered context AFTER the budget."""
    n_tokens_unbudgeted: int
    """Tokens before it. Both are reported so the budget's effect is visible."""
    scores: tuple[float, ...]
    truncation: dict[str, int]
    """Per-unit token counts actually rendered, in rank order."""

    @property
    def k_returned(self) -> int:
        return len(self.unit_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "k_returned": self.k_returned,
            "n_tokens": self.n_tokens,
            "n_tokens_unbudgeted": self.n_tokens_unbudgeted,
            "budget_utilisation": round(self.n_tokens / self.n_tokens_unbudgeted, 6)
            if self.n_tokens_unbudgeted
            else 0.0,
            "scores": [round(s, 6) for s in self.scores],
            "unit_ids": list(self.unit_ids),
        }


def retrieve(
    index: MemoryIndex,
    query: str,
    query_vector: list[float],
    k: int,
    *,
    unit: str = "session",
    budget_tokens: int | None = None,
) -> Retrieved:
    """Top-K by cosine similarity, in rank order, with the budget applied.

    TIES ARE BROKEN DETERMINISTICALLY by unit id. Cosine scores collide constantly
    on short texts, and an unstable tie-break makes a retriever that looks
    deterministic produce different answers run to run - which breaks replay
    (PRD §R5.2) and makes every seed-variance estimate meaningless.
    """
    scored = [
        (cosine(query_vector, vec), i)
        for i, (u, vec) in enumerate(zip(index.units, index.vectors, strict=True))
        if u.unit_type == unit
    ]
    scored.sort(key=lambda pair: (-pair[0], index.units[pair[1]].unit_id))
    top = scored[:k]

    unit_ids = tuple(index.units[i].unit_id for _, i in top)
    scores = tuple(s for s, _ in top)
    unbudgeted = sum(index.units[i].n_tokens for _, i in top)
    truncation = [
        index.units[i].n_tokens
        if budget_tokens is None
        else min(index.units[i].n_tokens, budget_tokens)
        for _, i in top
    ]
    return Retrieved(
        unit=unit,
        unit_ids=unit_ids,
        n_tokens=sum(truncation),
        n_tokens_unbudgeted=unbudgeted,
        scores=scores,
        truncation=dict(zip(unit_ids, truncation, strict=True)),
    )


def gold_rank(index: MemoryIndex, gold_source_ids: set[str], retrieved: Retrieved) -> int | None:
    """1-based rank of the best-ranked gold unit in the result, else None.

    Ranked over UNITS, not sources: a gold session chunked into four pieces is
    found when any piece appears, and the rank that matters is where the first
    piece landed. `None` is a real answer - "not in the top K" is the measurement,
    not a gap in it - and callers must handle it rather than coerce it to 0.
    """
    for rank, uid in enumerate(retrieved.unit_ids, start=1):
        u = index.units[index.unit_index[uid]]
        if u.source_id in gold_source_ids:
            return rank
    return None
