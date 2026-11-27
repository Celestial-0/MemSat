"""Top-K retrieval in rank order, with the budget applied per unit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from msat.memory.embed import cosine
from msat.memory.write import MemoryIndex


@dataclass(frozen=True, slots=True)
class Retrieved:
    unit: str
    unit_ids: tuple[str, ...]
    n_tokens: int
    n_tokens_unbudgeted: int
    scores: tuple[float, ...]
    truncation: dict[str, int]

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
    for rank, uid in enumerate(retrieved.unit_ids, start=1):
        u = index.units[index.unit_index[uid]]
        if u.source_id in gold_source_ids:
            return rank
    return None
