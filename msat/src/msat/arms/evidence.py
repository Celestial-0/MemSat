"""recall@K and gold-rank."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from msat.arms.render import Memory, RenderedPrompt

ABSTENTION_SUFFIX = "_abs"


class ItemKind(StrEnum):
    ANSWERABLE = "answerable"
    ABSTENTION = "abstention"


@dataclass(frozen=True, slots=True)
class GoldItem:
    qid: str
    sessions: frozenset[str]

    @property
    def n_gold(self) -> int:
        return len(self.sessions)

    @property
    def kind(self) -> ItemKind:
        return ItemKind.ABSTENTION if not self.sessions else ItemKind.ANSWERABLE

    @classmethod
    def from_record(cls, qid: str, record: Mapping[str, Any]) -> GoldItem:
        return cls(qid=qid, sessions=frozenset(record.get("answer_session_ids") or ()))

    @property
    def ceiling_at(self) -> Callable[[int], float | None]:
        n = self.n_gold
        return lambda k: min(1.0, k / n) if n else None


def classify(qid: str, gold: GoldItem) -> ItemKind:
    if qid.endswith(ABSTENTION_SUFFIX):
        return ItemKind.ABSTENTION
    if not gold.sessions:
        raise ValueError(
            f"{qid!r} is not an abstention item (no `_abs` suffix) but has no "
            f"`answer_session_ids`. Recall is undefined for it and it must not enter "
            f"the denominator - that is how a broken record turns into a flattering curve."
        )
    return ItemKind.ANSWERABLE


@dataclass(frozen=True, slots=True)
class RecallAtK:
    qid: str
    k: int
    recall: float | None
    n_gold: int
    n_gold_found: int
    gold_ranks: tuple[int, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def kind(self) -> ItemKind:
        return ItemKind.ABSTENTION if self.recall is None else ItemKind.ANSWERABLE

    @property
    def complete(self) -> bool:
        return self.n_gold > 0 and not self.missing


def gold_session_ranks(
    memories: Sequence[Memory],
    gold_sessions: Iterable[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    gold = set(gold_sessions)
    if not gold:
        return (), ()
    # Rank is first appearance; two chunks of one session share that session's rank.
    ranks: dict[str, int] = {}
    for rank, memory in enumerate(memories, start=1):
        sid = memory.session_id
        if sid in gold and sid not in ranks:
            ranks[sid] = rank
    ordered = tuple(ranks[s] for s in sorted(ranks, key=lambda s: ranks[s]))
    return ordered, tuple(sorted(gold - set(ranks)))


def recall_at_k(
    memories: Sequence[Memory],
    gold: GoldItem,
    *,
    qid: str | None = None,
) -> RecallAtK:
    qid = qid or gold.qid
    kind = classify(qid, gold)
    # K comes from the render, so a caller cannot score a depth it never used.
    k = len(memories)
    if kind is ItemKind.ABSTENTION:
        return RecallAtK(
            qid=qid,
            k=k,
            recall=None,
            n_gold=0,
            n_gold_found=0,
            gold_ranks=(),
            missing=(),
        )
    ranks, missing = gold_session_ranks(memories, gold.sessions)
    found = len(ranks)
    return RecallAtK(
        qid=qid,
        k=k,
        recall=found / gold.n_gold,
        n_gold=gold.n_gold,
        n_gold_found=found,
        gold_ranks=ranks,
        missing=missing,
    )


@dataclass
class RecallSummary:
    k: int
    answerable: list[RecallAtK] = field(default_factory=list)
    abstention: list[RecallAtK] = field(default_factory=list)

    @property
    def mean_recall(self) -> float | None:
        vals = [r.recall for r in self.answerable if r.recall is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def frac_complete(self) -> float | None:
        if not self.answerable:
            return None
        return sum(1 for r in self.answerable if r.complete) / len(self.answerable)

    @property
    def ceiling(self) -> float:
        if not self.answerable:
            return 0.0
        return sum(min(1.0, self.k / r.n_gold) for r in self.answerable) / len(self.answerable)

    def above_ceiling(self) -> list[str]:
        out = []
        for r in self.answerable:
            if r.recall is not None and r.recall > min(1.0, self.k / r.n_gold) + 1e-12:
                out.append(r.qid)
        return out

    def gold_rank_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for r in self.answerable:
            for rank in r.gold_ranks:
                hist[rank] = hist.get(rank, 0) + 1
        return dict(sorted(hist.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_answerable": len(self.answerable),
            "n_abstention": len(self.abstention),
            "mean_recall": self.mean_recall,
            "frac_complete": self.frac_complete,
            "ceiling": round(self.ceiling, 6),
            "above_ceiling": self.above_ceiling(),
            "gold_rank_histogram": self.gold_rank_histogram(),
        }


def summarise(
    prompts: Sequence[RenderedPrompt], gold_by_qid: Mapping[str, GoldItem]
) -> list[RecallSummary]:
    buckets: dict[int, RecallSummary] = {}
    for prompt in prompts:
        item = recall_at_k_from_prompt(prompt, gold_by_qid[prompt.qid])
        bucket = buckets.setdefault(item.k, RecallSummary(k=item.k))
        target = bucket.abstention if item.kind is ItemKind.ABSTENTION else bucket.answerable
        target.append(item)
    return [buckets[k] for k in sorted(buckets)]


def recall_at_k_from_prompt(prompt: RenderedPrompt, gold: GoldItem) -> RecallAtK:
    sessions = prompt.ordered_session_ids()
    if not sessions:
        raise ValueError(
            f"rendered prompt for {prompt.qid or '<unknown qid>'} carries no "
            f"session_ids; recall@K is undefined for it"
        )
    memories = [
        Memory(
            memory_id=mid,
            session_id=sid,
            source_question_id=prompt.qid,
            text="",
            n_tokens=0,
        )
        for mid, sid in zip(prompt.memory_ids, sessions, strict=True)
    ]
    return recall_at_k(memories, gold, qid=prompt.qid)
