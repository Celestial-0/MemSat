"""`recall@K` and gold-rank — the metric the paper's claims rest on.

PRD §R4.2 names the manifest as "the ground truth for recall@K and gold-rank
analysis, computed from the actual pipeline, never a reimplementation". That last
clause is the whole reason this module exists as a separate file with no access to
the retriever.

THE HEADLINE, STATED PLAINLY
----------------------------
    recall@K = |gold SESSIONS present among the K rendered memories| / |gold sessions|

It is a property of the RENDERED PROMPT, not of what the retriever proposed. A
retriever can rank the gold session first and the renderer can then drop it under a
budget; `recall@K` must describe what the reader actually saw, because that is
what the accuracy number is caused by.

TWO DIVISIONS THAT ARE EASY TO GET WRONG
----------------------------------------
1. **Sessions, not chunks.** One session yields many chunks. Scoring chunks lets a
   single gold session inflate the numerator by however many pieces it was split
   into, and `recall@K` then rises with the chunk size while retrieval gets no
   better. `test_chunks_of_one_session_count_once` pins this.

2. **Rank is 1-based and over SESSIONS.** The first distinct gold session a reader
   meets is at rank r; every later chunk of it is the SAME rank. A naive
   `index + 1` over the memory list reports the second chunk of a rank-1 session as
   rank 2 and the gold-rank histogram is then a function of chunking.

R8.4 — ABSTENTION ITEMS ARE NOT MEASURED HERE
---------------------------------------------
An `_abs` item has `answer_session_ids == []` by construction, so `recall@K` is
0/0. Two tempting wrong answers: NaN (which silently drops every abstention item
from the mean and shrinks the denominator) or 0.0 (which says a perfect retriever
scored zero). Both are lies. R8.4 requires them reported separately, so
`recall_at_k` returns `None` and `recall_summary` keeps them in their own bucket
with their count.

`ceiling(K) = Pr(|gold| <= K)` (PRD §ceiling) is the matching upper bound: a
PERFECT retriever still only reaches `min(1, K/|gold|)` on each item, so a recall
curve above the ceiling curve is a bug in this module, not a finding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from msat.arms.render import Memory, RenderedPrompt

ABSTENTION_SUFFIX = "_abs"
"""LongMemEval's abstention items are suffixed `_abs`. PRD §R8.4 gives them an
explicit scoring rule and requires them reported separately from recall@K."""


class ItemKind(StrEnum):
    ANSWERABLE = "answerable"
    ABSTENTION = "abstention"


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One question's evidence, resolved.

    `sessions` is `answer_session_ids`. It is VERIFIED against the haystack in
    `datasets/schema.py`, so by the time a GoldItem exists, every id in it names a
    session that is actually present in the corpus.
    """

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
        """`max recall@K` for a PERFECT retriever, as a callable of K."""
        n = self.n_gold
        return lambda k: min(1.0, k / n) if n else None


def classify(qid: str, gold: GoldItem) -> ItemKind:
    """The `_abs` rule is checked FIRST and by suffix, not by emptiness alone.

    Both signals are consulted because each is load-bearing on its own: a
    non-`_abs` item with no gold sessions is a DATA INTEGRITY failure, and
    accepting it would quietly compute a recall curve over a broken denominator.
    """
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
    """One item's retrieval outcome."""

    qid: str
    k: int
    recall: float | None
    """None for an abstention item. R8.4: never computed, never imputed."""
    n_gold: int
    n_gold_found: int
    gold_ranks: tuple[int, ...] = ()
    """1-based rank of each gold SESSION in the rendered prompt, ascending.

    Empty for a gold session that was not retrieved at all - which is different
    from a gold session at rank 0 and must not be encoded as such.
    """
    missing: tuple[str, ...] = ()

    @property
    def kind(self) -> ItemKind:
        return ItemKind.ABSTENTION if self.recall is None else ItemKind.ANSWERABLE

    @property
    def complete(self) -> bool:
        """Every gold session retrieved. Distinct from recall == 1.0 in general."""
        return self.n_gold > 0 and not self.missing


def gold_session_ranks(
    memories: Sequence[Memory],
    gold_sessions: Iterable[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """`(ranks, missing)` — rank is over SESSIONS in first-appearance order.

    Two chunks of the same gold session both land at the session's rank. Scanning
    once and recording only the first appearance is what keeps the gold-rank
    histogram a property of retrieval rather than of the chunk size.
    """
    gold = set(gold_sessions)
    if not gold:
        return (), ()
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
    """`recall@K` for one rendered prompt.

    `k` is `len(memories)`, NOT a parameter. Taking K as an argument is how a
    caller computes recall for a K it did not actually render and reports a curve
    that no prompt corresponds to; the rendered length is the only K that exists.
    """
    qid = qid or gold.qid
    kind = classify(qid, gold)
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
    """A recall curve over many items, with the two populations kept apart."""

    k: int
    answerable: list[RecallAtK] = field(default_factory=list)
    abstention: list[RecallAtK] = field(default_factory=list)

    @property
    def mean_recall(self) -> float | None:
        """Mean over ANSWERABLE items only. None when there are none."""
        vals = [r.recall for r in self.answerable if r.recall is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def frac_complete(self) -> float | None:
        """Fraction of answerable items with EVERY gold session retrieved.

        Reported next to mean recall because the two answer different questions:
        mean recall is sensitive to multi-session items, completeness is not.
        """
        if not self.answerable:
            return None
        return sum(1 for r in self.answerable if r.complete) / len(self.answerable)

    @property
    def ceiling(self) -> float:
        """`sum(min(1, K/|gold|)) / n` — a PERFECT retriever's mean recall here."""
        if not self.answerable:
            return 0.0
        return sum(min(1.0, self.k / r.n_gold) for r in self.answerable) / len(self.answerable)

    def above_ceiling(self) -> list[str]:
        """Items whose recall beats the theoretical maximum.

        Should be impossible. Reported rather than raised, because the right
        response to an impossible recall value is to see ALL of them, not to crash
        on the first - but it is never `None`, never 0.0, and never clamped.
        """
        out = []
        for r in self.answerable:
            if r.recall is not None and r.recall > min(1.0, self.k / r.n_gold) + 1e-12:
                out.append(r.qid)
        return out

    def gold_rank_histogram(self) -> dict[int, int]:
        """How many gold SESSIONS landed at each rank. The lost-in-the-middle view."""
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
    """One summary per K actually rendered, in ascending K.

    Derived from the prompts themselves rather than from a declared grid, so a K
    that was never rendered cannot appear on the curve with a made-up value.
    """
    buckets: dict[int, RecallSummary] = {}
    for prompt in prompts:
        item = recall_at_k_from_prompt(prompt, gold_by_qid[prompt.qid])
        bucket = buckets.setdefault(item.k, RecallSummary(k=item.k))
        target = bucket.abstention if item.kind is ItemKind.ABSTENTION else bucket.answerable
        target.append(item)
    return [buckets[k] for k in sorted(buckets)]


def recall_at_k_from_prompt(prompt: RenderedPrompt, gold: GoldItem) -> RecallAtK:
    """Recall for a rendered prompt.

    Reads the SESSION sequence the prompt actually rendered, ordered by `ranks` -
    not a re-retrieval. That is what makes the manifest the ground truth per
    §R4.2 rather than a restatement of the retriever's own opinion.
    """
    sessions = prompt.ordered_session_ids()
    if not sessions:
        # A prompt with no sessions cannot be scored, and `Memory.is_gold_for`
        # would return False for every session in an EMPTY gold set. Refuse rather
        # than report a recall of 0 for a prompt nobody can score.
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
