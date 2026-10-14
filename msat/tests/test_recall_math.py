"""`recall@K` — the metric the paper's claims rest on.

Every fixture here is HAND-COMPUTED and small enough to verify by eye. That is the
whole point: the headline metric is the one number a referee will check, and a test
that computes it with the same code it is testing proves only that the code is
self-consistent.

The two divisions that are easy to get wrong, and which most of these tests pin:

  * gold is counted in SESSIONS, never chunks
  * an abstention item gets `None`, never 0.0 and never NaN
"""

from __future__ import annotations

import pytest

from msat.arms.evidence import (
    GoldItem,
    ItemKind,
    RecallSummary,
    classify,
    gold_session_ranks,
    recall_at_k,
    recall_at_k_from_prompt,
    summarise,
)
from msat.arms.render import Memory, RenderedPrompt
from msat.store.responses import ConditionKey

COND = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")


def _mem(session: str, chunk: int = 0) -> Memory:
    return Memory(
        memory_id=f"{session}#{chunk}",
        session_id=session,
        source_question_id="q",
        text="t",
        n_tokens=10,
    )


def _gold(*sessions: str) -> GoldItem:
    return GoldItem(qid="q1", sessions=frozenset(sessions))


# ── the arithmetic, hand-checked ───────────────────────────────────────────────


def test_all_gold_retrieved_is_one() -> None:
    r = recall_at_k([_mem("a"), _mem("b"), _mem("c")], _gold("c"), qid="q1")
    assert r.recall == 1.0
    assert r.gold_ranks == (3,)
    assert r.complete is True
    assert r.missing == ()


def test_no_gold_retrieved_is_zero() -> None:
    r = recall_at_k([_mem("x"), _mem("y")], _gold("a", "b"), qid="q1")
    assert r.recall == 0.0
    assert r.gold_ranks == ()
    assert r.missing == ("a", "b")
    assert r.complete is False


def test_partial_retrieval_is_a_fraction() -> None:
    """2 gold, 1 found -> 0.5. Not 1/2 of the memories, not 0."""
    r = recall_at_k([_mem("x"), _mem("a"), _mem("y")], _gold("a", "b"), qid="q1")
    assert r.recall == 0.5
    assert r.n_gold == 2
    assert r.n_gold_found == 1
    assert r.missing == ("b",)


def test_recall_is_computed_over_what_was_rendered_not_what_was_asked() -> None:
    """A renderer that dropped a memory under budget has changed K.

    Taking K as a parameter is how a caller reports recall for a K no prompt
    corresponds to. Here `k` comes from `len(memories)`.
    """
    r = recall_at_k([_mem("a")], _gold("a", "b"), qid="q1")
    assert r.k == 1
    assert r.recall == 0.5, "1 of 2 gold among ONE rendered memory"


# ── sessions, not chunks ───────────────────────────────────────────────────────


def test_chunks_of_one_session_count_once() -> None:
    """The single most likely bug in a recall implementation.

    Gold is a session. A session split into 3 chunks contributes 3 memory ids, so
    matching on ids lets one gold session fill the numerator and recall rises with
    the chunk size while retrieval gets no better.
    """
    mems = [_mem("gold", 0), _mem("other"), _mem("gold", 1), _mem("gold", 2)]
    ranks, missing = gold_session_ranks(mems, ["gold"])
    assert ranks == (1,), "one gold session, one rank - its FIRST appearance"
    assert missing == ()
    assert recall_at_k(mems, _gold("gold"), qid="q1").recall == 1.0


def test_a_gold_session_split_across_the_list_reports_its_first_rank() -> None:
    ranks, _ = gold_session_ranks(
        [_mem("a"), _mem("b"), _mem("c"), _mem("a", 1), _mem("d")], ["a", "b"]
    )
    assert ranks == (1, 2)


def test_gold_split_by_chunk_size_holds_recall_still() -> None:
    """Chunking must not move the metric at all."""
    fine = [_mem("gold", i) for i in range(6)] + [_mem("x")]
    coarse = [_mem("gold"), _mem("x")]
    assert recall_at_k(fine, _gold("gold"), qid="q1").recall == (
        recall_at_k(coarse, _gold("gold"), qid="q1").recall
    )


def test_gold_session_ranks_are_ascending() -> None:
    ranks, _ = gold_session_ranks([_mem("z"), _mem("a"), _mem("m")], ["z", "a", "m"])
    assert ranks == (1, 2, 3)


# ── R8.4 · abstention items ────────────────────────────────────────────────────


def test_an_abstention_item_gets_none_never_zero() -> None:
    """0/0 is not 0.0.

    Scoring it 0.0 says a perfect retriever retrieved nothing on every abstention
    item. NaN would drop them from the mean and shrink the denominator. R8.4
    requires them reported separately, which `None` plus the separate bucket does.
    """
    r = recall_at_k(
        [_mem("a"), _mem("b")], GoldItem(qid="q5_abs", sessions=frozenset()), qid="q5_abs"
    )
    assert r.recall is None
    assert r.kind is ItemKind.ABSTENTION
    assert r.n_gold == 0


def test_the_abstention_rule_is_the_suffix_not_emptiness() -> None:
    """A non-`_abs` item with no gold is a DATA failure and must not enter the
    denominator."""
    with pytest.raises(ValueError, match="not an abstention item"):
        classify("q5", GoldItem(qid="q5", sessions=frozenset()))


def test_an_abstention_suffix_with_unexpected_gold_is_still_abstention() -> None:
    """The suffix decides. An item labelled `_abs` is scored by R1.2's abstention
    rule regardless of what `answer_session_ids` happens to contain."""
    assert classify("q9_abs", GoldItem(qid="q9_abs", sessions=frozenset({"s"}))) is (
        ItemKind.ABSTENTION
    )


# ── the summary ────────────────────────────────────────────────────────────────


def _summary(items: list[tuple[list[Memory], GoldItem, str]]) -> RecallSummary:
    out = RecallSummary(k=len(items[0][0]))
    for mems, gold, qid in items:
        r = recall_at_k(mems, gold, qid=qid)
        (out.abstention if r.kind is ItemKind.ABSTENTION else out.answerable).append(r)
    return out


def test_the_two_populations_are_kept_apart() -> None:
    s = _summary(
        [
            ([_mem("a")], _gold("a"), "q1"),
            ([_mem("x")], _gold("a", "b"), "q2"),
            ([_mem("z")], GoldItem(qid="q3_abs", sessions=frozenset()), "q3_abs"),
        ]
    )
    assert len(s.answerable) == 2
    assert len(s.abstention) == 1
    # 1.0 and 0.0 mean 0.5 -- the abstention item is NOT counted as a 0.
    assert s.mean_recall == pytest.approx(0.5)
    assert s.as_dict()["n_abstention"] == 1


def test_mean_recall_is_none_with_no_answerable_items() -> None:
    s = _summary([([_mem("a")], GoldItem(qid="q1_abs", sessions=frozenset()), "q1_abs")])
    assert s.mean_recall is None
    assert s.frac_complete is None
    assert s.ceiling == 0.0


def test_completeness_is_reported_next_to_mean_recall() -> None:
    """Mean recall is sensitive to multi-session items; completeness is not, and
    a curve needs both."""
    s = _summary(
        [
            ([_mem("a"), _mem("b")], _gold("a", "b"), "q1"),  # 1.0, complete
            ([_mem("a")], _gold("a", "b"), "q2"),  # 0.5, incomplete
        ]
    )
    assert s.mean_recall == pytest.approx(0.75)
    assert s.frac_complete == pytest.approx(0.5)


def test_the_ceiling_is_the_perfect_retriever_and_recall_cannot_beat_it() -> None:
    """ceiling(K) = mean(min(1, K/|gold|)). With K=2 and one item holding 4 gold
    sessions, a perfect retriever reaches 0.5 - and nothing may exceed it."""
    s = _summary([([_mem("a"), _mem("b")], _gold("a", "b", "c", "d"), "q1")])
    assert s.ceiling == pytest.approx(0.5)
    assert s.mean_recall == pytest.approx(0.5)
    assert s.above_ceiling() == []


def test_a_recall_above_the_ceiling_is_reported_not_clamped() -> None:
    """Should be impossible with real data, and `above_ceiling()` is empty in every
    honest case - see the ceiling test above.

    The detector is still tested, against an INCONSISTENT record. Clamping would hide
    the bug that produced the impossible value; raising would hide every other one
    alongside it. Listing it is the only option that keeps the rest of the summary
    usable.
    """
    from msat.arms.evidence import RecallAtK

    impossible = RecallAtK(qid="q1", k=2, recall=1.0, n_gold=5, n_gold_found=5)
    assert RecallSummary(k=2, answerable=[impossible]).as_dict()["above_ceiling"] == ["q1"]


def test_the_gold_rank_histogram_counts_sessions() -> None:
    s = _summary(
        [
            ([_mem("a"), _mem("b")], _gold("a", "b"), "q1"),
            ([_mem("x"), _mem("b")], _gold("b"), "q2"),
        ]
    )
    # rank 1: q1's a, q2's x is not gold -> one gold session at rank 1
    assert s.gold_rank_histogram() == {1: 1, 2: 2}


# ── from the rendered prompt ───────────────────────────────────────────────────


def _prompt(session_ids: list[str], qid: str = "q1") -> RenderedPrompt:
    return RenderedPrompt(
        arm="A",
        condition=COND,
        system="",
        user="",
        memory_ids=tuple(f"{s}#0" for s in session_ids),
        session_ids=tuple(session_ids),
        ranks=tuple(range(1, len(session_ids) + 1)),
        k=len(session_ids),
        b=512,
        b_mode="budget",
        corpus_fraction=0.1,
        qid=qid,
    )


def test_recall_is_read_from_the_manifest_not_a_re_retrieval() -> None:
    r = recall_at_k_from_prompt(_prompt(["a", "b", "c"]), _gold("c"))
    assert r.recall == 1.0
    assert r.gold_ranks == (3,)


def test_a_prompt_with_no_session_ids_is_refused() -> None:
    """`is_gold_for` against an empty gold set would report 0.0 for a prompt nobody
    can score. Refusing beats a confident zero."""
    bad = RenderedPrompt(
        arm="A",
        condition=COND,
        system="",
        user="",
        memory_ids=(),
        session_ids=(),
        ranks=(),
        k=0,
        b=512,
        b_mode="budget",
        corpus_fraction=0.0,
        qid="q1",
    )
    with pytest.raises(ValueError, match="no session_ids"):
        recall_at_k_from_prompt(bad, _gold("a"))


# ── R4.2 · the manifest invariant that makes the metric trustworthy ─────────────


def test_misaligned_session_ids_are_rejected_at_construction() -> None:
    """Parallel tuples that disagree shift every rank without changing any count."""
    with pytest.raises(ValueError, match="parallel tuples"):
        RenderedPrompt(
            arm="A",
            condition=COND,
            system="",
            user="",
            memory_ids=("a#0", "b#0"),
            session_ids=("a",),
            ranks=(1, 2),
            k=2,
            b=512,
            b_mode="budget",
            corpus_fraction=0.0,
        )


def test_ranks_must_be_a_permutation_of_one_to_n() -> None:
    """A gap or a repeat means the ordering came from something other than the
    rendered sequence.

    A PERMUTATION is required, not dense-and-ascending: a future arm may
    legitimately list its memories in a different order than it ranks them, and the
    invariant must not forbid that. What it must forbid is a missing or repeated
    rank, which is what silently shifts gold into the wrong position.
    """
    for bad in ((1, 3), (1, 1), (2, 3)):
        with pytest.raises(ValueError, match="PERMUTATION"):
            RenderedPrompt(
                arm="A",
                condition=COND,
                system="",
                user="",
                memory_ids=("a#0", "b#0"),
                session_ids=("a", "b"),
                ranks=bad,
                k=2,
                b=512,
                b_mode="budget",
                corpus_fraction=0.0,
            )


def test_ordered_sessions_follows_the_ranks_not_the_list_position() -> None:
    """Memories listed out of rank order still score the right rank.

    This is exactly why the ranks check is a permutation rather than an equality
    with `1..n`: if the invariant forbade this, `ordered_session_ids` would be dead
    code that could never run.
    """
    p = RenderedPrompt(
        arm="A",
        condition=COND,
        system="",
        user="",
        memory_ids=("z#0", "a#0"),
        session_ids=("z", "a"),
        ranks=(2, 1),
        k=2,
        b=512,
        b_mode="budget",
        corpus_fraction=0.0,
        qid="q1",
    )
    assert p.ordered_session_ids() == ("a", "z")
    assert p.gold_ranks(frozenset({"z"})) == (2,)


# ── summarise ──────────────────────────────────────────────────────────────────


def test_summarise_produces_one_bucket_per_rendered_k() -> None:
    """K=1 bucket: q1 hits. K=2 bucket: q2 hits, q3 misses (its gold is absent)."""
    out = summarise(
        [_prompt(["a"], "q1"), _prompt(["a", "b"], "q2"), _prompt(["a", "b"], "q3")],
        {"q1": _gold("a"), "q2": _gold("a"), "q3": _gold("c")},
    )
    assert [s.k for s in out] == [1, 2]
    assert out[0].mean_recall == 1.0
    assert out[1].mean_recall == pytest.approx(0.5)
    assert out[1].as_dict()["n_answerable"] == 2


def test_summarise_never_invents_a_k_that_was_not_rendered() -> None:
    """A K absent from the prompts cannot appear on the curve with a made-up value."""
    out = summarise([_prompt(["a"], "q1")], {"q1": _gold("a")})
    assert [s.k for s in out] == [1]
