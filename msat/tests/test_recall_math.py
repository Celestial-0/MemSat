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
from msat.store import Condition

COND = Condition(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")


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
    r = recall_at_k([_mem("x"), _mem("a"), _mem("y")], _gold("a", "b"), qid="q1")
    assert r.recall == 0.5
    assert r.n_gold == 2
    assert r.n_gold_found == 1
    assert r.missing == ("b",)


def test_recall_is_computed_over_what_was_rendered_not_what_was_asked() -> None:
    r = recall_at_k([_mem("a")], _gold("a", "b"), qid="q1")
    assert r.k == 1
    assert r.recall == 0.5, "1 of 2 gold among ONE rendered memory"


def test_chunks_of_one_session_count_once() -> None:
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
    fine = [_mem("gold", i) for i in range(6)] + [_mem("x")]
    coarse = [_mem("gold"), _mem("x")]
    assert recall_at_k(fine, _gold("gold"), qid="q1").recall == (
        recall_at_k(coarse, _gold("gold"), qid="q1").recall
    )


def test_gold_session_ranks_are_ascending() -> None:
    ranks, _ = gold_session_ranks([_mem("z"), _mem("a"), _mem("m")], ["z", "a", "m"])
    assert ranks == (1, 2, 3)


def test_an_abstention_item_gets_none_never_zero() -> None:
    r = recall_at_k(
        [_mem("a"), _mem("b")], GoldItem(qid="q5_abs", sessions=frozenset()), qid="q5_abs"
    )
    assert r.recall is None
    assert r.kind is ItemKind.ABSTENTION
    assert r.n_gold == 0


def test_the_abstention_rule_is_the_suffix_not_emptiness() -> None:
    with pytest.raises(ValueError, match="not an abstention item"):
        classify("q5", GoldItem(qid="q5", sessions=frozenset()))


def test_an_abstention_suffix_with_unexpected_gold_is_still_abstention() -> None:
    assert classify("q9_abs", GoldItem(qid="q9_abs", sessions=frozenset({"s"}))) is (
        ItemKind.ABSTENTION
    )


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
    assert s.mean_recall == pytest.approx(0.5)
    assert s.as_dict()["n_abstention"] == 1


def test_mean_recall_is_none_with_no_answerable_items() -> None:
    s = _summary([([_mem("a")], GoldItem(qid="q1_abs", sessions=frozenset()), "q1_abs")])
    assert s.mean_recall is None
    assert s.frac_complete is None
    assert s.ceiling == 0.0


def test_completeness_is_reported_next_to_mean_recall() -> None:
    s = _summary(
        [
            ([_mem("a"), _mem("b")], _gold("a", "b"), "q1"),
            ([_mem("a")], _gold("a", "b"), "q2"),
        ]
    )
    assert s.mean_recall == pytest.approx(0.75)
    assert s.frac_complete == pytest.approx(0.5)


def test_the_ceiling_is_the_perfect_retriever_and_recall_cannot_beat_it() -> None:
    s = _summary([([_mem("a"), _mem("b")], _gold("a", "b", "c", "d"), "q1")])
    assert s.ceiling == pytest.approx(0.5)
    assert s.mean_recall == pytest.approx(0.5)
    assert s.above_ceiling() == []


def test_a_recall_above_the_ceiling_is_reported_not_clamped() -> None:
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
    assert s.gold_rank_histogram() == {1: 1, 2: 2}


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


def test_misaligned_session_ids_are_rejected_at_construction() -> None:
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


def test_summarise_produces_one_bucket_per_rendered_k() -> None:
    out = summarise(
        [_prompt(["a"], "q1"), _prompt(["a", "b"], "q2"), _prompt(["a", "b"], "q3")],
        {"q1": _gold("a"), "q2": _gold("a"), "q3": _gold("c")},
    )
    assert [s.k for s in out] == [1, 2]
    assert out[0].mean_recall == 1.0
    assert out[1].mean_recall == pytest.approx(0.5)
    assert out[1].as_dict()["n_answerable"] == 2


def test_summarise_never_invents_a_k_that_was_not_rendered() -> None:
    out = summarise([_prompt(["a"], "q1")], {"q1": _gold("a")})
    assert [s.k for s in out] == [1]
