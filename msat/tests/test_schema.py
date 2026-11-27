from __future__ import annotations

from pathlib import Path

import pytest

from msat.datasets.schema import (
    CorpusStats,
    DataIntegrityError,
    Question,
    Session,
    Turn,
    is_abstention,
    validate_record,
)
from tests.conftest import requires_s

_OK: dict[str, object] = {
    "ability": "multi-session",
    "answerable": True,
    "n_sessions": 4,
    "n_session_ids": 4,
    "n_dates": 4,
    "n_session_arrays": 4,
    "gold_session_ids": ("s0", "s1"),
    "session_ids": ("s0", "s1", "s2", "s3"),
    "n_gold_turns": 2,
}


def test_valid_record_passes() -> None:
    validate_record(qid="abc", **_OK)


def test_mismatched_positional_arrays_are_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="positional arrays disagree"):
        validate_record(qid="abc", **{**_OK, "n_session_ids": 3})


def test_gold_session_absent_from_corpus_is_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="absent from the corpus"):
        validate_record(qid="abc", **{**_OK, "gold_session_ids": ("missing",)})


def test_unknown_question_type_is_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="unknown question_type"):
        validate_record(qid="abc", **{**_OK, "ability": "made-up-type"})


def test_lost_gold_flags_are_rejected_for_answerable_items() -> None:
    with pytest.raises(DataIntegrityError, match="gold declarations disagree"):
        validate_record(qid="abc", **{**_OK, "n_gold_turns": 0})


def test_gold_turns_without_gold_ids_are_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="answer_session_ids` is empty"):
        validate_record(qid="abc", **{**_OK, "gold_session_ids": ()})


def test_abstention_items_are_exempt_from_the_gold_cross_check() -> None:
    validate_record(
        qid="0862e8bf_abs",
        **{**_OK, "ability": "multi-session", "answerable": False, "n_gold_turns": 0},
    )


def test_session_counts_must_agree_between_the_two_passes() -> None:
    with pytest.raises(DataIntegrityError, match="metadata pass and the record pass"):
        validate_record(qid="abc", **{**_OK, "n_sessions": 99})


def test_is_abstention() -> None:
    assert is_abstention("abc_abs")
    assert not is_abstention("abc")


def test_resolve_gold_indices() -> None:
    q = Question(
        qid="a",
        scale="S",
        question="q",
        question_date="",
        gold_answers=("x",),
        gold_session_ids=("s2", "s0"),
        ability="multi-session",
        answerable=True,
        corpus=CorpusStats(n_sessions=3, n_gold_sessions=2),
        sessions=(
            Session("s0", "", (Turn("user", "a"),)),
            Session("s1", "", (Turn("user", "b"),)),
            Session("s2", "", (Turn("user", "c"),)),
        ),
    )
    assert q.resolve_gold_indices() == (2, 0)


def test_resolve_gold_indices_raises_on_missing() -> None:
    q = Question(
        qid="a",
        scale="S",
        question="q",
        question_date="",
        gold_answers=("x",),
        gold_session_ids=("nope",),
        ability="multi-session",
        answerable=True,
        corpus=CorpusStats(n_sessions=1, n_gold_sessions=1),
        sessions=(Session("s0", "", (Turn("user", "a"),)),),
    )
    with pytest.raises(KeyError, match="absent from the corpus"):
        q.resolve_gold_indices()


def test_session_properties() -> None:
    s = Session(
        "s0",
        "date",
        (Turn("user", "hello", False), Turn("assistant", "world", True)),
    )
    assert s.n_turns == 2
    assert s.n_chars == len("hello") + len("world")
    assert s.n_gold_turns == 1
    assert s.has_gold_turn


def test_empty_corpus_is_detected() -> None:
    q = Question(
        qid="a",
        scale="S",
        question="q",
        question_date="",
        gold_answers=(),
        gold_session_ids=(),
        ability="multi-session",
        answerable=True,
        corpus=CorpusStats(n_sessions=0),
    )
    assert q.is_empty_corpus


@requires_s
def test_the_real_split_splits_470_30(path_s: Path) -> None:
    from msat.datasets import iter_metadata

    questions = list(iter_metadata(path_s))
    assert len(questions) == 500
    assert sum(q.answerable for q in questions) == 470
    assert sum(not q.answerable for q in questions) == 30


@requires_s
def test_every_gold_id_resolves_on_the_real_split(path_s: Path) -> None:
    from itertools import islice

    from msat.datasets import iter_questions

    for q in islice(iter_questions(path_s), 40):
        indices = q.resolve_gold_indices()
        assert indices, f"{q.qid} resolved no gold index"
        assert all(0 <= i < q.corpus.n_sessions for i in indices)
