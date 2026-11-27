"""Frozen record types for the LongMemEval loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

QUESTION_TYPES: Final[tuple[str, ...]] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)

ABSTENTION_SUFFIX: Final = "_abs"

SCALES: Final[tuple[str, ...]] = ("S", "M")

DATASET_NAME: Final = "longmemeval"
DATASET_VERSION: Final = "lme_cleaned"


def is_abstention(question_id: str) -> bool:
    return question_id.endswith(ABSTENTION_SUFFIX)


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False

    @property
    def n_chars(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_chars(self) -> int:
        return sum(t.n_chars for t in self.turns)

    @property
    def n_gold_turns(self) -> int:
        return sum(1 for t in self.turns if t.has_answer)

    @property
    def has_gold_turn(self) -> bool:
        return self.n_gold_turns > 0


@dataclass(frozen=True, slots=True)
class CorpusStats:
    n_sessions: int
    n_turns: int = 0
    n_chars: int = 0
    n_gold_sessions: int = 0
    n_gold_turns: int = 0


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    scale: str
    question: str
    question_date: str
    gold_answers: tuple[str, ...]
    gold_session_ids: tuple[str, ...]
    ability: str
    answerable: bool
    corpus: CorpusStats
    dataset: str = DATASET_NAME
    dataset_version: str = DATASET_VERSION
    conversation_id: str = ""
    sessions: tuple[Session, ...] = field(default=())

    @property
    def n_gold_sessions(self) -> int:
        return len(self.gold_session_ids)

    @property
    def session_ids(self) -> tuple[str, ...]:
        if self.sessions:
            return tuple(s.session_id for s in self.sessions)
        return ()

    @property
    def is_empty_corpus(self) -> bool:
        return self.corpus.n_sessions == 0

    def resolve_gold_indices(self) -> tuple[int, ...]:
        index = {sid: i for i, sid in enumerate(self.session_ids)}
        missing = [sid for sid in self.gold_session_ids if sid not in index]
        if missing:
            raise KeyError(
                f"{self.qid}: {len(missing)} gold session id(s) absent from the corpus "
                f"(first: {missing[0]!r}). Recall@K is undefined for this question."
            )
        return tuple(index[sid] for sid in self.gold_session_ids)


class DataIntegrityError(RuntimeError):
    """A raw file violates an invariant the study depends on."""


def validate_record(
    *,
    qid: str,
    ability: str,
    answerable: bool,
    n_sessions: int,
    n_session_ids: int,
    n_dates: int,
    n_session_arrays: int,
    gold_session_ids: tuple[str, ...],
    session_ids: tuple[str, ...],
    n_gold_turns: int,
) -> None:
    lengths = {
        "haystack_sessions": n_session_arrays,
        "haystack_session_ids": n_session_ids,
        "haystack_dates": n_dates,
    }
    if len(set(lengths.values())) != 1:
        raise DataIntegrityError(
            f"{qid}: positional arrays disagree — {lengths}. "
            "Recall@K and gold-rank analysis are undefined if these lengths differ."
        )
    if n_sessions != n_session_arrays:
        raise DataIntegrityError(
            f"{qid}: corpus.n_sessions ({n_sessions}) != len(haystack_sessions) "
            f"({n_session_arrays}). The metadata pass and the record pass disagree."
        )
    if ability not in QUESTION_TYPES and not is_abstention(qid):
        raise DataIntegrityError(
            f"{qid}: unknown question_type {ability!r}. Expected one of "
            f"{QUESTION_TYPES} (or an `_abs` item)."
        )
    if session_ids:
        corpus = set(session_ids)
        unknown = [g for g in gold_session_ids if g not in corpus]
        if unknown:
            raise DataIntegrityError(
                f"{qid}: {len(unknown)} gold session id(s) absent from the corpus "
                f"(first: {unknown[0]!r})."
            )

    if not answerable:
        return
    if gold_session_ids and n_gold_turns == 0:
        raise DataIntegrityError(
            f"{qid}: {len(gold_session_ids)} gold session(s) declared but zero turns "
            "carry `has_answer: true` — the two gold declarations disagree."
        )
    if n_gold_turns and not gold_session_ids:
        raise DataIntegrityError(
            f"{qid}: {n_gold_turns} turn(s) carry `has_answer: true` but "
            "`answer_session_ids` is empty."
        )
