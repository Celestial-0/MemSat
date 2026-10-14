"""Frozen record types for the LongMemEval loader.

These are the ground truth every downstream artifact is attributed to. Two
invariants are encoded here rather than in prose because a violation of either
silently corrupts the entire study:

1. **Positional alignment.** `haystack_sessions[i]`, `haystack_session_ids[i]`
   and `haystack_dates[i]` describe the same session. Recall@K is computed by
   rank over an ordering produced from one of these; if the three lists are not
   the same length, every gold-resolution answer is wrong.

2. **Gold is declared twice, independently.** `answer_session_ids` names the
   gold *sessions*; `has_answer` marks the gold *turns* inside them. The
   official data carries both. `validate_record` asserts they agree, which turns
   a would-be silent mismatch into a loud loader error.

Everything is `frozen=True, slots=True`. A `Question` is a fact about the
benchmark, not a mutable working object — the write path must not be able to
edit its own evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

#: The six question types in the official LongMemEval release, plus the
#: abstention suffix. Verified against the raw data, not assumed.
QUESTION_TYPES: Final[tuple[str, ...]] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)

ABSTENTION_SUFFIX: Final = "_abs"

#: Splits shipped by the official release and consumed here.
SCALES: Final[tuple[str, ...]] = ("S", "M")

DATASET_NAME: Final = "longmemeval"
DATASET_VERSION: Final = "lme_cleaned"


def is_abstention(question_id: str) -> bool:
    """Abstention items carry the `_abs` suffix and are scored by the benchmark's
    own rule — excluded from `recall@K`, kept in end-to-end accuracy (PRD §R8.4)."""
    return question_id.endswith(ABSTENTION_SUFFIX)


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversation turn."""

    role: str
    content: str
    has_answer: bool = False

    @property
    def n_chars(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class Session:
    """One retrieved "memory" at `unit = session`.

    The benchmark's evidence is session-level (~2.5K tokens each), but real
    systems retrieve facts or chunks — so `unit` lives in `retrieval.yaml` and
    both items-retrieved and tokens-retrieved are reported downstream (PRD §R1.6).
    """

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
    """Per-question corpus statistics.

    PRD §R3.1 is explicit that corpus-level percentiles are insufficient: K=32 is
    80% of a 40-session conversation, yet a mean-based guard passes. Every guard
    in this project reads `n_sessions` from *this* object for *this* question.
    """

    n_sessions: int
    n_turns: int = 0
    n_chars: int = 0
    n_gold_sessions: int = 0
    n_gold_turns: int = 0


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question, with its corpus.

    `sessions` is empty when the record was loaded in metadata-only mode (the
    ceiling pass); `corpus.n_sessions` is still exact, because the session count
    comes from `haystack_session_ids` and is verified against the
    `haystack_sessions` array length. See `datasets.longmemeval`.
    """

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
        """Corpus session ids in haystack order."""
        if self.sessions:
            return tuple(s.session_id for s in self.sessions)
        return ()

    @property
    def is_empty_corpus(self) -> bool:
        """A question with no haystack at all. Degenerate by construction (PRD §R3.5)."""
        return self.corpus.n_sessions == 0

    def resolve_gold_indices(self) -> tuple[int, ...]:
        """Positions of the gold sessions in haystack order.

        Raises if any declared gold id is absent from the corpus — a gold session
        that does not exist makes `recall@K` undefined rather than merely low.
        """
        index = {sid: i for i, sid in enumerate(self.session_ids)}
        missing = [sid for sid in self.gold_session_ids if sid not in index]
        if missing:
            raise KeyError(
                f"{self.qid}: {len(missing)} gold session id(s) absent from the corpus "
                f"(first: {missing[0]!r}). Recall@K is undefined for this question."
            )
        return tuple(index[sid] for sid in self.gold_session_ids)


class DataIntegrityError(RuntimeError):
    """A raw file violates an invariant the study depends on.

    Named separately from `ValueError` so callers can distinguish "this dataset is
    not what we thought" from "this record is malformed".
    """


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
    """Assert the structural invariants for one record, or raise.

    Checked in order of how badly each failure would corrupt the study.
    """
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

    # The two independent gold declarations must agree — but ONLY for answerable
    # items, and this is a measured property of the data rather than an assumption.
    #
    # Measured 2026-10-12 over S and oracle: **0 mismatches across all 470
    # answerable records**, and **21 of 30 abstention records** mismatch. That is
    # correct, not damage: an abstention question asks about information the corpus
    # does not contain, so `answer_session_ids` names the sessions that *would*
    # hold the answer and no turn can carry `has_answer: true`. Enforcing the
    # cross-check on `_abs` items would reject 21 valid records; skipping it
    # everywhere would hide a real lost-flag bug. Hence the explicit gate.
    #
    # `recall@K` is not computed on abstention items at all (PRD §R8.4).
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
