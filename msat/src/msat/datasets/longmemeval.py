"""Streaming loader for the cleaned LongMemEval release.

WHY STREAMING IS MANDATORY
--------------------------
`lme_m_cleaned.json` is **2.6 GB** and a *single record* in the S split exceeds
40 MB of JSON text. `json.load` raises `MemoryError` on the M split and is
hopeless on S. PRD §N2 requires streaming with a **quantified** RAM ceiling — a
test asserting "under some budget" with no number is unfailable, so the budget
lives in `RAM_CEILING_BYTES` and `tests/test_streaming.py` asserts against it.

TWO MODES, ONE FILE EACH WAY
---------------------------
`iter_metadata()`  — memory-flat. Walks the ijson event stream and keeps only
                     the fields the ceiling needs. Session *text* is received
                     as events and immediately discarded, so peak RSS does not
                     grow with corpus size. Used by `msat ceiling`, `data-lock`,
                     and every corpus statistic.

`iter_questions()` — materialises one whole record at a time via `ijson.items`.
                     Used by P3's write path, which genuinely needs the turns.

The metadata pass is not an optimisation detail: it is what lets a 2.6 GB file
be analysed on a laptop. It also *verifies* the full pass — `n_sessions` is
counted from `haystack_session_ids` and cross-checked against the
`haystack_sessions` array length, so a corrupt file fails loudly instead of
producing a confident wrong ceiling.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import ijson

from msat.datasets.schema import (
    CorpusStats,
    DataIntegrityError,
    Question,
    Session,
    Turn,
    is_abstention,
    validate_record,
)

#: Quantified RAM ceiling for the streaming loaders (PRD §N2).
#:
#: Derived from measurement, not taste: the metadata pass holds one partial
#: record, so its peak is set by the largest single record's *scalar* fields plus
#: the `haystack_session_ids` list (≈500 short strings on M). The full-record
#: pass is bounded by the largest single parsed record. Both were measured on the
#: S and M splits; see `tests/test_streaming.py::test_metadata_pass_is_flat`.
RAM_CEILING_BYTES: Final = 1_500 * 1024 * 1024  # 1.5 GiB

#: Raw inputs, by logical split. `FILLER` is Arm B's *only* permitted padding
#: source (PRD §R4, Arm B) and is not part of the S/M/oracle triple.
RAW_FILES: Final[dict[str, str]] = {
    "S": "lme_s_cleaned.json",
    "M": "lme_m_cleaned.json",
    "oracle": "lme_oracle.json",
    "filler": "5_filler_sess.json",
}

DEFAULT_DATA_DIRNAME: Final = "Research/context/raw"


def resolve_data_dir() -> Path:
    """Where the raw benchmarks live.

    `$MSAT_DATA_DIR` wins; otherwise `Research/context/raw` beside the `msat`
    package. The benchmarks live in the `Research` git submodule and are
    gitignored there, so they cannot simply be vendored into `msat/data/`
    (plan discrepancy D7).
    """
    override = os.environ.get("MSAT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root.parent / DEFAULT_DATA_DIRNAME).resolve()


def raw_path(split: str, data_dir: Path | None = None) -> Path:
    """Path to one raw file. Raises `KeyError` for an unknown split name."""
    try:
        filename = RAW_FILES[split]
    except KeyError:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(RAW_FILES)}") from None
    return (data_dir or resolve_data_dir()) / filename


def available_splits(data_dir: Path | None = None) -> dict[str, Path]:
    """Which raw files are actually present, as `{split: path}`."""
    root = data_dir or resolve_data_dir()
    return {k: v for k, v in ((s, root / f) for s, f in RAW_FILES.items()) if v.is_file()}


# ── metadata pass ──────────────────────────────────────────────────────────────
# Exact ijson prefixes. Explicit dispatch on whole prefixes rather than parsing
# them apart: the nested-array prefix for one session's turn list
# (`item.haystack_sessions.item`) *contains* `.item`, so any "is this a list
# element?" test based on substring matching silently swallows the event that
# counts sessions. Exact matching cannot misclassify a sibling field.
_P_QUESTION_ID: Final = "item.question_id"
_P_TYPE: Final = "item.question_type"
_P_QUESTION: Final = "item.question"
_P_QDATE: Final = "item.question_date"
_P_ANSWER: Final = "item.answer"
_P_ANSWER_ITEM: Final = "item.answer.item"

_P_GOLD_STARTS: Final = "item.answer_session_ids"
_P_GOLD_ITEM: Final = "item.answer_session_ids.item"
_P_IDS_STARTS: Final = "item.haystack_session_ids"
_P_IDS_ITEM: Final = "item.haystack_session_ids.item"
_P_DATES_STARTS: Final = "item.haystack_dates"
_P_DATES_ITEM: Final = "item.haystack_dates.item"

_P_SESSION: Final = "item.haystack_sessions.item"  # one session = array of turns
_P_TURN: Final = "item.haystack_sessions.item.item"  # one turn = map
_P_TURN_HAS_ANSWER: Final = "item.haystack_sessions.item.item.has_answer"
_P_TURN_CONTENT: Final = "item.haystack_sessions.item.item.content"

_SCALAR_PREFIXES: Final = {
    _P_QUESTION_ID: "question_id",
    _P_TYPE: "question_type",
    _P_QUESTION: "question",
    _P_QDATE: "question_date",
}
_SCALAR_EVENTS: Final = frozenset({"string", "number", "boolean"})


def iter_metadata(path: Path, *, validate: bool = True) -> Iterator[Question]:
    """Yield every question's metadata without ever holding its session text.

    Peak memory is one partial record's scalars, so this streams the 2.6 GB M
    split on a laptop (PRD §N2).

    Args:
        path: a cleaned LongMemEval JSON array.
        validate: run `validate_record` on every record. Off only for the
            `data-lock` row count, which does not care about structure.
    """
    with path.open("rb") as fh:
        parser = ijson.parse(fh, use_float=True)

        rec: dict[str, Any] = {}
        gold_ids: list[str] = []
        session_ids: list[str] = []
        in_answer_array = False
        n_dates = n_session_arrays = n_turns = n_chars = n_gold_turns = 0

        for prefix, event, value in parser:
            # ---- record boundaries -----------------------------------------
            if prefix == "item":
                if event == "start_map":
                    rec, in_answer_array = {}, False
                    gold_ids, session_ids = [], []
                    n_dates = n_session_arrays = n_turns = n_chars = n_gold_turns = 0
                    continue
                if event == "end_map":
                    yield _build_question(
                        rec,
                        path.name,
                        gold_ids,
                        session_ids,
                        n_dates,
                        n_session_arrays,
                        n_turns,
                        n_chars,
                        n_gold_turns,
                        validate,
                    )
                    rec = {}
                    continue
                continue  # `map_key` events on the record itself

            # ---- scalar fields ---------------------------------------------
            key = _SCALAR_PREFIXES.get(prefix)
            if key is not None:
                if event in _SCALAR_EVENTS:
                    rec[key] = value
                continue

            # ---- `answer`: scalar in most records, list in some -------------
            if prefix == _P_ANSWER:
                if event == "start_array":
                    in_answer_array, rec["answer"] = True, []
                elif event == "end_array":
                    in_answer_array = False
                elif event in _SCALAR_EVENTS:
                    rec["answer"] = value
                continue
            if prefix == _P_ANSWER_ITEM:
                if event in _SCALAR_EVENTS and in_answer_array:
                    rec["answer"].append(value)
                continue

            # ---- gold session ids ------------------------------------------
            if prefix == _P_GOLD_ITEM:
                if event == "string":
                    gold_ids.append(value)
                continue
            if prefix == _P_GOLD_STARTS:
                continue

            # ---- corpus session ids (the authoritative session count) ------
            if prefix == _P_IDS_ITEM:
                if event == "string":
                    session_ids.append(value)
                continue
            if prefix == _P_IDS_STARTS:
                continue

            # ---- dates: length only ----------------------------------------
            if prefix == _P_DATES_ITEM:
                if event == "string":
                    n_dates += 1
                continue
            if prefix == _P_DATES_STARTS:
                continue

            # ---- the haystack itself ---------------------------------------
            if prefix == _P_SESSION:
                if event == "start_array":
                    n_session_arrays += 1
                continue
            if prefix == _P_TURN:
                if event == "start_map":
                    n_turns += 1
                continue
            if prefix == _P_TURN_HAS_ANSWER:
                if event == "boolean" and value:
                    n_gold_turns += 1
                continue
            if prefix == _P_TURN_CONTENT:
                if event == "string":
                    n_chars += len(value)
                continue

            # Anything else (role strings, map_key events, nested maps we do not
            # model) is intentionally ignored. Session *text* arrives here and is
            # dropped without ever being accumulated — that is what keeps this pass
            # flat regardless of corpus size.
            continue


def _build_question(
    rec: dict[str, Any],
    filename: str,
    gold_ids: list[str],
    session_ids: list[str],
    n_dates: int,
    n_session_arrays: int,
    n_turns: int,
    n_chars: int,
    n_gold_turns: int,
    validate: bool,
) -> Question:
    qid = str(rec.get("question_id", ""))
    ability = str(rec.get("question_type", ""))
    gold = tuple(gold_ids)
    n_sessions = len(session_ids)

    if validate:
        validate_record(
            qid=qid,
            ability=ability,
            answerable=not is_abstention(qid),
            n_sessions=n_sessions,
            n_session_ids=n_sessions,
            n_dates=n_dates,
            n_session_arrays=n_session_arrays,
            gold_session_ids=gold,
            session_ids=tuple(session_ids),
            n_gold_turns=n_gold_turns,
        )

    raw_answer = rec.get("answer")
    answers = (
        tuple(raw_answer)
        if isinstance(raw_answer, list)
        else ((str(raw_answer),) if raw_answer else ())
    )

    return Question(
        qid=qid,
        scale=filename_to_scale(filename),
        question=str(rec.get("question", "")),
        question_date=str(rec.get("question_date", "")),
        gold_answers=answers,
        gold_session_ids=gold,
        ability=ability,
        answerable=not is_abstention(qid),
        corpus=CorpusStats(
            n_sessions=n_sessions,
            n_turns=n_turns,
            n_chars=n_chars,
            n_gold_sessions=len(gold),
            n_gold_turns=n_gold_turns,
        ),
        conversation_id=qid,
    )


def filename_to_scale(filename: str) -> str:
    """`lme_s_cleaned.json` -> `S`. Falls back to `""` for unknown names."""
    stem = Path(filename).stem.lower()
    for token, scale in (("_s_", "S"), ("_m_", "M"), ("s_cleaned", "S"), ("m_cleaned", "M")):
        if token in stem:
            return scale
    if stem.endswith("oracle") or "oracle" in stem:
        return "oracle"
    return ""


# ── full-record pass ───────────────────────────────────────────────────────────
def iter_questions(path: Path, *, validate: bool = True) -> Iterator[Question]:
    """Yield complete questions, session text included, one record at a time.

    Used by P3's write path. Memory is bounded by the largest single record —
    see `RAM_CEILING_BYTES`. For metadata and every corpus statistic, prefer
    `iter_metadata`, which is flat.
    """
    with path.open("rb") as fh:
        for raw in ijson.items(fh, "item", use_float=True):
            qid = str(raw.get("question_id", ""))
            ids = list(raw.get("haystack_session_ids", []))
            dates = list(raw.get("haystack_dates", []))
            session_arrays = list(raw.get("haystack_sessions", []))
            sessions = tuple(
                Session(
                    session_id=str(sid),
                    # `i` is bounded by len(ids) by the zip below, and ids/dates
                    # are validated to be the same length, so the index is safe.
                    date=str(dates[i]) if i < len(dates) else "",
                    turns=tuple(
                        Turn(
                            role=str(t.get("role", "")),
                            content=str(t.get("content", "")),
                            has_answer=bool(t.get("has_answer", False)),
                        )
                        for t in turns
                    ),
                )
                for i, (sid, turns) in enumerate(zip(ids, session_arrays, strict=False))
            )
            gold = tuple(raw.get("answer_session_ids") or ())
            n_gold_turns = sum(s.n_gold_turns for s in sessions)

            if validate:
                validate_record(
                    qid=qid,
                    ability=str(raw.get("question_type", "")),
                    answerable=not is_abstention(qid),
                    n_sessions=len(sessions),
                    n_session_ids=len(ids),
                    n_dates=len(dates),
                    n_session_arrays=len(session_arrays),
                    gold_session_ids=gold,
                    session_ids=tuple(s.session_id for s in sessions),
                    n_gold_turns=n_gold_turns,
                )

            raw_answer = raw.get("answer")
            answers = (
                tuple(str(a) for a in raw_answer)
                if isinstance(raw_answer, list)
                else ((str(raw_answer),) if raw_answer else ())
            )
            question_text = str(raw.get("question", ""))

            yield Question(
                qid=qid,
                scale=filename_to_scale(path.name),
                question=question_text,
                question_date=str(raw.get("question_date", "")),
                gold_answers=answers,
                gold_session_ids=gold,
                ability=str(raw.get("question_type", "")),
                answerable=not is_abstention(qid),
                corpus=CorpusStats(
                    n_sessions=len(sessions),
                    n_turns=sum(s.n_turns for s in sessions),
                    n_chars=sum(s.n_chars for s in sessions),
                    n_gold_sessions=len(gold),
                    n_gold_turns=n_gold_turns,
                ),
                conversation_id=qid,
                sessions=sessions,
            )


def iter_splits(
    splits: tuple[str, ...] = ("S", "M"), data_dir: Path | None = None
) -> Iterator[Question]:
    """Stream questions across several splits, tagging each with its scale."""
    root = data_dir or resolve_data_dir()
    for split in splits:
        path = raw_path(split, root)
        if not path.is_file():
            raise FileNotFoundError(
                f"raw file for split {split!r} not found: {path}\n"
                f"Set MSAT_DATA_DIR, or see `make env` for the resolved path."
            )
        yield from iter_metadata(path)


def count_records(path: Path) -> int:
    """Number of top-level objects in a JSON array, without parsing them.

    Used by `msat data-lock` for the row count (PRD §R2.3) without loading a
    2.6 GB file.
    """
    total = 0
    with path.open("rb") as fh:
        for prefix, event, _ in ijson.parse(fh):
            if prefix == "item" and event == "start_map":
                total += 1
    return total


def assert_data_integrity(paths: dict[str, Path]) -> dict[str, Any]:
    """Run `validate_record` over every present split and summarise.

    Returns a report; raises `DataIntegrityError` on the first violation, because
    a wrong ceiling is worse than no ceiling.
    """
    report: dict[str, Any] = {}
    for split, path in paths.items():
        n = 0
        n_answerable = 0
        n_abstention = 0
        sessions_min, sessions_max = None, 0
        sessions_total = 0
        gold_total = 0
        try:
            for q in iter_metadata(path):
                n += 1
                if q.answerable:
                    n_answerable += 1
                else:
                    n_abstention += 1
                s = q.corpus.n_sessions
                sessions_total += s
                sessions_max = max(sessions_max, s)
                sessions_min = s if sessions_min is None else min(sessions_min, s)
                gold_total += q.corpus.n_gold_sessions
        except DataIntegrityError:
            raise
        report[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_records": n,
            "n_answerable": n_answerable,
            "n_abstention": n_abstention,
            "sessions_per_question": {
                "min": sessions_min or 0,
                "max": sessions_max,
                "mean": round(sessions_total / n, 4) if n else 0.0,
            },
            "gold_sessions_total": gold_total,
        }
    return report
