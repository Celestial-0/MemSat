"""Arm B — length held constant while memory count varies (PRD §P7, §R9.10).

THE CLAIM UNDER TEST
--------------------
`count` (how many memories) and `length` (how many tokens) are different things,
and an accuracy difference between a 2-memory and an 18-memory condition is only
evidence about *count* if the two conditions were the same *length*. Arm B is the
manipulation that holds length fixed so count can be isolated.

THE EXIT GATE, IN TWO NUMBERS
------------------------------
  * `total_tokens` matches the reference arm within ±2% (`TOKEN_TARGET_TOLERANCE`)
  * ZERO gold sessions appear in the padding

The second is the dangerous one. A padding memory drawn from a session that happens
to be the answer is not padding - it is the evidence, re-entered under a different
name, and the "insufficient evidence" condition silently becomes a sufficient one.
`tests/test_no_leakage.py` holds the complementary claim: that the R9.10 size
classifier, which is the right tool for a *deletion*-style construction, separates
Arm B by count *by design*, and that Arm B's own assertion is the token match.

WHY THE FIXTURE FAILS LOUDLY
----------------------------
Every test here builds a real `FillerSource` through `FillerSource.load` from a real
file. If the filler were synthesised in the constructor, the gold-leakage assertion
would be checked against text the test chose, and a change to `FillerSource`'s
contract would not surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.arms.padding import (
    FILLER_FILENAME,
    TOKEN_TARGET_TOLERANCE,
    FillerMissingError,
    FillerSource,
    GoldLeakageError,
    PaddingBudget,
    TokenTargetMissedError,
    assert_no_gold_leakage,
    leakage_features,
    pad_to_length,
    size_leakage_matrix,
)
from msat.arms.render import Memory

#: 456 chars / 4 chars-per-token = 114 tokens per pad session.
#:
#: Deliberately SMALLER than a 200-token retrieved memory. That size difference is
#: the arm's entire mechanism: at a matched total token count, the number of
#: memories the reader sees differs *only* because pad sessions are a different
#: granularity from retrieved chunks. Make the two the same size and the count
#: difference collapses - the arm stops manipulating anything.
PAD = "padding corpus text that is long enough to be worth adding. " * 8

#: 4% of 100 000 = 4 000 tokens, ±2% -> [3 920, 4 080].
CORPUS = 100_000
FRACTION = 0.04

FILLER_KEYS = (
    "achieved_tokens",
    "achieved_fraction",
    "n_padded",
    "n_retrieved",
    "token_tolerance",
    "filler_path",
)


def _filler(tmp_path: Path, *, n: int = 60) -> FillerSource:
    p = tmp_path / FILLER_FILENAME
    p.write_text(json.dumps({"sessions": [PAD] * n}), encoding="utf-8")
    return FillerSource.load(p)


def _memories(count: int, *, tokens: int = 200, prefix: str = "s") -> list[Memory]:
    return [
        Memory(
            memory_id=f"{prefix}{i}#0",
            session_id=f"{prefix}{i}",
            source_question_id="q1",
            text="body " * 40,
            n_tokens=tokens,
        )
        for i in range(count)
    ]


def _budget() -> PaddingBudget:
    return PaddingBudget(corpus_tokens=CORPUS, fraction=FRACTION)


# ── the two numbers the gate turns on ─────────────────────────────────────────


def test_the_tolerance_is_the_two_percent_the_prd_names() -> None:
    """Pinned as a named constant because it is the acceptance criterion, not a
    tuning parameter. A default argument lets it drift silently."""
    assert TOKEN_TARGET_TOLERANCE == 0.02
    assert _budget().tolerance == 0.02


def test_a_short_condition_padded_to_the_target_matches_within_two_percent(
    tmp_path: Path,
) -> None:
    budget = _budget()
    padded, report = pad_to_length(_memories(2), budget, _filler(tmp_path))  # 400 tokens
    got = sum(m.n_tokens for m in padded)
    assert budget.lower <= got <= budget.upper
    assert report["achieved_tokens"] == got
    assert report["achieved_fraction"] == pytest.approx(FRACTION, abs=TOKEN_TARGET_TOLERANCE)


def test_the_report_says_what_was_reached(tmp_path: Path) -> None:
    """A reader cannot tell a 40% arm from a 38% one without it, which is why it is
    in the return value rather than a log line. Checked against the real report
    rather than a hand-written list, so a renamed key fails here."""
    _, report = pad_to_length(_memories(2), _budget(), _filler(tmp_path))
    assert set(FILLER_KEYS) <= set(report)
    assert report["n_retrieved"] == 2
    assert report["token_tolerance"] == TOKEN_TARGET_TOLERANCE


def test_a_full_condition_is_not_padded_further(tmp_path: Path) -> None:
    """Padding a full-length condition would push it PAST the target and break the
    match that Arm B exists to establish."""
    padded, report = pad_to_length(_memories(20), _budget(), _filler(tmp_path))
    assert report["n_padded"] == 0, "20 x 200 = 4 000, already the target"
    assert len(padded) == 20
    assert sum(m.n_tokens for m in padded) == _budget().target_tokens


def test_count_varies_while_length_does_not(tmp_path: Path) -> None:
    """The arm's defining property: the same total tokens, a different number of
    memories the reader sees.

    The count difference is real but it is NOT the one a reader would guess.
    Padding is made of smaller units than retrieved memories, so the arm that
    retrieved LESS ends up with MORE total memories. Asserting the *retrieved*
    counts is the honest statement of what was manipulated; the rendered counts
    differ too, in the opposite direction, which is itself worth pinning.
    """
    budget = _budget()
    filler = _filler(tmp_path)
    short, short_rep = pad_to_length(_memories(2), budget, filler)
    long, long_rep = pad_to_length(_memories(18), budget, filler)

    short_tokens = sum(m.n_tokens for m in short)
    long_tokens = sum(m.n_tokens for m in long)
    rel = abs(short_tokens - long_tokens) / budget.target_tokens
    assert rel <= TOKEN_TARGET_TOLERANCE, f"lengths differ by {rel:.2%}"

    assert short_rep["n_retrieved"] < long_rep["n_retrieved"], "the manipulation itself"
    assert short_rep["n_padded"] > long_rep["n_padded"], (
        "the arm that retrieved less is made up with more padding"
    )
    assert len(short) != len(long), "rendered memory counts must differ"


def test_arm_b_length_matching_against_arm_a_is_reported_not_silent(tmp_path: Path) -> None:
    """`size_leakage_matrix` is the cross-arm check. A silent pass and a silent
    failure look the same in a log, so warnings are returned as data."""
    budget = _budget()
    filler = _filler(tmp_path)
    a, _ = pad_to_length(_memories(18), budget, filler)
    b, _ = pad_to_length(_memories(2), budget, filler)

    matched = {
        ("A", 1, 128.0, "budget", "S", "gemma"): leakage_features(a),
        ("B", 1, 128.0, "budget", "S", "gemma"): leakage_features(b),
    }
    assert size_leakage_matrix(matched)["warnings"] == []

    drifted = {
        ("A", 1, 128.0, "budget", "S", "gemma"): leakage_features(a),
        ("B", 1, 128.0, "budget", "S", "gemma"): leakage_features(_memories(2)),
    }
    assert size_leakage_matrix(drifted)["warnings"], (
        "an unpadded arm is a 10x gap and is not length-matched"
    )


# ── the dangerous half: gold must not appear in the padding ───────────────────


def test_zero_gold_sessions_reach_the_padding(tmp_path: Path) -> None:
    """The single most consequential assertion in this file.

    A padding memory drawn from an answer session is not padding; it is the
    evidence re-entered under another name, and the "insufficient evidence"
    condition silently becomes a sufficient one. The experiment then measures
    nothing and reports a confident number."""
    padded, _ = pad_to_length(
        _memories(2), _budget(), _filler(tmp_path), gold_sessions=["goldA", "goldB"]
    )
    pads = {m.session_id for m in padded if m.source_question_id == "__filler__"}
    assert pads & {"goldA", "goldB"} == set()


def test_a_gold_session_in_the_retrieved_set_is_refused(tmp_path: Path) -> None:
    """Checked on the way IN, not only on the padding: the arm must be told which
    sessions are the answer so it can exclude them."""
    with pytest.raises(GoldLeakageError):
        pad_to_length(_memories(4), _budget(), _filler(tmp_path), gold_sessions=["s1"])


def test_the_leakage_assertion_names_what_it_checked() -> None:
    """`what=` is in the signature because a bare "gold leakage" does not tell a
    reader whether the padding or the retrieved set failed."""
    with pytest.raises(GoldLeakageError) as exc:
        assert_no_gold_leakage(_memories(3), ["s0"], what="retrieved")
    assert "retrieved" in str(exc.value)


def test_the_leak_check_is_on_session_id_so_a_chunked_gold_cannot_slip_through() -> None:
    """The failure mode that made this check session-based rather than id-based.

    A gold session split into three chunks contributes three different memory ids, so
    an id-based check sees none of them as gold and passes. `session_id` is the
    identity that survives chunking."""
    chunks = [Memory(f"sGOLD#{i}", "sGOLD", "q1", "body", 100) for i in range(3)]
    assert "sGOLD" not in {m.memory_id for m in chunks}, (
        "the premise: memory_id alone would NOT match the session name"
    )
    with pytest.raises(GoldLeakageError):
        assert_no_gold_leakage(chunks, ["sGOLD"], what="padding")


def test_an_empty_gold_set_is_not_an_error() -> None:
    """Every item genuinely has no gold sessions (pure abstention). Demanding a
    non-empty set would make those questions unrenderable."""
    assert_no_gold_leakage(_memories(3), [], what="padding")


# ── a filler that is too small is refused, not worked around ───────────────────


def test_a_missing_filler_file_is_a_named_error(tmp_path: Path) -> None:
    """B3: `5_filler_sess.json` is an input nobody has produced yet. The arm names
    the file it wants rather than inventing sessions."""
    assert FILLER_FILENAME == "5_filler_sess.json"
    with pytest.raises(FillerMissingError, match=FILLER_FILENAME):
        FillerSource.load(tmp_path / FILLER_FILENAME)


def test_another_filler_filename_is_refused(tmp_path: Path) -> None:
    """`5_filler_sess.json` and nothing else. Substituting the target's own
    unretrieved sessions would inject real context a budgeted arm would never have
    shown, and nothing downstream would detect it."""
    p = tmp_path / "my_own_padding.json"
    p.write_text(json.dumps({"sessions": ["x"]}), encoding="utf-8")
    with pytest.raises(FillerMissingError):
        FillerSource.load(p)


def test_a_filler_with_no_sessions_is_refused(tmp_path: Path) -> None:
    p = tmp_path / FILLER_FILENAME
    p.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    with pytest.raises(FillerMissingError):
        FillerSource.load(p)


def test_a_filler_too_small_to_reach_the_target_is_refused(tmp_path: Path) -> None:
    """Padding more anyway would silently measure a different B than the condition
    key declares - the failure mode where the artifact and the experiment disagree
    and neither knows."""
    p = tmp_path / FILLER_FILENAME
    p.write_text(json.dumps({"sessions": ["short"]}), encoding="utf-8")
    with pytest.raises(TokenTargetMissedError, match=r"4%"):
        pad_to_length(_memories(2), _budget(), FillerSource.load(p))
