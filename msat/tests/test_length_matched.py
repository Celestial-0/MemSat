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

PAD = "padding corpus text that is long enough to be worth adding. " * 8

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


def test_the_tolerance_is_the_two_percent_the_prd_names() -> None:
    assert TOKEN_TARGET_TOLERANCE == 0.02
    assert _budget().tolerance == 0.02


def test_a_short_condition_padded_to_the_target_matches_within_two_percent(
    tmp_path: Path,
) -> None:
    budget = _budget()
    padded, report = pad_to_length(_memories(2), budget, _filler(tmp_path))
    got = sum(m.n_tokens for m in padded)
    assert budget.lower <= got <= budget.upper
    assert report["achieved_tokens"] == got
    assert report["achieved_fraction"] == pytest.approx(FRACTION, abs=TOKEN_TARGET_TOLERANCE)


def test_the_report_says_what_was_reached(tmp_path: Path) -> None:
    _, report = pad_to_length(_memories(2), _budget(), _filler(tmp_path))
    assert set(FILLER_KEYS) <= set(report)
    assert report["n_retrieved"] == 2
    assert report["token_tolerance"] == TOKEN_TARGET_TOLERANCE


def test_a_full_condition_is_not_padded_further(tmp_path: Path) -> None:
    padded, report = pad_to_length(_memories(20), _budget(), _filler(tmp_path))
    assert report["n_padded"] == 0, "20 x 200 = 4 000, already the target"
    assert len(padded) == 20
    assert sum(m.n_tokens for m in padded) == _budget().target_tokens


def test_count_varies_while_length_does_not(tmp_path: Path) -> None:
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


def test_zero_gold_sessions_reach_the_padding(tmp_path: Path) -> None:
    padded, _ = pad_to_length(
        _memories(2), _budget(), _filler(tmp_path), gold_sessions=["goldA", "goldB"]
    )
    pads = {m.session_id for m in padded if m.source_question_id == "__filler__"}
    assert pads & {"goldA", "goldB"} == set()


def test_a_gold_session_in_the_retrieved_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GoldLeakageError):
        pad_to_length(_memories(4), _budget(), _filler(tmp_path), gold_sessions=["s1"])


def test_the_leakage_assertion_names_what_it_checked() -> None:
    with pytest.raises(GoldLeakageError) as exc:
        assert_no_gold_leakage(_memories(3), ["s0"], what="retrieved")
    assert "retrieved" in str(exc.value)


def test_the_leak_check_is_on_session_id_so_a_chunked_gold_cannot_slip_through() -> None:
    chunks = [Memory(f"sGOLD#{i}", "sGOLD", "q1", "body", 100) for i in range(3)]
    assert "sGOLD" not in {m.memory_id for m in chunks}, (
        "the premise: memory_id alone would NOT match the session name"
    )
    with pytest.raises(GoldLeakageError):
        assert_no_gold_leakage(chunks, ["sGOLD"], what="padding")


def test_an_empty_gold_set_is_not_an_error() -> None:
    assert_no_gold_leakage(_memories(3), [], what="padding")


def test_a_missing_filler_file_is_a_named_error(tmp_path: Path) -> None:
    assert FILLER_FILENAME == "5_filler_sess.json"
    with pytest.raises(FillerMissingError, match=FILLER_FILENAME):
        FillerSource.load(tmp_path / FILLER_FILENAME)


def test_another_filler_filename_is_refused(tmp_path: Path) -> None:
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
    p = tmp_path / FILLER_FILENAME
    p.write_text(json.dumps({"sessions": ["short"]}), encoding="utf-8")
    with pytest.raises(TokenTargetMissedError, match=r"4%"):
        pad_to_length(_memories(2), _budget(), FillerSource.load(p))
