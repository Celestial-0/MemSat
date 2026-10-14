"""P3 — the K-guard and B-guard matrix (PRD §R3.5), and the memory baseline.

The guards are the point of this phase. Everything else — chunking, embedding,
retrieval — is a baseline that is allowed to be weak, as long as it is
deterministic and audited.

The cross-product below is the R3.5 matrix. Each case is named after the failure
it prevents, because "test_k_guard_returns_false" tells a future reader nothing
about why the case exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_synthetic_fixture import LAYOUT, build

from msat.guards import (
    EMPTY_CORPUS,
    K_EXCEEDS_CORPUS,
    MISSING_QUESTION,
    PROMPT_FULL_CONTEXT,
    QuestionStats,
    b_guard,
    cell_verdict,
    corpus_summary,
    degeneracy_fraction,
    k_guard,
    prompt_tokens,
)
from msat.memory import HashEmbedder, gold_rank, retrieve, write


def stats(n_sessions: int, tokens_per_session: int, qid: str = "q") -> QuestionStats:
    return QuestionStats(
        question_id=qid,
        units={"session": n_sessions, "chunk": n_sessions * 3, "fact": n_sessions * 8},
        corpus_tokens=n_sessions * tokens_per_session,
    )


# ── the K-guard matrix ────────────────────────────────────────────────────────
def test_k_exactly_equal_to_the_corpus_is_not_degenerate() -> None:
    """K == n_units is the boundary cell, and it is the most informative one.

    Rejecting it would throw away the top of the curve - the point where depth has
    consumed the whole conversation - and with it any chance of seeing saturation.
    """
    v = k_guard(40, stats(40, 25), "session")
    assert v.ok, v.reason


def test_k_one_over_the_corpus_is_degenerate() -> None:
    v = k_guard(41, stats(40, 25), "session")
    assert not v.ok
    assert v.code == K_EXCEEDS_CORPUS
    assert v.detail["excess"] == 1


def test_k_well_over_a_small_corpus_is_degenerate() -> None:
    v = k_guard(64, stats(3, 40), "session")
    assert not v.ok
    assert v.code == K_EXCEEDS_CORPUS


def test_a_single_question_corpus_smaller_than_k_is_degenerate() -> None:
    """The K=64 defect in one line: this conversation holds three sessions."""
    v = k_guard(32, stats(3, 40), "session")
    assert not v.ok
    assert "3 sessions" in v.reason


def test_an_empty_corpus_is_degenerate_at_every_k() -> None:
    for k in (1, 2, 64):
        v = k_guard(k, stats(0, 0), "session")
        assert not v.ok
        assert v.code == EMPTY_CORPUS


def test_a_missing_question_is_degenerate_not_assumed_valid() -> None:
    """A condition that cannot be shown to exist is not a condition."""
    v = k_guard(4, None, "session")
    assert not v.ok
    assert v.code == MISSING_QUESTION


def test_a_percentile_guard_would_have_passed_the_case_that_failed() -> None:
    """Why the guard is per-question and never a corpus percentile.

    A percentile guard derives its threshold from the DISTRIBUTION of corpus
    sizes, not the mean. For these two questions the empirical p95 of
    `n_sessions` is its maximum (38-40), so K=32 sails through it — and K=32 is
    degenerate outright on the 3-session conversation. The guard only catches that
    question by accident of which question it looks at, and there is no way to
    know in advance which conversation is the short one.
    """
    k = 32
    corpus = [stats(40, 25, "large"), stats(3, 25, "small")]
    sizes = sorted(s.n_units("session") for s in corpus)
    p95 = sizes[-1]  # empirical p95 of a two-point sample is its maximum
    assert k < p95, "a percentile guard WOULD admit this K"

    assert not k_guard(k, corpus[0], "session").degenerate, "fine for the long one"
    assert k_guard(k, corpus[1], "session").degenerate, "degenerate for the short one"


def test_the_unit_changes_the_answer() -> None:
    """40 sessions is 120 chunks and 320 facts. The same K is a different
    condition at each unit, which is why `unit` is not optional (PRD §R1.6)."""
    s = stats(40, 25)
    assert k_guard(64, s, "session").degenerate, "64 of 40 sessions is unreachable"
    assert k_guard(64, s, "chunk").ok, "64 of 120 chunks is perfectly reachable"
    assert k_guard(200, s, "chunk").degenerate, "200 of 120 chunks is unreachable"
    assert k_guard(400, s, "fact").degenerate, "400 of 320 facts is unreachable"
    assert k_guard(320, s, "fact").ok, "320 of 320 facts is the boundary, and allowed"


# ── the B-guard matrix ────────────────────────────────────────────────────────
def test_b_beyond_the_corpus_is_full_context() -> None:
    """Every unit retrieved at full length IS the whole corpus, whatever B says
    above the mean memory length."""
    s = stats(10, 100)  # 1,000 corpus tokens, 100 per session
    v = b_guard(4096, s, k=10, mean_unit_tokens=100)
    assert not v.ok
    assert v.code == PROMPT_FULL_CONTEXT
    assert v.detail["prompt_tokens"] == pytest.approx(1000.0)


def test_b_guard_checks_the_prompt_not_the_budget() -> None:
    """The bug that shipped once, silently.

    A large B behind a small K does NOT fill the context. Evaluating the guard on
    `B` alone marks this degenerate and throws away a valid cell — and in
    `design.py` the prompt size itself was computed from K alone, so feasibility
    and cost were both wrong in the same direction.

    40 sessions x 100 tokens = a 4,000-token corpus. At B=1024 the per-memory cap
    never binds (1024 > the 100-token mean), so the prompt is just K x 100:
    K=2 renders 200 tokens and is fine; K=40 renders the whole corpus and is not.
    The budget never moved. Only K did. That is the demonstration.
    """
    s = stats(40, 100)
    assert b_guard(1024, s, k=2, mean_unit_tokens=100).ok, "200 of 4,000 tokens is not full"
    assert b_guard(1024, s, k=8, mean_unit_tokens=100).ok, "800 of 4,000 tokens is not full"
    assert b_guard(1024, s, k=40, mean_unit_tokens=100).degenerate, "the whole corpus is full"

    # And the budget DOES bite when it is below the mean memory length — which is
    # why the design's B values all sit under ~2,550 tokens.
    assert b_guard(40, s, k=40, mean_unit_tokens=100).ok, "40 x 40 = 1,600 of 4,000"


def test_b_guard_requires_the_mean_and_will_not_guess() -> None:
    """Deriving the mean from the corpus would make the guard disagree with the
    renderer, which is the one thing a guard must never do."""
    with pytest.raises(ValueError, match="mean_unit_tokens"):
        b_guard(1024, stats(40, 100), k=2)


def test_prompt_tokens_caps_at_what_the_corpus_actually_has() -> None:
    """You cannot retrieve 64 sessions from a 3-session corpus."""
    s = stats(3, 100)
    assert prompt_tokens(64, 50, s, "session", mean_unit_tokens=100) == pytest.approx(150.0)


# ── crossed cases ─────────────────────────────────────────────────────────────
def test_k_fine_b_degenerate_reports_the_budget() -> None:
    """K is reachable (2 of 40 sessions) but the budget fills the context anyway."""
    s = stats(2, 1000)  # 2 sessions, 2,000 corpus tokens, 1,000 each
    v = cell_verdict(2, 4096, s, "session", mean_unit_tokens=1000)
    assert not v.ok
    assert v.code == PROMPT_FULL_CONTEXT, "the BUDGET is the problem, say so"


def test_b_fine_k_degenerate_reports_the_depth() -> None:
    """B is tiny and harmless (40 tokens over a 3,000-token corpus) but K=64
    cannot be reached from a 3-session question."""
    v = cell_verdict(64, 40, stats(3, 1000), "session", mean_unit_tokens=1000)
    assert not v.ok
    assert v.code == K_EXCEEDS_CORPUS, "the DEPTH is the problem, say so"


def test_the_crossed_cases_stay_distinct() -> None:
    """Collapsing both into "degenerate" loses the diagnosis, and with it the fix."""
    a = cell_verdict(2, 4096, stats(2, 1000), "session", mean_unit_tokens=1000)
    b = cell_verdict(64, 40, stats(3, 1000), "session", mean_unit_tokens=1000)
    assert a.degenerate and b.degenerate
    assert a.code != b.code


def test_both_clean_is_accepted() -> None:
    assert cell_verdict(4, 64, stats(40, 100), "session", mean_unit_tokens=100).ok


# ── degeneracy_fraction (R3.3) ────────────────────────────────────────────────
def test_degeneracy_fraction_is_per_cell_not_project_wide() -> None:
    corpus = [stats(3, 100, "small"), stats(40, 100, "large")]
    # K=16 is fine for one question and degenerate for the other.
    frac = degeneracy_fraction(16, 64, corpus, "session", mean_unit_tokens=100)
    assert frac == pytest.approx(0.5)
    # K=2 is fine for both.
    assert degeneracy_fraction(2, 64, corpus, "session", mean_unit_tokens=100) == 0.0


def test_an_empty_corpus_is_wholly_degenerate() -> None:
    assert degeneracy_fraction(1, 1, [], "session", mean_unit_tokens=1) == 1.0


def test_corpus_summary_reports_the_binding_k() -> None:
    """A grid sized against the MEAN conversation is degenerate on the shortest
    one. `max_safe_k` is the shortest conversation's limit, not the mean's."""
    summary = corpus_summary([stats(3, 100, "a"), stats(40, 100, "b")], "session")
    assert summary["max_safe_k"] == 3
    assert summary["units_per_question"] == {"min": 3, "median": 40, "max": 40}


# ── the baseline itself: deterministic and audited ───────────────────────────
def test_the_write_path_is_deterministic() -> None:
    """A weak retriever is acceptable. A CHANGING one is not."""
    data = build()
    q = data["questions"][1]
    idx_a, rec_a = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    idx_b, rec_b = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    assert rec_a.write_hash == rec_b.write_hash, "the same corpus produced different provenance"
    assert idx_a.vectors == idx_b.vectors


def test_nothing_is_dropped_silently() -> None:
    """Every source session appears in exactly one record, kept or not."""
    data = build()
    q = data["questions"][0]
    _, rec = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    assert rec.n_units == len(q["sessions"])
    assert rec.n_written + rec.n_dropped == rec.n_units
    assert {u.unit_id for u in rec.units} == {f"session:{s['session_id']}" for s in q["sessions"]}


def test_a_blank_session_is_dropped_with_a_reason() -> None:
    _, rec = write("q", [("s0", "real text here"), ("s1", "   ")])
    assert rec.n_written == 1
    assert rec.dropped_by_reason == {"empty_source": 1}
    assert rec.encoding_ceiling == pytest.approx(0.5)


def test_a_duplicate_session_is_dropped_and_says_so() -> None:
    """Two identical sessions are one memory. Counting them twice would make the
    encoding ceiling look better than it is."""
    _, rec = write("q", [("s0", "same text"), ("s1", "same text")])
    assert rec.dropped_by_reason == {"duplicate": 1}


def test_the_encoding_ceiling_is_reported_and_is_one_on_a_clean_corpus() -> None:
    data = build()
    for q in data["questions"]:
        _, rec = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
        assert rec.encoding_ceiling == pytest.approx(1.0)


def test_retrieval_finds_the_gold_session_and_reports_both_counts() -> None:
    """R1.6: items AND tokens, both labelled with the unit."""
    data = build()
    q = data["questions"][2]
    sessions = [(s["session_id"], s["text"]) for s in q["sessions"]]
    idx, _ = write(q["question_id"], sessions)
    emb = HashEmbedder()
    qv = emb.encode([q["question"]])[0]

    got = retrieve(idx, q["question"], qv, k=8, unit="session", budget_tokens=10)

    assert got.unit == "session"
    assert got.k_returned == 8
    assert got.n_tokens == sum(min(10, n) for n in got.truncation.values())
    assert got.n_tokens < got.n_tokens_unbudgeted, "the budget did nothing"


def test_a_budget_above_the_unit_size_is_inert() -> None:
    """Truncating a unit to its own length is a no-op, and it is reported as one.

    This is why the design's B values all sit below the measured mean memory
    length (~2,563 tokens): a budget above it changes nothing, so it buys no
    contrast and costs a grid level. `n_tokens_unbudgeted` is reported beside
    `n_tokens` so the no-op is visible rather than inferred.
    """
    data = build()
    q = data["questions"][2]
    idx, _ = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    qv = HashEmbedder().encode([q["question"]])[0]
    got = retrieve(idx, q, qv, k=8, unit="session", budget_tokens=4096)
    assert got.n_tokens == got.n_tokens_unbudgeted
    assert got.as_dict()["budget_utilisation"] == pytest.approx(1.0)
    assert got.n_tokens_unbudgeted > 0
    assert got.scores == tuple(sorted(got.scores, reverse=True)), "results must be rank ordered"


def test_the_budget_reduces_density_without_reducing_k() -> None:
    """The whole reason B is per-memory: the axes are independent."""
    data = build()
    q = data["questions"][2]
    idx, _ = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    qv = HashEmbedder().encode([q["question"]])[0]
    tight = retrieve(idx, q, qv, k=16, unit="session", budget_tokens=16)
    loose = retrieve(idx, q, qv, k=16, unit="session", budget_tokens=1024)
    assert tight.k_returned == loose.k_returned == 16
    assert tight.n_tokens < loose.n_tokens
    assert tight.unit_ids == loose.unit_ids


def test_ties_break_deterministically_by_unit_id() -> None:
    """Identical text is DEDUPED at write time, so ties are built directly here.

    Both behaviours are correct and they are different: duplicate elimination is a
    property of the write path, tie-breaking is a property of the retriever.
    """
    from msat.memory.chunk import Unit
    from msat.memory.write import MemoryIndex

    idx = MemoryIndex()
    for i in range(8):
        # Same vector for every unit, distinct ids - the only way to get a tie.
        idx.add(Unit(f"session:s{i}", "session", f"s{i}", "text", 4), [1.0, 0.0])

    a = retrieve(idx, "query", [1.0, 0.0], k=4, unit="session")
    b = retrieve(idx, "query", [1.0, 0.0], k=4, unit="session")
    assert a.unit_ids == b.unit_ids
    assert a.unit_ids == ("session:s0", "session:s1", "session:s2", "session:s3")


def test_identical_sources_are_deduplicated_not_indexed_twice() -> None:
    """Counting them twice would make the encoding ceiling look better than it is."""
    _, rec = write("q", [(f"s{i}", "identical text") for i in range(8)])
    assert rec.n_written == 1
    assert rec.dropped_by_reason == {"duplicate": 7}
    assert rec.encoding_ceiling == pytest.approx(1 / 8)


def test_gold_rank_is_one_based_and_none_means_not_found() -> None:
    """`None` is a measurement, not a gap: "not in the top K" is the result."""
    data = build()
    q = data["questions"][0]
    idx, _ = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    qv = HashEmbedder().encode([q["question"]])[0]
    found = retrieve(idx, q, qv, k=3, unit="session")
    assert gold_rank(idx, {q["sessions"][0]["session_id"]}, found) == 1
    assert gold_rank(idx, {"nothing_like_it"}, found) is None


def test_retrieval_reports_a_short_corpus_rather_than_inventing_units() -> None:
    """K=64 against a 3-session corpus returns 3. The guard rejects the cell, and
    the retriever does not paper over it."""
    data = build()
    q = data["questions"][0]
    idx, _ = write(q["question_id"], [(s["session_id"], s["text"]) for s in q["sessions"]])
    qv = HashEmbedder().encode([q["question"]])[0]
    got = retrieve(idx, q, qv, k=64, unit="session")
    assert got.k_returned == len(q["sessions"]) == 3


def test_the_fixture_covers_the_sizes_the_guards_exist_for() -> None:
    """A single-size fixture cannot exercise a K-guard."""
    sizes = sorted(n for _, n, _ in LAYOUT)
    assert sizes == [3, 12, 40]
