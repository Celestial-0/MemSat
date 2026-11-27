from __future__ import annotations

import os
from itertools import islice
from pathlib import Path

import psutil
import pytest

from msat.datasets.longmemeval import (
    RAM_CEILING_BYTES,
    count_records,
    iter_metadata,
    iter_questions,
)
from tests.conftest import requires_m, requires_s

_MB = 1024 * 1024


def _peak_rss() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


@requires_s
def test_s_split_streams_under_the_ram_ceiling(path_s: Path) -> None:
    before = _peak_rss()
    n = 0
    for _ in iter_metadata(path_s):
        n += 1
    peak = _peak_rss()
    assert n == 500
    assert peak < RAM_CEILING_BYTES, (
        f"streaming the S split peaked at {peak / _MB:.0f} MB, "
        f"over the {RAM_CEILING_BYTES / _MB:.0f} MB ceiling (PRD §N2)"
    )
    assert peak - before < 200 * _MB


@requires_s
def test_metadata_memory_does_not_grow_with_records(path_s: Path) -> None:
    gen = iter_metadata(path_s)
    samples: list[int] = []
    for i, _ in enumerate(gen):
        if i + 1 in (5, 25, 50, 100, 250, 500):
            samples.append(_peak_rss())
        if i + 1 == 500:
            break
    early = max(samples[:2])
    late = samples[-1]
    assert late - early < 100 * _MB, (
        f"RSS grew from {early / _MB:.0f} MB to {late / _MB:.0f} MB across the file; "
        "the metadata pass is not flat"
    )


@requires_m
@pytest.mark.slow
def test_m_split_streams_under_the_ram_ceiling(path_m: Path) -> None:
    assert path_m.stat().st_size > 2 * 1024**3, "M split is expected to exceed 2 GB"
    n = sum(1 for _ in iter_metadata(path_m))
    peak = _peak_rss()
    assert n == 500
    assert peak < RAM_CEILING_BYTES, (
        f"streaming the 2.7 GB M split peaked at {peak / _MB:.0f} MB, over the "
        f"{RAM_CEILING_BYTES / _MB:.0f} MB ceiling"
    )


@requires_s
def test_ram_ceiling_is_a_real_number() -> None:
    assert RAM_CEILING_BYTES > 0
    assert RAM_CEILING_BYTES < 8 * 1024**3


@requires_s
def test_count_records_matches_the_loader(path_s: Path) -> None:
    assert count_records(path_s) == sum(1 for _ in iter_metadata(path_s))


@requires_s
def test_both_passers_agree_on_the_corpus(path_s: Path) -> None:
    meta = {q.qid: q.corpus.n_sessions for q in iter_metadata(path_s)}
    full = {q.qid: q.corpus.n_sessions for q in iter_questions(path_s)}
    assert meta == full
    assert len(meta) == 500


@requires_s
def test_metadata_pass_discards_session_text(path_s: Path) -> None:
    for q in islice(iter_metadata(path_s), 20):
        assert q.sessions == ()
        assert q.corpus.n_sessions > 0


@requires_s
def test_full_pass_carries_session_text(path_s: Path) -> None:
    for q in islice(iter_questions(path_s), 5):
        assert q.sessions, "full pass must materialise sessions"
        assert all(s.turns for s in q.sessions)
        assert q.corpus.n_turns == sum(s.n_turns for s in q.sessions)
        assert q.corpus.n_chars == sum(s.n_chars for s in q.sessions)
