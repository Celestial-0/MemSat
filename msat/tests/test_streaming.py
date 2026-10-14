"""The streaming loader must survive the 2.6 GB M split (PRD §N2, §R2.1).

The requirement is not "the loader works". It is that the loader works **under a
quantified RAM ceiling** — a test asserting "under a budget" with no number in it
is unfailable, and an unfailable test is worse than no test because it reads as
coverage.

`RAM_CEILING_BYTES` in `datasets.longmemeval` is that number, and these tests
assert against it directly.
"""

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
    """277 MB file, flat memory. Runs in the default suite — it takes ~2 s."""
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
    # The file is 265 MB; a loader that held it would sit near that.
    assert peak - before < 200 * _MB


@requires_s
def test_metadata_memory_does_not_grow_with_records(path_s: Path) -> None:
    """Flat means flat: RSS after 50 records must not exceed RSS after 5.

    A loader that accumulates `haystack_sessions` passes a single ceiling check
    on a small file and then explodes on the big one. This asserts the *shape* of
    the memory curve, not just its peak.
    """
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
    """2.74 GB file under 1.5 GB RSS. Marked `slow`; run it with `make test-all`.

    This is the test that would have caught `json.load` raising `MemoryError`.
    """
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
    """Guard against the ceiling being weakened to make a test pass."""
    assert RAM_CEILING_BYTES > 0
    # A ceiling above the size of the largest raw input would be meaningless.
    assert RAM_CEILING_BYTES < 8 * 1024**3


@requires_s
def test_count_records_matches_the_loader(path_s: Path) -> None:
    assert count_records(path_s) == sum(1 for _ in iter_metadata(path_s))


@requires_s
def test_both_passers_agree_on_the_corpus(path_s: Path) -> None:
    """`iter_questions` builds real records; `iter_metadata` walks events.

    They are different code paths over the same file. If they disagree about
    `n_sessions`, one of them is wrong and every ceiling derived from it is wrong.
    """
    meta = {q.qid: q.corpus.n_sessions for q in iter_metadata(path_s)}
    full = {q.qid: q.corpus.n_sessions for q in iter_questions(path_s)}
    assert meta == full
    assert len(meta) == 500


@requires_s
def test_metadata_pass_discards_session_text(path_s: Path) -> None:
    """Metadata-mode questions carry no turns — that is what makes the pass flat."""
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
