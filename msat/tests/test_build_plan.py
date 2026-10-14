"""`build-plan` — the manifest, the guards, and the zero-spend guarantee.

Architecture §P3's claim is that degenerate cells are rejected BEFORE anything is
spent. The `no_network` fixture is what makes that falsifiable: it blocks the
socket module outright, so any code path that reached for the network would raise
rather than quietly succeed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat.cli import app
from msat.config import load_bundle
from msat.guards import QuestionStats, cell_verdict, degeneracy_fraction
from msat.manifest import ConditionKey, build_manifest, stats_from_corpus, write_manifest
from msat.scripts_fixture import load_synthetic

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def bundle():
    return load_bundle(CONFIGS)


@pytest.fixture
def corpus(cpt: float = 4.0) -> tuple[dict[str, list], dict[str, list]]:
    from msat.selftest import synthetic_corpus

    return synthetic_corpus(cpt)


# ── the manifest ──────────────────────────────────────────────────────────────
def test_every_declared_cell_is_enumerated(bundle, corpus) -> None:
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    expected = (
        len(bundle.design.K_grid)
        * len(bundle.design.B_grid)
        * len(bundle.design.scales)
        * len(bundle.design.readers)
    )
    assert len(m.cells) == expected
    assert m.config_hash == bundle.config_hash


def test_condition_keys_never_collide(bundle, corpus) -> None:
    """PRD §R5.6. `b_mode` is part of the key: the same number under two modes is
    two different conditions, and collapsing them would merge two cells into one."""
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    keys = [c.condition.key for c in m.cells]
    assert len(keys) == len(set(keys))
    assert m.collisions == []


def test_the_manifest_round_trips_through_json(bundle, corpus, tmp_path: Path) -> None:
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    path = write_manifest(m, tmp_path / "manifest.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["config_hash"] == m.config_hash
    assert len(reloaded["cells"]) == len(m.cells)
    assert reloaded["summary"]["n_accepted"] == len(m.accepted)


# ── rejection, not warning ────────────────────────────────────────────────────
def test_a_fully_degenerate_cell_is_rejected(bundle, corpus) -> None:
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)

    rejected = m.fully_degenerate
    assert rejected, "the fixture's K=64 exceeds its corpora and must be rejected"
    for cell in rejected:
        assert cell.status == "rejected"
        assert cell.degeneracy_fraction >= 1.0
        assert cell.reason.startswith("degeneracy")
        assert cell.example_reason, "a rejection must say what went wrong, not just how often"


def test_status_is_never_a_warning(bundle, corpus) -> None:
    """There is no third state. A cell either runs or it does not."""
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    assert {c.status for c in m.cells} <= {"accepted", "rejected"}


def test_over_deep_cells_are_rejected_on_the_small_fixture_questions(bundle, corpus) -> None:
    """The K=64 defect, caught at design time.

    The fixture's small and mid questions hold 3 and 12 sessions. K=64 exceeds
    both, so the condition is full-context whatever K says. On the REAL corpus this
    also holds on S (mean 47.7 sessions, `frac_k_gt_corpus = 1.0` at K=64) — P1
    measured it and `design.yaml` still declared K=64 as a headline level.
    """
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    small = [c for c in m.rejected if c.condition.scale == "S"]
    assert small, "S-scale K above 12 should be rejected"
    assert {c.condition.k for c in small} == {16, 32}, (
        "S holds 3- and 12-session questions, so every K above 12 is unreachable. "
        "K=64 is no longer in the grid at all (design.yaml trimmed it), so 16 is "
        "now the deepest K this rejects."
    )


def test_a_partly_degenerate_cell_is_kept_with_its_fraction(bundle, corpus) -> None:
    """Rejecting a cell that is degenerate for half its questions throws away real
    signal; keeping it silently overstates precision. It is accepted and carries
    the fraction so P4.3 can exclude those questions from paired comparisons."""
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    partial = [c for c in m.accepted if c.n_degenerate]
    for cell in partial:
        assert 0.0 < cell.degeneracy_fraction < 1.0
        assert "degenerate" in cell.reason


def test_a_scale_with_no_corpus_stats_rejects_rather_than_assumes_valid(bundle) -> None:
    """A condition that cannot be shown to exist is not a condition that can be
    counted as valid."""
    m = build_manifest(bundle, {"S": [], "M": []}, chars_per_token=4.0)
    assert len(m.accepted) == 0
    assert len(m.rejected) == len(m.cells)
    assert "NO CORPUS STATS" in " ".join(m.notes)


# ── the zero-spend guarantee ──────────────────────────────────────────────────
def test_build_plan_spends_nothing(no_network, cli_runner, tmp_path: Path) -> None:
    """Architecture §P3: rejection happens before any API call.

    `no_network` replaces the socket module, so a network call raises rather than
    quietly succeeding. This is the assertion that makes "before any API call"
    testable rather than aspirational.
    """
    result = cli_runner.invoke(
        app, ["build-plan", "--corpus", "synthetic", "--out", str(tmp_path / "m.json")]
    )
    # Exit 3 = EXIT_DEGENERATE_CELL, which is the CORRECT outcome: the fixture's
    # over-deep cells are degenerate for every question on at least one scale.
    assert result.exit_code in (0, 3), result.output
    assert (tmp_path / "m.json").exists(), "a manifest must be written even when cells are rejected"


def test_build_plan_rejects_rather_than_warning_on_the_fixture(cli_runner, tmp_path: Path) -> None:
    """Exit 3, not 0. A warning is a thing you ship straight past."""
    result = cli_runner.invoke(
        app, ["build-plan", "--corpus", "synthetic", "--out", str(tmp_path / "m.json")]
    )
    assert result.exit_code == 3
    assert "REJECTED" in result.output


# ── guards feed the manifest correctly ────────────────────────────────────────
def test_degeneracy_fraction_matches_the_guards_called_directly() -> None:
    """The manifest must not recompute the guards differently."""
    corpus = [
        QuestionStats("big", {"session": 40}, 4000),
        QuestionStats("small", {"session": 3}, 300),
    ]
    from_manifest = degeneracy_fraction(16, 128, corpus, "session", mean_unit_tokens=100)
    by_hand = (
        sum(cell_verdict(16, 128, s, "session", mean_unit_tokens=100).degenerate for s in corpus)
        / 2
    )
    assert from_manifest == by_hand == pytest.approx(0.5)


def test_the_characters_per_token_assumption_is_recorded(bundle, corpus) -> None:
    """Every B-guard verdict depends on an assumed ratio, so the manifest says so
    rather than letting a reader take its numbers for measured."""
    corpus_map, raw = corpus
    m = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    assert any("ASSUMED" in n for n in m.notes)


def test_stats_from_corpus_carries_the_unit(monkeypatch: Any) -> None:
    """R1.6: the unit K counts must be available for every unit, because a guard
    evaluated in the wrong unit rejects the wrong cells."""
    stats = stats_from_corpus("q", 40, 160000, chars_per_token=4.0)
    assert stats.n_units("session") == 40
    assert stats.n_units("chunk") == 120
    assert stats.n_units("fact") == 320
    assert stats.corpus_tokens == 40000


def test_the_fixture_covers_the_sizes_that_make_the_guards_fire() -> None:
    sizes = sorted(len(q["sessions"]) for q in load_synthetic())
    assert sizes == [3, 12, 40]
    assert min(sizes) < 16, "K=16 must exceed the smallest fixture corpus for a rejection"


def test_condition_key_includes_b_mode() -> None:
    a = ConditionKey("A+G", 4, 128.0, "per_memory_tokens", "S", "gemma")
    b = ConditionKey("A+G", 4, 128.0, "absolute_tokens", "S", "gemma")
    assert a.key != b.key, "the same number under two modes is two conditions"


def test_manifest_is_deterministic(bundle, corpus) -> None:
    """Same corpus, same manifest — replay (R5.2) depends on it."""
    corpus_map, raw = corpus
    a = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    b = build_manifest(bundle, corpus_map, chars_per_token=4.0, raw_counts=raw)
    assert json.dumps(a.as_dict(), sort_keys=True) == json.dumps(b.as_dict(), sort_keys=True)


def test_design_yaml_is_the_only_grid_source() -> None:
    """P2.4, restated where it bites.

    Checks the PARSED structure, not the text. `budget.yaml` explains in a comment
    exactly why the old grep guard failed, so a text search finds the string
    `B_grid` there and reports a false positive — which is what the structural
    check in `config.py` was written to avoid.
    """
    from msat.config import _declares_budget_levels

    raw = yaml.safe_load((CONFIGS / "design.yaml").read_text(encoding="utf-8"))
    assert raw["K_grid"] and raw["B_grid"]

    for name in ("budget.yaml", "budget_guard.yaml", "pricing/latest.yaml", "arms.yaml"):
        parsed = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
        assert not _declares_budget_levels(parsed), f"{name} must not redeclare the grid"
