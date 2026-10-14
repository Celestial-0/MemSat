"""The committed ceiling artifacts (PRD §R2).

The exit gate for P1 is: *`msat ceiling` reproduces `ceiling.json` **and**
`corpus_fraction.json` offline; `data.lock` matches.* Everything here either
verifies that reproduction or pins a property of the arithmetic that a future
refactor could silently break.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.ceiling import (
    CEILING_GRID,
    build_ceiling_artifact,
    build_ceiling_multiscale,
    build_corpus_fraction_artifact,
    by_type_table,
    ceiling_curve,
    corpus_fraction,
    degeneracy_fraction,
    dump_canonical,
    partial_credit_curve,
    saturation_k,
    survivors_at,
    verify_artifact,
)
from msat.datasets.schema import CorpusStats, Question
from tests.conftest import requires_m, requires_s


def _q(qid: str, n_sessions: int, n_gold: int, ability: str = "multi-session") -> Question:
    return Question(
        qid=qid,
        scale="S",
        question="q",
        question_date="",
        gold_answers=("a",),
        gold_session_ids=tuple(f"g{i}" for i in range(n_gold)),
        ability=ability,
        answerable=not qid.endswith("_abs"),
        corpus=CorpusStats(n_sessions=n_sessions, n_gold_sessions=n_gold),
    )


# ── R2.6 · partial credit must honour its grid ────────────────────────────────
def test_partial_credit_honours_its_grid() -> None:
    """Every requested K gets an entry — not only the last one.

    PRD §R2.6 exists because an earlier revision of the legacy script returned
    only the final K. The current `context/scripts/ceiling_analysis.py` is
    already correct on this point, so this test **pins the behaviour** instead of
    fixing a bug that is not there.
    """
    gold = [1, 2, 3, 4]
    grid = (1, 2, 4, 8)
    partial = partial_credit_curve(gold, grid)
    assert set(partial) == {"1", "2", "4", "8"}


def test_partial_credit_values_are_correct_by_hand() -> None:
    """min(g, K)/g averaged over questions."""
    gold = [1, 2, 3]
    assert partial_credit_curve(gold, (1,))["1"] == pytest.approx((1 + 0.5 + 1 / 3) / 3)
    assert partial_credit_curve(gold, (3,))["3"] == pytest.approx(1.0)


def test_ceiling_curve_is_monotone_non_decreasing() -> None:
    curve = ceiling_curve([1, 2, 3, 6], CEILING_GRID)
    values = [curve[str(k)] for k in sorted(CEILING_GRID)]
    assert values == sorted(values)


def test_saturation_k_is_the_min_crossing() -> None:
    gold = [1, 1, 2, 6]
    curve = ceiling_curve(gold, CEILING_GRID)
    assert saturation_k(curve) == 6


def test_saturation_k_returns_none_when_grid_is_too_short() -> None:
    assert saturation_k(ceiling_curve([9], (1, 2))) is None


# ── R2.5 · survivors_at is the fraction with corpus >= K, NOT "degenerate" ────
def test_survivors_at_and_degeneracy_overlap_only_at_equality() -> None:
    """The two keys use `>=` on opposite sides, so they overlap where corpus == K.

    PRD §R2.5 defines `survivors_at[K]` as the fraction with corpus **≥ K**;
    PRD §R3.1 defines degeneracy as **K ≥ corpus size**. Both are non-strict, so a
    question with exactly K sessions counts in *both* — correctly, since K == n
    sessions does retrieve the whole corpus and is therefore genuinely degenerate.

    The two only sum to 1 when no question has corpus exactly K; the exact
    relation is `survivors + degeneracy - frac_equal == 1`.
    """
    sizes = [10, 20, 30, 40]
    for k in (1, 10, 25, 40):
        surv = survivors_at(sizes, (k,))[str(k)]
        degen = degeneracy_fraction(sizes, k)
        frac_equal = sum(1 for s in sizes if s == k) / len(sizes)
        assert surv + degen - frac_equal == pytest.approx(1.0)


def test_degeneracy_is_per_question_not_corpus_wide() -> None:
    """PRD §R3.1: K=32 is 80% of a 40-session conversation, yet a mean-based guard passes.

    A corpus of [10, 90] has mean 50, so a mean-based guard would call K=32 fine.
    Per-question, one of the two questions is degenerate — a 50% degeneracy that
    a mean cannot see.
    """
    sizes = [10, 90]
    assert degeneracy_fraction(sizes, 32) == pytest.approx(0.5)
    assert sum(sizes) / len(sizes) == 50.0  # the mean a percentile guard would use


def test_k_equal_to_corpus_is_degenerate() -> None:
    """Exactly-equal is degenerate: K == n_sessions still retrieves everything."""
    assert degeneracy_fraction([40], 40) == pytest.approx(1.0)


# ── R2.2 · type stratification ────────────────────────────────────────────────
def test_by_type_classification() -> None:
    rows = [
        ("single-session-user", 1),
        ("single-session-user", 1),
        ("multi-session", 3),
        ("multi-session", 4),
    ]
    table = by_type_table(rows, CEILING_GRID)
    assert table["single-session-user"]["classification"] == "retriever-independent"
    assert table["multi-session"]["classification"] == "retriever-limited"
    assert table["multi-session"]["probe"]["1"] == pytest.approx(0.0)


def test_by_type_mixed_when_some_but_not_all_fit_in_one_session() -> None:
    table = by_type_table([("temporal-reasoning", 1), ("temporal-reasoning", 3)], CEILING_GRID)
    assert table["temporal-reasoning"]["classification"] == "mixed"


# ── R2.7 · corpus fraction (gap G12) ──────────────────────────────────────────
def test_corpus_fraction_detects_full_context() -> None:
    """K == corpus size is full-context, ratio exactly 1.0."""
    result = corpus_fraction([8, 16], (8, 16))
    assert result["8"]["frac_full_context"] == pytest.approx(0.5)  # the 8-session question
    assert result["16"]["frac_full_context"] == pytest.approx(0.5)  # the 16-session one


def test_corpus_fraction_reports_k_greater_than_corpus() -> None:
    """K above the corpus is a guard violation. Reported, never clamped."""
    result = corpus_fraction([10], (64,))
    assert result["64"]["frac_k_gt_corpus"] == pytest.approx(1.0)
    assert result["64"]["frac_full_context"] == pytest.approx(0.0)
    assert result["64"]["quantiles"]["max"] > 1.0


def test_same_k_is_a_different_system_at_different_scales() -> None:
    """The artifact's whole reason to exist.

    K=64 is full-context for no S conversation (median corpus 48 sessions) and
    for every M conversation (minimum corpus 460).
    """
    s_sizes = [40, 48, 50]
    m_sizes = [460, 475, 490]
    assert corpus_fraction(s_sizes, (64,))["64"]["frac_k_gt_corpus"] == pytest.approx(1.0)
    assert corpus_fraction(m_sizes, (64,))["64"]["frac_full_context"] == pytest.approx(0.0)


# ── determinism (R2.4) ────────────────────────────────────────────────────────
def test_dump_canonical_is_byte_stable() -> None:
    artifact = build_ceiling_artifact([_q("a", 50, 1), _q("b", 50, 3)])
    first = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    second = (
        json.dumps(
            build_ceiling_artifact([_q("a", 50, 1), _q("b", 50, 3)]),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    assert first == second


def test_verify_detects_drift_and_points_at_the_byte(tmp_path: Path) -> None:
    artifact = build_ceiling_artifact([_q("a", 50, 1)])
    path = tmp_path / "ceiling.json"
    dump_canonical(artifact, path)

    ok, detail = verify_artifact(artifact, path)
    assert ok, detail

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["ceiling"]["1"] = 0.123456
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok, detail = verify_artifact(artifact, path)
    assert not ok
    assert "DRIFT" in detail
    assert "first difference at byte" in detail


def test_verify_reports_a_missing_artifact(tmp_path: Path) -> None:
    ok, detail = verify_artifact({"a": 1}, tmp_path / "nope.json")
    assert not ok
    assert "missing" in detail


def test_artifact_carries_key_semantics() -> None:
    """A reader must not have to guess what `survivors_at` means (PRD §R2.5)."""
    artifact = build_ceiling_artifact([_q("a", 50, 1)])
    semantics = artifact["_key_semantics"]
    assert "survivors_at" in semantics
    assert "NOT" in semantics["survivors_at"]
    assert artifact["_provenance"]["api_calls"] == 0


def test_multiscale_reports_scale_invariance() -> None:
    artifact = build_ceiling_multiscale(
        {"S": [_q("a", 50, 1), _q("b", 50, 2)], "M": [_q("a", 475, 1), _q("b", 475, 2)]}
    )
    assert artifact["scale_invariance"]["identical_across_scales"] is True
    assert artifact["scales"]["S"]["analytic_saturation_k"] == 2


# ── committed artifacts ───────────────────────────────────────────────────────
def _committed(project_root: Path, name: str) -> Path:
    return project_root / "results" / name


@requires_s
@requires_m
def test_committed_ceiling_reproduces_byte_for_byte(
    project_root: Path, path_s: Path, path_m: Path
) -> None:
    """PRD §R2.4 — the committed artifact must regenerate identically.

    This is the P1 exit gate. If the raw data changes, or the arithmetic
    changes, this fails and the paper's numbers are no longer backed by the
    committed file.
    """
    from msat.datasets import iter_metadata, raw_path

    scales = {s: list(iter_metadata(raw_path(s))) for s in ("S", "M")}
    artifact = build_ceiling_multiscale(scales)
    path = _committed(project_root, "ceiling.json")
    if not path.is_file():
        pytest.fail(f"{path} is not committed — run `make ceiling`")
    ok, detail = verify_artifact(artifact, path)
    assert ok, detail


@requires_s
@requires_m
def test_committed_corpus_fraction_reproduces_byte_for_byte(project_root: Path) -> None:
    """PRD §R2.7 — same guarantee for the G12 artifact."""
    from msat.datasets import iter_metadata, raw_path

    scales = {s: list(iter_metadata(raw_path(s))) for s in ("S", "M")}
    artifact = build_corpus_fraction_artifact(scales)
    path = _committed(project_root, "corpus_fraction.json")
    if not path.is_file():
        pytest.fail(f"{path} is not committed — run `make ceiling`")
    ok, detail = verify_artifact(artifact, path)
    assert ok, detail


def test_ceiling_grid_is_a_superset_of_any_design_grid() -> None:
    """PRD §R0.1 — the ceiling's reporting grid must not become a second design grid.

    Once `configs/design.yaml` exists, every experimental cell needs a ceiling to
    be plotted against, so the design grid must be covered by `CEILING_GRID`.
    """
    design = Path(__file__).resolve().parent.parent / "configs" / "design.yaml"
    if not design.is_file():
        pytest.skip("configs/design.yaml lands in Phase 2")
    import yaml

    k_grid = tuple(yaml.safe_load(design.read_text(encoding="utf-8"))["K_grid"])
    missing = [k for k in k_grid if k not in CEILING_GRID]
    assert not missing, f"design K_grid points absent from CEILING_GRID: {missing}"
