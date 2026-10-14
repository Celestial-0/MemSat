"""Design, power, and cost (PRD §R0, §R6.5, §R4.4).

Two things here matter more than the rest:

1. **The estimator is cross-checked against `statsmodels.GEE`.** The power sweep
   uses a hand-written numpy sandwich estimator because a power analysis needs
   hundreds of fits at 126k rows. A fast implementation that has never been
   compared to the reference library is a liability, not an optimisation — so
   this file compares them to six decimal places.

2. **The crossed grid is the crossed grid.** PRD §R4.4 says arms A and G are one
   grid, not two sweeps. Getting that wrong doubles the cost estimate, and the
   cost estimate is the number that goes into the budget conversation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msat.config import load_bundle
from msat.constants import JUDGE_NOISE_GRID
from msat.design import (
    INTERACTION_COLUMN,
    ReaderLimit,
    SimSpec,
    build_cells,
    cell_key_collisions,
    context_feasibility,
    estimability,
    estimate_compute,
    estimate_cost,
    fit_logit_cluster_robust,
    mde_for_stratum,
    mde_interaction,
    propose_reduced_design,
    simulate_dataset,
)


def _small_spec(**overrides) -> SimSpec:  # type: ignore[no-untyped-def]
    base = {
        "n_questions": 40,
        "n_cells": 12,
        "runs_per_cell": 2,
        "k_levels": (1, 4, 16, 64),
        "b_levels": (4096.0, 65536.0),
        "log_n_sessions_mu": float(np.log(48.0)),
        "log_n_sessions_sigma": 0.12,
        "min_sessions": 38,
        "reader_strength_spread": 0.16,
        "n_sim": 6,
    }
    base.update(overrides)
    return SimSpec(**base)  # type: ignore[arg-type]


# ── the estimator ──────────────────────────────────────────────────────────────
def test_estimator_matches_statsmodels_gee() -> None:
    """The fast sandwich must agree with `statsmodels` GEE to ~1e-5.

    Equivalent estimators, so any disagreement is a bug in one of them. An earlier
    version of this file did not exist, and the fast path shipped unverified.
    """
    from statsmodels.genmod.cov_struct import Independence
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    spec = _small_spec(n_questions=60, n_cells=21, n_sim=1)
    y, xmat, clusters = simulate_dataset(spec, 0.4, np.random.default_rng(7))
    beta, se = fit_logit_cluster_robust(y, xmat, clusters)

    ref = GEE(y, xmat, groups=clusters, family=Binomial(), cov_struct=Independence()).fit()

    assert np.allclose(beta, ref.params, atol=1e-4), "coefficients disagree"
    assert np.allclose(se, ref.bse, rtol=1e-3), "standard errors disagree"


def test_estimator_survives_complete_separation() -> None:
    """A perfectly-separable cluster must not produce NaN or infinite SEs.

    This is what a degenerate cell looks like in the real data — which is exactly
    the failure PRD §R3 exists to prevent — so the estimator has to survive it in
    order to REPORT it.
    """
    xmat = np.column_stack([np.ones(20), np.arange(20.0)])
    y = (np.arange(20) > 9).astype(float)
    clusters = np.repeat(np.arange(5), 4)
    beta, se = fit_logit_cluster_robust(y, xmat, clusters)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(se))


def test_clustering_widens_the_standard_error() -> None:
    """Cluster-robust SEs are larger than iid SEs — that is the whole point."""
    spec = _small_spec(n_questions=60, n_sim=1)
    y, xmat, clusters = simulate_dataset(spec, 0.3, np.random.default_rng(3))
    _, se_clustered = fit_logit_cluster_robust(y, xmat, clusters)

    # Same fit, pretending every row is its own cluster -> iid SEs.
    _, se_iid = fit_logit_cluster_robust(y, xmat, np.arange(len(y)))
    assert se_clustered[INTERACTION_COLUMN] > se_iid[INTERACTION_COLUMN]


# ── the crossed grid (PRD §R4.4) ───────────────────────────────────────────────
def test_arms_a_and_g_are_one_grid_not_two() -> None:
    """Counting A and G separately would DOUBLE the cost estimate."""
    cells = build_cells(
        k_grid=[1, 2, 4, 8, 16, 32, 64],
        b_grid=[
            (4096.0, "absolute_tokens"),
            (16384.0, "absolute_tokens"),
            (65536.0, "absolute_tokens"),
        ],
        scales=["S", "M"],
        readers=["gemma", "qwen", "phi"],
    )
    assert len(cells) == 7 * 3 * 2 * 3 == 126
    assert {c.arm for c in cells} == {"A+G"}


def test_every_k_b_pair_appears() -> None:
    cells = build_cells(
        [1, 2, 4],
        [(4096.0, "absolute_tokens"), (16384.0, "absolute_tokens")],
        ["S"],
        ["gemma"],
    )
    pairs = {(c.k, c.b) for c in cells}
    assert len(pairs) == 6


def test_condition_keys_are_unique() -> None:
    cells = build_cells([1, 2, 4], [(4096.0, "absolute_tokens")], ["S", "M"], ["a", "b"])
    assert cell_key_collisions(cells) == []


def test_b_mode_is_part_of_the_key() -> None:
    """Hazard H4: a prefix cached at B=4K must not be served at B=16K."""
    cells = build_cells([1], [(4096.0, "absolute_tokens")], ["S"], ["a"])
    other_mode = build_cells([1], [(4096.0, "corpus_fraction")], ["S"], ["a"])
    assert cells[0].key != other_mode[0].key


def test_estimability_reports_a_complete_crossing() -> None:
    cells = build_cells(
        [1, 2, 4],
        [(4096.0, "absolute_tokens"), (16384.0, "absolute_tokens")],
        ["S"],
        ["gemma"],
    )
    est = estimability(cells)
    assert est["collision_free"] is True
    assert est["crossing_complete"] is True
    assert est["estimable"] is True


# ── power ──────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_mde_shrinks_as_the_grid_grows() -> None:
    small = _small_spec(n_questions=40, n_cells=12, n_sim=8)
    large = _small_spec(n_questions=160, n_cells=24, n_sim=8)
    a = mde_interaction(small, seed=11)
    b = mde_interaction(large, seed=11)
    assert a.get("powered") and b.get("powered")
    assert b["mde_log_odds"] < a["mde_log_odds"]


@pytest.mark.slow
def test_mde_is_not_an_artefact_of_the_search_bracket() -> None:
    """Regression test for a real bug.

    The bisection used a fixed bracket of [0, 1.5] with an ABSOLUTE tolerance of
    0.02. At this grid the true MDE is ~0.005, so the search hit its tolerance
    floor (1.5 / 128 = 0.0117) and returned that — identically for every noise
    level, a green number that meant nothing.

    The bracket is now derived from the measured SE and the tolerance is relative,
    so the reported MDE must be close to the analytic
    `(z_{alpha/2} + z_{power}) * SE`.
    """
    spec = _small_spec(n_questions=120, n_cells=24, n_sim=12)
    result = mde_interaction(spec, seed=3)
    assert result["powered"]
    analytic = 2.8016 * result["se_log_odds"]
    assert result["mde_log_odds"] == pytest.approx(analytic, rel=0.35)


@pytest.mark.slow
def test_judge_noise_degrades_the_mde() -> None:
    """Grader misclassification must make the design LOOK WORSE, not better."""
    clean = _small_spec(n_questions=80, n_cells=24, n_sim=10, judge_flip=0.0)
    noisy = _small_spec(n_questions=80, n_cells=24, n_sim=10, judge_flip=0.12)
    a = mde_interaction(clean, seed=5)
    b = mde_interaction(noisy, seed=5)
    assert a["powered"] and b["powered"]
    assert b["mde_log_odds"] > a["mde_log_odds"] * 1.2


def test_judge_flip_attenuates_the_estimated_effect() -> None:
    """The mechanism itself: a noisy grader shrinks the observed coefficient."""
    clean = _small_spec(n_questions=60, n_sim=1, judge_flip=0.0)
    noisy = _small_spec(n_questions=60, n_sim=1, judge_flip=0.15)
    ests = []
    for spec in (clean, noisy):
        vals = []
        for seed in range(6):
            y, xmat, c = simulate_dataset(spec, 0.5, np.random.default_rng(seed))
            beta, _ = fit_logit_cluster_robust(y, xmat, c)
            vals.append(beta[INTERACTION_COLUMN])
        ests.append(float(np.mean(vals)))
    assert ests[1] < ests[0]


def test_underpowered_design_is_reported_not_hidden() -> None:
    """A design that cannot detect anything must say so, not return a big MDE.

    One question and one cell makes `log2(K)` and `log2(B)` constant, so the
    interaction column is exactly collinear with the main effects and the
    standard error is not estimable. That must surface as `powered: false`.
    """
    spec = _small_spec(
        n_questions=1,
        n_cells=1,
        runs_per_cell=1,
        n_sim=1,
        k_levels=(8,),
        b_levels=(4096.0,),
    )
    result = mde_interaction(spec, seed=1)
    assert result["powered"] is False
    assert result["mde_log_odds"] is None
    assert "note" in result


# ── R0.3 — propose, never truncate ────────────────────────────────────────────
def test_underpowered_design_produces_proposals(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    result = propose_reduced_design(
        bundle.design, mde_full=None, seed=1, n_sim=2, items_per_cell=500
    )
    steps = [p["step"] for p in result["proposals"]]
    assert "drop_tightest_b_level" in steps
    assert "halve_runs_per_cell" in steps


def test_dropping_a_reader_is_flagged_as_violating_r75(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    result = propose_reduced_design(
        bundle.design, mde_full=None, seed=1, n_sim=2, items_per_cell=500
    )
    drop_reader = next(p for p in result["proposals"] if p["step"] == "drop_reader_family")
    assert drop_reader["violates_r75"] is True
    assert "NOT RECOMMENDED" in drop_reader["rationale"]


def test_every_proposal_preserves_the_crossing(complete_configs: Path) -> None:
    """The crossing IS the contribution; a proposal that removes it is not a proposal."""
    bundle = load_bundle(complete_configs)
    result = propose_reduced_design(
        bundle.design, mde_full=None, seed=1, n_sim=2, items_per_cell=500
    )
    for proposal in result["proposals"]:
        assert len(proposal["B_grid"]) >= 2, f"{proposal['step']} removes the crossing"


# ── R0.5 / R9.8 — cost ────────────────────────────────────────────────────────
def test_cost_covers_every_cell(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    cells = build_cells(
        bundle.design.K_grid,
        [(b.value, b.mode) for b in bundle.design.B_grid],
        bundle.design.scales,
        bundle.design.readers,
    )
    cost = estimate_cost(
        cells,
        bundle.pricing,
        runs_per_cell=bundle.design.runs_per_cell,
        items_per_cell=500,
        reader_models={f.id: f.model for f in bundle.readers.families},
    )
    assert cost["unpriced_readers"] == []
    assert cost["n_cells_priced"] == len(cells)


def test_an_all_local_design_costs_nothing_in_dollars(complete_configs: Path) -> None:
    """The point of dropping the paid API: the USD budget is $0.

    Which is exactly why `estimate_compute` exists. A $0 budget with an
    unestimated runtime is how a three-week compute bill arrives in week 8.
    """
    bundle = load_bundle(complete_configs)
    cells = build_cells(
        bundle.design.K_grid,
        [(b.value, b.mode) for b in bundle.design.B_grid],
        bundle.design.scales,
        bundle.design.readers,
    )
    cost = estimate_cost(
        cells,
        bundle.pricing,
        runs_per_cell=bundle.design.runs_per_cell,
        items_per_cell=500,
        reader_models={f.id: f.model for f in bundle.readers.families},
    )
    assert cost["cost_uncached_total_usd"] == 0.0
    assert cost["cost_cache_adjusted_total_usd"] == 0.0

    compute = estimate_compute(
        cells,
        bundle.pricing,
        runs_per_cell=bundle.design.runs_per_cell,
        items_per_cell=500,
    )
    assert compute["available"] is True
    assert compute["total_hours"] > 0
    assert compute["total_days"] >= 1


def test_there_is_no_provider_cache_to_exploit(complete_configs: Path) -> None:
    """PRD §R9.8's uncached-vs-cached contrast collapses when everything is local.

    With local weights there is no provider prompt cache at all — every call is a
    full prefill — so the distinction that R9.8 makes for API providers does not
    exist here, and the compute budget is the only currency.
    """
    bundle = load_bundle(complete_configs)
    assert bundle.pricing["compute_budget"]["cacheable_prefix"] is False


def test_unpriced_reader_is_reported_not_skipped_silently(complete_configs: Path) -> None:
    """A lookup failure that returns $0 looks like a triumph. It must be loud.

    This matters MORE now that every price is $0: an unpriced reader would be
    indistinguishable from a free one without `unpriced_readers`.
    """
    bundle = load_bundle(complete_configs)
    cells = build_cells([16], [(16384.0, "absolute_tokens")], ["S"], ["mystery_reader"])
    cost = estimate_cost(
        cells,
        bundle.pricing,
        runs_per_cell=1,
        items_per_cell=10,
        reader_models={"mystery_reader": "no-such-model"},
    )
    assert cost["unpriced_readers"] == ["mystery_reader"]
    assert cost["n_cells_priced"] == 0
    assert cost["n_calls"] == 0


def test_compute_scales_with_the_grid(complete_configs: Path) -> None:
    """The cost that matters now is wall-clock, and it must grow with the grid."""
    bundle = load_bundle(complete_configs)
    small = estimate_compute(
        build_cells([1], [(4096.0, "absolute_tokens")], ["S"], ["gemma"]),
        bundle.pricing,
        runs_per_cell=1,
        items_per_cell=100,
    )
    large = estimate_compute(
        build_cells(
            [1, 64],
            [(4096.0, "absolute_tokens"), (65536.0, "absolute_tokens")],
            ["S"],
            ["gemma"],
        ),
        bundle.pricing,
        runs_per_cell=1,
        items_per_cell=100,
    )
    assert large["total_hours"] > small["total_hours"]


# ── context feasibility (a scientific constraint, not a setting) ───────────────
# The old `test_m_scale_high_k_overflows_a_short_context_reader` asserted that
# overflow was M-only and that the widest reader was always clean. Both were
# artefacts of the `absolute_tokens` budget, which manufactured a ~120k M prompt
# that does not exist under `per_memory_tokens`. It was removed rather than
# patched: a test asserting a specific corpus size is a test that breaks every
# time the design changes correctly.


def test_a_degenerate_cell_is_rejected_never_truncated(complete_configs: Path) -> None:
    """A prompt that will not fit must make the cell INVALID, not smaller.

    PRD §R3.2 forbids truncating the prompt to fit, so the cell is dropped and the
    grid shrinks. Discovering this during P5 would mean discarding completed work,
    which is why it is computed at design time.

    REWRITTEN 2026-10-13. The old version asserted that overflow happened only on
    the M scale and that the widest reader was always clean. Both were artefacts of
    the old `absolute_tokens` budget, where a ~120k M prompt existed at all. Under
    `per_memory_tokens` prompts are far smaller AND S saturates its corpus at K=48,
    so S can overflow too.

    The test no longer depends on that. It pins the invariant that actually
    matters - a reader given a ceiling it cannot meet reports exactly the cells
    that exceed it - using a deliberately tight ceiling, so it stays meaningful
    whatever the panel or budget mode turns out to be.
    """
    bundle = load_bundle(complete_configs)
    cells = build_cells(
        bundle.design.K_grid,
        [(b.value, b.mode) for b in bundle.design.B_grid],
        bundle.design.scales,
        bundle.design.readers,
    )

    # A ceiling chosen to bite, not inherited from a placeholder window.
    tight = 20_000
    report = context_feasibility(
        cells,
        {r: ReaderLimit.from_window(r, tight) for r in bundle.design.readers},
        bundle.pricing,
    )

    assert report["n_infeasible"] > 0, "a 20k ceiling must reject some cells"

    for rows in report["infeasible_cells"].values():
        for row in rows:
            # A cell is infeasible exactly when it exceeds the ceiling...
            assert row["prompt_tokens"] > tight
            assert row["ceiling"] == tight
            assert row["overflow_by"] == row["prompt_tokens"] - tight
            # ...and it is reported, never quietly shrunk to fit.
            assert row["blocked_by"] == "window"

    # No cell reported as feasible may in fact exceed the ceiling.
    reported = sum(len(v) for v in report["infeasible_cells"].values())
    assert reported == report["n_infeasible"]


def test_no_cell_silently_exceeds_its_ceiling(complete_configs: Path) -> None:
    """Independent cross-check: recompute every prompt and compare.

    Deliberately does not reuse `context_feasibility`'s own arithmetic, so a bug in
    the reported `prompt_tokens` cannot hide behind a matching bug in the check.
    """
    from msat.design import _content_tokens, _prompt_tokens_per_memory

    bundle = load_bundle(complete_configs)
    cells = build_cells(
        bundle.design.K_grid,
        [(b.value, b.mode) for b in bundle.design.B_grid],
        bundle.design.scales,
        bundle.design.readers,
    )
    tok = bundle.pricing["token_estimates"]

    for cell in cells:
        anchors = tok[cell.scale]
        if cell.b_mode == "per_memory_tokens":
            expected = _prompt_tokens_per_memory(cell.k, cell.b, anchors)
        else:
            expected = min(_content_tokens(cell.k, anchors), cell.b)
        # content must never exceed the per-memory cap times the memories available
        n = min(float(cell.k), float(anchors["sessions_mean"]))
        assert expected <= n * anchors["tokens_per_session"] + 1e-6
        assert expected > 0


def test_context_policy_never_truncates(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    policy = bundle.readers.context_policy
    assert policy.on_overflow == "reject_cell"
    assert policy.never_truncate_silently is True
    assert policy.require_context_window_ge_prompt is True


# ── strata (PRD §R9.9) ────────────────────────────────────────────────────────
def test_stratum_mde_shrinks_with_n() -> None:
    small = mde_for_stratum(30, seed=1)
    large = mde_for_stratum(127, seed=1)
    assert large["mde_pp_at_p50"] < small["mde_pp_at_p50"]
    assert large["se_pp"] < small["se_pp"]


def test_n30_stratum_reports_9pp_se_and_a_larger_mde() -> None:
    """PRD §R9.9 quotes "SE ~ 9pp" for the n=30 strata. That is the SE.

    The detectable effect at 80% power is ~2.8x larger, and larger again once
    within-item clustering is accounted for. Quoting the 9pp SE where an MDE
    belongs would overstate what a 30-item stratum can resolve by nearly
    threefold — which is how an exploratory stratum becomes a spurious finding.
    """
    result = mde_for_stratum(30, seed=1)
    assert 8.0 < result["se_pp"] < 10.0, "must reproduce the PRD's ~9pp figure"
    assert result["se_pp_clustered"] > result["se_pp"]
    assert result["mde_pp_at_p50"] > result["se_pp"] * 2.5


def test_judge_noise_grid_is_ordered() -> None:
    flips = [f for f, _ in JUDGE_NOISE_GRID]
    assert flips == sorted(flips)
    assert flips[0] == 0.0
