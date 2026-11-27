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


def test_estimator_matches_statsmodels_gee() -> None:
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
    xmat = np.column_stack([np.ones(20), np.arange(20.0)])
    y = (np.arange(20) > 9).astype(float)
    clusters = np.repeat(np.arange(5), 4)
    beta, se = fit_logit_cluster_robust(y, xmat, clusters)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(se))


def test_clustering_widens_the_standard_error() -> None:
    spec = _small_spec(n_questions=60, n_sim=1)
    y, xmat, clusters = simulate_dataset(spec, 0.3, np.random.default_rng(3))
    _, se_clustered = fit_logit_cluster_robust(y, xmat, clusters)

    _, se_iid = fit_logit_cluster_robust(y, xmat, np.arange(len(y)))
    assert se_clustered[INTERACTION_COLUMN] > se_iid[INTERACTION_COLUMN]


def test_arms_a_and_g_are_one_grid_not_two() -> None:
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
    spec = _small_spec(n_questions=120, n_cells=24, n_sim=12)
    result = mde_interaction(spec, seed=3)
    assert result["powered"]
    analytic = 2.8016 * result["se_log_odds"]
    assert result["mde_log_odds"] == pytest.approx(analytic, rel=0.35)


@pytest.mark.slow
def test_judge_noise_degrades_the_mde() -> None:
    clean = _small_spec(n_questions=80, n_cells=24, n_sim=10, judge_flip=0.0)
    noisy = _small_spec(n_questions=80, n_cells=24, n_sim=10, judge_flip=0.12)
    a = mde_interaction(clean, seed=5)
    b = mde_interaction(noisy, seed=5)
    assert a["powered"] and b["powered"]
    assert b["mde_log_odds"] > a["mde_log_odds"] * 1.2


def test_judge_flip_attenuates_the_estimated_effect() -> None:
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
    bundle = load_bundle(complete_configs)
    result = propose_reduced_design(
        bundle.design, mde_full=None, seed=1, n_sim=2, items_per_cell=500
    )
    for proposal in result["proposals"]:
        assert len(proposal["B_grid"]) >= 2, f"{proposal['step']} removes the crossing"


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
    bundle = load_bundle(complete_configs)
    assert bundle.pricing["compute_budget"]["cacheable_prefix"] is False


def test_unpriced_reader_is_reported_not_skipped_silently(complete_configs: Path) -> None:
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


def test_a_degenerate_cell_is_rejected_never_truncated(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    cells = build_cells(
        bundle.design.K_grid,
        [(b.value, b.mode) for b in bundle.design.B_grid],
        bundle.design.scales,
        bundle.design.readers,
    )

    largest_prompt = max(min(c.k, 48.0) * min(2563.0, c.b) for c in cells)
    tight = int(largest_prompt * 0.5)
    assert tight < largest_prompt, "the tight ceiling must actually bite"
    report = context_feasibility(
        cells,
        {r: ReaderLimit.from_window(r, tight) for r in bundle.design.readers},
        bundle.pricing,
    )

    assert report["n_infeasible"] > 0, (
        f"a {tight}-token ceiling against a {largest_prompt}-token worst prompt "
        f"must reject some cells"
    )

    for rows in report["infeasible_cells"].values():
        for row in rows:
            assert row["prompt_tokens"] > tight
            assert row["reader_capacity_tokens"] == tight
            assert row["overflow_by"] == row["prompt_tokens"] - tight
            assert row["blocked_by"] == "window"

    reported = sum(len(v) for v in report["infeasible_cells"].values())
    assert reported == report["n_infeasible"]


def test_no_cell_silently_exceeds_its_ceiling(complete_configs: Path) -> None:
    from msat.design import content_tokens, prompt_tokens_per_memory

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
            expected = prompt_tokens_per_memory(cell.k, cell.b, anchors)
        else:
            expected = min(content_tokens(cell.k, anchors), cell.b)
        n = min(float(cell.k), float(anchors["sessions_mean"]))
        assert expected <= n * anchors["tokens_per_session"] + 1e-6
        assert expected > 0


def test_context_policy_never_truncates(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    policy = bundle.readers.context_policy
    assert policy.on_overflow == "reject_cell"
    assert policy.never_truncate_silently is True
    assert policy.require_context_window_ge_prompt is True


def test_stratum_mde_shrinks_with_n() -> None:
    small = mde_for_stratum(30, seed=1)
    large = mde_for_stratum(127, seed=1)
    assert large["mde_pp_at_p50"] < small["mde_pp_at_p50"]
    assert large["se_pp"] < small["se_pp"]


def test_n30_stratum_reports_9pp_se_and_a_larger_mde() -> None:
    result = mde_for_stratum(30, seed=1)
    assert 8.0 < result["se_pp"] < 10.0, "must reproduce the PRD's ~9pp figure"
    assert result["se_pp_clustered"] > result["se_pp"]
    assert result["mde_pp_at_p50"] > result["se_pp"] * 2.5


def test_judge_noise_grid_is_ordered() -> None:
    flips = [f for f, _ in JUDGE_NOISE_GRID]
    assert flips == sorted(flips)
    assert flips[0] == 0.0
