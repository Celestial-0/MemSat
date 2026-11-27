"""Is the K x B interaction estimable at this grid?"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from msat.ceiling import dump_canonical
from msat.config import ConfigBundle, ReaderFamily
from msat.constants import JUDGE_NOISE_GRID
from msat.design import (
    ALPHA,
    TARGET_POWER,
    ReaderLimit,
    SimSpec,
    build_cells,
    context_feasibility,
    estimability,
    estimate_compute,
    estimate_cost,
    mde_column,
    mde_for_stratum,
    mde_interaction,
    propose_reduced_design,
)
from msat.hardware.vram import DEFAULT_HEADROOM_GB

_COLUMNS: dict[str, int] = {
    "log2K": 1,
    "log2B": 2,
    "interaction_log2K_x_log2B": 3,
    "corpus_fraction": 4,
    "gold_rank": 5,
    "reader_strength": 6,
}


VRAM_GB: float = 8.0
OVERHEAD_GB: float = DEFAULT_HEADROOM_GB


def _reader_limit(family: ReaderFamily, *, headroom_gb: float = OVERHEAD_GB) -> ReaderLimit:
    from msat.hardware.capacity import compute_reader_limit

    return compute_reader_limit(family, card_gb=VRAM_GB, headroom_gb=headroom_gb)


def _corpus_distribution(bundle: ConfigBundle) -> dict[str, Any]:
    from msat.datasets import iter_metadata, raw_path, resolve_data_dir

    out: dict[str, Any] = {}
    root = resolve_data_dir()
    for scale in bundle.design.scales:
        path = raw_path(scale, root)
        if not path.is_file():
            continue
        sizes = np.array([q.corpus.n_sessions for q in iter_metadata(path)], dtype=float)
        sizes = sizes[sizes > 0]
        if sizes.size == 0:
            continue
        log_sizes = np.log(sizes)
        out[scale] = {
            "n": int(sizes.size),
            "mu": float(log_sizes.mean()),
            "sigma": float(log_sizes.std(ddof=1)) if sizes.size > 1 else 0.0,
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
        }
    return out


def _infeasible_kb(
    context: dict[str, Any],
) -> set[tuple[int, float]]:
    bad: set[tuple[int, float]] = set()
    for rows in (context.get("infeasible_cells") or {}).values():
        for row in rows:
            bad.add((int(row["K"]), float(row["B"])))
    return bad


def largest_feasible_rectangle(
    k_grid: Sequence[int],
    b_grid: Sequence[float],
    bad_kb: set[tuple[int, float]],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    best_n = 0
    best: tuple[tuple[int, ...], tuple[float, ...]] = ((), ())
    n_k, n_b = len(k_grid), len(b_grid)
    for ki in range(1, n_k + 1):
        for ks in range(0, n_k - ki + 1):
            ksub = tuple(k_grid[ks : ks + ki])
            for bj in range(1, n_b + 1):
                for bs in range(0, n_b - bj + 1):
                    bsub = tuple(b_grid[bs : bs + bj])
                    if any((k, b) in bad_kb for k in ksub for b in bsub):
                        continue
                    if ki * bj > best_n or (ki * bj == best_n and ki > len(best[0])):
                        best_n = ki * bj
                        best = (ksub, bsub)
    return best


def _build_spec(
    bundle: ConfigBundle,
    corpus: dict[str, Any],
    *,
    n_sim: int,
    sigma_run: float = 0.0,
    scales: tuple[str, ...] | None = None,
    k_grid: tuple[int, ...] | None = None,
    b_grid: tuple[float, ...] | None = None,
) -> SimSpec:
    design = bundle.design
    used = scales or tuple(design.scales)
    mus = [corpus[s]["mu"] for s in used if s in corpus]
    sigmas = [corpus[s]["sigma"] for s in used if s in corpus]
    mins = [corpus[s]["min"] for s in used if s in corpus]
    mu = float(np.mean(mus)) if mus else math.log(48.0)
    sigma = float(np.mean(sigmas)) if sigmas else 0.12
    min_sessions = min(mins) if mins else 38

    items = 500 if design.items.include_abstention else design.items.limit or 470
    n_cells = (
        len(k_grid or design.K_grid)
        * len(b_grid or design.b_values)
        * len(used)
        * len(design.readers)
    )
    return SimSpec(
        n_questions=items,
        n_cells=n_cells,
        runs_per_cell=design.runs_per_cell,
        k_levels=tuple(k_grid or design.K_grid),
        b_levels=tuple(b_grid or design.b_values),
        log_n_sessions_mu=mu,
        log_n_sessions_sigma=sigma,
        min_sessions=min_sessions,
        reader_strength_spread=0.16,
        n_sim=n_sim,
        sigma_run=sigma_run,
    )


def run_power_analysis(
    bundle: ConfigBundle,
    *,
    out_dir: Path,
    n_sim: int = 30,
    seed: int = 20261012,
    sensitivity: bool = True,
) -> dict[str, Any]:
    design = bundle.design
    corpus = _corpus_distribution(bundle)
    b_grid = [(b.value, b.mode) for b in design.B_grid]
    cells = build_cells(design.K_grid, b_grid, design.scales, design.readers)
    items = 500 if design.items.include_abstention else design.items.limit or 470

    context = context_feasibility(
        cells,
        {
            f.id: _reader_limit(
                f,
                headroom_gb=float(
                    bundle.raw["readers.yaml"]
                    .get("context_policy", {})
                    .get("headroom_gb", OVERHEAD_GB)
                ),
            )
            for f in bundle.readers.families
        },
        bundle.pricing,
    )
    bad_kb = _infeasible_kb(context)
    feas_k, feas_b = largest_feasible_rectangle(
        tuple(design.K_grid), tuple(design.b_values), bad_kb
    )
    trimmed = bool(feas_k) and (feas_k != tuple(design.K_grid) or feas_b != tuple(design.b_values))

    spec = _build_spec(bundle, corpus, n_sim=n_sim, k_grid=feas_k or None, b_grid=feas_b or None)
    nominal_spec = _build_spec(bundle, corpus, n_sim=n_sim)

    interaction = mde_interaction(spec, seed=seed)
    nominal_interaction = mde_interaction(nominal_spec, seed=seed)

    main_effects = {
        name: mde_column(spec, col, seed + i)
        for i, (name, col) in enumerate(_COLUMNS.items())
        if name != "interaction_log2K_x_log2B"
    }

    sensitivity_rows: list[dict[str, Any]] = []
    if sensitivity:
        for i, (flip, sigma) in enumerate(JUDGE_NOISE_GRID):
            s = replace(spec, judge_flip=flip, sigma_run=sigma, n_sim=max(8, n_sim // 2))
            result = mde_interaction(s, seed=seed + 1000 * (i + 1))
            sensitivity_rows.append(
                {
                    "judge_flip": flip,
                    "sigma_run": sigma,
                    "interpretation": _noise_label(flip, sigma),
                    "effect_attenuation": round(1.0 - 2.0 * flip, 4),
                    **result,
                }
            )

    strata = {str(n): mde_for_stratum(n, seed=seed + n) for n in (30, 50, 100, 121, 127, items)}

    feas_set = {(int(k), float(b)) for k in feas_k for b in feas_b}
    runnable_cells = [c for c in cells if (c.k, float(c.b)) in feas_set] or cells
    reader_models = {f.id: f.model for f in bundle.readers.families}
    judges_n = (
        0
        if bundle.mode == "fast"
        else int(bundle.pricing.get("judge", {}).get("judges_per_output", 2))
    )
    cost = estimate_cost(
        runnable_cells,
        bundle.pricing,
        runs_per_cell=design.runs_per_cell,
        items_per_cell=items,
        judges_per_output=judges_n,
        reader_models=reader_models,
    )
    compute = estimate_compute(
        runnable_cells,
        bundle.pricing,
        runs_per_cell=design.runs_per_cell,
        items_per_cell=items,
        judges_per_output=judges_n,
    )

    feasible_grid: dict[str, Any] = {
        "is_full_grid": not bad_kb,
        "infeasible_KB_pairs": [[int(k), float(b)] for k, b in sorted(bad_kb)],
        "K_grid": list(feas_k),
        "B_grid": list(feas_b),
        "n_KB_pairs": len(feas_k) * len(feas_b),
        "mde_nominal_grid": nominal_interaction,
        "mde_feasible_grid": interaction,
        "n_cells_effective": spec.n_cells,
        "n_observations_effective": spec.n_observations,
    }
    if trimmed:
        feasible_grid["mde_inflation"] = round(
            interaction["mde_log_odds"] / nominal_interaction["mde_log_odds"], 4
        )
        feasible_grid["why_primary_is_the_feasible_one"] = (
            "The nominal MDE covers cells no reader can serve on this card. R9.12 "
            "refuses to compare readers over unequal K support, so the primary "
            "estimate is the one on the largest feasible rectangle."
        )
    else:
        feasible_grid["why_primary_is_the_feasible_one"] = (
            "the whole nominal grid is executable on this card, so the two coincide"
        )

    guard = bundle.budget_guard
    budget_state = _check_budget(guard.max_usd, cost["cost_uncached_total_usd"])

    est = estimability(cells)

    powered = bool(interaction.get("powered"))
    report: dict[str, Any] = {
        "_provenance": {
            "generated_by": "msat estimate-power",
            "api_calls": 0,
            "config_hash": bundle.config_hash,
            "preregistration_hash": bundle.preregistration_hash,
            "preregistered_sign": bundle.preregistration.interaction_sign_prediction,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "n_sim_per_point": n_sim,
            "seed": seed,
        },
        "_method": {
            "estimator": "logit GLM with cluster-robust (sandwich) SE by question; "
            "equivalent to statsmodels GEE with independence working correlation",
            "cross_checked_against": "statsmodels.genmod GEE — agreement to 6 dp "
            "(tests/test_design_power.py::test_estimator_matches_statsmodels_gee)",
            "NOT_used": "MixedLM — it is LINEAR and silently accepts a 0/1 outcome "
            "with no error (PRD §6.1 verified trap)",
            "predictors_centred": "log2(K) at 3.0, log2(B) at 14.0 — a "
            "reparameterisation of PRD §R9.1, not a different model; uncentred "
            "log2(B) spans 12..16 and would sit the baseline near 98% accuracy",
            "covariates": "corpus_fraction and the question count are fitted to the "
            "REAL per-question corpus distribution measured in P1",
            "modelled_not_measured": "gold_rank (a property of the retrieval "
            "manifest, which does not exist until P5)",
            "pairing": "every question is measured at every cell — treating "
            "conditions as independent would inflate the SE",
        },
        "grid": {
            "K_grid": list(design.K_grid),
            "B_grid": [{"value": b.value, "mode": b.mode} for b in design.B_grid],
            "scales": list(design.scales),
            "readers": list(design.readers),
            "runs_per_cell": design.runs_per_cell,
            "items_per_cell": items,
            "n_cells": design.n_cells,
            "n_cells_nominal": design.n_cells,
            "n_cells_effective": spec.n_cells,
            "n_observations": spec.n_observations,
            "grid_effective_K": list(feas_k),
            "grid_effective_B": list(feas_b),
            "cells_dropped_for_vram": design.n_cells - spec.n_cells,
        },
        "corpus_distribution_measured": corpus,
        "interaction": interaction,
        "feasible_grid": feasible_grid,
        "main_effects": main_effects,
        "judge_noise_sensitivity": sensitivity_rows,
        "strata": strata,
        "estimability": est,
        "context_feasibility": context,
        "cost": cost,
        "compute_budget": compute,
        "budget_guard": budget_state,
    }

    if not powered:
        report["reduced_design_proposals"] = propose_reduced_design(
            design,
            mde_full=interaction.get("mde_log_odds"),
            seed=seed,
            n_sim=n_sim,
            items_per_cell=items,
        )
    report["verdict"] = _verdict(powered, interaction, budget_state, sensitivity_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    dump_canonical(report, out_dir / "power.json")
    return report


def _noise_label(flip: float, sigma: float) -> str:
    if flip <= 0.0 and sigma <= 0.0:
        return "perfect measurement — an upper bound, not a forecast"
    if flip <= 0.02:
        return "seed variance only; grader essentially exact"
    if flip <= 0.06:
        return "mild grader disagreement"
    if flip <= 0.12:
        return "poor grader agreement (kappa ~ 0.118 territory, per 2511.02246)"
    return "severe grader noise — measurement, not sample size, is binding"


def _check_budget(max_usd: float | None, estimate: float) -> dict[str, Any]:
    if max_usd is None:
        return {
            "state": "pending_guide_signoff",
            "enforceable": False,
            "estimate_usd": estimate,
            "note": "blocker B7 — the approved ceiling was set for a 1-factor sweep. "
            "This estimate is the INPUT to the budget conversation; no run may be "
            "launched until the guide sets it.",
        }
    return {
        "state": "set",
        "enforceable": True,
        "max_usd": max_usd,
        "estimate_usd": estimate,
        "within_budget": estimate <= max_usd,
        "headroom_usd": round(max_usd - estimate, 2),
        "multiple_of_budget": round(estimate / max_usd, 3) if max_usd else None,
    }


def _verdict(
    powered: bool,
    interaction: dict[str, Any],
    budget: dict[str, Any],
    sensitivity: list[dict[str, Any]],
) -> dict[str, Any]:
    mde = interaction.get("mde_log_odds")
    if not powered:
        headline = (
            f"UNDERPOWERED: the K x B interaction is not detectable at this grid "
            f"({interaction.get('note', '')}). PRD §R0.3 requires a PROPOSAL — see "
            f"`reduced_design_proposals`."
        )
    else:
        headline = (
            f"Interaction estimable: MDE = {mde} log-odds "
            f"({interaction['mde_pp_at_p50']} pp at p=0.5, "
            f"{interaction['mde_pp_at_p30']} pp at p=0.3). "
            f"SE = {interaction['se_log_odds']}."
        )

    binding = "sensitivity sweep unavailable"
    if sensitivity and sensitivity[0].get("mde_log_odds") and sensitivity[-1].get("mde_log_odds"):
        clean = sensitivity[0]["mde_log_odds"]
        noisy = sensitivity[-1]["mde_log_odds"]
        ratio = noisy / clean if clean else float("inf")
        binding = (
            f"Measurement quality, not sample size, is the binding constraint: the "
            f"MDE degrades from {clean} to {noisy} log-odds ({ratio:.1f}x) as grader "
            f"misclassification rises from 0% to "
            f"{100 * sensitivity[-1]['judge_flip']:.0f}%. Buying more cells would not "
            f"help; buying a better grader would."
        )
    return {
        "powered": powered,
        "headline": headline,
        "binding_constraint": binding,
        "budget": "BLOCKED"
        if budget["state"] != "set"
        else ("within budget" if budget["within_budget"] else "OVER BUDGET"),
    }


def load_power(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
