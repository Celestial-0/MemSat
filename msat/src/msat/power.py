"""`msat design-power` — is the K x B interaction estimable at this grid?

PRD §R0. This is the single highest-leverage command in the project. It answers,
before a single dollar is spent, the question whose answer invalidates weeks of
downstream work if it goes unnoticed: **is the headline interaction detectable?**

What it produces
----------------
`results/power.json` containing the MDE for the interaction, the main effects and
the exploratory strata; a judge-noise sensitivity sweep; a full cost estimate
per cell; and the pre-registration hash (PRD §R0.4) — all in one artifact, so
none of it can be quoted without the caveats attached to it.

Three things it deliberately does NOT do
----------------------------------------
1. **Predict an effect.** Power says how large an effect the design could detect.
   A well-powered design for an effect that is not there is a publishable null,
   which is exactly why the sign prediction is registered separately.
2. **Silently truncate.** If the grid is underpowered, PRD §R0.3 requires a
   PROPOSAL. This module emits ranked proposals and stops.
3. **Treat statistical power as the binding constraint.** With ~126k paired
   observations the interaction is trivially detectable *under the simulated
   noise model*. The honest conclusion is that measurement quality binds, not
   sample size — so the sensitivity sweep varies exactly that.
"""

from __future__ import annotations

import json
import math
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

#: Column names matching the simulated design matrix, for the report.
_COLUMNS: dict[str, int] = {
    "log2K": 1,
    "log2B": 2,
    "interaction_log2K_x_log2B": 3,
    "corpus_fraction": 4,
    "gold_rank": 5,
    "reader_strength": 6,
}


#: The card the design is sized against (blocker B7's wall-clock budget).
VRAM_GB: float = 8.0
OVERHEAD_GB: float = DEFAULT_HEADROOM_GB
"""Single source of truth: `hardware.vram.DEFAULT_HEADROOM_GB`.

This used to be a local `0.5` whose docstring claimed to cover "CUDA context,
activations, allocator slack" -- the identical scope `vram.py` covered with `3.0`.
One constant said deep cells were servable and the other said they were not, which
is the whole of D11: not a modelling disagreement, a duplicated number. Both paths
now resolve to the same value, so the feasibility report and the runtime capacity
preflight cannot disagree about a reader's ceiling.

Which VALUE is right remains an open decision, and `design.headroom_that_fits`
reports the threshold at which the full grid becomes executable rather than leaving
it to preference. A too-small number here does not fail safely: `retrieval.yaml`
sets `never_truncate_silently: true` and `on_overflow: reject_cell`, so the
optimistic value turns into an OOM at hour 14 instead of a refusal at hour 0.
"""


def _reader_limit(family: ReaderFamily) -> ReaderLimit:
    """Build a reader's ceiling from its architecture facts.

    Uses the VRAM ceiling when the family declares `vram`, and the declared
    context window otherwise. The window is the weaker of the two, so omitting
    `vram` does not fail loudly - it makes feasibility optimistic in exactly the
    direction that is hardest to notice, which is why `readers.yaml` documents
    the consequence next to each family.
    """
    if family.vram is None:
        return ReaderLimit.from_window(family.id, family.context_window)
    return ReaderLimit.from_vram(
        family.id,
        context_window=family.context_window,
        weight_gb=family.vram.weight_gb,
        kv_mb_per_1k=family.vram.kv_mb_per_1k,
        vram_gb=VRAM_GB,
        overhead_gb=OVERHEAD_GB,
    )


def _corpus_distribution(bundle: ConfigBundle) -> dict[str, Any]:
    """Fit the per-question corpus-size distribution to the REAL data (P1).

    Using the measured distribution rather than an assumed one means
    `corpus_fraction` in the simulation is the covariate the primary model will
    actually see — which matters, because on the S split K=64 is full-context for
    100% of questions and on M for none.
    """
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
    # Corpus stats are averaged across scales on the log scale: the S and M
    # distributions differ by ~10x in size but have similar relative spread.
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
    """Compute and emit `power.json`. Returns the report."""
    design = bundle.design
    corpus = _corpus_distribution(bundle)
    b_grid = [(b.value, b.mode) for b in design.B_grid]
    cells = build_cells(design.K_grid, b_grid, design.scales, design.readers)
    items = 500 if design.items.include_abstention else design.items.limit or 470

    spec = _build_spec(bundle, corpus, n_sim=n_sim)

    # ── the headline number ────────────────────────────────────────────────
    interaction = mde_interaction(spec, seed=seed)

    # ── main effects, for context ──────────────────────────────────────────
    main_effects = {
        name: mde_column(spec, col, seed + i)
        for i, (name, col) in enumerate(_COLUMNS.items())
        if name != "interaction_log2K_x_log2B"
    }

    # ── judge-noise sensitivity ────────────────────────────────────────────
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

    # ── exploratory strata ─────────────────────────────────────────────────
    strata = {str(n): mde_for_stratum(n, seed=seed + n) for n in (30, 50, 100, 121, 127, items)}

    # ── cost (now $0) and COMPUTE (the real budget) ─────────────────────────
    reader_models = {f.id: f.model for f in bundle.readers.families}
    judges_n = int(bundle.pricing.get("judge", {}).get("judges_per_output", 2))
    cost = estimate_cost(
        cells,
        bundle.pricing,
        runs_per_cell=design.runs_per_cell,
        items_per_cell=items,
        judges_per_output=judges_n,
        reader_models=reader_models,
    )
    compute = estimate_compute(
        cells,
        bundle.pricing,
        runs_per_cell=design.runs_per_cell,
        items_per_cell=items,
        judges_per_output=judges_n,
    )

    # ── context feasibility: can each reader actually serve each cell? ───────
    context = context_feasibility(
        cells,
        {f.id: _reader_limit(f) for f in bundle.readers.families},
        bundle.pricing,
    )

    # ── budget guard (R0.5 / N8) ───────────────────────────────────────────
    guard = bundle.budget_guard
    budget_state = _check_budget(guard.max_usd, cost["cost_uncached_total_usd"])

    # ── estimability ───────────────────────────────────────────────────────
    est = estimability(cells)

    powered = bool(interaction.get("powered"))
    report: dict[str, Any] = {
        "_provenance": {
            "generated_by": "msat design-power",
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
            "n_observations": spec.n_observations,
        },
        "corpus_distribution_measured": corpus,
        "interaction": interaction,
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
    """PRD §R0.5 / §N8 — a mismatch is a HARD error, and an unset ceiling is not
    silently treated as infinite."""
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

    # Data-driven, not asserted: compare the perfect-measurement MDE against the
    # realistic one and report the degradation.
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
    """Read back a previously written `power.json`."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
