"""Grid, power, and estimability (PRD §R0).

`msat design-power` answers one question **before a single API call is spent**:

> Is the `log2(K) x log2(B)` INTERACTION estimable at this cell count?

An underpowered interaction (risk A13) is the most likely way this design
produces an uninterpretable null, and it is the one expensive mistake that is
free to avoid today. Discovering it in week 8 means a re-run; discovering it
this afternoon means a smaller grid.

Method
------
Monte-Carlo power: simulate the DECLARED primary model (PRD §R9.1) on synthetic
outcomes at the declared cell count, fit it, and read the standard error of the
interaction term. MDE is the effect that reaches 80% power at alpha = 0.05.

The fit is a logistic regression with **cluster-robust standard errors by
question** — which is exactly `statsmodels` GEE with an independence working
correlation, the estimator the PRD records as "verified working". It is
implemented directly in numpy because the real design is ~118k rows and a
power sweep needs hundreds of fits; `tests/test_design_power.py` checks this
implementation against `statsmodels.GEE` on a small case, so the fast path
cannot silently diverge from the reference.

Covariate distributions are taken from the REAL corpus where P1 measured them
(`n_sessions`, `n_gold_sessions` per question), so `corpus_fraction` in the
simulation is the actual covariate distribution rather than a convenient guess.
`gold_rank` is modelled, since it is a property of the retrieval manifest that
does not exist until P5 — and that is stated in the report rather than hidden.

What this is NOT: it does not predict an effect. It says how large an effect the
design could detect. A design can be well-powered for an effect that is not
there, and that outcome is a publishable null — which is why the sign
prediction is registered separately, before data collection (PRD §R0.4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from scipy import stats
from scipy.special import expit

from msat.hardware.vram import DEFAULT_HEADROOM_GB

ALPHA: Final = 0.05
TARGET_POWER: Final = 0.80
Z_ALPHA_2: Final = float(stats.norm.ppf(1 - ALPHA / 2))
Z_POWER: Final = float(stats.norm.ppf(TARGET_POWER))

#: Inflation applied to a naive binomial SE when the quantity of interest is a
#: *stratum* mean. Three runs per cell are not three independent items, so
#: `sqrt(p(1-p)/n)` understates the true standard error.
CLUSTER_INFLATION: Final = 1.25

#: Simulated variance components. `sigma_question` is the between-question
#: intercept SD; it is what makes the cluster-robust SE larger than an
#: independent-observation SE would give, and it is the dominant reason a 470-item
#: benchmark cannot detect small effects.
SIGMA_QUESTION: Final = 0.70

#: True main-effect coefficients used for the simulation. These are PLANTED, not
#: estimated — power depends on the effect being tested, not on any particular
#: baseline, and inventing a fitted model here would be circular.
TRUE_MAIN_EFFECTS: Final[dict[str, float]] = {
    "intercept": 0.20,
    "log2K": 0.15,
    "log2B": 0.25,
    "corpus_fraction": -0.10,
    "gold_rank_z": -0.20,
    "reader_strength": 0.80,
}


# ── the condition grid ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Cell:
    """One distinct condition. The key is the whole grid, never just K (PRD §R5.6)."""

    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str

    @property
    def key(self) -> tuple[str, int, float, str, str, str]:
        return (self.arm, self.k, round(self.b, 6), self.b_mode, self.scale, self.reader)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "scale": self.scale,
            "reader": self.reader,
        }


def build_cells(
    k_grid: Sequence[int],
    b_grid: Sequence[tuple[float, str]],
    scales: Sequence[str],
    readers: Sequence[str],
    arms: Sequence[str] = ("A+G",),
) -> list[Cell]:
    """Every cell in `K_grid x B_grid x scales x readers`.

    PRD §R4.4: arms A and G are ONE crossed grid. Each condition retrieves top-K
    in rank order (that is what "A" means) and then truncates to B tokens (that is
    what "G" means), so a crossed cell *is* A and G simultaneously.

    The default arm label is therefore `A+G` and the cell count is
    `len(K) x len(B) x len(scales) x len(readers)` — **not** twice that. Labelling
    the grid `("A", "G")` here would silently double the cost estimate, and with
    it the number that goes into the budget conversation (blocker B7). A reviewer
    comparing our estimate against a one-factor sweep would find it exactly 2x
    larger for no scientific reason.

    `arms` remains a parameter so a genuinely separate arm (B, C, D, E, F) can be
    enumerated the same way in Phase 5.
    """
    return [
        Cell(arm=arm, k=k, b=b, b_mode=mode, scale=scale, reader=reader)
        for arm in arms
        for scale in scales
        for reader in readers
        for k in k_grid
        for b, mode in b_grid
    ]


def cell_key_collisions(cells: Sequence[Cell]) -> list[tuple[Any, ...]]:
    """Duplicate condition keys. Non-empty means a silent correctness bug (H4)."""
    seen: dict[tuple[Any, ...], int] = {}
    for cell in cells:
        seen[cell.key] = seen.get(cell.key, 0) + 1
    return [k for k, n in seen.items() if n > 1]


# ── the estimator ──────────────────────────────────────────────────────────────
def fit_logit_cluster_robust(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic MLE with a cluster-robust (sandwich) covariance.

    Equivalent to `statsmodels.GEE(..., cov_struct=Independence())`, which is the
    estimator PRD §6.1 records as "verified working" for the binomial case — and
    emphatically NOT `MixedLM`, which is LINEAR and silently accepts a 0/1
    outcome without error.

    `ridge` is a vanishing L2 penalty on the non-intercept coefficients. It exists
    because plain IRLS DIVERGES under quasi-complete separation: when a cluster's
    outcomes are all 1 (or all 0), `mu` reaches 1, the weight `mu(1-mu)` hits the
    floor, and `(y-mu)/w` explodes to ~1e10, driving the coefficient to infinity
    and every standard error to zero. That is not a numerical curiosity — it is
    what a saturated cell looks like, which is exactly the degeneracy this project
    exists to detect. At n ~ 1e5 the penalty is ~1e-3 log-odds and does not move
    the estimate; without it, one degenerate cell poisons the whole power sweep.

    Returns `(beta, std_errors)`.
    """
    beta = np.zeros(X.shape[1], dtype=float)
    n = float(X.shape[0])
    penalty = ridge * n * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # never penalise the intercept

    for _ in range(max_iter):
        eta = X @ beta
        mu = expit(eta)
        weights = np.clip(mu * (1.0 - mu), 1e-8, None)
        z = eta + (y - mu) / weights
        xtw = X.T * weights
        a = xtw @ X + penalty
        b_vec = xtw @ z
        try:
            new_beta = np.linalg.solve(a, b_vec)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            break
        converged = np.max(np.abs(new_beta - beta)) < tol
        beta = new_beta
        if converged:
            break

    mu = expit(X @ beta)
    weights = np.clip(mu * (1.0 - mu), 1e-8, None)
    a_inv = np.linalg.pinv((X.T * weights) @ X + penalty)
    resid = y - mu
    scores = X * resid[:, None]
    meat = _cluster_meat(scores, clusters)
    cov = a_inv @ meat @ a_inv.T
    # Small negative variances come from floating-point cancellation in the
    # sandwich; clamp rather than propagate a NaN std error.
    var = np.clip(np.diag(cov), 0.0, None)
    return beta, np.sqrt(var)


# ── simulation ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SimSpec:
    """Everything the power simulation needs, all of it declared.

    `k_levels` / `b_levels` are the GRID, and cells are laid out as a full
    factorial over them — exactly as the real design is. Randomly assigning K and
    B to cells looks equivalent and is not: with a small cell count it can draw a
    design in which K (or B) is constant, which makes `log2K * log2B` an exact
    linear function of `log2B` and the interaction unidentifiable. That failure is
    silent — the fit "succeeds" and returns enormous nonsense coefficients.

    `log_n_sessions_mu` / `log_n_sessions_sigma` describe the per-question corpus
    size. The CLI fits them to the REAL distribution measured in P1 rather than
    assuming them, so `corpus_fraction` in the simulation is the actual covariate
    the primary model will see.
    """

    n_questions: int
    n_cells: int
    runs_per_cell: int
    k_levels: tuple[int, ...]
    b_levels: tuple[float, ...]
    log_n_sessions_mu: float
    log_n_sessions_sigma: float
    min_sessions: int
    reader_strength_spread: float
    n_sim: int
    with_interaction: bool = True
    sigma_run: float = 0.0
    """Within-cell, across-run latent noise (seed + judge disagreement) on the logit
    scale. Kept separate from `judge_flip` because they act differently."""
    judge_flip: float = 0.0
    """Probability that the grader records the WRONG verdict for an observation.

    This is the parameter that actually decides the answer, and it is a
    MISCLASSIFICATION model, not a variance inflation.

    Modelling judge noise as extra within-cluster variance does nothing, and the
    reason is instructive: cluster-robust standard errors are *designed* to stay
    valid under arbitrary within-cluster correlation, so inflating it barely moves
    the SE. An earlier version of this sweep did exactly that and reported an MDE
    identical to four decimals at every noise level — a green number that meant
    nothing.

    Judge noise is not within-cluster correlation. It is the MEASUREMENT
    attenuating the signal: a grader with low agreement converts some correct
    answers to incorrect and some incorrect to correct, which pulls observed
    accuracy toward chance and shrinks the recoverable effect by roughly
    `(1 - 2 * flip)`. 2511.02246 reports kappa = 0.118 for LLM judges, and
    2609.38021 flipped 8 of 500 verdicts between two passes of the SAME
    deterministic system under the SAME judge.
    """

    @property
    def n_observations(self) -> int:
        return self.n_questions * self.n_cells * self.runs_per_cell

    @property
    def k_min(self) -> int:
        return min(self.k_levels)

    @property
    def k_max(self) -> int:
        return max(self.k_levels)

    @property
    def b_min(self) -> float:
        return min(self.b_levels)

    @property
    def b_max(self) -> float:
        return max(self.b_levels)

    def cell_levels(self) -> tuple[list[float], list[float]]:
        """K and B per cell, as a full factorial repeated to fill `n_cells`."""
        combos = [(float(k), float(b)) for k in self.k_levels for b in self.b_levels]
        ks, bs = [], []
        for i in range(self.n_cells):
            k, b = combos[i % len(combos)]
            ks.append(k)
            bs.append(b)
        return ks, bs


@dataclass
class PowerResult:
    delta: float
    power: float
    se: float
    mean_estimate: float
    n_obs: int
    reps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta": round(self.delta, 6),
            "power": round(self.power, 4),
            "se_log_odds": round(self.se, 6),
            "mean_estimate": round(self.mean_estimate, 6),
            "n_observations": self.n_obs,
            "n_reps": self.reps,
            **self.extra,
        }


#: Column order of the simulated design matrix. The interaction is index 3.
INTERACTION_COLUMN: Final = 3

#: Grid midpoints used to CENTRE the predictors.
#:
#: PRD §R9.1 writes the model on raw `log2(K)` and `log2(B)`. `log2(B)` ranges
#: 12..16 here, so a 0.25-per-log2-unit main effect would sit the baseline at +4
#: log-odds — a 98% accuracy "benchmark" before any effect is planted, and a
#: degenerate design that cannot detect anything.
#:
#: Centring is a REPARAMETERISATION, not a different model: both predictors and
#: their product are recoverable by linear transformation, the fitted values are
#: DEPRECATED and deliberately unused. These were the means of the ORIGINAL
#: grids; `design.yaml` changed both axes on 2026-10-13 and nothing watching
#: these constants complained. `_design_matrix` now calls `grid_center` on the
#: grid it is handed, so no module-level centre can drift again. Kept named so
#: an existing import fails loudly rather than silently picking up a stale
#: value - delete once no caller references them.
K_CENTER: Final = 3.0  # mean of log2([1,2,4,8,16,32,64])
B_CENTER: Final = 14.0  # mean of log2([4096,16384,65536]) - THE GRID THIS IS FROM NO LONGER EXISTS


def grid_center(log_values: Sequence[float] | np.ndarray) -> float:
    """Centre a moderator on its OWN grid, rather than a constant in this file.

    `K_CENTER` and `B_CENTER` were hardcoded as the means of the original grids.
    When the B axis moved from `[4096, 16384, 65536]` to `[128, 512, 2048]`,
    `B_CENTER = 14.0` silently became the mean of a grid that no longer existed,
    and nothing complained - because a hardcoded constant cannot notice that the
    thing it describes has changed.

    Centring is a reparametrisation, so the interaction coefficient is invariant
    and this is not a numerical bug. It is a lie in a constant, and the class of
    bug is worth eliminating: every one of today's findings was a value that had
    drifted away from the thing it described, and nothing was watching.
    """
    # `len(...) == 0`, not `if not ...`: callers pass numpy arrays, and
    # truth-testing an array raises "truth value is ambiguous". The empty-grid
    # guard has to work for both the Sequence signature and the array call site.
    if len(log_values) == 0:
        raise ValueError("cannot centre an empty grid")
    return float(np.mean(np.asarray(log_values, dtype=float)))


# ── K_delta on the ACTUAL grid (PRD §R9.4) ─────────────────────────────────────
#
# `A(K+1) - A(K) <= 0.5pp` is INVALID on a doubling grid. Every step is 2x, so
# the rule silently becomes "0.5pp per doubling" for some steps and "0.5pp per
# quarter-doubling" for others, and fires far too early wherever the grid is
# uneven. On a grid with a 16 -> 48 -> 64 tail the raw per-step threshold is 3.8x
# stricter on one step than on the next.
#
# The fix is to state the threshold in units that mean the same thing everywhere -
# percentage points per unit of log2(K) - and convert at the actual step.


def k_step_index(k_grid: Sequence[int], k: int) -> float:
    """Position of `k` on the declared grid, normalised to [0, 1].

    Step POSITION, not log2(K): the analysis cares how far along the design this
    point sits, which is what "normalised by step position" means.
    """
    ordered = sorted(k_grid)
    if k not in ordered:
        raise ValueError(f"K={k} is not on the declared grid {ordered}")
    if len(ordered) == 1:
        return 0.0
    return ordered.index(k) / (len(ordered) - 1)


def k_log2_step(k_grid: Sequence[int], k: int) -> float:
    """Size of the step ARRIVING at `k`, in log2 units.

    On the shipped doubling grid every step is 1.0. On an uneven grid the steps
    differ, and a threshold in raw percentage points means different things at
    each. At the smallest K there is no arriving step, so the departing one is
    used - otherwise the first cell would be exempt from every rule.
    """
    ordered = sorted(k_grid)
    if k not in ordered:
        raise ValueError(f"K={k} is not on the declared grid {ordered}")
    i = ordered.index(k)
    if i > 0:
        return float(np.log2(k / ordered[i - 1]))
    if len(ordered) < 2:
        return 0.0
    return float(np.log2(ordered[1] / ordered[0]))


def k_delta_in_pp(base_pp: float, k_grid: Sequence[int], k: int) -> float:
    """The `base_pp` threshold, converted to the step that actually occurs at `k`.

    `base_pp` is read as percentage points per unit of log2(K). On a uniform
    doubling grid this returns `base_pp` unchanged, which is the behaviour the
    original rule intended; on an uneven grid it scales with the real step so the
    rule stops firing early on short steps.
    """
    return base_pp * k_log2_step(k_grid, k)


def _design_matrix(
    log2k: np.ndarray,
    log2b: np.ndarray,
    corpus_frac: np.ndarray,
    gold_rank_z: np.ndarray,
    reader_strength: np.ndarray,
    with_interaction: bool,
) -> np.ndarray:
    # Centred on the grid THIS CALL was given, via `grid_center` - not on a
    # constant in this file. `K_CENTER`/`B_CENTER` were the means of grids that
    # no longer exist (`B_CENTER = 14.0` is the mean of log2[4096,16384,65536],
    # removed 2026-10-13), and `_design_matrix` kept using them, so every power
    # and MDE number was centred 5 log2 units off on the B axis. `grid_center`
    # was written to prevent exactly this and had no caller. It has one now.
    #
    # Reparametrisation, so the interaction coefficient is invariant - but the
    # old constant also silently broke `_prompt_tokens`-vs-`_interp_tokens`
    # comparisons that assume a common centring, and nothing was watching.
    k = log2k - grid_center(log2k)
    b = log2b - grid_center(log2b)
    cols = [np.ones_like(k), k, b]
    if with_interaction:
        cols.append(k * b)
    cols += [corpus_frac, gold_rank_z, reader_strength]
    return np.column_stack(cols)


def _cluster_meat(scores: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """`sum_c (sum_{i in c} s_i)(sum_{i in c} s_i)'`, accumulated without a Python
    loop over clusters (the design has ~500 of them and thousands of fits)."""
    if clusters.size == 0:  # pragma: no cover - empty input
        return np.zeros((scores.shape[1], scores.shape[1]))
    n_clusters = int(clusters.max()) + 1
    per_cluster = np.empty((n_clusters, scores.shape[1]), dtype=float)
    for j in range(scores.shape[1]):
        per_cluster[:, j] = np.bincount(clusters, weights=scores[:, j], minlength=n_clusters)
    return per_cluster.T @ per_cluster


def simulate_dataset(
    spec: SimSpec, delta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One simulated study at the DECLARED cell count.

    Row layout is `question x cell x run`, so every question is measured at every
    cell. That pairing is the entire reason the design has any power: treating
    conditions as independent samples inflates the standard error and would make
    a real interaction look undetectable.

    Returns `(y, X, clusters)`.
    """
    per_cell = spec.n_cells * spec.runs_per_cell

    # --- cell-level covariates: FULL FACTORIAL, not random assignment -------
    k_all, b_all = spec.cell_levels()
    k_vals = np.asarray(k_all, dtype=float)
    b_vals = np.asarray(b_all, dtype=float)
    reader_strength = rng.normal(0.77, spec.reader_strength_spread, size=spec.n_cells)

    log2k = np.tile(np.repeat(np.log2(k_vals), spec.runs_per_cell), spec.n_questions)
    log2b = np.tile(np.repeat(np.log2(b_vals), spec.runs_per_cell), spec.n_questions)
    reader_col = np.tile(np.repeat(reader_strength, spec.runs_per_cell), spec.n_questions)

    # --- question-level corpus size -----------------------------------------
    n_sessions = rng.lognormal(spec.log_n_sessions_mu, spec.log_n_sessions_sigma, spec.n_questions)
    n_sessions = np.clip(n_sessions, spec.min_sessions, None)

    # corpus_fraction is K / |corpus| for THIS question at THIS cell — the actual
    # covariate the primary model uses (PRD §R2.8), not a constant.
    k_matrix = np.tile(np.repeat(k_vals, spec.runs_per_cell), spec.n_questions)
    corpus_frac = (k_matrix / np.repeat(n_sessions, per_cell)).reshape(-1)

    # gold_rank is a property of the retrieval manifest, which does not exist
    # until P5. It is MODELLED here, and the report says so.
    gold_rank_z = rng.normal(0.0, 1.0, size=spec.n_questions * per_cell)

    X = _design_matrix(log2k, log2b, corpus_frac, gold_rank_z, reader_col, spec.with_interaction)
    clusters = np.repeat(np.arange(spec.n_questions), per_cell)

    beta = np.array(
        [
            TRUE_MAIN_EFFECTS["intercept"],
            TRUE_MAIN_EFFECTS["log2K"],
            TRUE_MAIN_EFFECTS["log2B"],
        ]
        + ([delta] if spec.with_interaction else [])
        + [
            TRUE_MAIN_EFFECTS["corpus_fraction"],
            TRUE_MAIN_EFFECTS["gold_rank_z"],
            TRUE_MAIN_EFFECTS["reader_strength"],
        ]
    )
    b_q = rng.normal(0.0, SIGMA_QUESTION, size=spec.n_questions)
    noise = np.repeat(rng.normal(0.0, spec.sigma_run, spec.n_questions), per_cell)
    p_true = expit(X @ beta + b_q[clusters] + noise)
    y_true = rng.binomial(1, p_true).astype(float)
    if spec.judge_flip > 0.0:
        # The GRADER records the wrong verdict. This attenuates the observed effect
        # toward chance rather than inflating the standard error — see the
        # `judge_flip` docstring for why that distinction is the whole point.
        flip = rng.random(y_true.shape[0]) < spec.judge_flip
        y = np.where(flip, 1.0 - y_true, y_true)
    else:
        y = y_true
    return y, X, clusters


def simulate_once(spec: SimSpec, delta: float, rng: np.random.Generator) -> tuple[float, float]:
    """One simulated study. Returns `(interaction_estimate, its std_error)`."""
    y, X, clusters = simulate_dataset(spec, delta, rng)
    beta, se = fit_logit_cluster_robust(y, X, clusters)
    if not spec.with_interaction:
        return float("nan"), float("nan")
    return float(beta[INTERACTION_COLUMN]), float(se[INTERACTION_COLUMN])


def power_at_delta(spec: SimSpec, delta: float, rng: np.random.Generator) -> PowerResult:
    """Empirical power for a planted interaction of size `delta`."""
    ests: list[float] = []
    ses: list[float] = []
    for _ in range(spec.n_sim):
        est, se = simulate_once(spec, delta, rng)
        ests.append(est)
        ses.append(se)
    est_arr = np.asarray(ests)
    se_arr = np.asarray(ses)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = est_arr / np.where(se_arr > 0, se_arr, np.nan)
    power = float(np.mean(np.abs(z) > Z_ALPHA_2))
    return PowerResult(
        delta=delta,
        power=power,
        se=float(np.nanmean(se_arr)),
        mean_estimate=float(np.nanmean(est_arr)),
        n_obs=spec.n_observations,
        reps=spec.n_sim,
    )


def mde_interaction(
    spec: SimSpec, *, seed: int, tol: float = 0.05, max_power_probes: int = 4
) -> dict[str, Any]:
    """Minimum detectable interaction, by bisection on empirical power.

    The bracket is derived from the measured SE rather than fixed. An earlier
    version bisected on `[0, 1.5]`, which is ~375x the true answer at this grid —
    so the search hit its *tolerance floor* (`1.5 / 128 = 0.0117`) and returned
    that, identically, for every noise level. A green number that is an artefact of
    the search interval is worse than no number.

    `tol` is therefore RELATIVE to the initial bracket, and the bracket starts at
    roughly 4x the SE-implied MDE.
    """
    rng = np.random.default_rng(seed)

    # One probe at delta = 0 gives the SE, which sets a sensible bracket.
    se0 = power_at_delta(spec, 0.0, rng).se
    if not np.isfinite(se0) or se0 <= 0:
        return {"mde_log_odds": None, "powered": False, "note": "SE not estimable"}

    hi = max(4.0 * (Z_ALPHA_2 + Z_POWER) * se0, 1e-3)
    for _ in range(max_power_probes):
        if power_at_delta(spec, hi, rng).power >= TARGET_POWER:
            break
        hi *= 2.0
    else:
        return {
            "mde_log_odds": None,
            "powered": False,
            "note": f"power < {TARGET_POWER} even at delta = {hi:.3f} log-odds; "
            "the grid cannot detect this interaction at any plausible size",
            "n_observations": spec.n_observations,
        }

    lower = 0.0
    for _ in range(60):
        mid = 0.5 * (lower + hi)
        if power_at_delta(spec, mid, rng).power >= TARGET_POWER:
            hi = mid
        else:
            lower = mid
        if hi - lower < tol * max(hi, 1e-6):
            break

    final = power_at_delta(spec, hi, rng)
    return {
        "mde_log_odds": round(hi, 5),
        "mde_pp_at_p50": round(_logodds_to_pp(hi), 4),
        "mde_pp_at_p30": round(_logodds_to_pp(hi, math.log(0.3 / 0.7)), 4),
        "power_at_mde": round(final.power, 4),
        "se_log_odds": round(final.se, 6),
        "n_observations": final.n_obs,
        "n_reps_per_point": spec.n_sim,
        "powered": True,
    }


def _logodds_to_pp(delta: float, base_logit: float = 0.0) -> float:
    """Effect size in probability points at a named operating point.

    A log-odds coefficient is not interpretable on its own: the same coefficient
    is ~25 pp at p = 0.5 and ~8 pp at p = 0.9. Both are reported rather than one
    flattering default, because "the interaction MDE" means nothing without the
    accuracy level it applies at.
    """
    return float(100.0 * (expit(base_logit + delta) - expit(base_logit)))


def mde_column(spec: SimSpec, column: int, seed: int) -> dict[str, Any]:
    """MDE for a single coefficient, from its standard error at delta = 0."""
    rng = np.random.default_rng(seed)
    ses: list[float] = []
    for _ in range(spec.n_sim):
        y, X, clusters = simulate_dataset(spec, 0.0, rng)
        _, se_vec = fit_logit_cluster_robust(y, X, clusters)
        ses.append(float(se_vec[column]))
    mean_se = float(np.mean(ses))
    return {
        "se_log_odds": round(mean_se, 6),
        "mde_log_odds": round(Z_ALPHA_2 * mean_se, 4),
        "mde_pp_at_p50": round(_logodds_to_pp(Z_ALPHA_2 * mean_se), 4),
    }


def mde_for_stratum(n_items: int, *, seed: int, n_sim: int = 20) -> dict[str, Any]:
    """MDE for an exploratory stratum, e.g. one `question_type` at n = 30.

    PRD §R9.9 calls these out explicitly because they are the analyses most likely
    to be over-read: a per-type curve on 30 items has an enormous standard error,
    and reporting its p-value without its MDE is how a null stratum becomes a
    spurious finding.

    Two standard errors are reported, not one, because they answer different
    questions and quoting either alone misleads:

    - `se_pp` — the plain binomial SE at p = 0.5. This is the number PRD §R9.9
      quotes for n = 30 ("SE ~ 9pp"), and it is reproduced here exactly.
    - `se_pp_clustered` — inflated by 1.25, because the three runs per cell are not
      three independent items. The MDE uses THIS one.

    Note the gap between SE and MDE. The *detectable effect* at 80% power is
    `(z_{alpha/2} + z_power) ~ 2.8` times the standard error, so the n=30 strata
    can only resolve effects of roughly 30 pp. Quoting the 9pp SE where an MDE
    belongs would overstate a 30-item stratum by nearly threefold.
    """
    _ = seed, n_sim
    se = math.sqrt(0.25 / max(n_items, 1))
    se_clustered = se * CLUSTER_INFLATION
    mde = (Z_ALPHA_2 + Z_POWER) * se_clustered
    return {
        "n_items": n_items,
        "se_pp": round(100.0 * se, 2),
        "se_pp_clustered": round(100.0 * se_clustered, 2),
        "mde_pp_at_p50": round(100.0 * mde, 2),
        "note": (
            "Exploratory strata are reported descriptively. At n=30 the plain "
            "binomial SE is ~9pp (PRD §R9.9) but the minimum detectable effect at "
            "80% power is ~28pp. A p-value quoted without these beside it is not "
            "interpretable."
        ),
    }


# ── cost (PRD §R0.5, §R9.8) ────────────────────────────────────────────────────
def _interp_tokens(k: int, low_k: int, low_tok: float, high_k: int, high_tok: float) -> float:
    """Log-linear interpolation between two measured anchor points."""
    if k <= low_k:
        return low_tok
    if k >= high_k:
        return high_tok
    frac = (math.log2(k) - math.log2(low_k)) / (math.log2(high_k) - math.log2(low_k))
    return math.exp(math.log(low_tok) + frac * (math.log(high_tok) - math.log(low_tok)))


def estimate_cost(
    cells: Sequence[Cell],
    pricing: dict[str, Any],
    *,
    runs_per_cell: int,
    items_per_cell: int,
    judges_per_output: int = 2,
    reader_models: dict[str, str] | None = None,
    pricing_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Planned cost, uncached (PRD §R9.8 — the PRIMARY figure).

    Agent-memory prefixes are not cacheable: the retrieved block is query-specific
    and sits after any cache breakpoint. A cached-cost figure would therefore be a
    fiction, so it is reported alongside only as a contrast.

    `reader_models` maps a reader *family id* to the pricing *model key*. Without
    it every lookup misses and the estimate is silently $0 — a number that looks
    like a triumph and is actually a lookup failure. Unmapped readers are counted
    and reported rather than skipped, so the failure cannot hide.
    """
    models = pricing["models"]
    tok = pricing["token_estimates"]
    judge_cfg = pricing.get("judge", {})
    judge_in = judge_cfg.get("input_tokens_approx", 4000)
    judge_out = judge_cfg.get("output_tokens", 100)
    mapping = reader_models or {}

    per_cell: list[dict[str, Any]] = []
    unmapped: set[str] = set()
    total = 0.0
    total_cached = 0.0

    for cell in cells:
        scale = cell.scale if cell.scale in tok else "S"
        anchors = tok[scale]
        # A K=1 condition has a small prompt; K=64 a large one. Anchors are 1 and 64.
        prompt_tok = _interp_tokens(
            cell.k, 1, anchors["prompt_tokens_at_K1"], 64, anchors["prompt_tokens_at_K64"]
        )
        out_tok = anchors["output_tokens"]

        model_key = mapping.get(cell.reader, cell.reader)
        price = models.get(model_key)
        if price is None:
            unmapped.add(cell.reader)
            per_cell.append(
                {
                    **cell.as_dict(),
                    "error": f"no pricing entry for {model_key!r}",
                }
            )
            continue

        calls = items_per_cell * runs_per_cell
        cell_cost = calls * (prompt_tok * price["input"] + out_tok * price["output"]) / 1e6
        # Judges run on EVERY output (PRD §R8.2) and are easy to forget.
        judge_cost = (
            calls
            * judges_per_output
            * (judge_in * price["input"] + judge_out * price["output"])
            / 1e6
        )
        total += cell_cost + judge_cost
        total_cached += calls * (prompt_tok * price["cache_read"] + out_tok * price["output"]) / 1e6
        per_cell.append(
            {
                **cell.as_dict(),
                "pricing_model": model_key,
                "prompt_tokens": int(prompt_tok),
                "output_tokens": int(out_tok),
                "calls": calls,
                "cost_usd": round(cell_cost + judge_cost, 4),
            }
        )

    priced = [c for c in per_cell if "cost_usd" in c]
    return {
        "_note": "ESTIMATE. Prices are provisional until verified against an invoice.",
        "cost_uncached_total_usd": round(total, 2),  # PRIMARY (PRD §R9.8)
        "cost_cache_adjusted_total_usd": round(total_cached, 2),
        "n_cells": len(cells),
        "n_cells_priced": len(priced),
        "n_calls": len(priced) * items_per_cell * runs_per_cell,
        "unpriced_readers": sorted(unmapped),
        "mean_cost_per_cell_usd": round(total / len(priced), 4) if priced else 0.0,
        "most_expensive_cell": max(priced, key=lambda c: c["cost_usd"], default=None),
        "per_cell": per_cell,
    }


# ── reduced design (PRD §R0.3) ─────────────────────────────────────────────────
def propose_reduced_design(
    design: Any,
    *,
    mde_full: float | None,
    seed: int,
    n_sim: int,
    items_per_cell: int,
    keep_scales: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Propose a SMALLER grid that preserves estimability. Never truncate silently.

    If the interaction is underpowered at the full grid, PRD §R0.3 requires a
    PROPOSAL, not a quiet truncation. The ladder below is ordered by what it costs
    scientifically, cheapest sacrifice first:

      1. drop a B level        — preserves the crossing, halves the cost per cell
      2. drop a scale          — S or M, keeping one
      3. drop runs per cell    — costs seed-variance resolution (PRD §R7.4)

    Dropping a READER is last and is flagged explicitly: PRD §R7.5 makes two
    families mandatory, and 2606.21807 shows a single-reader curve is not
    attributable. It is never proposed without saying so.
    """
    proposals: list[dict[str, Any]] = []

    b_levels = list(design.B_grid)
    if len(b_levels) > 2:
        kept = b_levels[1:]  # drop the tightest budget first
        proposals.append(
            {
                "step": "drop_tightest_b_level",
                "K_grid": list(design.K_grid),
                "B_grid": [(b.value, b.mode) for b in kept],
                "scales": list(design.scales),
                "readers": list(design.readers),
                "runs_per_cell": design.runs_per_cell,
                "rationale": "keeps the full K range and the crossing; removes the "
                "level most likely to be reader-starved rather than retrieval-limited",
            }
        )

    if len(design.scales) > 1:
        for drop in design.scales:
            proposals.append(
                {
                    "step": f"drop_scale_{drop}",
                    "K_grid": list(design.K_grid),
                    "B_grid": [(b.value, b.mode) for b in b_levels],
                    "scales": [s for s in design.scales if s != drop],
                    "readers": list(design.readers),
                    "runs_per_cell": design.runs_per_cell,
                    "rationale": f"keeps both readers and the full grid; drops {drop} "
                    "(G10 depth x scale becomes descriptive only)",
                }
            )

    if design.runs_per_cell > 1:
        proposals.append(
            {
                "step": "halve_runs_per_cell",
                "K_grid": list(design.K_grid),
                "B_grid": [(b.value, b.mode) for b in b_levels],
                "scales": list(design.scales),
                "readers": list(design.readers),
                "runs_per_cell": max(1, design.runs_per_cell // 2),
                "rationale": "cuts cost ~50% but halves seed-variance resolution "
                "(PRD §R7.4); only if the noise floor is already known to be tight",
            }
        )

    proposals.append(
        {
            "step": "drop_reader_family",
            "K_grid": list(design.K_grid),
            "B_grid": [(b.value, b.mode) for b in b_levels],
            "scales": list(design.scales),
            "readers": [design.readers[0]],
            "runs_per_cell": design.runs_per_cell,
            "rationale": "⚠️ NOT RECOMMENDED — PRD §R7.5 requires two reader families. "
            "2606.21807 found compression gain falls with reader baseline and flips 31% "
            "of pairwise rankings, so a single-reader curve is not attributable. Listed "
            "only so the cost of this option is explicit rather than discovered later.",
            "violates_r75": True,
        }
    )

    _ = (mde_full, seed, n_sim, items_per_cell, keep_scales)
    return {
        "principle": "PRD §R0.3 — propose, never truncate. Each proposal keeps the "
        "K x B crossing intact; the crossing IS the contribution.",
        "proposals": proposals,
    }


def _content_tokens(k: int, anchors: dict[str, float]) -> float:
    """Tokens contributed by retrieving top-K memories, at this scale.

    PREFERRED PATH - measured. P1 measured `corpus_chars.mean_per_question` and
    `sessions_per_question.mean`, so the content at depth K is exactly:

        content(K) = min(K, sessions_mean) * tokens_per_session

    with the `min` because you cannot retrieve 64 sessions from a 47.7-session
    corpus: S saturates at K=48 and every K beyond it is the same prompt. A log
    interpolation between two anchors misses that cap entirely, which is how the
    S-scale prompts came out 10x too small.

    FALLBACK - anchors. Log-interpolated between K=1 and K=64. Kept so unit tests
    can supply a tiny two-point curve, and for configs that have not been
    re-derived yet. `validate_token_anchors` exists to make that fallback loud
    rather than silent.
    """
    sessions = anchors.get("sessions_mean")
    per_session = anchors.get("tokens_per_session")
    if sessions is not None and per_session is not None:
        return min(float(k), float(sessions)) * float(per_session)
    return _interp_tokens(k, 1, anchors["prompt_tokens_at_K1"], 64, anchors["prompt_tokens_at_K64"])


def _budget_tokens(b: float, b_mode: str, corpus_tokens: float | None) -> float:
    """Resolve a budget level to ABSOLUTE tokens. The mode travels with the value.

    PRD §R1.7: `4096` tokens is a different fraction of a 3M-character corpus than of
    a 300K-character one, so `log2(B)` stops meaning one thing across the grid.
    That is exactly why the mode is carried rather than assumed.

    `corpus_fraction` was in the schema from the start and was NEVER implemented -
    the value was read as a token count, so a fraction of 0.02 produced a 0.02-token
    prompt and the cell silently rendered as the bare question. A declared-but-dead
    config option is worse than a missing one, because the schema says it is valid.
    """
    if b_mode == "absolute_tokens":
        return float(b)
    if b_mode == "corpus_fraction":
        if corpus_tokens is None:
            raise ValueError(
                f"b_mode={b_mode!r} needs `corpus_tokens` for this scale; add "
                "`token_estimates.<scale>.corpus_tokens` to configs/pricing/latest.yaml"
            )
        if not 0.0 < b <= 1.0:
            raise ValueError(f"corpus_fraction must be in (0, 1], got {b}")
        return float(b) * float(corpus_tokens)
    if b_mode == "per_memory_tokens":
        raise ValueError(
            "per_memory_tokens is resolved per memory, not as one total; call "
            "_prompt_tokens_per_memory, not _budget_tokens"
        )
    raise ValueError(f"unknown b_mode {b_mode!r}")


def _prompt_tokens_per_memory(k: int, b: float, anchors: dict[str, float]) -> float:
    """Tokens for a PER-MEMORY budget: retrieve top-K, give each memory B tokens.

        n        = min(K, sessions_mean)          # you cannot retrieve what isn't there
        per_mem  = min(tokens_per_session, B)    # each memory is capped at B
        prompt   = n * per_mem

    WHY THIS IS THE MODE THE DESIGN NEEDS
    -------------------------------------
    Under the old truncating semantics `prompt = min(content(K), B)`, the budget was
    a CUTOFF ON K expressed in tokens. The two axes were the same axis in different
    units, and the number of *active* budget levels was monotone in K — a staircase
    that provably caps at 2 distinct outcomes at K=1 and K=2 no matter which B
    values you choose. A factor that cannot vary independently cannot carry an
    interaction, and the K x B interaction is the whole claim.

    Under a per-memory budget the prompt scales WITH K: `K x B`. Doubling K doubles
    the prompt at any B, and changing B changes the density without changing how
    many memories were retrieved. The two axes become independent, so the crossing
    is real and the interaction is estimable.

    It is also the more meaningful manipulation. "Retrieve 16 memories and show the
    first 512 tokens of each" is a genuinely different retrieval strategy from
    "retrieve 4 memories and show all of them", and it is the density question the
    cited compression literature (2606.21807: compression gain shrinks as the reader
    gets stronger) is actually about. Truncating the concatenated list cannot
    distinguish those two strategies — it produces the same prompt for both.
    """
    sessions = anchors.get("sessions_mean")
    per_session = anchors.get("tokens_per_session")
    if sessions is None or per_session is None:
        raise ValueError(
            "b_mode='per_memory_tokens' needs `sessions_mean` and `tokens_per_session` "
            "for this scale. They come from P1's measured corpus; add them to "
            "`token_estimates.<scale>` in configs/pricing/latest.yaml."
        )
    n = min(float(k), float(sessions))
    return n * min(float(per_session), float(b))


def _prompt_tokens(
    k: int,
    b: float | None,
    anchors: dict[str, float],
    overhead: float = 0.0,
    *,
    b_mode: str = "absolute_tokens",
    corpus_tokens: float | None = None,
) -> float:
    """Prompt size for a cell: K fills the budget, the budget caps it, plus overhead.

    `prompt = min(tokens_from_K_memories, budget_tokens) + overhead`

    WHY THIS IS NOT JUST `_interp_tokens(k)`
    ---------------------------------------
    It was, until 2026-10-13. `B` was accepted by `Cell` and never read by this
    function, so every prompt size in the project was a function of K alone. Two
    consequences, both silent:

    - **Feasibility was wrong.** M/K=64 was modelled at 120k tokens, so cells
      were rejected against a ceiling that the budget level had already ruled out.
      B=65536 caps that prompt at 65,536 no matter how deep K goes.
    - **The compute estimate was inflated.** `estimate_compute` calls this too, so
      the M-scale cost was priced at up to 120k tokens per call for cells whose
      budget level allows at most 65,536. The 314.5 h figure carries that error.

    The trap is that `K` alone looks sufficient — retrieval depth obviously
    determines how much text goes into the prompt. It does not, once there is a
    token budget: K is how much you *ask* for, B is how much you *get*. When K's
    content exceeds B, raising K changes nothing, which is precisely the
    saturation effect PRD §R9 is about. Modelling that with K alone would make
    the interaction unfalsifiable.

    `b=None` means "no budget level declared" and returns the uncapped size; it
    exists for tests that probe K's contribution in isolation, not for cells.
    """
    content = _content_tokens(k, anchors)
    if b is None:
        return content + overhead
    if b_mode == "per_memory_tokens":
        return _prompt_tokens_per_memory(k, float(b), anchors) + overhead
    return min(content, _budget_tokens(b, b_mode, corpus_tokens)) + overhead


def duplicate_conditions(
    cells: Sequence[Cell],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    """Cells that render a BYTE-IDENTICAL prompt to another cell. (PRD §R5.6)

    PRD §R4.4: a crossed cell retrieves top-K **in rank order**, then truncates to B
    tokens. So the rendered prompt is:

        top-K in full            if content(K) <= B
        the FIRST B tokens of    if content(K) >  B
        one shared rank-ordered stream

    The second branch is the trap. Once the budget truncates, *every* K past the
    cutoff produces the same B-token prefix of the same stream, so K=4/B=4096 and
    K=64/B=4096 are the same prompt. And when content(K) is below B the budget is
    inert, so B=4096 and B=65536 are the same prompt for the same K.

    **Two independent budget levels are therefore one level.** The grid still
    reports `collision_free` because `Cell.key` includes B, so nothing else in the
    pipeline can catch this — every duplicate is charged full compute AND counted
    as an independent condition, which is precisely how the power analysis comes to
    overstate the design.

    Note there is no `prompt_overhead` parameter: the per-prompt overhead is
    additive and identical for every cell, so it shifts all prompts by the same
    constant and can never change whether two cells are equal. A parameter here
    would imply it could.

    Reported, never silently dropped: dropping a duplicate cell is a scientific
    decision about which arms exist, and this function's job is to make it visible
    rather than to make it.
    """
    tok = pricing["token_estimates"]

    def signature(cell: Cell) -> tuple:
        anchors = tok[cell.scale if cell.scale in tok else "S"]
        content = _content_tokens(cell.k, anchors)
        # `reader` is part of the identity: (K,B,scale,gemma) and (K,B,scale,qwen)
        # render the SAME prompt but are different conditions — the reader is the
        # treatment. Collapsing them would understate the distinct-condition count
        # and make the number look worse than it is.
        if cell.b_mode == "per_memory_tokens":
            # prompt = min(K, sessions) * min(per_session, B): monotone in both K
            # and B, so only genuine corpus exhaustion (K beyond the session count)
            # makes two cells collide.
            return (cell.reader, cell.scale, "per_mem", cell.k, int(cell.b))
        budget = _budget_tokens(cell.b, cell.b_mode, anchors.get("corpus_tokens"))
        if content <= budget:
            # Budget inert: the prompt is the full top-K, so B does not appear.
            return (cell.reader, cell.scale, "untruncated", cell.k)
        # Budget binds: the prompt is a B-token prefix, so only B matters.
        return (cell.reader, cell.scale, "truncated", int(budget))

    groups: dict[tuple, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(signature(cell), []).append(cell)

    duplicate_sets: list[dict[str, Any]] = [
        {
            "members": [c.as_dict() for c in members],
            "n_members": len(members),
            "n_redundant": len(members) - 1,
        }
        for members in groups.values()
        if len(members) > 1
    ]
    n_redundant = sum(d["n_redundant"] for d in duplicate_sets)
    return {
        "n_cells": len(cells),
        "n_unique_conditions": len(groups),
        "n_duplicate_sets": len(duplicate_sets),
        "n_redundant_cells": n_redundant,
        "redundant_fraction": round(n_redundant / len(cells), 4) if cells else 0.0,
        "duplicate_sets": duplicate_sets,
        "why_invisible": (
            "Cell.key includes B, so cell_key_collisions() reports these grids as "
            "collision-free. The keys differ; the rendered prompt does not."
        ),
    }


def validate_token_anchors(
    pricing: dict[str, Any],
    measured_chars_per_question: dict[str, float],
    sessions_per_question: dict[str, float],
    *,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    """Cross-check the configured token anchors against P1's measured corpus.

    The anchors in `token_estimates` were order-of-magnitude placeholders written
    before P1 measured anything. P1 measured `corpus_chars.mean_per_question` and
    `sessions_per_question.mean` for both scales, which makes the anchors
    CHECKABLE — and they do not survive the check.

    S and M sessions are the same size (~10.2k chars each: 489,297/47.7 and
    4,838,071/475.3). A config claiming S/K=64 is 12,000 tokens while M/K=64 is
    120,000 is asserting a 10x difference in per-session size that is not in the
    data. That single number decides which cells fit in VRAM.

    `chars_per_token` defaults to 4.0 (the usual English BPE figure). It is an
    ASSUMPTION and is reported as such: verifying it needs the real tokenizer,
    which needs the model files, which live on the execution box. The check below
    is robust to it — a 5x discrepancy is far outside any plausible ratio.
    """
    tok = pricing["token_estimates"]
    findings: list[dict[str, Any]] = []

    for scale, chars in sorted(measured_chars_per_question.items()):
        if scale not in tok or scale not in sessions_per_question:
            continue
        sessions = sessions_per_question[scale]
        chars_per_session = chars / sessions
        implied_tokens_per_session = chars_per_session / chars_per_token
        corpus_tokens = chars / chars_per_token

        declared_k1 = float(tok[scale]["prompt_tokens_at_K1"])
        # content(K=64) is bounded by the corpus: you cannot retrieve 64 sessions
        # from a 47.7-session corpus.
        declared_k64 = float(tok[scale]["prompt_tokens_at_K64"])
        expected_k64 = min(64.0, sessions) * implied_tokens_per_session

        for label, declared, expected in (
            ("K1", declared_k1, implied_tokens_per_session),
            ("K64", declared_k64, expected_k64),
        ):
            if expected <= 0:
                continue
            ratio = declared / expected
            findings.append(
                {
                    "scale": scale,
                    "corpus_chars_per_question": round(chars, 1),
                    "sessions_per_question": round(sessions, 2),
                    "chars_per_session": round(chars_per_session, 1),
                    "implied_corpus_tokens": round(corpus_tokens, 1),
                    "anchor": f"prompt_tokens_at_{label}",
                    "declared": round(declared, 1),
                    "expected_from_measured_corpus": round(expected, 1),
                    "ratio": round(ratio, 3),
                    "ok": 0.5 <= ratio <= 2.0,
                }
            )

    bad = [f for f in findings if not f["ok"]]
    return {
        "chars_per_token_assumed": chars_per_token,
        "findings": findings,
        "n_off": len(bad),
        "consistent": not bad,
        "verdict": (
            f"{len(bad)} of {len(findings)} anchors are more than 2x off the corpus "
            "measured in P1. Feasibility and cost computed from them are wrong."
            if bad
            else "all anchors consistent with the measured corpus"
        ),
    }


@dataclass(frozen=True, slots=True)
class ReaderLimit:
    """What one reader can actually serve, and which ceiling binds.

    Two ceilings exist and conflating them is how a run dies at hour 14:

    1. `context_window` — the checkpoint's declared `max_position_embeddings`.
       A property of the weights. Every local candidate declares 131k-262k, so
       this rarely binds.
    2. `max_prompt_tokens` — how many prompt tokens actually FIT once the weights
       are resident. KV cache grows linearly with prompt length, and an 8 GB card
       holding a ~9B model at 4-bit has ~2.5 GB left, which is 20k-66k tokens.

    **THIS is the binding constraint, and it is a property of the hardware, not
    the model.** A 262k-window reader on an 8 GB card serves ~66k tokens. Reading
    the window alone marks every long cell feasible and then OOMs during P5.
    """

    id: str
    context_window: int
    max_prompt_tokens: int
    weight_gb: float | None = None
    kv_mb_per_1k: float | None = None
    vram_gb: float | None = None

    @property
    def ceiling(self) -> int:
        return min(self.context_window, self.max_prompt_tokens)

    @property
    def binding(self) -> str:
        """Which ceiling will actually fail first. Reported per cell, never assumed."""
        return "vram" if self.max_prompt_tokens < self.context_window else "window"

    @classmethod
    def from_window(cls, reader_id: str, window: int) -> ReaderLimit:
        """A reader with no separate VRAM ceiling in play (large GPU, or hosted)."""
        return cls(id=reader_id, context_window=window, max_prompt_tokens=window)

    @classmethod
    def from_vram(
        cls,
        reader_id: str,
        *,
        context_window: int,
        weight_gb: float,
        kv_mb_per_1k: float,
        vram_gb: float = 8.0,
        overhead_gb: float = DEFAULT_HEADROOM_GB,
    ) -> ReaderLimit:
        """Derive the ceiling from published architecture numbers, not a guess."""
        return cls(
            id=reader_id,
            context_window=context_window,
            max_prompt_tokens=vram_max_prompt_tokens(
                weight_gb=weight_gb,
                kv_mb_per_1k=kv_mb_per_1k,
                vram_gb=vram_gb,
                overhead_gb=overhead_gb,
            ),
            weight_gb=weight_gb,
            kv_mb_per_1k=kv_mb_per_1k,
            vram_gb=vram_gb,
        )


def vram_max_prompt_tokens(
    *,
    weight_gb: float,
    kv_mb_per_1k: float,
    vram_gb: float = 8.0,
    overhead_gb: float = DEFAULT_HEADROOM_GB,
) -> int:
    """Largest prompt that fits in VRAM, from KV cache arithmetic.

        KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes/elem
        prompt_tokens  = (vram_gb - weight_gb - overhead_gb) / kv_mb_per_1k * 1000

    Lives here rather than in config because the number must be DERIVED from
    published architecture facts and a stated card size. A hand-typed capacity in
    YAML is a claim nobody re-checks, and this is the number that decides which
    cells exist.

    Returns 0 when the weights alone do not fit — a reader that cannot be resident
    at all is not "slow", it cannot run, and that must not read as a large number.

    `overhead_gb` used to default to a LOCAL 0.5 while `hardware.vram` used 3.0 for
    the same physical quantity, and both docstrings claimed the same four
    components. The design path then marked deep cells feasible that the runtime
    capacity preflight rejected — the D11 contradiction, and it was a duplicated
    constant rather than a modelling disagreement. The default is now
    `DEFAULT_HEADROOM_GB` so there is one number in the project.
    """
    if kv_mb_per_1k <= 0:
        raise ValueError(f"kv_mb_per_1k must be positive, got {kv_mb_per_1k}")
    available_mb = (vram_gb - weight_gb - overhead_gb) * 1024.0
    if available_mb <= 0:
        return 0
    return int(available_mb / kv_mb_per_1k * 1000.0)


def headroom_that_fits(
    *,
    weight_gb: float,
    kv_mb_per_1k: float,
    required_prompt_tokens: int,
    vram_gb: float = 8.0,
) -> float:
    """The largest headroom under which `required_prompt_tokens` still fits.

    D11 asks whether the whole crossed grid is executable on this card, and the
    answer is not a yes or a no: it is a threshold. This returns the largest
    `overhead_gb` that still serves the deepest prompt this reader must run, so the
    decision is made against a number rather than a preference.

    Negative means the weights alone do not fit at any headroom.
    """
    if kv_mb_per_1k <= 0:
        raise ValueError(f"kv_mb_per_1k must be positive, got {kv_mb_per_1k}")
    kv_gb = required_prompt_tokens * kv_mb_per_1k / 1024.0 / 1024.0
    return vram_gb - weight_gb - kv_gb


def context_feasibility(
    cells: Sequence[Cell],
    readers: Mapping[str, ReaderLimit],
    pricing: dict[str, Any],
    *,
    prompt_overhead: int | None = None,
) -> dict[str, Any]:
    """Which cells each reader cannot serve, and which ceiling rejects them.

    This is a SCIENTIFIC constraint, not an engineering inconvenience. PRD §R3.2
    says a degenerate cell is rejected, never silently truncated — and a prompt
    that overflows the reader's capacity is exactly that: the reader would be
    answering a different question than the one asked.

    Both ceilings are checked and the binding one is named per cell. Reporting
    only the model window is the failure mode: every local candidate declares
    131k-262k, so a window-only check finds nothing wrong and the 8 GB card OOMs
    during P5 instead.

    `prompt_overhead` is the fixed per-prompt cost (instructions, question,
    generation priming). It is added to EVERY cell, so a cell that fits only
    without it does not fit — the alternative is marking a cell feasible and
    failing on the last token.
    """
    tok = pricing["token_estimates"]
    overhead = float(
        prompt_overhead
        if prompt_overhead is not None
        else pricing.get("prompt_overhead_tokens", 500)
    )

    infeasible: dict[str, list[dict[str, Any]]] = {}
    max_prompt: dict[str, int] = {}

    for reader, limit in readers.items():
        worst = 0
        for cell in cells:
            if cell.reader != reader:
                continue
            anchors = tok[cell.scale if cell.scale in tok else "S"]
            prompt = int(
                _prompt_tokens(
                    cell.k,
                    cell.b,
                    anchors,
                    overhead,
                    b_mode=cell.b_mode,
                    corpus_tokens=anchors.get("corpus_tokens"),
                )
            )
            worst = max(worst, prompt)
            ceiling = limit.ceiling
            if prompt > ceiling:
                infeasible.setdefault(reader, []).append(
                    {
                        **cell.as_dict(),
                        "prompt_tokens": prompt,
                        "context_window": limit.context_window,
                        "vram_capacity_tokens": limit.max_prompt_tokens,
                        "ceiling": ceiling,
                        "blocked_by": limit.binding,
                        "overflow_by": prompt - ceiling,
                    }
                )
        max_prompt[reader] = worst

    return {
        "readers": {
            r: {
                "context_window": lim.context_window,
                "vram_capacity_tokens": lim.max_prompt_tokens,
                "ceiling": lim.ceiling,
                "blocked_by": lim.binding,
                "largest_prompt": max_prompt.get(r, 0),
                "weight_gb": lim.weight_gb,
                "kv_mb_per_1k": lim.kv_mb_per_1k,
            }
            for r, lim in sorted(readers.items())
        },
        "infeasible_cells": infeasible,
        "n_infeasible": sum(len(v) for v in infeasible.values()),
        "prompt_overhead_tokens": overhead,
        "policy": "reject_cell — never truncate silently (PRD §R3.2)",
    }


def estimate_compute(
    cells: Sequence[Cell],
    pricing: dict[str, Any],
    *,
    runs_per_cell: int,
    items_per_cell: int,
    judges_per_output: int = 2,
) -> dict[str, Any]:
    """Wall-clock GPU hours — the real budget once there is no paid API.

    With local weights every call is a full prefill (there is no provider cache),
    so prefill dominates and the M scale dominates the schedule. A $0 budget with
    an unestimated runtime is how a three-week compute bill arrives in week 8.

    Throughput figures are **assumed**, not measured. They are stated so the guide
    can challenge them; `design-power` must be re-run against measured numbers
    before P5.
    """
    cb = pricing.get("compute_budget")
    if not cb:
        return {"available": False}

    tp = cb["throughput_assumed"]
    prefill_rate = float(tp["prefill_tokens_per_sec"])
    decode_rate = float(tp["decode_tokens_per_sec"])
    out_reader = float(cb["output_tokens"]["reader"])
    out_judge = float(cb["output_tokens"]["judge"])
    judge_cfg = pricing.get("judge", {})
    judge_in = float(judge_cfg.get("input_tokens_approx", 4000))
    tok = pricing["token_estimates"]
    # Same overhead as `context_feasibility`: a cost estimate that omits it is
    # not comparable to a feasibility check that includes it.
    overhead = float(pricing.get("prompt_overhead_tokens", 500))

    per_reader: dict[str, dict[str, float]] = {}
    total_prefill = 0.0
    total_decode = 0.0
    judge_prefill = 0.0
    judge_decode = 0.0
    n_calls = 0

    for cell in cells:
        anchors = tok[cell.scale if cell.scale in tok else "S"]
        prompt = _prompt_tokens(
            cell.k,
            cell.b,
            anchors,
            overhead,
            b_mode=cell.b_mode,
            corpus_tokens=anchors.get("corpus_tokens"),
        )
        calls = items_per_cell * runs_per_cell
        n_calls += calls
        bucket = per_reader.setdefault(cell.reader, {"hours": 0.0, "calls": 0.0})
        hours = calls * (prompt / prefill_rate + out_reader / decode_rate) / 3600.0
        bucket["hours"] += hours
        bucket["calls"] += calls
        total_prefill += calls * prompt
        total_decode += calls * out_reader
        # Judges run on EVERY output (PRD §R8.2) — the dominant hidden cost.
        judge_prefill += calls * judges_per_output * judge_in
        judge_decode += calls * judges_per_output * out_judge

    reader_hours = sum(b["hours"] for b in per_reader.values())
    judge_hours = (judge_prefill / prefill_rate + judge_decode / decode_rate) / 3600.0
    # PRD §R8.6: a second full pass of the same system under the same judge.
    audit_hours = judge_hours * float(judge_cfg.get("reliability_audit_multiplier", 1.0))
    total = reader_hours + judge_hours + audit_hours

    budget = float(cb["max_wall_clock_hours"])
    return {
        "available": True,
        "assumptions": {
            "prefill_tokens_per_sec": prefill_rate,
            "decode_tokens_per_sec_aggregate": decode_rate,
            "dtype": cb.get("dtype"),
            "batch_size": cb.get("batch_size"),
            "status": "ASSUMED — must be benchmarked on the target machine before P5",
        },
        "n_calls": n_calls,
        "reader_hours": round(reader_hours, 1),
        "judge_hours": round(judge_hours, 1),
        "judge_audit_hours": round(audit_hours, 1),
        "total_hours": round(total, 1),
        "total_days": round(total / 24.0, 1),
        "max_wall_clock_hours": budget,
        "within_budget": total <= budget,
        "multiple_of_budget": round(total / budget, 2) if budget else None,
        "per_reader": {
            k: {"hours": round(v["hours"], 1), "calls": int(v["calls"])}
            for k, v in sorted(per_reader.items())
        },
        "judge_share_of_total": round(judge_hours / total, 3) if total else 0.0,
    }


def estimability(
    cells: Sequence[Cell],
    *,
    degenerate: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Gate branch (iv): is the interaction estimable at EVERY cell?

    A cell that is degenerate contributes nothing to the interaction while still
    consuming budget and, worse, silently flattening the term. `degeneracy_fraction`
    is supplied by the P3 per-question guards; until then this reports the
    structural facts it CAN establish.
    """
    collisions = cell_key_collisions(cells)
    by_scale: dict[str, set[tuple[int, float]]] = {}
    for c in cells:
        by_scale.setdefault(c.scale, set()).add((c.k, round(c.b, 6)))
    return {
        "n_cells": len(cells),
        "n_unique_keys": len({c.key for c in cells}),
        "key_collisions": [list(k) for k in collisions],
        "collision_free": not collisions,
        "cells_per_scale": {s: len(v) for s, v in sorted(by_scale.items())},
        "crossing_complete": all(len(v) > 1 for v in by_scale.values()),
        "degeneracy_fraction": degenerate or {},
        "estimable": not collisions,
    }
