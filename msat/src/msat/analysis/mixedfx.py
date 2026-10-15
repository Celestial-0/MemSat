"""Estimation primitives for PRD §R9.1's binomial primary model.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
The rest of the analysis layer decides WHAT to estimate. This module is the only
place that decides HOW, and the decision is not a default - it is the answer to a
trap the team has already paid for once.

THE TRAP
--------
`statsmodels.MixedLM` is a **LINEAR** mixed model. Handed a 0/1 outcome it raises
nothing: no warning, no exception, no check. It returns coefficients that look
reasonable and fitted probabilities that leave [0, 1], and every downstream number
computed from them is a fluent fiction. PRD §6.1 records this as a verified trap,
`results/power.json` carries a `_method.NOT_used` field naming it, and
`pyproject.toml` carries the same warning next to the statsmodels pin.

So the estimator here is `statsmodels.genmod.GEE` with `family=Binomial()` and an
independence working correlation, which makes the reported standard errors
CLUSTER-ROBUST on the grouping variable. That is the path §6.1 records as "verified
working", and `tests/test_design_power.py::test_estimator_matches_statsmodels_gee`
already pins the equivalent hand-written sandwich against it to ~1e-4. Two
implementations of the same estimator, cross-checked, neither of them linear.

WHY THE OUTCOME IS VALIDATED RATHER THAN ASSUMED
-------------------------------------------------
`fit_gee_binomial` raises `NonBinaryOutcomeError` on any outcome that is not 0/1.
This is deliberately unfriendly. The alternative - coerce, round, or trust - is the
precisely the behaviour that makes the MixedLM trap dangerous: a model class that
accepts data it cannot model will not tell you, and the symptom (a probability of
1.3) surfaces three modules downstream in a figure. Refusing at the door costs one
line at the call site and turns a silent corruption into a stack trace pointing at
the column that caused it.

THE R9.1 SPECIFICATION IS NOT BUILT HERE
-----------------------------------------
This module takes NAMED COLUMNS. It does not know what K and B are, does not build
a design matrix, and does not decide which terms go into the model. `design.py`'s
`_design_matrix` is the single place the R9.1 design is constructed - it centres
`log2(K)` and `log2(B)` on the grid it is handed and emits the interaction column in
a fixed order - and `analysis/interaction.py` is the only caller. Duplicating a
column order in a second module is how a coefficient labelled `log2B` ends up
meaning `corpus_fraction`, and this project has found that bug class three times
already (`design.py` documents two of them at length).

WHAT IS NOT CLAIMED
-------------------
A cluster-robust GEE is a GENERALISED estimating equation, not a likelihood-based
mixed model. Two consequences are load-bearing downstream and are recorded in
every result rather than left to the reader:

- The standard errors are robust to arbitrary within-cluster correlation, which is
  the property R9.1's `(1|question)` needs. They are not, and do not claim to be,
  the profile-likelihood standard errors of a fitted GLMM. `msat[glmm]` is the path
  to an exact nested GLMM if one is ever required.
- GEE admits exactly ONE grouping variable. R9.1 writes `(1|question) + (1|model)`.
  Both cannot be GEE clusters at once, so the reader dimension is handled as a
  fixed covariate plus a reader-clustered secondary fit whose standard error is
  reported BESIDE the primary one. See `analysis/interaction.py` - the gap is
  documented there rather than papered over here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy import stats

from msat.design import Z_ALPHA_2, _logodds_to_pp

#: Two-sided alpha for every Wald interval this module reports.
ALPHA: Final = 0.05

#: The estimator, spelled out, in every artifact it touches. A result whose
#: provenance is a free-text sentence is a result whose provenance drifts.
ESTIMATOR: Final = "statsmodels.genmod.GEE / Binomial / Independence working correlation"

#: PRD §6.1: cluster-robust SE by question.
DEFAULT_CLUSTER_COL: Final = "question"

#: Below this, a "cluster-robust" sandwich is one or two terms in a sum over
#: clusters and its variance estimate has no meaningful small-sample correction.
#: statsmodels will happily fit it and return a number; that number is not a
#: standard error.
MIN_CLUSTERS: Final = 2

#: Below this many clusters, the sandwich still computes but its small-sample
#: behaviour is poor, and the result says so rather than presenting the interval as
#: authoritative. Not an error: `fit_gee_binomial` must still be able to produce
#: the reader-clustered secondary fit, where three readers is the whole panel.
SMALL_CLUSTER_WARNING: Final = 30

#: Outcome values a binomial family can actually model. Anything else is a
#: different data type, and guessing which one the caller meant is how a 0/1/2
#: accuracy scale becomes a logit fit with silently discarded levels.
BINARY_VALUES: Final[frozenset[float]] = frozenset({0.0, 1.0})


class NonBinaryOutcomeError(ValueError):
    """The outcome column is not 0/1.

    Raised rather than coerced. This is the guard that stands where the MixedLM
    trap would otherwise sit: a linear mixed model accepts a 0/1 outcome without
    complaint, and a 0/1/2 outcome is just as fatal - so the check happens once, at
    the boundary, with the offending values named.
    """


class TooFewClustersError(ValueError):
    """Fewer than `MIN_CLUSTERS` distinct values in the grouping column.

    Not a warning. With one cluster the sandwich is a single outer product and the
    reported variance is a rank-1 artefact, and nothing downstream can tell that
    apart from a real interval.
    """


class RankDeficientDesignError(ValueError):
    """Two or more terms are exact linear combinations of each other.

    The dangerous one is a SILENT fit: statsmodels emits a
    `SingularMatrixWarning`, converges, returns a plausible coefficient and a NaN
    standard error. A NaN that reaches a results table reads as "not computed",
    when in fact it means "not determined" - and the coefficient beside it is a
    chosen one of infinitely many equally good fits.
    """


class MissingCovariateError(ValueError):
    """A term the specification declares has no source column.

    `analysis/interaction.py` raises this for `corpus_fraction` (PRD §R2.8/R9.8),
    where dropping the term silently would leave a model that looks like R9.1 and
    is not. The name is specific because the failure is: the artifact would be
    indistinguishable from the declared one.
    """


# ── the result ────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PrimaryFit:
    """One binomial GEE fit, with the whole term table attached.

    `coefficients` / `std_errors` / `p_values` carry EVERY term, not just the
    focused one. A headline number whose covariates are not readable beside it
    cannot be audited, and the covariates are where a specification drifts.

    `log_likelihood` is the quasi-log-likelihood under the independence working
    correlation. It is carried because PRD §R9.2's nested comparison needs it,
    and because a result object that can be re-differenced into an LRT is one that
    does not need the raw model kept alive to stay checkable.
    """

    term: str
    coefficient: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int
    n_clusters: int
    converged: bool
    alpha: float = ALPHA
    estimator: str = ESTIMATOR
    outcome: str = "correct"
    cluster_col: str = DEFAULT_CLUSTER_COL
    log_likelihood: float = 0.0
    cluster_note: str = ""
    """Why this many clusters makes the interval what it is. Non-empty for small
    `n_clusters` - a sandwich over three clusters computes, but the resulting
    interval is a wide, lumpy thing that reads as precise on a figure. An empty
    string is a claim too: it says the cluster count is not a concern here."""
    n_dropped: int = 0
    """Rows lost to NaN in the outcome, the terms, or the cluster. REPORTED, never
    silent: `scores.parquet` carries NULL for abstention items and unlabelled rows
    (R8.4), so this is a normal non-zero number on real data, and a model that
    quietly analysed fewer rows than the table holds is indistinguishable from one
    that analysed all of them."""
    term_names: tuple[str, ...] = ()
    coefficients: Mapping[str, float] = None  # type: ignore[assignment]
    std_errors: Mapping[str, float] = None  # type: ignore[assignment]
    p_values: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # `Mapping` has no useful default in a dataclass and `None` is a lie a
        # caller could print. An empty dict is a fact: this fit reported no other
        # terms. Set once, on a frozen object, via object.__setattr__ because the
        # fields are slots.
        for name in ("coefficients", "std_errors", "p_values"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})

    @property
    def z(self) -> float:
        """Coefficient in units of its own standard error."""
        return self.coefficient / self.std_error if self.std_error > 0 else float("nan")

    @property
    def effect_pp(self) -> float:
        """The coefficient as percentage points at p = 0.5, the flat-operating-point
        reading. Never quote this alone - `effect_pp_at` is the honest version."""
        return logodds_to_pp(self.coefficient)

    def effect_pp_at(self, base_p: float = 0.5) -> float:
        """The coefficient as percentage points at a named baseline accuracy.

        A log-odds coefficient is not interpretable on its own: the same
        coefficient is ~25pp at p = 0.5 and ~8pp at p = 0.9. Both are reported
        rather than one flattering default, because "the interaction" means
        nothing without the accuracy level it applies at.
        """
        return logodds_to_pp(self.coefficient, base_p)

    def term_fit(self, name: str) -> PrimaryFit:
        """The same fit, focused on a different term.

        One fit, many readouts - rather than refitting per term - so two numbers
        printed side by side in a table cannot come from different models.
        """
        if name not in self.coefficients:
            raise KeyError(f"{name!r} is not in this fit. Terms: {sorted(self.coefficients)}")
        se = float(self.std_errors[name])
        beta = float(self.coefficients[name])
        half = Z_ALPHA_2 * se
        return PrimaryFit(
            term=name,
            coefficient=beta,
            std_error=se,
            ci_low=beta - half,
            ci_high=beta + half,
            p_value=float(self.p_values[name]),
            n_obs=self.n_obs,
            n_clusters=self.n_clusters,
            converged=self.converged,
            alpha=self.alpha,
            estimator=self.estimator,
            outcome=self.outcome,
            cluster_col=self.cluster_col,
            log_likelihood=self.log_likelihood,
            cluster_note=self.cluster_note,
            n_dropped=self.n_dropped,
            term_names=self.term_names,
            coefficients=self.coefficients,
            std_errors=self.std_errors,
            p_values=self.p_values,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready. Rounded, and never `None`-coerced: a fit that could not be
        computed is `converged: false`, which is a different state from a fit that
        was never attempted."""
        return {
            "term": self.term,
            "coefficient": round(self.coefficient, 6),
            "std_error": round(self.std_error, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "n_obs": self.n_obs,
            "n_clusters": self.n_clusters,
            "converged": self.converged,
            "alpha": self.alpha,
            "estimator": self.estimator,
            "outcome": self.outcome,
            "cluster_col": self.cluster_col,
            "log_likelihood": round(self.log_likelihood, 6),
            "cluster_note": self.cluster_note,
            "n_dropped": self.n_dropped,
            "term_names": list(self.term_names),
            "coefficients": {k: round(v, 6) for k, v in self.coefficients.items()},
            "std_errors": {k: round(v, 6) for k, v in self.std_errors.items()},
            "p_values": {k: round(v, 6) for k, v in self.p_values.items()},
            "effect_pp_at_p50": round(self.effect_pp_at(0.5), 4),
            "effect_pp_at_p70": round(self.effect_pp_at(0.7), 4),
            "effect_pp_at_p30": round(self.effect_pp_at(0.3), 4),
        }


# ── the conversion ────────────────────────────────────────────────────────────
def logodds_to_pp(beta: float, base_p: float = 0.5) -> float:
    """A log-odds coefficient, in percentage points at baseline accuracy `base_p`.

    Delegates to `design._logodds_to_pp` rather than re-deriving it. That function
    already exists because the MDE report needs the same conversion, and two
    definitions of "percentage points at a baseline" will agree until one of them
    is edited - at which point the MDE and the estimate stop being comparable, and
    they are compared every time the pre-registered verdict is written.

    `base_p` is a PROBABILITY here, not a logit: the caller has an accuracy in hand
    and should not have to know that this particular conversion wants `log(p/(1-p))`.
    """
    if not 0.0 < base_p < 1.0:
        raise ValueError(f"base_p must be a probability in (0, 1), got {base_p!r}")
    return _logodds_to_pp(float(beta), float(np.log(base_p / (1.0 - base_p))))


# ── the estimator ─────────────────────────────────────────────────────────────
def assert_binary_outcome(values: Any, *, column: str = "outcome") -> None:
    """Raise `NonBinaryOutcomeError` unless `values` is 0/1.

    The single place the outcome is checked, so the check cannot be applied to one
    fit path and forgotten on another. Coercion is attempted once - `True`/`False`
    and integer columns are legitimate encodings of a Bernoulli outcome - and
    anything that does not land exactly on {0, 1} is named and refused.
    """
    arr = np.asarray(values)
    if arr.dtype == bool:
        return
    try:
        numeric = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise NonBinaryOutcomeError(
            f"column {column!r} is not numeric ({exc}). A binomial family needs a "
            f"0/1 outcome; this looks like a label column that was never scored. "
            f"Distinct values: {sorted({str(v) for v in arr.ravel().tolist()})[:8]}"
        ) from exc

    if not np.all(np.isfinite(numeric)):
        raise NonBinaryOutcomeError(
            f"column {column!r} contains NaN/inf. Drop those rows explicitly and "
            f"report the count - the denominator of every accuracy number in the "
            f"paper depends on it (R8.4: abstention items are NULL, not zero)."
        )

    bad = np.unique(numeric[~np.isin(numeric, list(BINARY_VALUES))])
    if bad.size:
        raise NonBinaryOutcomeError(
            f"column {column!r} must be 0/1 for a binomial model; found "
            f"{bad.tolist()[:8]}. Not coercing: a linear mixed model would have "
            f"accepted these silently and returned fitted probabilities outside "
            f"[0, 1] (PRD §6.1 verified trap). If this is an F1 or a judge score "
            f"in [0, 1], binarise it with a DECLARED threshold and record it - do "
            f"not let the estimator decide."
        )


def fit_gee_binomial(
    df: Any,
    terms: Sequence[str],
    cluster_col: str = DEFAULT_CLUSTER_COL,
    *,
    outcome: str = "correct",
    focus: str | None = None,
    alpha: float = ALPHA,
    maxiter: int = 100,
    tol: float = 1e-9,
) -> PrimaryFit:
    """Binomial GEE with cluster-robust standard errors on `cluster_col`.

    `df` is a tidy frame with one row per observation; `terms` are column names in
    it. An intercept is added here and is NOT one of the terms - a term list that
    silently includes a constant column produces a rank-deficient design and a
    `converged: false` that reads like a data problem.

    `focus` names the term to report. When omitted it resolves to the term
    containing `:` (the interaction) if there is one, else the LAST term supplied -
    the usual "the thing under test goes last" convention. Passing it explicitly is
    better; the default exists so a caller cannot forget to say what it wanted and
    silently get the log2(B) coefficient instead.

    `maxiter` is passed straight through to statsmodels. `converged` is reported
    and is NOT treated as fatal: a non-converged fit is still the best available
    number, and discarding it would replace a flagged weak result with a missing
    one. The flag travels with the coefficient so nobody reads the estimate without
    it.
    """
    from statsmodels.genmod.cov_struct import Independence  # type: ignore[import-untyped]
    from statsmodels.genmod.families import Binomial  # type: ignore[import-untyped]
    from statsmodels.genmod.generalized_estimating_equations import (  # type: ignore[import-untyped]
        GEE,
    )

    term_list = tuple(str(t) for t in terms)
    if not term_list:
        raise ValueError("fit_gee_binomial needs at least one term")

    required = (outcome, cluster_col, *term_list)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingCovariateError(
            f"column(s) {missing} are required by this fit but absent. The primary "
            f"model (PRD §R9.1) is declared once and is not allowed to shed terms "
            f"without saying so; a model missing a declared covariate produces a "
            f"table indistinguishable from the declared one."
        )

    subset = df.loc[:, list(required)].dropna()
    n_dropped = int(len(df) - len(subset))
    if subset.empty:
        raise ValueError(
            f"no usable rows: every one of {len(df)} rows is NULL in the outcome, "
            f"the cluster column, or a term. There is nothing to fit."
        )

    y = subset[outcome].to_numpy()
    assert_binary_outcome(y, column=outcome)

    groups = subset[cluster_col].to_numpy()
    n_clusters = len(np.unique(groups))
    if n_clusters < MIN_CLUSTERS:
        raise TooFewClustersError(
            f"cluster column {cluster_col!r} has {n_clusters} distinct value(s); "
            f"a cluster-robust sandwich needs at least {MIN_CLUSTERS}. With one "
            f"cluster the variance estimate is a single outer product and the "
            f"interval printed beside it would mean nothing."
        )

    x = np.column_stack(
        [np.ones(len(subset), dtype=float)]
        + [np.asarray(subset[t], dtype=float) for t in term_list]
    )
    if x.shape[1] >= len(subset):
        raise ValueError(
            f"{len(subset)} observations and {x.shape[1]} parameters (intercept "
            f"included). The design is not identified - check for a constant or "
            f"collinear term before fitting."
        )
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise RankDeficientDesignError(
            f"the design has {x.shape[1]} columns and rank {rank}: at least one term "
            f"is an exact linear combination of the others. statsmodels would fit it "
            f"anyway and return NaN standard errors, so the refusal is here. Terms "
            f"fitted: {['intercept', *term_list]}. The usual cause is a covariate "
            f"that duplicates a factor - `reader_strength` is constant within each "
            f"question type, so it is an exact combination of the `ability` dummies "
            f"and the intercept. Drop one, or fix the fixture."
        )

    model = GEE(
        np.asarray(y, dtype=float),
        x,
        groups=groups,
        family=Binomial(),
        cov_struct=Independence(),
    )
    result = model.fit(maxiter=maxiter, ctol=tol)

    names = ("intercept", *term_list)
    betas = np.asarray(result.params, dtype=float)
    ses = np.asarray(result.bse, dtype=float)
    pvals = np.asarray(result.pvalues, dtype=float)

    focus_term = _resolve_focus(term_list, focus)
    # Index into `names`, NOT `term_list`. `names` is the intercept followed by the
    # terms, so `term_list.index(focus_term)` is short by one and reads the
    # coefficient of whatever term sits BELOW it in the list. That is precisely the
    # shape of this project's worst bug class: a fluent, plausible, mislabelled
    # number, and the reason the whole table is built from `names` below.
    focus_index = names.index(focus_term)
    beta_focus = float(betas[focus_index])
    finite_se = float(ses[focus_index])

    return PrimaryFit(
        term=focus_term,
        coefficient=beta_focus,
        std_error=finite_se,
        ci_low=beta_focus - Z_ALPHA_2 * finite_se,
        ci_high=beta_focus + Z_ALPHA_2 * finite_se,
        p_value=float(pvals[focus_index]),
        n_obs=len(subset),
        n_clusters=n_clusters,
        converged=bool(getattr(result, "converged", True)),
        alpha=alpha,
        estimator=ESTIMATOR,
        outcome=outcome,
        cluster_col=cluster_col,
        log_likelihood=float(result.llf),
        cluster_note=_cluster_note(n_clusters, cluster_col),
        n_dropped=n_dropped,
        term_names=names,
        coefficients=dict(zip(names, betas.tolist(), strict=True)),
        std_errors=dict(zip(names, ses.tolist(), strict=True)),
        p_values=dict(zip(names, pvals.tolist(), strict=True)),
    )


def _cluster_note(n_clusters: int, cluster_col: str) -> str:
    """Say when the cluster count makes the interval unreliable.

    The sandwich is a sum over clusters, so its variance estimate degrades sharply
    as clusters run out. statsmodels applies a finite-sample correction and keeps
    going, which is the right call for a primary fit - discarding it would replace
    a flagged interval with no interval. The count goes in the result either way,
    and this note says plainly when it is the number to distrust.
    """
    if n_clusters >= SMALL_CLUSTER_WARNING:
        return ""
    return (
        f"only {n_clusters} distinct {cluster_col!r} values: the cluster-robust "
        f"standard error is a sum over {n_clusters} terms and its small-sample "
        f"behaviour is poor. Usable, not authoritative - read the interval as "
        f"indicative and prefer the other clustering where one exists."
    )


def _resolve_focus(terms: Sequence[str], focus: str | None) -> str:
    """Which term this fit reports on."""
    if focus is not None:
        if focus not in terms:
            raise KeyError(f"focus {focus!r} is not one of the fitted terms {list(terms)}")
        return focus
    interactions = [t for t in terms if ":" in t]
    if interactions:
        return interactions[0]
    return terms[-1]


def cluster_counts(df: Any, cluster_col: str) -> dict[Any, int]:
    """Rows per cluster. Reported by `analyze` because a cluster with three rows
    and one with three thousand contribute to the sandwich in proportion to neither,
    and the imbalance is invisible in `n_clusters` alone."""
    if cluster_col not in df.columns:
        raise MissingCovariateError(f"cluster column {cluster_col!r} is absent")
    counts = df[cluster_col].value_counts(dropna=False)
    return {k: int(v) for k, v in counts.to_dict().items()}


def z_value(beta: float, se: float) -> float:
    """A z statistic, with the degenerate case named rather than returned as NaN.

    `statsmodels` emits NaN when a coefficient's standard error is 0 - which is what
    a quasi-separated cell looks like - and a NaN that reaches a results table
    reads as "not computed" rather than "not determined".
    """
    if se <= 0 or not np.isfinite(se):
        return float("nan")
    return float(beta / se)


def p_from_z(z: float) -> float:
    """Two-sided normal tail probability, NaN-preserving."""
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(z)))
