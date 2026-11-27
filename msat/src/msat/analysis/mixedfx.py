"""Estimation primitives for the binomial primary model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy import stats
from scipy.special import expit

from msat.design import Z_ALPHA_2

ALPHA: Final = 0.05

ESTIMATOR: Final = "statsmodels.genmod.GEE / Binomial / Independence working correlation"

DEFAULT_CLUSTER_COL: Final = "question"

MIN_CLUSTERS: Final = 2

SMALL_CLUSTER_WARNING: Final = 30

BINARY_VALUES: Final[frozenset[float]] = frozenset({0.0, 1.0})


class NonBinaryOutcomeError(ValueError):
    """Raised when the outcome column contains values other than 0 and 1."""


class TooFewClustersError(ValueError):
    """Raised when the grouping column has fewer than `MIN_CLUSTERS` distinct values."""


class RankDeficientDesignError(ValueError):
    """Raised when terms are exact linear combinations, which would otherwise fit silently."""


class MissingCovariateError(ValueError):
    """Raised when a declared term has no source column."""


@dataclass(frozen=True, slots=True)
class PrimaryFit:
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
    n_dropped: int = 0
    term_names: tuple[str, ...] = ()
    coefficients: Mapping[str, float] = None  # type: ignore[assignment]
    std_errors: Mapping[str, float] = None  # type: ignore[assignment]
    p_values: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("coefficients", "std_errors", "p_values"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})

    @property
    def z(self) -> float:
        return self.coefficient / self.std_error if self.std_error > 0 else float("nan")

    @property
    def effect_pp(self) -> float:
        return logodds_to_pp(self.coefficient)

    def effect_pp_at(self, base_p: float = 0.5) -> float:
        return logodds_to_pp(self.coefficient, base_p)

    def term_fit(self, name: str) -> PrimaryFit:
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


def logodds_to_pp(beta: float, base_p: float = 0.5) -> float:
    if not 0.0 < base_p < 1.0:
        raise ValueError(f"base_p must be a probability in (0, 1), got {base_p!r}")
    base_logit = float(np.log(base_p / (1.0 - base_p)))
    return float(100.0 * (expit(base_logit + float(beta)) - expit(base_logit)))


def assert_binary_outcome(values: Any, *, column: str = "outcome") -> None:
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
    if n_clusters >= SMALL_CLUSTER_WARNING:
        return ""
    return (
        f"only {n_clusters} distinct {cluster_col!r} values: the cluster-robust "
        f"standard error is a sum over {n_clusters} terms and its small-sample "
        f"behaviour is poor. Usable, not authoritative - read the interval as "
        f"indicative and prefer the other clustering where one exists."
    )


def _resolve_focus(terms: Sequence[str], focus: str | None) -> str:
    if focus is not None:
        if focus not in terms:
            raise KeyError(f"focus {focus!r} is not one of the fitted terms {list(terms)}")
        return focus
    interactions = [t for t in terms if ":" in t]
    if interactions:
        return interactions[0]
    return terms[-1]


def cluster_counts(df: Any, cluster_col: str) -> dict[Any, int]:
    if cluster_col not in df.columns:
        raise MissingCovariateError(f"cluster column {cluster_col!r} is absent")
    counts = df[cluster_col].value_counts(dropna=False)
    return {k: int(v) for k, v in counts.to_dict().items()}


def z_value(beta: float, se: float) -> float:
    if se <= 0 or not np.isfinite(se):
        return float("nan")
    return float(beta / se)


def p_from_z(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(z)))
