"""PRD §R9.1 - the primary model, DECLARED ONCE, and the endpoints built on it.

    correct ~ log2(K)*log2(B) + log2(K) + log2(B) + corpus_fraction + gold_rank
             + ability + reader_strength + (1|question) + (1|model)

`configs/preregistration.yaml` says of this specification: *"PRD §R9.1 - declared
once, here and nowhere else."* That is a constraint on the CODEBASE, not a figure
of speech: this module owns the single `InteractionSpec` instance
(`PRIMARY_SPEC`), and every other surface - the power simulation's column order,
`results.md`, the figures - reads the term list from it instead of restating it.
A second copy of this formula is a second model, and a reviewer who finds two
cannot tell which one the headline number came from.

THE ESTIMATOR IS GEE, AND THAT IS NOT A DEFAULT
-----------------------------------------------
`statsmodels.MixedLM` is LINEAR and silently accepts a 0/1 outcome - no error,
coefficients that cannot mean what they appear to, fitted probabilities outside
[0, 1]. PRD §6.1 records it as a verified trap. See `analysis/mixedfx.py` for the
estimator and the refusal that stands where the trap would otherwise sit.

R9.1'S TWO RANDOM INTERRECTS, AND WHAT GEE CAN ACTUALLY DO WITH THEM
--------------------------------------------------------------------
GEE takes exactly ONE grouping variable, so `(1|question) + (1|model)` cannot be
fitted as two GEE clusters. The reading implemented here, and the one to challenge
if you disagree:

- `(1|question)` is the PRIMARY cluster. This is what §6.1 verified, and it is the
  correlation structure that matters: three replicates of one cell on one question
  are not three independent observations.
- `(1|model)` is realised as the continuous `reader_strength` covariate PLUS a
  reader-clustered SECONDARY fit whose standard error is reported beside the
  primary one (`primary_analysis`). Not absorbed silently and not dropped: with two
  or three reader families a random intercept is barely identified anyway, and the
  reader-clustered SE is the honest conservative alternative. When the reader panel
  has fewer than `MIN_CLUSTERS` levels the secondary is reported as
  unavailable, with the count, rather than as a number that looks like evidence.

This is a documented approximation of the written specification, not a fulfilment
of it. An exact nested GLMM is `msat[glmm]`.

CENTRING, AND WHY THE HEADLINE NUMBER SURVIVES IT
--------------------------------------------------
`design._design_matrix` centres `log2(K)` and `log2(B)` on the grid it is handed.
That is mandatory, not cosmetic: `log2(B)` spans 7..9 under the per-memory budget,
and a 0.25-per-log2-unit main effect would sit the baseline near 60% before any
effect is planted. Centring with BOTH main effects retained leaves the interaction
coefficient numerically INVARIANT - shifting `k -> k - kappa` and `b -> b - mu`
gives `k*b = k'b' + mu*k' + kappa*b' + kappa*mu`, so `beta_kb` is unchanged and
only the main effects move. The headline number therefore means the same thing in
this parameterisation as in the raw one written in the PRD, and the code verifies
that reconstruction rather than trusting it.

R9.5: WHY STRATIFIED IS PRIMARY AND POOLED IS SECONDARY
-------------------------------------------------------
The ceiling is strongly TYPE-CONDITIONAL (see `ceiling.py`, and the measured
`results/ceiling.json`): single-hop and multi-hop questions have different evidence
structures and therefore different saturation points, by design. A pooled
saturation point is a WEIGHTED AVERAGE OF TWO REGIMES. It is not a noisier version
of the stratified answer - it is an answer to a different question, namely "what
does the mixture saturate at", and it is reported only so that the mixture is
visible, never because it is the quantity of interest. `strata_report` labels each
curve's `role` in its own output for that reason.

R9.10: THE LEAKAGE CHECK IS WIRED, NOT RE-LITIGATED
---------------------------------------------------
`leakage_check` runs `analysis/leakage.py`'s classifier unchanged. What it adds is
the applicability question. For Arm B the classifier separates the conditions
PERFECTLY BY CONSTRUCTION, because `n_memories` *is* the label - Arm B holds token
count fixed and varies memory count. That is not a leak (the count is printed in
the prompt as a numbered list); it is the manipulation restated. `tests/
test_no_leakage.py::test_arm_b_as_designed_is_separable_by_memory_count` pins the
separability and `24_open_decisions.md` D10 records the decision. So this module
DETECTS that case, labels R9.10 not-applicable, and checks the assertion that does
apply to Arm B - the token match in `arms/padding.size_leakage_matrix`. The
deletion-style construction R9.10 was written about still fails the check loudly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

# pandas ships no stubs and is used here through four calls (`get_dummies`,
# `groupby().mean()`, `dropna()`, `to_numpy()`). `DataFrame` is passed around as
# `Any` for the same reason `design.py` stays in numpy: the frame is a transport,
# not a domain object, and pinning its type buys no safety at the boundary.
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats

from msat.analysis.leakage import (
    DEFAULT_AUROC_THRESHOLD,
    LeakageFeatures,
    LeakageReport,
    assert_no_leakage,
)
from msat.analysis.mixedfx import (
    ALPHA,
    MissingCovariateError,
    PrimaryFit,
    TooFewClustersError,
    fit_gee_binomial,
    logodds_to_pp,
)
from msat.arms.padding import TOKEN_TARGET_TOLERANCE, size_leakage_matrix
from msat.design import _design_matrix, grid_center, k_delta_in_pp, k_log2_step, k_step_index

# ── the declared vocabulary ───────────────────────────────────────────────────
# Names are constants, not literals repeated at call sites. `power.py` labels its
# own columns `log2K` / `log2B` and this module used to be free to spell them
# differently; two spellings of one column is how a report ends up with a row
# labelled `log2B` holding the reader strength.

TERM_K: Final = "log2K"
TERM_B: Final = "log2B"
TERM_INTERACTION: Final = "log2K:log2B"
TERM_QUADRATIC_K: Final = "log2K_sq"

R91_DECLARED: Final = (
    "correct ~ log2(K)*log2(B) + log2(K) + log2(B) + corpus_fraction + gold_rank "
    "+ ability + reader_strength + (1|question) + (1|model)"
)
"""PRD §R9.1 verbatim. Carried in every report so the artifact states the model
it actually fitted rather than the model someone remembers."""

#: `corpus_fraction` is the one covariate whose absence is FATAL (PRD §R2.8/G12,
#: R9.8 "a covariate in EVERY model"). Dropping it silently leaves a table that is
#: indistinguishable from the declared specification while answering a different
#: question - how accuracy depends on depth once you have also controlled for how
#: much of the corpus you are reading, which is most of the effect.
REQUIRED_TERMS: Final[frozenset[str]] = frozenset({"corpus_fraction"})

#: R9.1's full term list, `ability` included (it arrives as reference-coded
#: dummies). The yardstick `AnalysisFrame.dropped_terms` is measured against, so
#: it has to be the COMPLETE declaration rather than whatever a given `InteractionSpec`
#: happens to list - otherwise a short spec reports nothing dropped and a reader
#: cannot tell it was fitted on less than R9.1.
R91_TERMS: Final[tuple[str, ...]] = (
    TERM_INTERACTION,
    TERM_K,
    TERM_B,
    "corpus_fraction",
    "gold_rank",
    "reader_strength",
    "ability",
)

#: Marginal-gain threshold for `K_delta`, in percentage points PER UNIT OF log2(K).
#: Not per grid step - see R9.4 and `saturation_stats`.
DELTA_THRESHOLD_PP: Final = 0.5

#: `K_95` asks for the smallest depth reaching this share of the best OBSERVED
#: accuracy on this grid.
K95_FRACTION: Final = 0.95

#: `Frontier K` restricts its argmax to depths within this share of `A_max`.
#: Cost-efficiency over a depth that gave up 5% of the accuracy is not efficiency.
FRONTIER_FLOOR: Final = 0.99

#: How close to 0 or 1 an AUROC has to be before the feature is called separating,
#: and how close to 0.5 before it is called uninformative. Two-sided on purpose:
#: which condition is "label 1" is an artefact of `sorted()` on the condition names,
#: so a one-sided rule would fire for the arm whose memories are the LARGER ones and
#: not for the other - the check would pass on half the designs and flag the other
#: half for the same reason.
PERFECT_SEPARATION_TOLERANCE: Final = 0.01
LENGTH_MATCHED_AUROC_TOLERANCE: Final = 0.10


def _separates(auroc: float) -> bool:
    """True when a feature separates the conditions in EITHER direction."""
    return min(auroc, 1.0 - auroc) <= PERFECT_SEPARATION_TOLERANCE


def _at_chance(auroc: float) -> bool:
    """True when a feature carries no information about the condition."""
    return abs(auroc - 0.5) <= LENGTH_MATCHED_AUROC_TOLERANCE


# ── the specification ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class InteractionSpec:
    """PRD §R9.1 as data. One instance exists: `PRIMARY_SPEC`."""

    terms: tuple[str, ...] = (
        TERM_INTERACTION,
        TERM_K,
        TERM_B,
        "corpus_fraction",
        "gold_rank",
        "reader_strength",
    )
    outcome: str = "correct"
    cluster_col: str = "question"
    k_col: str = "K"
    b_col: str = "B"
    corpus_fraction_col: str = "corpus_fraction"
    gold_rank_col: str = "gold_rank"
    reader_strength_col: str = "reader_strength"
    ability_col: str = "ability"
    reader_col: str = "reader"
    alpha: float = ALPHA
    include_ability: bool = True
    quadratic_term: str = TERM_QUADRATIC_K
    declared: str = R91_DECLARED

    def term_names(self, ability_levels: Sequence[str] = ()) -> tuple[str, ...]:
        """The fitted term list, including one dummy per ability level.

        Reference-coded (first level dropped): with `ability` as a full dummy set
        plus an intercept the design is singular, and statsmodels reports a
        rank-deficient design with a `converged` flag rather than an error. The
        dropped level is reported in `AnalysisFrame.dropped_levels` so the encoding
        is auditable.
        """
        return (*self.terms, *(f"ability[{level}]" for level in ability_levels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "requirement": "PRD §R9.1",
            "terms": list(self.terms),
            "outcome": self.outcome,
            "cluster_col": self.cluster_col,
            "random_effects": {
                "question": "GEE cluster (independence working correlation)",
                "model": (
                    f"reader_strength covariate + reader-clustered secondary SE; "
                    f"GEE admits one grouping variable, so this is an APPROXIMATION "
                    f"of `{self.reader_col}` being a second random intercept"
                ),
            },
            "centred": "log2(K) and log2(B) on the observed grid (design.grid_center)",
            "estimator": "statsmodels.genmod.GEE / Binomial / cluster-robust",
            "not_used": "statsmodels.MixedLM - LINEAR, silently accepts a 0/1 outcome",
            "alpha": self.alpha,
            "quadratic_term": self.quadratic_term,
        }


PRIMARY_SPEC: Final = InteractionSpec()
"""THE single declaration of PRD §R9.1. Import from here; do not rebuild one."""


# ── the analysis frame ────────────────────────────────────────────────────────
@dataclass
class AnalysisFrame:
    """R9.1's design, named, plus the context `fit_primary` needs afterwards."""

    frame: Any
    """Columns: the interaction and main terms, the covariates, and the raw `K`,
    `B`, `question`, `ability`, `reader` columns the secondary analyses group by."""
    spec: InteractionSpec
    k_center: float
    b_center: float
    term_names: tuple[str, ...]
    dropped_terms: tuple[str, ...] = ()
    dropped_levels: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": len(self.frame),
            "term_names": list(self.term_names),
            "k_center": round(self.k_center, 6),
            "b_center": round(self.b_center, 6),
            "dropped_terms": list(self.dropped_terms),
            "dropped_levels": list(self.dropped_levels),
            "notes": list(self.notes),
        }


def _standardise(values: Any) -> Any:
    """z-score a column, refusing a constant one.

    `design._design_matrix` names its fifth column `gold_rank_z` and expects a
    standardised input. Standardising also puts `gold_rank` on the same log-odds
    scale as the other covariates, which is what makes its coefficient readable
    next to `reader_strength`'s. A constant column returns NaN here, and NaN in the
    design makes the fit silently drop it - so it is an error instead.
    """
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std())
    if sd <= 0.0:
        raise ValueError(
            "a covariate is constant across every row, so its coefficient is not "
            "identifiable. Standardising produced NaN and the fit would have "
            "dropped the term without saying so."
        )
    return (arr - float(arr.mean())) / sd


def build_analysis_frame(
    df: Any,
    spec: InteractionSpec = PRIMARY_SPEC,
    *,
    strict: bool = True,
    with_quadratic: bool = False,
) -> AnalysisFrame:
    """Turn a tidy scored frame into R9.1's named design.

    `design._design_matrix` is called rather than reimplemented: it owns the column
    ORDER and the centring, and a second implementation of either would be a second
    model. The interaction column it returns at index 3 is reconstructed here and
    compared, because a silent column-order change would relabel a coefficient
    without changing a single number - which is the cheapest and most damaging bug
    this codebase can host.

    `strict=True` (default) raises `MissingCovariateError` for any declared term
    with no source column. `strict=False` drops the non-REQUIRED ones and records
    them in `dropped_terms`, which is the only sanctioned way to shed a term and
    the report always carries it.
    """
    columns = set(df.columns)
    k = np.asarray(df[spec.k_col], dtype=float)
    b = np.asarray(df[spec.b_col], dtype=float)
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(b)):
        raise ValueError(
            f"{spec.k_col!r}/{spec.b_col!r} contain NaN. log2 is undefined there and "
            f"the row cannot be placed on the grid."
        )
    if np.any(k <= 0) or np.any(b <= 0):
        raise ValueError(
            f"{spec.k_col!r}/{spec.b_col!r} must be strictly positive - log2(0) is "
            f"undefined and K=0 is not a retrieval condition."
        )

    def _column(name: str, source: str, *, required: bool) -> Any:
        """Fetch a declared term's source column, or raise.

        The record of what was dropped is NOT kept here - it is derived at the end
        from the fitted term list, so a term that survived the data but was left out
        of the specification is still reported. Accumulating failures as they happen
        would miss that case.
        """
        if source in columns:
            return df[source]
        if strict or required:
            raise MissingCovariateError(
                f"{name!r} is declared in PRD §R9.1 but column {source!r} is absent. "
                f"corpus_fraction is REQUIRED in every model (PRD §R2.8/G12, R9.8); "
                f"the others are declared terms and dropping one silently produces a "
                f"table indistinguishable from the specification it claims to be. "
                f"Join it in, or pass strict=False to drop it ON THE RECORD."
            )
        return None

    corpus_frac = _column("corpus_fraction", spec.corpus_fraction_col, required=True)
    gold_rank = _column("gold_rank", spec.gold_rank_col, required=False)
    strength = _column("reader_strength", spec.reader_strength_col, required=False)

    gold_rank_z = (
        None
        if gold_rank is None
        else _standardise(gold_rank.to_numpy()[df[spec.k_col].notna().to_numpy()])
    )
    strength_arr = None if strength is None else np.asarray(strength, dtype=float)

    xmat = _design_matrix(
        np.log2(k),
        np.log2(b),
        np.asarray(corpus_frac, dtype=float),
        np.zeros_like(k) if gold_rank_z is None else gold_rank_z,
        np.zeros_like(k) if strength_arr is None else strength_arr,
        True,
    )
    k_center = grid_center(np.log2(k))
    b_center = grid_center(np.log2(b))

    # Cross-check the reused design matrix against the formula it claims to encode.
    reconstructed = (np.log2(k) - k_center) * (np.log2(b) - b_center)
    if not np.allclose(xmat[:, 3], reconstructed, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            "design._design_matrix column 3 is not (log2K - kc)*(log2B - bc). The "
            "column ORDER changed under this module and every label after it is "
            "wrong. Fix the order in one place rather than here."
        )

    # A term the frame cannot produce must not appear as a zero column: `_design_matrix`
    # has no slot for it, and a zero column is rank-deficient by construction.
    data: dict[str, Any] = {
        "log2K": np.log2(k) - k_center,
        "log2B": np.log2(b) - b_center,
        TERM_INTERACTION: xmat[:, 3],
        "corpus_fraction": xmat[:, 4],
    }
    if gold_rank_z is not None:
        data["gold_rank"] = gold_rank_z
    if strength_arr is not None:
        data["reader_strength"] = strength_arr

    out = pd.DataFrame(data, index=df.index)

    ability_levels: list[str] = []
    dropped_levels: list[str] = []
    if spec.include_ability:
        if spec.ability_col not in columns:
            if strict:
                raise MissingCovariateError(
                    f"'ability' is declared in PRD §R9.1 but column "
                    f"{spec.ability_col!r} is absent. Question type is the stratum "
                    f"R9.5 makes primary; without it there are no strata to stratify "
                    f"by. Pass strict=False to drop it on the record."
                )
        else:
            raw = df[spec.ability_col].astype(str)
            levels = sorted(raw.unique())
            ability_levels = levels[1:] if len(levels) > 1 else []
            dropped_levels = levels[:1]
            dummies = pd.get_dummies(raw, prefix="ability", dtype=float)
            for level in ability_levels:
                out[f"ability[{level}]"] = dummies[f"ability_{level}"].to_numpy()

    if with_quadratic:
        # Built on the CENTRED log2(K) so the quadratic does not re-introduce the
        # raw-scale offset that centring exists to remove.
        out[spec.quadratic_term] = (np.log2(k) - k_center) ** 2

    for col in (spec.cluster_col, spec.outcome, spec.k_col, spec.reader_col):
        if col in columns:
            out[col] = df[col].to_numpy()
    if spec.ability_col in columns:
        out[spec.ability_col] = df[spec.ability_col].to_numpy()
    out["log2K_raw"] = np.log2(k)
    out["log2B_raw"] = np.log2(b)

    term_names = tuple(t for t in spec.term_names(ability_levels) if t in out.columns)

    # `dropped_terms` is derived from the FITTED list against R9.1's full
    # declaration, not accumulated as failures happen. The distinction matters: a
    # term the frame could not produce and a term this `spec` never declared are
    # different facts, and only the first is a defect. Anything R9.1 declares and
    # this fit does not carry is written down, because a model that quietly shed
    # `corpus_fraction` produces a table indistinguishable from the declared one.
    #
    # `ability` arrives as reference-coded dummies, so it is "carried" when any
    # `ability[...]` column is present - matching on the bare name would report
    # R9.1's own covariate as dropped on every single fit.
    def _carried(term: str) -> bool:
        return term in term_names or (
            term == "ability" and any(t.startswith("ability[") for t in term_names)
        )

    dropped_terms = tuple(t for t in R91_TERMS if not _carried(t))

    notes = tuple(
        [
            "log2(K) and log2(B) are centred on the OBSERVED grid. The interaction "
            "coefficient is invariant to this shift when both main effects are "
            "retained, so the headline number matches the raw-scale specification.",
        ]
        + (
            [f"dropped {dropped_levels[0]!r} as the reference level of `ability`"]
            if dropped_levels
            else []
        )
    )

    return AnalysisFrame(
        frame=out,
        spec=spec,
        k_center=k_center,
        b_center=b_center,
        term_names=term_names,
        dropped_terms=dropped_terms,
        dropped_levels=tuple(dropped_levels),
        notes=tuple(notes),
    )


# ── R9.1 ──────────────────────────────────────────────────────────────────────
def fit_primary(
    df: Any,
    spec: InteractionSpec = PRIMARY_SPEC,
    *,
    strict: bool = True,
    maxiter: int = 100,
) -> PrimaryFit:
    """Fit PRD §R9.1 and return the INTERACTION term.

    The focus is the interaction because that is the headline (G9); every other
    coefficient is on the returned object, readable without a refit.

    Raises `MissingCovariateError` (a `ValueError`) rather than fitting a smaller
    model. See `build_analysis_frame`.
    """
    built = build_analysis_frame(df, spec, strict=strict)
    return fit_gee_binomial(
        built.frame,
        built.term_names,
        spec.cluster_col,
        outcome=spec.outcome,
        focus=TERM_INTERACTION,
        alpha=spec.alpha,
        maxiter=maxiter,
    )


def fit_reader_clustered(
    df: Any,
    spec: InteractionSpec = PRIMARY_SPEC,
    *,
    strict: bool = True,
) -> PrimaryFit:
    """The same model, clustered by READER instead of by question.

    The honest half of the `(1|model)` story (see the module docstring): GEE has
    one grouping variable, so the reader dimension is measured by refitting. With
    two or three reader families this raises `TooFewClustersError` - which is the
    correct answer, not a failure to route around.
    """
    built = build_analysis_frame(df, spec, strict=strict)
    return fit_gee_binomial(
        built.frame,
        built.term_names,
        spec.reader_col,
        outcome=spec.outcome,
        focus=TERM_INTERACTION,
        alpha=spec.alpha,
    )


def primary_analysis(
    df: Any,
    spec: InteractionSpec = PRIMARY_SPEC,
    *,
    strict: bool = True,
    baseline_p: float = 0.5,
) -> dict[str, Any]:
    """The R9.1 block `analyze` writes: the fit, both SEs, and the reading of it.

    The reader-clustered SE is a FIELD, not a footnote, including when it is
    unavailable - a table showing one standard error next to two reader families
    invites the reader to assume the reader dimension was accounted for.
    """
    built = build_analysis_frame(df, spec, strict=strict)
    primary = fit_primary(df, spec, strict=strict)

    reader_note = ""
    try:
        secondary = fit_reader_clustered(df, spec, strict=strict)
        reader_block: dict[str, Any] = {
            "available": True,
            "std_error": round(secondary.std_error, 6),
            "n_clusters": secondary.n_clusters,
            "note": secondary.cluster_note,
        }
    except TooFewClustersError as exc:
        n_readers = int(df[spec.reader_col].nunique()) if spec.reader_col in df.columns else 0
        reader_block = {"available": False, "std_error": None, "n_clusters": n_readers}
        reader_note = str(exc)

    return {
        "specification": spec.to_dict(),
        "frame": built.to_dict(),
        "primary": primary.to_dict(),
        "reader_clustered_secondary": reader_block,
        "reader_clustered_note": reader_note,
        "interaction_pp": {
            f"p{int(baseline_p * 100)}": round(primary.effect_pp_at(baseline_p), 4),
            "p30": round(primary.effect_pp_at(0.3), 4),
            "p70": round(primary.effect_pp_at(0.7), 4),
        },
        "mde_note": (
            "Compare this against the MDE from `msat design-power` "
            "(results/power.json -> interaction.mde_log_odds). A p-value alone does "
            "not say whether the design could have detected what it did not find."
        ),
    }


# ── R9.2 — monotone vs inverted-U ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LrtResult:
    """A nested-model comparison, with every modelling choice stated on it."""

    stat: float
    df: int
    p_value: float
    df_null: int
    df_alt: int
    quadratic_coefficient: float
    at_boundary: bool
    converged: bool
    nesting: str
    boundary_handling: str
    random_effect_handling: str
    quasi: bool = True
    verdict: str = ""
    terms_null: tuple[str, ...] = ()
    terms_alt: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": round(self.stat, 6),
            "df": self.df,
            "p_value": round(self.p_value, 6),
            "df_null": self.df_null,
            "df_alt": self.df_alt,
            "quadratic_coefficient": round(self.quadratic_coefficient, 6),
            "at_boundary": self.at_boundary,
            "converged": self.converged,
            "nesting": self.nesting,
            "boundary_handling": self.boundary_handling,
            "random_effect_handling": self.random_effect_handling,
            "quasi": self.quasi,
            "verdict": self.verdict,
            "terms_null": list(self.terms_null),
            "terms_alt": list(self.terms_alt),
            "aic_not_used": (
                "AIC is a model-selection criterion, not a test: comparing it "
                "between nested models gives no p-value and no error rate. PRD §R9.2 "
                "forbids it, so it is not computed here."
            ),
        }


def lrt_monotone_vs_inverted_u(
    df: Any,
    spec: InteractionSpec = PRIMARY_SPEC,
    *,
    strict: bool = True,
) -> LrtResult:
    """LRT of the monotone K response against an inverted-U (R9.2).

    PARAMETERISED, both halves:

        null  : R9.1 exactly (linear in log2 K)
        alt   : R9.1 + `log2K_sq`, the centred quadratic

    df = 1, exactly: the alternative is the null with ONE coefficient fixed at
    zero, which is regular nesting.

    BOUNDARY. An inverted-U means the linear predictor is CONCAVE in log2(K), so
    the quadratic coefficient is constrained to be <= 0 and the null value 0 sits on
    the BOUNDARY of that constraint. When the unconstrained fit prefers a convex
    term (`beta_sq > 0`) the constrained optimum is at the boundary and the
    nominal chi-square_1 null distribution is wrong: the mixture
    `0.5 * delta_0 + 0.5 * chi2_1` applies, and the reported p-value is an UPPER
    bound on the true one. `at_boundary` says which was used. Reporting the
    nominal value there would manufacture evidence for curvature that the data
    does not support.

    RANDOM EFFECTS. GEE estimates none. Both fits carry the identical grouping
    (`question`) and working correlation, so no random-effect estimation enters the
    statistic and nothing differs between the two arms but the quadratic term.

    QUASI-LRT. This is a generalised estimating equation, so the statistic is
    `-2 * (llf_alt - llf_null)` under the independence working correlation: a
    QUASI-likelihood ratio. It is asymptotic and the p-value is approximate. Stated
    here rather than left for a reader to discover.
    """
    built = build_analysis_frame(df, spec, strict=strict, with_quadratic=True)
    null_terms = tuple(t for t in built.term_names if t != spec.quadratic_term)
    alt_terms = (*null_terms, spec.quadratic_term)

    null_fit = fit_gee_binomial(
        built.frame, null_terms, spec.cluster_col, outcome=spec.outcome, alpha=spec.alpha
    )
    alt_fit = fit_gee_binomial(
        built.frame,
        alt_terms,
        spec.cluster_col,
        outcome=spec.outcome,
        focus=spec.quadratic_term,
        alpha=spec.alpha,
    )

    stat = 2.0 * (alt_fit.log_likelihood - null_fit.log_likelihood)
    beta_sq = float(alt_fit.coefficients[spec.quadratic_term])
    at_boundary = beta_sq > 0.0

    if stat < -1e-8:
        # The alt fit scored WORSE than the null on its own objective. Usually a
        # non-converged iteration. Reported, not clipped silently.
        verdict = (
            f"INVALID: the quadratic model's objective is {stat:.3f}, i.e. below the "
            f"null model's. The fit did not converge (converged={alt_fit.converged}); "
            f"no test is reported because there is nothing to test."
        )
        return LrtResult(
            stat=0.0,
            df=1,
            p_value=float("nan"),
            df_null=len(null_terms),
            df_alt=len(alt_terms),
            quadratic_coefficient=beta_sq,
            at_boundary=at_boundary,
            converged=bool(alt_fit.converged and null_fit.converged),
            nesting="regular (one coefficient fixed at zero); df = 1",
            boundary_handling="n/a - statistic not computable",
            random_effect_handling="identical GEE grouping in both fits",
            verdict=verdict,
            terms_null=null_terms,
            terms_alt=alt_terms,
        )

    stat = max(stat, 0.0)
    nominal = float(stats.chi2.sf(stat, 1))
    p_value = 0.5 * nominal if at_boundary else nominal

    if at_boundary:
        boundary = (
            "beta(log2K_sq) > 0, i.e. the unconstrained optimum lies OUTSIDE the "
            "concave region an inverted-U requires. The constraint beta <= 0 is "
            "active, 0 is on its boundary, and the null distribution is the "
            "Chernoff mixture 0.5*delta_0 + 0.5*chi2_1. The reported p-value is "
            "0.5x the nominal one and is an UPPER BOUND."
        )
    else:
        boundary = (
            "beta(log2K_sq) < 0: the optimum is interior to the concavity "
            "constraint, so the asymptotic chi-square_1 reference applies."
        )

    verdict = (
        f"Null (monotone in log2 K) vs alternative (+ log2K_sq): "
        f"quasi-LR = {stat:.3f} on 1 df, p = {p_value:.4f}"
        f"{' [boundary-corrected, upper bound]' if at_boundary else ''}. "
        + (
            "The monotone response is NOT rejected."
            if p_value > spec.alpha
            else "The monotone response is rejected in favour of curvature."
        )
        + " This is a quasi-likelihood ratio from a GEE; the p-value is asymptotic."
    )

    return LrtResult(
        stat=stat,
        df=1,
        p_value=p_value,
        df_null=len(null_terms),
        df_alt=len(alt_terms),
        quadratic_coefficient=beta_sq,
        at_boundary=at_boundary,
        converged=bool(alt_fit.converged and null_fit.converged),
        nesting="regular (one coefficient fixed at zero); df = 1",
        boundary_handling=boundary,
        random_effect_handling=(
            f"identical GEE grouping on {spec.cluster_col!r} and independence working "
            f"correlation in both fits; GEE estimates no random effects, so none "
            f"enter the statistic"
        ),
        verdict=verdict,
        terms_null=null_terms,
        terms_alt=alt_terms,
    )


# ── R9.3 / R9.4 — the four saturation statistics ──────────────────────────────
@dataclass(frozen=True, slots=True)
class KCurve:
    """Accuracy against depth, ON THE ACTUAL GRID, with optional cost.

    `ks` must be strictly increasing. Every statistic below reads its step size off
    this sequence, which is the whole of R9.4: a rule that assumes consecutive K
    values differ by one is wrong on every grid that is not 1,2,3,4,...
    """

    ks: tuple[int, ...]
    accuracy: tuple[float, ...]
    cost: tuple[float, ...] | None = None
    n_per_k: tuple[int, ...] = ()
    label: str = "pooled"

    def __post_init__(self) -> None:
        if len(self.ks) != len(self.accuracy):
            raise ValueError("KCurve: ks and accuracy must be the same length")
        if len(self.ks) < 2:
            raise ValueError(
                "KCurve needs at least two grid points. One point has no slope, no "
                "marginal gain and no knee - and a statistic reported from it would "
                "be the grid point itself wearing a costume."
            )
        if any(a >= b for a, b in zip(self.ks, self.ks[1:], strict=False)):
            raise ValueError(
                f"KCurve: ks must be strictly increasing with no duplicates, got "
                f"{self.ks}. A repeated depth would make the marginal gain at that "
                f"step 0/0 and the knee's (j-i) denominator zero."
            )
        if self.n_per_k and len(self.n_per_k) != len(self.ks):
            raise ValueError("KCurve: n_per_k must match ks")
        if self.cost is not None and len(self.cost) != len(self.ks):
            raise ValueError("KCurve: cost must match ks")
        bad = [a for a in self.accuracy if not 0.0 <= a <= 1.0]
        if bad:
            raise ValueError(
                f"accuracy must be a proportion in [0, 1]; got {bad[:4]}. A value "
                f"outside it means the metric was a percentage or a count."
            )

    @property
    def grid(self) -> tuple[int, ...]:
        return self.ks

    @property
    def accuracy_max(self) -> float:
        return max(self.accuracy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ks": list(self.ks),
            "accuracy": [round(a, 6) for a in self.accuracy],
            "cost": None if self.cost is None else [round(c, 6) for c in self.cost],
            "n_per_k": list(self.n_per_k),
        }


@dataclass(frozen=True, slots=True)
class StatisticInterval:
    """One saturation statistic: its point value and its bootstrap interval."""

    name: str
    k: int | None
    definition: str
    ci_low: int | None = None
    ci_high: int | None = None
    ci_method: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return self.k is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "K": self.k,
            "definition": self.definition,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            **dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class SaturationStats:
    """All four statistics, always present as slots. R9.3 requires all four."""

    curve: KCurve
    k_95: StatisticInterval
    k_knee: StatisticInterval
    k_delta: StatisticInterval
    frontier: StatisticInterval
    role: str = "unspecified"
    """`primary` or `secondary` (R9.5). Carried on the result so a table that drops
    the role cannot silently promote the pooled curve."""
    note: str = ""

    @property
    def ks(self) -> dict[str, int | None]:
        return {
            "K_95": self.k_95.k,
            "K_knee": self.k_knee.k,
            "K_delta": self.k_delta.k,
            "Frontier K": self.frontier.k,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": self.curve.label,
            "note": self.note,
            "ks": self.ks,
            "k_95": self.k_95.to_dict(),
            "k_knee": self.k_knee.to_dict(),
            "k_delta": self.k_delta.to_dict(),
            "frontier_k": self.frontier.to_dict(),
            "curve": self.curve.to_dict(),
            "n_statistics_reported": 4,
        }


def _k_95(curve: KCurve) -> int | None:
    """Smallest depth reaching `K95_FRACTION` of the best OBSERVED accuracy.

    An ARGUMENT over a noisy grid, which is why its interval is the bootstrap
    distribution of the argument rather than an interval for a smooth parameter
    (R9.3 asks for the method to be stated, not assumed).
    """
    target = K95_FRACTION * curve.accuracy_max
    if target <= 0.0:
        return None
    for k, a in zip(curve.ks, curve.accuracy, strict=True):
        if a >= target:
            return k
    return None


def _k_knee(curve: KCurve) -> tuple[int | None, str]:
    """Earliest structural break: `argmax_j max_{i<j} (A_i - A_j) / (j - i)`.

    Normalised by `(j - i)` in INDEX units, which is the convention R9.4 points at
    ("K_knee already normalises by (j-i); match it"). On a grid whose steps are all
    one doubling, index units and log2 units coincide, so this is the only knee
    definition that does not silently change meaning when the grid is irregular.
    """
    acc = curve.accuracy
    best_drop = -math.inf
    best_k: int | None = None
    for j in range(1, len(acc)):
        drop = max((acc[i] - acc[j]) / (j - i) for i in range(j))
        if drop > best_drop:
            best_drop = drop
            best_k = curve.ks[j]
    if best_k is None or best_drop <= 0.0:
        return (
            curve.ks[-1],
            "no positive drop anywhere on the grid: the response is monotone "
            "non-decreasing, so the knee is the last grid point by definition.",
        )
    return best_k, ""


def _k_delta(
    curve: KCurve, threshold_pp: float = DELTA_THRESHOLD_PP
) -> tuple[int | None, int | None, dict[str, Any]]:
    """`K_delta`, normalised by STEP POSITION (R9.4).

    The naive rule `A(K+1) - A(K) <= 0.5pp` compares a 1x step against a 2x step on
    the same threshold and fires far too early wherever the grid is uneven - on
    `[16, 48, 64]` it is 3.8x stricter on one step than the next. The fix, per R9.4
    and `design.k_delta_in_pp`, is to state the threshold in units that mean the
    same thing everywhere - percentage points PER UNIT OF log2(K) - and convert at
    the step that actually occurs.

    Returns `(K_delta, K_delta_sustained, detail)`. `K_delta` is the declared
    statistic (first step below threshold). `K_delta_sustained` is the first depth
    from which EVERY later step is also below it, which is what a practitioner
    means by "retrieval depth stops mattering here"; a single quiet step is noise,
    and reporting only the first would make the number fire on one lucky cell.
    """
    grid = curve.ks
    acc = curve.accuracy
    steps: list[dict[str, Any]] = []
    for j in range(len(grid) - 1):
        k_from, k_to = grid[j], grid[j + 1]
        step = k_log2_step(grid, k_to)
        gain_pp = 100.0 * (acc[j + 1] - acc[j])
        threshold = k_delta_in_pp(threshold_pp, grid, k_to)
        steps.append(
            {
                "from_K": k_from,
                "to_K": k_to,
                "log2_step": round(step, 6),
                "gain_pp": round(gain_pp, 4),
                "threshold_pp": round(threshold, 4),
                "gain_pp_per_log2": round(gain_pp / step, 4) if step > 0 else None,
                "below_threshold": bool(gain_pp <= threshold),
            }
        )

    below = [i for i, s in enumerate(steps) if s["below_threshold"]]
    k_delta = grid[below[0]] if below else None
    sustained = None
    for i in below:
        if all(s["below_threshold"] for s in steps[i:]):
            sustained = grid[i]
            break

    detail = {
        "steps": steps,
        "threshold_pp_per_log2K": threshold_pp,
        "normalised": (
            "gain_pp <= threshold_pp_per_log2K * log2(K_to / K_from), i.e. the "
            "threshold is converted at the step that actually occurs. The naive "
            "raw-step rule is 3.8x stricter on one step of [16,48,64] than on the "
            "next and fires early (PRD §R9.4)."
        ),
    }
    if k_delta is not None:
        detail["k_delta_step_position"] = round(k_step_index(grid, k_delta), 6)
        detail["k_delta_step_position_note"] = (
            "position of K_delta on the grid, normalised to [0, 1] "
            "(design.k_step_index). A K_delta at 0.2 and one at 0.8 are not "
            "comparable numbers without it."
        )
    return k_delta, sustained, detail


def _frontier_k(curve: KCurve, floor: float = FRONTIER_FLOOR) -> tuple[int | None, str]:
    """`argmax (A(K) - A_max) / Cost(K)` restricted to `A(K) >= floor * A_max`.

    Reads "what is the cheapest depth that still gets essentially everything".
    Over a curve with no cost column there is no efficiency to maximise, and the
    answer is `None` with a reason - not the last grid point, which would read as
    a measurement.
    """
    if curve.cost is None:
        return None, "no cost column on the curve; the efficiency ratio is undefined"
    if any(c <= 0 for c in curve.cost):
        return None, "cost column has non-positive entries; the ratio is undefined"
    best_k: int | None = None
    best_ratio = -math.inf
    for k, a, c in zip(curve.ks, curve.accuracy, curve.cost, strict=True):
        if a < floor * curve.accuracy_max:
            continue
        ratio = (a - curve.accuracy_max) / c
        if ratio > best_ratio:
            best_ratio = ratio
            best_k = k
    if best_k is None:
        return None, (
            f"no grid point reaches {floor:.0%} of A_max, so the restricted argmax "
            f"is over an empty set"
        )
    return best_k, ""


def _percentile_ci(values: Sequence[int], level: float = 0.95) -> tuple[int | None, int | None]:
    """Percentile interval over the bootstrap replicates that produced a value.

    `None` for a statistic the resamples could not compute is reported as an absent
    bound, not as a zero-width interval at the point estimate.
    """
    if not values:
        return None, None
    arr = np.sort(np.asarray(values, dtype=float))
    lo = float(np.quantile(arr, (1.0 - level) / 2.0, method="linear"))
    hi = float(np.quantile(arr, 1.0 - (1.0 - level) / 2.0, method="linear"))
    return math.floor(lo + 0.5), math.floor(hi + 0.5)


def saturation_stats(
    curve: KCurve,
    *,
    boot: Sequence[KCurve] = (),
    threshold_pp: float = DELTA_THRESHOLD_PP,
    frontier_floor: float = FRONTIER_FLOOR,
    ci_level: float = 0.95,
    role: str = "unspecified",
) -> SaturationStats:
    """All four statistics on one curve, with bootstrap intervals when given.

    `boot` is a sequence of RESAMPLED curves - see `resample_curves`, which
    resamples questions rather than rows, because three replicates of one cell are
    not three independent observations.

    THE CI METHOD, STATED (R9.3). All four are bounded-integer ARGUMENTS over a
    noisy grid, so none of them has a sampling distribution with a smooth shape. The
    interval reported is the `ci_level` PERCENTILE interval of the bootstrap
    replicates of the argument itself. For `K_95` in particular that distribution is
    lumpy - the estimator can only take grid values - and the interval can be
    degenerate when the curve never dips. A degenerate interval means "this
    statistic did not move under resampling", not "this depth is certain".
    """
    boot_list = [c for c in boot if isinstance(c, KCurve)]

    k95 = _k_95(curve)
    knee, knee_note = _k_knee(curve)
    delta, sustained, delta_detail = _k_delta(curve, threshold_pp)
    frontier, frontier_reason = _frontier_k(curve, frontier_floor)

    method = (
        f"{len(boot_list)}-replicate percentile bootstrap over QUESTIONS, "
        f"{ci_level:.0%} interval, applied to the argument itself"
        if boot_list
        else ""
    )

    def _boot_values(fn: Any) -> tuple[list[int], tuple[int | None, int | None]]:
        vals = []
        for c in boot_list:
            v = fn(c)
            if v is not None:
                vals.append(int(v))
        return vals, _percentile_ci(vals, ci_level)

    b95, ci95 = _boot_values(_k_95)
    bknee, ciknee = _boot_values(lambda c: _k_knee(c)[0])
    bdelta, cidata = _boot_values(lambda c: _k_delta(c, threshold_pp)[0])
    bfront, cfront = _boot_values(lambda c: _frontier_k(c, frontier_floor)[0])

    delta_detail["k_delta_sustained"] = sustained
    delta_detail["n_bootstrap_replicates_used"] = len(bdelta)

    return SaturationStats(
        curve=curve,
        k_95=StatisticInterval(
            name="K_95",
            k=k95,
            definition=(
                f"smallest K with A(K) >= {K95_FRACTION:.0%} of the best observed "
                f"accuracy on this grid ({curve.accuracy_max:.4f})"
            ),
            ci_low=ci95[0],
            ci_high=ci95[1],
            ci_method=method,
            detail={
                "target_accuracy": round(K95_FRACTION * curve.accuracy_max, 6),
                "n_bootstrap_replicates_used": len(b95),
            },
            unavailable_reason=(
                "" if k95 is not None else "no grid point reaches the 95% target (A_max <= 0)"
            ),
        ),
        k_knee=StatisticInterval(
            name="K_knee",
            k=knee,
            definition=(
                "earliest structural break: argmax_j max_{i<j} (A(i)-A(j))/(j-i), "
                "normalised by step index"
            ),
            ci_low=ciknee[0],
            ci_high=ciknee[1],
            ci_method=method,
            detail={"n_bootstrap_replicates_used": len(bknee), "note": knee_note},
        ),
        k_delta=StatisticInterval(
            name="K_delta",
            k=delta,
            definition=(
                f"smallest K whose next step gains <= "
                f"{threshold_pp} pp per unit log2(K), converted at the actual step"
            ),
            ci_low=cidata[0],
            ci_high=cidata[1],
            ci_method=method,
            detail=delta_detail,
            unavailable_reason=(
                "" if delta is not None else "every step on the grid gains more than the threshold"
            ),
        ),
        frontier=StatisticInterval(
            name="Frontier K",
            k=frontier,
            definition=(
                f"argmax (A(K) - A_max)/Cost(K) over depths with "
                f"A(K) >= {frontier_floor:.0%} * A_max"
            ),
            ci_low=cfront[0],
            ci_high=cfront[1],
            ci_method=method,
            detail={"n_bootstrap_replicates_used": len(bfront)},
            unavailable_reason="" if frontier is not None else frontier_reason,
        ),
        role=role,
        note="",
    )


# ── building and resampling curves ─────────────────────────────────────────────
def curve_from_frame(
    df: Any,
    *,
    k_col: str = "K",
    y_col: str = "correct",
    question_col: str = "question",
    cost_col: str | None = None,
    label: str = "pooled",
) -> KCurve:
    """Mean accuracy at each depth on the grid the data actually contains.

    `K` is taken from the observed values, not from a config. A statistic computed
    on a grid with a level the run never reached is a statistic about a cell nobody
    ran.
    """
    usable = df[[k_col, y_col]].dropna()
    if usable.empty:
        raise ValueError("no scored rows to build a curve from")
    ks = sorted(int(k) for k in usable[k_col].unique())
    means = usable.groupby(k_col)[y_col].mean()
    counts = usable.groupby(k_col)[y_col].size()
    cost = None
    if cost_col is not None and cost_col in df.columns:
        cost_by_k = df.groupby(k_col)[cost_col].mean()
        cost = tuple(float(cost_by_k[k]) for k in ks)
    return KCurve(
        ks=tuple(ks),
        accuracy=tuple(float(means[k]) for k in ks),
        cost=cost,
        n_per_k=tuple(int(counts[k]) for k in ks),
        label=label,
    )


def resample_curves(
    df: Any,
    *,
    n_boot: int = 400,
    seed: int = 0,
    k_col: str = "K",
    y_col: str = "correct",
    question_col: str = "question",
    strat_col: str | None = None,
    cost_col: str | None = None,
    label: str = "pooled",
) -> list[KCurve]:
    """Question-level bootstrap replicates of a curve.

    THE RESAMPLING UNIT IS THE QUESTION, never the row. Rows from one question share
    a corpus, a gold answer, a difficulty and a reader draw; resampling them
    independently treats three replicates of one cell as three independent
    observations and produces an interval roughly sqrt(3) too narrow. Every accuracy
    in this project is a within-question mean, so the question is the exchangeable
    unit.

    With `strat_col`, questions are resampled WITHIN stratum. Stratified analysis is
    PRIMARY (R9.5), and an unstratified bootstrap on a stratum whose questions all
    land in one of two resamples is not a confidence interval for that stratum.

    Returns the replicates only. `n_boot` is the number asked for; a caller that
    needs the realised count should read `len(...)`, because a replicate that
    cannot be built is dropped rather than faked.
    """
    if n_boot <= 0:
        return []
    usable = df[[k_col, y_col, question_col]].dropna()
    if usable.empty:
        return []
    rng = np.random.default_rng(seed)

    ks = tuple(sorted(int(k) for k in usable[k_col].unique()))
    cells: dict[Any, dict[int, list[float]]] = {}
    for question, group in usable.groupby(question_col):
        by_k = cells.setdefault(question, {})
        for k, value in zip(group[k_col], group[y_col], strict=True):
            by_k.setdefault(int(k), []).append(float(value))

    strata: dict[Any, list[Any]] = {}
    if strat_col is not None and strat_col in df.columns:
        labels = df.groupby(question_col)[strat_col].first()
        for question, label_value in labels.items():
            strata.setdefault(label_value, []).append(question)

    out: list[KCurve] = []
    for _ in range(n_boot):
        if strata:
            sampled: list[Any] = []
            for members in strata.values():
                draw = rng.integers(0, len(members), size=len(members))
                sampled.extend(members[i] for i in draw)
        else:
            all_questions = list(cells)
            draw = rng.integers(0, len(all_questions), size=len(all_questions))
            sampled = [all_questions[i] for i in draw]

        sums = dict.fromkeys(ks, 0.0)
        counts = dict.fromkeys(ks, 0)
        for question in sampled:
            for k, values in cells.get(question, {}).items():
                if k in sums:
                    sums[k] += sum(values)
                    counts[k] += len(values)
        if any(counts[k] == 0 for k in ks):
            continue
        out.append(
            KCurve(
                ks=ks,
                accuracy=tuple(sums[k] / counts[k] for k in ks),
                n_per_k=tuple(counts[k] for k in ks),
                label=label,
            )
        )
    return out


# ── R9.5 — stratified primary, pooled secondary ───────────────────────────────
PRIMARY_ROLE: Final = "primary"
SECONDARY_ROLE: Final = "secondary"

WHY_STRATIFIED_IS_PRIMARY: Final = (
    "PRD §R9.5: the recall ceiling is strongly TYPE-CONDITIONAL - single-hop and "
    "multi-hop questions have different evidence structures and therefore different "
    "saturation points, by construction rather than by noise (ceiling.py; the "
    "measured results/ceiling.json). A pooled saturation point is a WEIGHTED "
    "AVERAGE OF TWO REGIMES: it is not a noisier version of the stratified answer, "
    "it is an answer to a different question. The pooled curve is reported as "
    "SECONDARY and labelled as such everywhere it appears, so the mixture is "
    "visible without being the quantity of interest."
)


@dataclass(frozen=True, slots=True)
class StratumResult:
    """One curve's saturation statistics, with its role attached."""

    label: str
    role: str
    n_questions: int
    n_obs: int
    stats: SaturationStats
    analytic_saturation_k: int | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = self.stats.to_dict()
        out.update(
            {
                "label": self.label,
                "role": self.role,
                "n_questions": self.n_questions,
                "n_obs": self.n_obs,
                "analytic_saturation_k": self.analytic_saturation_k,
                "notes": list(self.notes),
            }
        )
        return out


@dataclass(frozen=True, slots=True)
class StrataReport:
    """R9.5's output: strata (primary) and the pooled curve (secondary)."""

    strata: tuple[StratumResult, ...]
    pooled: StratumResult
    strat_col: str
    why: str = WHY_STRATIFIED_IS_PRIMARY
    primary_is: str = "stratified"
    secondary_is: str = "pooled"

    def by_label(self) -> dict[str, StratumResult]:
        return {s.label: s for s in self.strata}

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_analysis": self.primary_is,
            "secondary_analysis": self.secondary_is,
            "why": self.why,
            "strat_col": self.strat_col,
            "strata": [s.to_dict() for s in self.strata],
            "pooled_secondary": self.pooled.to_dict(),
        }


def strata_report(
    df: Any,
    *,
    strat_col: str = "ability",
    k_col: str = "K",
    y_col: str = "correct",
    question_col: str = "question",
    cost_col: str | None = None,
    n_boot: int = 200,
    seed: int = 0,
    ceiling_by_stratum: Mapping[str, Mapping[str, float]] | None = None,
    threshold_pp: float = DELTA_THRESHOLD_PP,
) -> StrataReport:
    """Saturation statistics PER STRATUM (primary) and pooled (secondary). R9.5.

    `ceiling_by_stratum` maps stratum -> `{K: ceiling}` and, when supplied, adds the
    ANALYTIC saturation point `ceiling.saturation_k` beside each measured one. That
    pairing is the point of R9.5: the analytic K* is the depth past which retrieval
    cannot add evidence, and comparing it to the measured knee per type is what
    separates "the reader stopped using what it has" from "the corpus ran out".

    Strata with too few questions to support a statistic are still reported, with
    `n_questions` on them, and their bootstrap simply fails to move. Silently
    dropping a small stratum would hide exactly the exploratory strata R9.9 warns
    are most likely to be over-read.
    """
    from msat.ceiling import saturation_k

    if strat_col not in df.columns:
        raise MissingCovariateError(
            f"{strat_col!r} is absent, so there are no strata to stratify by. R9.5 "
            f"makes the stratified analysis primary; a pooled-only report is the "
            f"SECONDARY read and is not a substitute."
        )

    strata: list[StratumResult] = []
    for level, group in df.groupby(strat_col):
        label = str(level)
        curve = curve_from_frame(
            group,
            k_col=k_col,
            y_col=y_col,
            question_col=question_col,
            cost_col=cost_col,
            label=label,
        )
        boot = resample_curves(
            group,
            n_boot=n_boot,
            seed=seed,
            k_col=k_col,
            y_col=y_col,
            question_col=question_col,
            cost_col=cost_col,
            label=label,
        )
        stats_block = saturation_stats(
            curve, boot=boot, threshold_pp=threshold_pp, role=PRIMARY_ROLE
        )
        analytic = None
        if ceiling_by_stratum is not None and label in ceiling_by_stratum:
            analytic = saturation_k(dict(ceiling_by_stratum[label]))
        strata.append(
            StratumResult(
                label=label,
                role=PRIMARY_ROLE,
                n_questions=int(group[question_col].nunique()) if question_col in group else 0,
                n_obs=len(group),
                stats=stats_block,
                analytic_saturation_k=analytic,
                notes=("PRIMARY (PRD §R9.5)",),
            )
        )

    pooled_curve = curve_from_frame(
        df,
        k_col=k_col,
        y_col=y_col,
        question_col=question_col,
        cost_col=cost_col,
        label="pooled",
    )
    pooled = StratumResult(
        label="pooled",
        role=SECONDARY_ROLE,
        n_questions=int(df[question_col].nunique()) if question_col in df.columns else 0,
        n_obs=len(df),
        stats=saturation_stats(
            pooled_curve,
            boot=resample_curves(
                df,
                n_boot=n_boot,
                seed=seed,
                k_col=k_col,
                y_col=y_col,
                question_col=question_col,
                cost_col=cost_col,
                label="pooled",
            ),
            threshold_pp=threshold_pp,
            role=SECONDARY_ROLE,
        ),
        notes=("SECONDARY / DESCRIPTIVE (PRD §R9.5)", WHY_STRATIFIED_IS_PRIMARY),
    )

    strata.sort(key=lambda s: s.label)
    return StrataReport(strata=tuple(strata), pooled=pooled, strat_col=strat_col)


# ── R0.4 — the pre-registered verdict ─────────────────────────────────────────
FORBIDDEN_ZERO_CLAIMS: Final[tuple[str, ...]] = (
    "exactly zero",
    "is zero",
    "proves there is no interaction",
    "no interaction exists",
)
"""Phrases this module must never emit as its own CLAIM.

`configs/preregistration.yaml` carries the distinction in `not_claimed`: *"that
the interaction is exactly zero - only that it is small relative to the MDE."* The
first half is not in the data's gift. A confidence interval that covers zero says
the design could not resolve the coefficient; it does not say the coefficient is
zero, and a paper that prints the first as the second has claimed something it
cannot support. The forbidden list is checked by the test suite against every
string the verdict emits as a claim - the `not_claimed` line that DISCLAIMS it is
carried verbatim and is not a claim.
"""


def load_preregistration(config_root: Path | str | None = None) -> dict[str, Any]:
    """Read `configs/preregistration.yaml`.

    Read directly rather than through `config.load_bundle`, deliberately: the
    SHIPPED bundle is incomplete until the reader checkpoints are confirmed
    (`tests/conftest.py::complete_configs` exists for exactly that reason), and the
    pre-registered prediction must remain readable in that state - it is a fact
    recorded before any run, not a property of the current bundle.

    The claim that it was registered *in advance* rests on the SHA-256 in
    `results/power.json` (`_provenance.preregistration_hash`), not on this read.
    """
    import yaml

    root = Path(config_root) if config_root is not None else Path("configs")
    path = root / "preregistration.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The pre-registered sign prediction (PRD §R0.4) "
            f"lives there; without it the primary estimate has nothing to be "
            f"compared against and the verdict cannot be written."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "interaction_sign_prediction" not in data:
        raise ValueError(
            f"{path} has no `interaction_sign_prediction`. The file is the "
            f"pre-registration (PRD §R0.4); a prediction it cannot name is not one."
        )
    return data


@dataclass(frozen=True, slots=True)
class PreregistrationVerdict:
    """The interaction, read against the prediction made before the data existed."""

    prediction: str
    estimate_log_odds: float
    mde_log_odds: float | None
    ratio_to_mde: float | None
    consistent: bool | None
    p_value: float
    ci_low: float
    ci_high: float
    claim: str
    """What is ASSERTED. Never contains a zero claim; checked by the test suite."""
    reading: str = ""
    not_claimed: tuple[str, ...] = ()
    claims_exactly_zero: bool = False
    source: str = "configs/preregistration.yaml"
    mde_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "estimate_log_odds": round(self.estimate_log_odds, 6),
            "mde_log_odds": self.mde_log_odds,
            "ratio_to_mde": None if self.ratio_to_mde is None else round(self.ratio_to_mde, 4),
            "consistent": self.consistent,
            "p_value": round(self.p_value, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "claim": self.claim,
            "reading": self.reading,
            "not_claimed": list(self.not_claimed),
            "claims_exactly_zero": self.claims_exactly_zero,
            "source": self.source,
            "mde_source": self.mde_source,
        }


def _coerce_mde(mde: float | Mapping[str, Any] | None) -> tuple[float | None, str]:
    """Accept the MDE as a number or as the `power.json` block that contains one.

    `results/power.json` nests it under `interaction.mde_log_odds`; passing the
    whole report is the natural thing for a caller to do and it is unambiguous,
    whereas making every caller dig one level down is where a wrong number gets
    typed.
    """
    if mde is None:
        return None, "no MDE supplied"
    if isinstance(mde, Mapping):
        block = mde.get("interaction", mde)
        # `powered` is checked BEFORE the value. `mde_interaction` returns
        # `{"powered": false, "mde_log_odds": null, ...}` when the grid cannot
        # detect the interaction at any plausible size, and a verdict that read a
        # number out of that block would be a verdict against an MDE that does not
        # exist.
        if isinstance(block, Mapping) and block.get("powered") is False:
            return None, (
                f"power.json reports the design UNPOWERED for this interaction "
                f"({block.get('note', 'no note')}). No MDE exists to compare against."
            )
        value = block.get("mde_log_odds") if isinstance(block, Mapping) else None
        if value is None:
            return None, "the supplied mapping carries no `mde_log_odds`"
        return float(value), "power.json -> interaction.mde_log_odds"
    if mde <= 0:
        return None, f"MDE must be positive; got {mde}"
    return float(mde), "supplied directly"


def preregistration_verdict(
    primary_fit: PrimaryFit,
    mde: float | Mapping[str, Any] | None,
    *,
    prereg: Mapping[str, Any] | None = None,
    config_root: Path | str | None = None,
    baseline_p: float = 0.5,
) -> PreregistrationVerdict:
    """Compare the estimated interaction to `configs/preregistration.yaml`. R0.4.

    THE CLAIM THIS FUNCTION IS NOT PERMITTED TO MAKE
    -------------------------------------------------
    "The interaction is exactly zero." It has no access to the truth of that, and
    the pre-registration already wrote the honest version into `not_claimed`: the
    coefficient is small *relative to the minimum detectable effect*. Those are
    different statements - a coefficient of 0.4 log-odds can be both non-significant
    and scientifically enormous at n = 500 - and the second one is what a reader of
    the paper will remember. `claims_exactly_zero` is a field rather than a
    convention so it can be asserted in a test, and `not_claimed` is carried through
    verbatim from the file so the disclaimer travels with the number.

    `mde` is the minimum detectable effect on the SAME scale as the coefficient
    (log-odds). If it is absent or the design is flagged unpowered, `consistent` is
    `None` - not `True`. "We could not have detected this" and "there was nothing
    there" must not print the same.
    """
    config = dict(prereg) if prereg is not None else load_preregistration(config_root)
    prediction = str(config.get("interaction_sign_prediction", "approximately_zero"))
    reading = str(config.get("sign_readings", {}).get(prediction, ""))
    not_claimed = tuple(str(x) for x in config.get("not_claimed", ()))
    mde_value, mde_source = _coerce_mde(mde)

    beta = float(primary_fit.coefficient)
    effect_pp = logodds_to_pp(beta, baseline_p)
    consistent: bool | None = None
    ratio: float | None = None

    if mde_value is None:
        claim = (
            f"The estimated log2(K) x log2(B) interaction is {beta:+.4f} log-odds "
            f"({effect_pp:+.2f} pp at p = {baseline_p:g}), "
            f"95% CI [{primary_fit.ci_low:+.4f}, {primary_fit.ci_high:+.4f}]. "
            f"The pre-registered prediction was {prediction!r}, and NO verdict can "
            f"be stated: the minimum detectable effect is unavailable ({mde_source}). "
            f"A design that cannot resolve the coefficient cannot confirm or refute "
            f"it."
        )
    else:
        ratio = abs(beta) / mde_value
        consistent = bool(ratio <= 1.0)
        ci_covers_zero = primary_fit.ci_low <= 0.0 <= primary_fit.ci_high
        claim = (
            f"The estimated log2(K) x log2(B) interaction is {beta:+.4f} log-odds "
            f"({effect_pp:+.2f} pp at p = {baseline_p:g}), "
            f"95% CI [{primary_fit.ci_low:+.4f}, {primary_fit.ci_high:+.4f}]"
            f"{' (which covers zero)' if ci_covers_zero else ' (which excludes zero)'}. "
            f"The pre-registered prediction was {prediction!r}. "
            f"|estimate| = {abs(beta):.4f} is {ratio:.2f}x the minimum detectable "
            f"effect of {mde_value:.4f} log-odds"
            + (
                ", so the estimate is SMALL RELATIVE TO THE MDE and the "
                "prediction is CONSISTENT with it."
                if consistent
                else ", so the estimate EXCEEDS THE MDE and the prediction is NOT supported."
            )
            + " This is a statement about resolution, NOT about the coefficient "
            "being absent: no finite design distinguishes a small coefficient from "
            "no coefficient, and this one does not claim to."
        )

    return PreregistrationVerdict(
        prediction=prediction,
        estimate_log_odds=beta,
        mde_log_odds=mde_value,
        ratio_to_mde=ratio,
        consistent=consistent,
        p_value=float(primary_fit.p_value),
        ci_low=float(primary_fit.ci_low),
        ci_high=float(primary_fit.ci_high),
        claim=claim,
        reading=reading,
        not_claimed=not_claimed,
        claims_exactly_zero=False,
        mde_source=mde_source,
    )


# ── R9.10 — the leakage check, wired ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LeakageCheck:
    """R9.10's classifier, run, plus whether the assertion applies to this arm."""

    report: LeakageReport
    n_items: int
    arms: tuple[str, ...]
    state: str
    """One of four NAMED states, never a pair of booleans:

    - `no_separation` - size does not predict the condition. R9.10 satisfied.
    - `size_leak` - size does predict it. R9.10 violated; the run is flagged.
      This is 2609.32269's case: conditions built by DELETING passages, so every
      size feature carries the label.
    - `count_labelled_length_matched` - the CALLER DECLARED this a count-labelled
      arm (`count_labelled=True`) and `total_tokens` matches across conditions.
      Arm B as designed: memory count is the label and the reader is handed it as a
      numbered list, so there is nothing for size to leak.
    - `count_labelled_length_drifted` - declared count-labelled, and length is NOT
      held constant. Not a label leak, but the arm's own assertion failed and the
      count-vs-length separation is not being measured.

    Four states rather than two because the last two are neither "fine" nor "a
    leak", and folding them into either produces a verdict in words a reader has to
    unpick."""
    observed_count_separation: bool
    """OBSERVATION, not a decision: `n_memories` separates in either direction.

    Reported so the reason for a `count_labelled` DECLARATION is visible. An earlier
    version inferred the exemption from this flag and it was wrong in both
    directions - a deletion-style construction and a length-drifted Arm B are
    indistinguishable from size features alone, which is precisely R9.10's point.
    The exemption is now declared by the caller, from `arms/padding`'s own knowledge
    of how the arm was built."""
    applicable: bool
    """True when R9.10's assertion binds verbatim."""
    verdict: str
    token_match: Mapping[str, Any] = field(default_factory=dict)
    token_match_verdict: str = ""

    @property
    def passed(self) -> bool:
        """One boolean for the caller, derived from the named state.

        A field rather than a recomputation so the check cannot be read two ways.
        """
        if self.state in ("count_labelled_length_matched", "count_labelled_length_drifted"):
            return not self.token_match.get("warnings")
        return self.state == "no_separation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": "PRD §R9.10",
            "n_items": self.n_items,
            "arms": list(self.arms),
            "state": self.state,
            "passed": self.passed,
            "observed_count_separation": self.observed_count_separation,
            "applicable": self.applicable,
            "verdict": self.verdict,
            "classifier": self.report.as_dict(),
            "token_match": dict(self.token_match),
            "token_match_verdict": self.token_match_verdict,
        }


def leakage_check(
    df: Any,
    *,
    label_col: str = "arm",
    n_memories_col: str = "n_memories",
    n_tokens_col: str = "n_tokens",
    n_turns_col: str | None = "n_turns",
    count_labelled: bool | None = None,
    threshold: float = DEFAULT_AUROC_THRESHOLD,
    seed: int = 0,
    n_null: int = 50,
) -> LeakageCheck:
    """Run R9.10's classifier over a scored frame and say whether it applies.

    Columns are the four size features `analysis/leakage.py` uses, plus the
    condition label. Nothing here modifies the detector: `assert_no_leakage` is
    called unchanged and its verdict is reproduced verbatim in `classifier`.

    WHAT IS ADDED, AND WHY
    -----------------------
    R9.10's classifier separates Arm B PERFECTLY, and that is not a leak: memory
    count is the manipulation, and the reader is handed the count anyway as a
    numbered list. Calling it "Arm B is INVALID" - which is literally what
    `assert_no_leakage`'s FAIL text says - would be wrong, and flagging every run on
    it would make the check a nuisance that gets switched off.

    So the exemption is a DECLARATION, not an inference. `count_labelled=True` is
    set by the caller from `arms/padding`'s own knowledge of how the arm was built,
    exactly as `tests/test_no_leakage.py::test_arm_b_as_designed_is_separable_by_
    memory_count` declares it in prose. An earlier version of this function INFERRED
    the exemption from the observed AUROCs and was wrong in both directions, for one
    reason: a deletion-style construction and a length-drifted Arm B are
    INDISTINGUISHABLE from size features alone - which is exactly R9.10's argument
    for why the classifier exists. Inferring the exemption from the evidence the
    check is meant to police is circular.

    The default is `None` = R9.10 applies verbatim, so a caller who forgets to
    declare gets the strict reading.
    """
    required = [label_col, n_memories_col, n_tokens_col]
    if n_turns_col is not None:
        required.append(n_turns_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingCovariateError(
            f"R9.10's classifier needs the size features {missing}. They come from "
            f"arms.padding.leakage_features at render time; a run whose manifest "
            f"lacks them cannot be checked for leakage, and skipping the check "
            f"silently is the one outcome R9.10 exists to prevent."
        )

    labels = sorted({str(v) for v in df[label_col].tolist()})
    label_rank = {label: i for i, label in enumerate(labels)}
    if len(label_rank) < 2:
        raise ValueError(
            f"{label_col!r} has {len(label_rank)} distinct value(s). R9.10 asks "
            f"whether SIZE separates two conditions; with one condition there is "
            f"nothing to separate."
        )

    features = [
        LeakageFeatures(
            label=label_rank[str(row[label_col])],
            n_memories=int(row[n_memories_col]),
            n_tokens=int(row[n_tokens_col]),
            n_turns=int(row[n_turns_col]) if n_turns_col else int(row[n_memories_col]),
        )
        for _, row in df.iterrows()
    ]

    report = assert_no_leakage(features, threshold=threshold, seed=seed, n_null=n_null)

    per_feature = report.per_feature_auroc
    count_auroc = per_feature.get("n_memories", 0.5)
    length_auroc = per_feature.get("n_tokens", 0.5)
    observed_count_separation = _separates(count_auroc)

    token_match: Mapping[str, Any] = {}
    token_verdict = ""
    if count_labelled:
        state = (
            "count_labelled_length_matched"
            if _at_chance(length_auroc)
            else "count_labelled_length_drifted"
        )
        by_condition: dict[tuple[str, ...], dict[str, float]] = {}
        for _, row in df.iterrows():
            key = (str(row[label_col]),)
            bucket = by_condition.setdefault(
                key, {"n_memories": 0.0, "total_tokens": 0.0, "n": 0.0}
            )
            bucket["n_memories"] += float(row[n_memories_col])
            bucket["total_tokens"] += float(row[n_tokens_col])
            bucket["n"] += 1.0
        matrix = size_leakage_matrix(
            {
                key: {
                    "n_memories": v["n_memories"] / v["n"],
                    "total_tokens": v["total_tokens"] / v["n"],
                }
                for key, v in by_condition.items()
            }
        )
        token_match = matrix
        warnings = list(matrix.get("warnings", ()))
        token_verdict = (
            f"This arm is DECLARED count-labelled, so R9.10's size classifier is not "
            f"the assertion that binds: memory count IS the label, and it is printed "
            f"in the prompt as a numbered list, so there is nothing for size to leak "
            f"that the reader does not already have (AUROC {count_auroc:.3f} on "
            f"`n_memories`). What binds is the LENGTH MATCH (AUROC {length_auroc:.3f} "
            f"on `n_tokens`): "
            + (
                f"FAILED - {'; '.join(warnings)}"
                if warnings
                else f"PASSED - total_tokens matches across conditions within "
                f"+/-{TOKEN_TARGET_TOLERANCE:.0%}, so length is held constant."
            )
        )
        verdict = token_verdict
    else:
        state = "no_separation" if report.passed else "size_leak"
        verdict = (
            report.verdict
            if report.passed
            else (
                report.verdict
                + " Conditions separate on a size feature that is not the label, which "
                "is the leak 2609.32269 reports."
            )
        )

    return LeakageCheck(
        report=report,
        n_items=len(features),
        arms=tuple(labels),
        state=state,
        observed_count_separation=observed_count_separation,
        applicable=not count_labelled,
        verdict=verdict,
        token_match=token_match,
        token_match_verdict=token_verdict,
    )
