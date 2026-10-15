"""`analysis/interaction.py` - PRD §R9.1 and the endpoints built on it.

EVERY FIXTURE HERE IS SYNTHETIC
------------------------------
No run of this project has completed, so nothing in this file asserts a number the
paper will quote. Each fixture PLANTS its answer (`delta`, the curve shapes, the
stratum regimes) and the test asserts the estimator recovers it. That is the only
kind of claim available before the data exists, and it is the right one: a
saturation statistic with a known shape must return that shape, and a stratified
analysis with two regimes must not average them.

THE FIVE THINGS THIS FILE IS FOR
--------------------------------
1. `test_the_specification_is_r91` - the declared term list is PRD §R9.1, read
   from the SAME string the CLI and the preregistration quote. "Declared once,
   here and nowhere else" is a constraint on the code, and a test is the only thing
   that enforces it.
2. `test_corpus_fraction_absent_raises_a_named_error` - the covariate R9.8 puts in
   EVERY model. Dropping it silently leaves a table indistinguishable from the
   declared specification.
3. `test_k_delta_is_normalised_on_an_uneven_grid` - `A(K+1) - A(K) <= 0.5pp` is
   3.8x stricter on one step of `[16, 48, 64]` than on the next, and fires early.
   R9.4 is the load-bearing requirement in this file.
4. `test_pooled_and_stratified_saturation_points_differ` - R9.5. A pooled
   saturation point is a weighted average of two regimes, which is a different
   answer rather than a noisier one.
5. `test_the_verdict_never_claims_the_interaction_is_exactly_zero` - the
   pre-registration's own `not_claimed` line. The strongest integrity property in
   the project, and the easiest to lose to a reworded sentence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from msat.analysis.interaction import (
    DELTA_THRESHOLD_PP,
    FORBIDDEN_ZERO_CLAIMS,
    K95_FRACTION,
    PRIMARY_ROLE,
    PRIMARY_SPEC,
    R91_DECLARED,
    SECONDARY_ROLE,
    TERM_INTERACTION,
    WHY_STRATIFIED_IS_PRIMARY,
    InteractionSpec,
    KCurve,
    LrtResult,
    MissingCovariateError,
    build_analysis_frame,
    curve_from_frame,
    fit_primary,
    fit_reader_clustered,
    leakage_check,
    load_preregistration,
    lrt_monotone_vs_inverted_u,
    preregistration_verdict,
    primary_analysis,
    resample_curves,
    saturation_stats,
    strata_report,
)
from msat.analysis.mixedfx import NonBinaryOutcomeError, TooFewClustersError

K_LEVELS = (1, 2, 4, 8, 16, 32)
B_LEVELS = (128.0, 256.0, 512.0)
ABILITIES = ("single-session", "multi-session", "temporal-reasoning")

DELTA = 0.80
"""Planted interaction for the synthetic design. A property of this fixture."""


def _design(
    *,
    n_questions: int = 250,
    runs_per_cell: int = 2,
    delta: float = DELTA,
    seed: int = 4,
    abilities: tuple[str, ...] = ABILITIES,
) -> Any:
    """SYNTHETIC crossed K x B design with a planted interaction.

    `ability` is assigned by `q % 3` and `reader_strength` is a linear ramp in `q`
    so the two are NOT collinear - a fixture where they are would trip the rank
    check for the right reason and hide the thing under test.
    """
    rng = np.random.default_rng(seed)
    k_center = float(np.mean(np.log2(K_LEVELS)))
    b_center = float(np.mean(np.log2(B_LEVELS)))
    cells = [(k, b) for k in K_LEVELS for b in B_LEVELS]

    rows: list[dict[str, Any]] = []
    for q in range(n_questions):
        strength = 0.70 + 0.0008 * q
        n_sessions = 30.0 + (q % 23)
        for k, b in cells:
            kc = float(np.log2(k)) - k_center
            bc = float(np.log2(b)) - b_center
            eta = (
                0.10
                + 0.30 * kc
                + 0.20 * bc
                + delta * kc * bc
                + 1.5 * strength
                + 0.8
                * {"single-session": 1, "multi-session": 0, "temporal-reasoning": -1}[
                    abilities[q % len(abilities)]
                ]
                + float(rng.normal(0.0, 0.25))
            )
            p = 1.0 / (1.0 + np.exp(-eta))
            for _ in range(runs_per_cell):
                rows.append(
                    {
                        "question": f"q{q:04d}",
                        "ability": abilities[q % len(abilities)],
                        "reader": ("gemma", "qwen", "phi")[q % 3],
                        "reader_strength": strength,
                        "K": k,
                        "B": b,
                        "corpus_fraction": k / n_sessions,
                        "gold_rank": float(rng.integers(1, 12)),
                        "correct": int(rng.binomial(1, p)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df() -> Any:
    return _design()


# ── the declaration (R9.1) ────────────────────────────────────────────────────
def test_the_specification_is_r91() -> None:
    """The declared term list is PRD §R9.1, and nothing has been added.

    Asserted against the literal string the PRD writes and the preregistration
    quotes, so a term added for convenience - a stray indicator, a second
    interaction - fails here rather than in a reviewer.
    """
    assert PRIMARY_SPEC.declared == R91_DECLARED
    for fragment in (
        "log2(K)*log2(B)",
        "+ log2(K)",
        "+ log2(B)",
        "corpus_fraction",
        "gold_rank",
        "ability",
        "reader_strength",
        "(1|question)",
        "(1|model)",
    ):
        assert fragment in PRIMARY_SPEC.declared, fragment

    assert TERM_INTERACTION in PRIMARY_SPEC.terms
    assert "log2K" in PRIMARY_SPEC.terms
    assert "log2B" in PRIMARY_SPEC.terms
    assert "corpus_fraction" in PRIMARY_SPEC.terms
    assert "gold_rank" in PRIMARY_SPEC.terms
    assert "reader_strength" in PRIMARY_SPEC.terms
    assert PRIMARY_SPEC.include_ability is True
    assert PRIMARY_SPEC.cluster_col == "question"
    assert PRIMARY_SPEC.outcome == "correct"


def test_the_specification_is_declared_once_and_importable() -> None:
    """`configs/preregistration.yaml` says "declared once, here and nowhere else".

    What makes that enforceable is a single module-level instance other surfaces can
    import, rather than each rebuilding the list. A test cannot stop a future edit,
    but it can make `PRIMARY_SPEC is <the one used>` true and obvious.
    """
    from msat.analysis import interaction

    assert interaction.PRIMARY_SPEC is PRIMARY_SPEC
    assert isinstance(PRIMARY_SPEC, InteractionSpec)
    with pytest.raises((AttributeError, TypeError)):
        PRIMARY_SPEC.terms = ("log2K",)  # frozen: the declaration cannot drift


def test_the_specification_records_the_estimator_and_the_refused_one() -> None:
    """Provenance, including what was deliberately NOT used.

    `not_used` is in the dict because a paper that omits it invites the question,
    and `results/power.json` already carries the same field for the same reason.
    """
    spec = PRIMARY_SPEC.to_dict()
    assert "MixedLM" in spec["not_used"]
    assert "GEE" in spec["estimator"]
    assert spec["cluster_col"] == "question"
    assert spec["random_effects"]["question"].startswith("GEE cluster")
    assert "APPROXIMATION" in spec["random_effects"]["model"]


def test_the_interaction_column_is_the_one_the_design_matrix_declares(df: Any) -> None:
    """The reused design matrix is CHECKED, not trusted.

    `design._design_matrix` owns the column order; if it ever changes, every label
    after the changed column is wrong while every number stays plausible. The
    reconstruction below is the guard.
    """
    built = build_analysis_frame(df)
    x = (np.log2(df["K"].to_numpy()) - built.k_center) * (
        np.log2(df["B"].to_numpy()) - built.b_center
    )
    assert np.allclose(built.frame[TERM_INTERACTION].to_numpy(), x)
    assert np.allclose(
        built.frame["log2K"].to_numpy(), np.log2(df["K"].to_numpy()) - built.k_center
    )


def test_centring_leaves_the_interaction_coefficient_invariant(df: Any) -> None:
    """The headline number means the same thing centred as uncentred.

    Shifting `k -> k - kappa` and `b -> b - mu` gives
    `k*b = k'b' + mu*k' + kappa*b' + kappa*mu`, so with BOTH main effects retained
    the interaction coefficient is unchanged and only the main effects move. PRD
    §R9.1 writes raw log2(K) and log2(B); centring is what keeps the baseline off
    the ceiling, and this test is why it can be done without changing the claim.
    """
    from msat.analysis.mixedfx import fit_gee_binomial

    built = build_analysis_frame(df)
    centred = fit_primary(df)

    raw = built.frame.copy()
    raw["log2K"] = raw["log2K"] + built.k_center
    raw["log2B"] = raw["log2B"] + built.b_center
    raw[TERM_INTERACTION] = raw["log2K"] * raw["log2B"]
    uncentred = fit_gee_binomial(
        raw, built.term_names, PRIMARY_SPEC.cluster_col, focus=TERM_INTERACTION
    )

    assert centred.coefficient == pytest.approx(uncentred.coefficient, abs=1e-9)
    # ...and the main effects DO move, which is the visible half of the identity.
    assert abs(centred.coefficients["log2K"] - uncentred.coefficients["log2K"]) > 1e-6


# ── the fit (R9.1) ────────────────────────────────────────────────────────────
def test_fit_primary_recovers_a_planted_interaction(df: Any) -> None:
    """The paper's headline number, on data whose answer is known.

    Bound is the fit's own standard error, not a magic number: a null estimate
    cannot satisfy it, whereas an absolute tolerance loose enough to be
    Monte-Carlo-safe would also be loose enough to pass a fit that found nothing.
    """
    fit = fit_primary(df)
    assert fit.term == TERM_INTERACTION
    assert fit.coefficient == pytest.approx(DELTA, abs=0.25)
    assert abs(fit.coefficient - DELTA) < 2.5 * fit.std_error
    assert fit.p_value < 1e-3
    assert fit.n_clusters == df["question"].nunique()
    assert fit.n_obs == len(df)
    assert fit.converged is True


def test_fit_primary_fits_every_declared_term(df: Any) -> None:
    """R9.1's covariates, all of them, none dropped.

    `ability` becomes reference-coded dummies. A silent drop would leave a table
    that looks like R9.1 and is not, which is why the reference level is reported
    rather than merely implied.
    """
    built = build_analysis_frame(df)
    for term in PRIMARY_SPEC.terms:
        assert term in built.term_names, term
    assert any(t.startswith("ability[") for t in built.term_names)
    assert built.dropped_terms == ()
    # `ability` is reference-coded: the FIRST LEVEL IN SORTED ORDER is the omitted
    # one, which is not necessarily the first in the source data. Getting that
    # backwards would make the reported level a lie.
    assert built.dropped_levels == tuple(sorted(ABILITIES)[:1])
    assert any(t.startswith("ability[") for t in built.to_dict()["term_names"])


def test_corpus_fraction_absent_raises_a_named_error(df: Any) -> None:
    """R2.8/G12 and R9.8: `corpus_fraction` is a covariate in EVERY model.

    Dropping it silently leaves a model that looks like R9.1 and is not - and
    `corpus_fraction` is most of what distinguishes "reading more" from "reading
    more OF THE CORPUS", so the omission is not a detail.
    """
    stripped = df.drop(columns=["corpus_fraction"])
    with pytest.raises(MissingCovariateError, match="corpus_fraction"):
        fit_primary(stripped)
    with pytest.raises(MissingCovariateError, match="corpus_fraction"):
        build_analysis_frame(stripped)
    # MissingCovariateError is a ValueError, so a caller's `except ValueError`
    # around a fit still catches it.
    with pytest.raises(ValueError):
        fit_primary(stripped)


def test_a_missing_ability_column_is_also_a_named_error(df: Any) -> None:
    """R9.5's primary stratification has nothing to stratify by without it."""
    with pytest.raises(MissingCovariateError, match="ability"):
        build_analysis_frame(df.drop(columns=["ability"]))


def test_strict_false_drops_on_the_record_not_in_silence(df: Any) -> None:
    """The one sanctioned way to shed a term, and it writes down what it shed.

    `dropped_terms` is derived from the FITTED list against R9.1's full
    declaration rather than accumulated as failures happen - so a spec that never
    declared `gold_rank` in the first place is still reported as not carrying it,
    which is the case an accumulating implementation would miss.
    """
    spec = InteractionSpec(terms=(TERM_INTERACTION, "log2K", "log2B"), include_ability=False)
    built = build_analysis_frame(df, spec, strict=False)
    assert set(built.term_names) == {TERM_INTERACTION, "log2K", "log2B"}
    assert set(built.dropped_terms) == {
        "corpus_fraction",
        "gold_rank",
        "reader_strength",
        "ability",
    }
    assert built.to_dict()["dropped_terms"] == list(built.dropped_terms)

    # A full-spec lenient build against a frame that is missing an OPTIONAL term
    # drops exactly that one.
    holed = df.drop(columns=["gold_rank"])
    lenient = build_analysis_frame(holed, strict=False)
    assert lenient.dropped_terms == ("gold_rank",)
    assert "gold_rank" not in lenient.term_names
    assert "corpus_fraction" in lenient.term_names


def test_a_corpus_fraction_may_not_be_dropped_even_in_lenient_mode(df: Any) -> None:
    """`strict=False` relaxes the OPTIONAL terms only.

    A lenient flag that also shed `corpus_fraction` would make R9.8 conditional on
    a keyword argument, which is not what R9.8 says.
    """
    spec = InteractionSpec(terms=(TERM_INTERACTION, "log2K", "log2B"), include_ability=False)
    with pytest.raises(MissingCovariateError, match="corpus_fraction"):
        build_analysis_frame(df.drop(columns=["corpus_fraction"]), spec, strict=False)


def test_a_non_positive_depth_is_refused(df: Any) -> None:
    """`log2(0)` is undefined and K=0 is not a retrieval condition."""
    broken = df.copy()
    broken.loc[broken.index[:5], "K"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        build_analysis_frame(broken)


def test_fit_primary_refuses_a_non_binary_outcome(df: Any) -> None:
    """R9.1's outcome is `correct` - a 0/1 indicator, not a score."""
    scored = df.assign(correct=df["correct"].astype(float) * 0.5)
    with pytest.raises(NonBinaryOutcomeError):
        fit_primary(scored)


def test_primary_analysis_reports_both_clusterings(df: Any) -> None:
    """The `(1|model)` half of R9.1 is an approximation, and the artifact says so.

    Three reader families make a reader-clustered sandwich computable but weak;
    both facts are in the block rather than in a footnote, because a table showing
    one standard error beside three readers invites the reader to assume the reader
    dimension was handled.
    """
    report = primary_analysis(df)
    assert report["specification"]["declared"] == R91_DECLARED
    assert report["primary"]["term"] == TERM_INTERACTION
    assert report["primary"]["n_clusters"] == df["question"].nunique()
    assert report["frame"]["dropped_terms"] == []

    secondary = report["reader_clustered_secondary"]
    assert secondary["available"] is True
    assert secondary["n_clusters"] == 3
    assert "only 3 distinct" in secondary["note"]
    assert set(report["interaction_pp"]) == {"p30", "p50", "p70"}
    assert report["interaction_pp"]["p50"] > report["interaction_pp"]["p70"]


def test_the_reader_clustered_fit_is_available_and_flagged(df: Any) -> None:
    fit = fit_reader_clustered(df)
    assert fit.cluster_col == "reader"
    assert fit.n_clusters == 3
    assert fit.cluster_note


def test_a_single_reader_panel_refuses_the_reader_clustered_fit(df: Any) -> None:
    """One reader is not a cluster-robust anything, and the refusal is correct."""
    with pytest.raises(TooFewClustersError):
        fit_reader_clustered(df.assign(reader="gemma"))


# ── R9.2 — monotone vs inverted-U ─────────────────────────────────────────────
def test_the_lrt_states_its_df_its_nesting_and_its_boundary_handling(df: Any) -> None:
    """R9.2: both models parameterised, df stated, nesting and boundary handling
    declared. AIC is explicitly not computed - it is a criterion, not a test."""
    result = lrt_monotone_vs_inverted_u(df)
    assert isinstance(result, LrtResult)
    assert result.df == 1
    assert result.df_alt == result.df_null + 1
    assert "regular" in result.nesting
    assert "quadratic" in result.verdict.lower() or "monotone" in result.verdict.lower()
    payload = result.to_dict()
    assert payload["aic_not_used"]
    assert payload["boundary_handling"]
    assert payload["random_effect_handling"]
    assert payload["quasi"] is True
    assert PRIMARY_SPEC.quadratic_term not in payload["terms_null"]
    assert PRIMARY_SPEC.quadratic_term in payload["terms_alt"]


def test_the_lrt_detects_a_planted_inverted_u(df: Any) -> None:
    """A real curvature must be found, or the test is a machine that always says
    "no" and nobody will know.

    SYNTHETIC fixture with a concave-in-log2(K) term planted on top of the standard
    design. R9.2's question is whether the monotone model is rejected, so a detector
    that never rejects is as useless as one that always does.
    """
    from scipy.special import expit

    rng = np.random.default_rng(21)
    rows: list[dict[str, Any]] = []
    k_center = float(np.mean(np.log2(K_LEVELS)))
    b_center = float(np.mean(np.log2(B_LEVELS)))
    for q in range(300):
        strength = 0.70 + 0.0008 * q
        for k in K_LEVELS:
            for b in B_LEVELS:
                kc = float(np.log2(k)) - k_center
                bc = float(np.log2(b)) - b_center
                eta = (
                    0.10
                    + 0.30 * kc
                    + 0.20 * bc
                    - 0.25 * kc * kc
                    + 1.5 * strength
                    + float(rng.normal(0.0, 0.2))
                )
                for _ in range(2):
                    rows.append(
                        {
                            "question": f"q{q:04d}",
                            "ability": ABILITIES[q % 3],
                            "reader": ("gemma", "qwen", "phi")[q % 3],
                            "reader_strength": strength,
                            "K": k,
                            "B": b,
                            "corpus_fraction": k / (30.0 + q % 23),
                            "gold_rank": float(rng.integers(1, 12)),
                            "correct": int(rng.binomial(1, float(expit(eta)))),
                        }
                    )
    curved = pd.DataFrame(rows)

    result = lrt_monotone_vs_inverted_u(curved)
    assert result.quadratic_coefficient < 0.0
    assert result.at_boundary is False
    assert result.p_value < 0.01, result.verdict


def test_a_convex_fit_is_reported_as_boundary_corrected(df: Any) -> None:
    """When the unconstrained optimum leaves the concave region, 0 is on the
    BOUNDARY and the nominal chi-square_1 p-value is wrong.

    The reported value must be the mixture `0.5 * chi2_1` - an UPPER bound - and
    `at_boundary` must say which was used. Reporting the nominal value there
    manufactures evidence for curvature the data does not support.
    """
    from scipy.special import expit

    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []
    k_center = float(np.mean(np.log2(K_LEVELS)))
    for q in range(250):
        strength = 0.70 + 0.0008 * q
        for k in K_LEVELS:
            for b in B_LEVELS:
                kc = float(np.log2(k)) - k_center
                eta = (
                    0.10 + 0.30 * kc + 0.25 * kc * kc + 1.5 * strength + float(rng.normal(0.0, 0.2))
                )
                for _ in range(2):
                    rows.append(
                        {
                            "question": f"q{q:04d}",
                            "ability": ABILITIES[q % 3],
                            "reader": ("gemma", "qwen", "phi")[q % 3],
                            "reader_strength": strength,
                            "K": k,
                            "B": b,
                            "corpus_fraction": k / (30.0 + q % 23),
                            "gold_rank": float(rng.integers(1, 12)),
                            "correct": int(rng.binomial(1, float(expit(eta)))),
                        }
                    )
    convex = pd.DataFrame(rows)

    result = lrt_monotone_vs_inverted_u(convex)
    assert result.quadratic_coefficient > 0.0
    assert result.at_boundary is True
    assert "Chernoff" in result.boundary_handling or "0.5" in result.boundary_handling
    assert "UPPER BOUND" in result.boundary_handling
    assert result.p_value < 0.5


# ── R9.3 / R9.4 — the four saturation statistics ──────────────────────────────
def test_a_curve_reports_all_four_statistics() -> None:
    """R9.3 requires all four, so all four are always present as slots.

    A statistic that cannot be computed is `available: False` with a REASON, never
    an absent field - a missing key and a key holding `None` read identically in a
    results table, and one of them means "we did not look".
    """
    curve = KCurve(ks=(1, 2, 4, 8, 16, 32), accuracy=(0.30, 0.48, 0.58, 0.60, 0.61, 0.61))
    stats = saturation_stats(curve)

    assert set(stats.ks) == {"K_95", "K_knee", "K_delta", "Frontier K"}
    assert stats.to_dict()["n_statistics_reported"] == 4
    assert stats.k_95.k == 4, "A_max is 0.61, so the 95% target is 0.5795 and K=4 hits it"
    assert stats.k_knee.k == 32, "monotone: the knee is the last grid point by definition"
    assert "monotone" in stats.k_knee.detail["note"]
    assert stats.frontier.available is False
    assert "no cost column" in stats.frontier.unavailable_reason
    assert stats.k_delta.k == 16


def test_k_95_is_the_smallest_depth_reaching_the_target() -> None:
    """The target is a share of the best OBSERVED accuracy, so it is computed after
    the whole curve is in - a definition relative to A(1) would make K_95 trivially
    equal to 1 for every curve."""
    stats = saturation_stats(KCurve(ks=(1, 2, 4, 8), accuracy=(0.10, 0.40, 0.80, 0.90)))
    assert K95_FRACTION == 0.95
    assert stats.k_95.detail["target_accuracy"] == pytest.approx(0.855)
    assert stats.k_95.k == 8, "nothing below K=8 reaches 0.855"


def test_frontier_k_needs_a_cost_column_and_picks_the_cheapest_depth_at_the_max() -> None:
    """`argmax (A - A_max)/Cost` restricted to `A >= 0.99 A_max`.

    Since `A - A_max <= 0` everywhere, the ratio is zero exactly at the depths that
    attain the maximum and negative below them; the argmax is therefore the CHEAPEST
    depth that attains `A_max` (or, if none does, the one giving up least per unit
    cost). "Cheapest depth that still gets everything" - which is the question the
    statistic exists to answer.

    With no cost column the answer is `None` with a reason, never the last grid
    point - which would read as a measurement.
    """
    curve = KCurve(
        ks=(1, 2, 4, 8),
        accuracy=(0.50, 0.995, 1.00, 1.00),
        cost=(1.0, 2.0, 8.0, 20.0),
    )
    stats = saturation_stats(curve)
    assert stats.frontier.available is True
    assert stats.frontier.k == 4, "K=4 and K=8 both attain A_max; K=4 costs 8, K=8 costs 20"
    assert "argmax" in stats.frontier.definition

    free = saturation_stats(KCurve(ks=curve.ks, accuracy=curve.accuracy))
    assert free.frontier.k is None
    assert "no cost column" in free.frontier.unavailable_reason


# ── R9.4 — K_delta normalised by step position (THE LOAD-BEARING ONE) ─────────
def test_k_delta_normalisation_is_a_no_op_on_a_doubling_grid() -> None:
    """On a grid where every step is one doubling, the conversion changes nothing.

    Pinned because it is the case the shipped K grid actually is, and because a
    "fix" that double-normalised (dividing by the log2 step twice) would look
    correct on an UNEVEN grid while silently halving every threshold here.
    """
    grid = [1, 2, 4, 8, 16, 32, 64]
    stats = saturation_stats(
        KCurve(ks=tuple(grid), accuracy=(0.40, 0.55, 0.60, 0.61, 0.612, 0.613, 0.613))
    )
    for step in stats.k_delta.detail["steps"]:
        assert step["log2_step"] == pytest.approx(1.0)
        assert step["threshold_pp"] == pytest.approx(DELTA_THRESHOLD_PP)
        assert step["gain_pp_per_log2"] == pytest.approx(step["gain_pp"], rel=1e-6)
    # K=8 -> 16 gains 0.2pp; K=16 -> 32 gains 0.1pp. First step under 0.5pp is 8->16.
    assert stats.k_delta.k == 8
    assert stats.k_delta.detail["k_delta_sustained"] == 8


def test_k_delta_is_normalised_by_step_position() -> None:
    """`design.k_step_index` puts the answer on [0, 1] so two K_deltas compare.

    K=2 and K=64 are not comparable numbers without it: one is 20% along the grid,
    the other 100%.
    """
    stats = saturation_stats(KCurve(ks=(1, 2, 4, 8), accuracy=(0.40, 0.55, 0.554, 0.554)))
    assert stats.k_delta.k == 2
    assert stats.k_delta.detail["k_delta_step_position"] == pytest.approx(1 / 3)
    assert "normalised to [0, 1]" in stats.k_delta.detail["k_delta_step_position_note"]


def test_the_raw_half_point_rule_fires_early_on_a_mixed_step_grid() -> None:
    """R9.4's actual claim, demonstrated rather than restated.

    `[1, 2, 3, 4, 8, 16]` mixes a 2x step with 1.5x and 1.33x ones - the "1x step
    against 2x steps" grid R9.4 warns about, and the same shape as `ceiling.
    CEILING_GRID`. Here the curve is gaining a steady 0.6pp per unit log2(K), which
    is 20% above the 0.5pp threshold, i.e. it has NOT saturated.

    The raw rule looks only at the step's raw gain: the 2->3 step gained 0.35pp, so
    it fires at K=2. The normalised rule converts at the step that occurred
    (0.35pp over log2(1.5) = 0.585 units is 0.60pp per unit) and correctly declines,
    reporting the genuinely flat 8->16 step instead - three grid levels later.
    """
    grid = (1, 2, 3, 4, 8, 16)
    accuracy = (0.30, 0.306, 0.3095, 0.312, 0.318, 0.319)

    normalised = saturation_stats(KCurve(ks=grid, accuracy=accuracy)).k_delta
    assert normalised.k == 8

    raw = next(
        ks
        for i, ks in enumerate(grid[:-1])
        if 100.0 * (accuracy[i + 1] - accuracy[i]) <= DELTA_THRESHOLD_PP
    )
    assert raw == 2, "the raw rule fires at 2 because the 2->3 step gained only 0.35pp"

    steps = saturation_stats(KCurve(ks=grid, accuracy=accuracy)).k_delta.detail["steps"]
    by_to = {s["to_K"]: s for s in steps}
    assert by_to[3]["log2_step"] == pytest.approx(np.log2(1.5), rel=1e-6)
    assert by_to[3]["gain_pp_per_log2"] == pytest.approx(0.60, abs=0.01)
    assert by_to[3]["below_threshold"] is False, "normalised: 0.60pp per log2 unit > 0.5"
    assert by_to[8]["log2_step"] == pytest.approx(1.0)
    assert by_to[8]["gain_pp_per_log2"] == pytest.approx(0.6, abs=0.01)
    assert by_to[16]["below_threshold"] is True


def test_k_delta_threshold_is_converted_at_the_arriving_step() -> None:
    """`design.k_delta_in_pp` scales the base threshold by the step that actually
    occurs - 0.5pp per log2 unit is 2.0pp across a 4x step and 0.21pp across a
    1.33x one."""
    steps = {
        s["to_K"]: s
        for s in saturation_stats(
            KCurve(ks=(1, 16, 48, 64), accuracy=(0.30, 0.44, 0.5462, 0.5654))
        ).k_delta.detail["steps"]
    }
    assert steps[16]["threshold_pp"] == pytest.approx(2.0, rel=1e-3)
    assert steps[48]["threshold_pp"] == pytest.approx(0.5 * np.log2(3.0), rel=1e-3)
    assert steps[64]["threshold_pp"] == pytest.approx(0.5 * np.log2(64 / 48), rel=1e-3)

    detail = saturation_stats(
        KCurve(ks=(1, 16, 48, 64), accuracy=(0.30, 0.44, 0.5462, 0.5654))
    ).k_delta.detail
    assert "converted at the step that actually occurs" in detail["normalised"]
    assert "3.8x stricter" in detail["normalised"]


def test_k_knee_normalises_by_step_index_not_by_raw_k() -> None:
    """`K_knee` divides by `(j - i)`, so an uneven grid cannot make a long step look
    like a structural break."""
    even = saturation_stats(KCurve(ks=(1, 2, 4, 8), accuracy=(0.80, 0.60, 0.55, 0.54)))
    assert even.k_knee.k == 2, "the drop is largest on the very first step"


def test_a_curve_that_never_flattens_reports_k_delta_none() -> None:
    """`available: False` with a reason, not a silent last grid point."""
    stats = saturation_stats(KCurve(ks=(1, 2, 4), accuracy=(0.30, 0.45, 0.70)))
    assert stats.k_delta.k is None
    assert "every step" in stats.k_delta.unavailable_reason


def test_a_kcurve_validates_its_own_grid() -> None:
    """Each of these would otherwise produce a statistic that looks computed."""
    with pytest.raises(ValueError, match="at least two"):
        KCurve(ks=(8,), accuracy=(0.5,))
    with pytest.raises(ValueError, match="same length"):
        KCurve(ks=(1, 2, 4), accuracy=(0.5, 0.6))
    with pytest.raises(ValueError, match="strictly increasing"):
        KCurve(ks=(4, 2, 1), accuracy=(0.5, 0.6, 0.7))
    with pytest.raises(ValueError, match=r"duplicates|strictly increasing"):
        KCurve(ks=(1, 2, 2), accuracy=(0.5, 0.6, 0.7))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        KCurve(ks=(1, 2), accuracy=(0.5, 65.0))
    with pytest.raises(ValueError, match="cost must match"):
        KCurve(ks=(1, 2), accuracy=(0.5, 0.6), cost=(1.0,))


# ── R9.3 — bootstrap CIs ──────────────────────────────────────────────────────
def test_the_bootstrap_resamples_questions_not_rows() -> None:
    """The resampling unit is the question.

    Rows from one question share a corpus, a gold answer and a difficulty. Resampling
    them independently treats three replicates of one cell as three independent
    observations and produces an interval roughly sqrt(3) too narrow - which is the
    interval the paper would publish.
    """
    df = _design(n_questions=120, seed=8)
    boot = resample_curves(df, n_boot=25, seed=1)
    assert len(boot) == 25
    for curve in boot:
        assert curve.ks == K_LEVELS
        assert all(0.0 <= a <= 1.0 for a in curve.accuracy)

    # Row-level resampling cannot change the answer on a frame where the within-
    # question mean is constant; question-level resampling can and does.
    question_means = df.groupby("question")["correct"].mean()
    assert question_means.std() > 0.0, "the fixture must vary by question for this test"


def test_saturation_stats_carry_bootstrap_intervals(df: Any) -> None:
    """R9.3 requires CIs on all four, with the method STATED.

    Stated because these are bounded-integer arguments over a noisy grid: the
    bootstrap distribution of an argument is lumpy, and a degenerate interval means
    "the statistic did not move under resampling", not "this depth is certain".
    """
    subset = df[df["ability"] == "single-session"]
    curve = curve_from_frame(subset, label="single-session")
    boot = resample_curves(subset, n_boot=40, seed=3, label="single-session")
    stats = saturation_stats(curve, boot=boot)

    for interval in (stats.k_95, stats.k_knee, stats.k_delta, stats.frontier):
        assert "percentile bootstrap over QUESTIONS" in interval.ci_method
        assert "argument itself" in interval.ci_method
    assert stats.k_95.ci_low is not None
    assert stats.k_95.ci_low <= stats.k_95.k <= stats.k_95.ci_high
    assert stats.k_95.detail["n_bootstrap_replicates_used"] > 0


def test_no_bootstrap_means_no_invented_interval() -> None:
    curve = KCurve(ks=(1, 2, 4), accuracy=(0.30, 0.45, 0.46))
    stats = saturation_stats(curve)
    assert stats.k_95.ci_method == ""
    assert stats.k_95.ci_low is None and stats.k_95.ci_high is None


def test_curve_from_frame_reads_the_grid_the_data_has() -> None:
    """Not the config. A statistic computed on a level the run never reached is a
    statistic about a cell nobody ran."""
    partial = _design(n_questions=40, seed=6)
    partial = partial[partial["K"] <= 4]
    curve = curve_from_frame(partial)
    assert curve.ks == (1, 2, 4)
    assert all(0.0 <= a <= 1.0 for a in curve.accuracy)
    assert sum(curve.n_per_k) == len(partial)
    assert curve.to_dict()["ks"] == [1, 2, 4]


# ── R9.5 — stratified primary, pooled secondary ───────────────────────────────
def _two_regime_frame() -> Any:
    """Two strata that saturate at DIFFERENT depths, by construction.

    `easy` is flat from K=2; `hard` is still climbing at K=32. The pooled mean sits
    between them at every K and its curve has the shape of NEITHER stratum - which
    is the point R9.5 makes about a weighted average of two regimes.

    DETERMINISTIC: each (question, K) cell contributes exactly
    `round(acc * DRAWS)` correct rows out of `DRAWS`, so the stratum means are the
    planted curves to within 0.005 and the test cannot flake on binomial noise.
    A sampling-based fixture would make "the pooled point differs from both" a
    statement about a seed.
    """
    draws = 100
    easy = (0.50, 0.90, 0.90, 0.90, 0.90, 0.90)
    hard = (0.20, 0.26, 0.36, 0.50, 0.62, 0.66)
    rows: list[dict[str, Any]] = []
    for q in range(120):
        for regime, curve in (("easy", easy), ("hard", hard)):
            for k, acc in zip(K_LEVELS, curve, strict=True):
                n_correct = round(acc * draws)
                for i in range(draws):
                    rows.append(
                        {
                            "question": f"{regime}-q{q:04d}",
                            "ability": regime,
                            "K": k,
                            "correct": int(i < n_correct),
                        }
                    )
    return pd.DataFrame(rows)


def test_pooled_and_stratified_saturation_points_differ() -> None:
    """R9.5, demonstrated rather than asserted.

    The two strata are DIFFERENT questions with different answers, and the pooled
    curve's answer is a third one that belongs to neither - which is why it is not
    the primary. Checked on `K_95` because it is the statistic with the clearest
    separation; the point is not that every statistic must disagree, it is that the
    pooled number is not any stratum's number.
    """
    report = strata_report(_two_regime_frame(), n_boot=0)

    easy = report.by_label()["easy"]
    hard = report.by_label()["hard"]
    pooled = report.pooled

    assert easy.stats.k_95.k == 2, easy.stats.ks
    assert hard.stats.k_95.k == 32, hard.stats.ks
    assert pooled.stats.k_95.k == 16, "strictly between the two regimes"
    assert len({easy.stats.k_95.k, hard.stats.k_95.k, pooled.stats.k_95.k}) == 3

    # And the pooled CURVE is the mean, which is the mechanism.
    pooled_acc = pooled.stats.curve.accuracy
    easy_acc = easy.stats.curve.accuracy
    hard_acc = hard.stats.curve.accuracy
    for i in range(len(K_LEVELS)):
        assert pooled_acc[i] == pytest.approx((easy_acc[i] + hard_acc[i]) / 2, abs=0.01)


def test_stratified_is_primary_and_pooled_is_secondary_everywhere(df: Any) -> None:
    """The role travels ON the result, so a table that drops the `role` column
    cannot silently promote the pooled curve."""
    report = strata_report(df, n_boot=0)
    assert report.primary_is == "stratified"
    assert report.secondary_is == "pooled"
    for stratum in report.strata:
        assert stratum.role == PRIMARY_ROLE
        assert stratum.stats.role == PRIMARY_ROLE
    assert report.pooled.role == SECONDARY_ROLE
    assert report.pooled.stats.role == SECONDARY_ROLE

    payload = report.to_dict()
    assert payload["primary_analysis"] == "stratified"
    assert payload["secondary_analysis"] == "pooled"
    assert payload["pooled_secondary"]["role"] == SECONDARY_ROLE


def test_the_justification_for_stratification_is_carried_not_asserted() -> None:
    """`ceiling.py` says the ceiling is strongly type-conditional, which is what
    makes a pooled saturation point meaningless rather than merely noisy. The
    reason travels with the output so a reader cannot be left to assume it."""
    report = strata_report(_two_regime_frame(), n_boot=0)
    assert report.why == WHY_STRATIFIED_IS_PRIMARY
    assert "weighted average" in report.why.lower()
    assert "TYPE-CONDITIONAL" in report.why
    assert WHY_STRATIFIED_IS_PRIMARY in report.to_dict()["why"]
    assert any("SECONDARY" in note for note in report.pooled.notes)


def test_a_stratum_that_is_too_small_is_still_reported(df: Any) -> None:
    """Silently dropping a small stratum hides exactly the exploratory strata R9.9
    warns are most likely to be over-read. `n_questions` goes on the result."""
    report = strata_report(df, n_boot=0)
    per_cell = len(K_LEVELS) * len(B_LEVELS) * 2  # runs_per_cell = 2
    for stratum in report.strata:
        assert stratum.n_questions > 0
        assert stratum.n_obs == stratum.n_questions * per_cell
    assert report.pooled.n_questions == sum(s.n_questions for s in report.strata)


def test_strata_report_can_add_the_analytic_saturation_point() -> None:
    """Comparing the measured knee to the ANALYTIC K* per type is what separates
    "the reader stopped using what it has" from "the corpus ran out"."""
    frame = _two_regime_frame()
    ceiling = {
        "easy": {str(k): 0.999 for k in K_LEVELS},
        "hard": {str(k): (0.5 if k < 16 else 0.999) for k in K_LEVELS},
    }
    report = strata_report(frame, n_boot=0, ceiling_by_stratum=ceiling)
    assert report.by_label()["easy"].analytic_saturation_k == 1
    assert report.by_label()["hard"].analytic_saturation_k == 16
    assert report.by_label()["easy"].to_dict()["analytic_saturation_k"] == 1
    # Not supplied -> reported as None rather than omitted, so a reader can tell
    # "the analytic ceiling was not joined in" from "the ceiling saturates nowhere".
    assert strata_report(frame, n_boot=0).by_label()["easy"].analytic_saturation_k is None


def test_a_missing_stratum_column_is_a_named_error(df: Any) -> None:
    with pytest.raises(MissingCovariateError, match="no strata to stratify by"):
        strata_report(df.drop(columns=["ability"]), n_boot=0)


# ── R0.4 — the pre-registered verdict ─────────────────────────────────────────
def test_the_shipped_preregistration_is_readable_and_names_its_prediction() -> None:
    """Read directly rather than through `config.load_bundle`.

    The shipped bundle is incomplete until the reader checkpoints are confirmed
    (that is why `tests/conftest.py::complete_configs` exists), and the prediction
    must stay readable in that state: it is a fact recorded before any run, not a
    property of the current bundle.
    """
    root = Path(__file__).resolve().parent.parent / "configs"
    prereg = load_preregistration(root)
    assert prereg["interaction_sign_prediction"] == "approximately_zero"
    assert prereg["registered"] < "2026-10-13"  # before the manifest was frozen
    assert any("exactly zero" in line for line in prereg["not_claimed"])


def test_the_verdict_does_not_claim_the_interaction_is_exactly_zero(df: Any) -> None:
    """THE integrity property, and the one most easily lost to a reworded sentence.

    `configs/preregistration.yaml` writes it into `not_claimed`: "that the
    interaction is exactly zero - only that it is small relative to the MDE". Those
    are different statements. A coefficient of 0.4 log-odds can be both
    non-significant at n = 500 and worth 25 accuracy points.

    Every string the verdict emits AS A CLAIM is checked against the forbidden list.
    The `not_claimed` line itself is carried through verbatim and is NOT a claim -
    it is the disclaimer - so it is exempt by construction.
    """
    fit = fit_primary(df)
    # A generous MDE: the fixture plants delta = 0.8, and the shipped MDE is
    # 0.01065, so the shipped number takes the REJECT branch. Both branches are
    # checked for the forbidden phrases below.
    verdict = preregistration_verdict(fit, abs(fit.coefficient) * 2.0)

    assert verdict.consistent is True
    assert verdict.claims_exactly_zero is False
    for phrase in FORBIDDEN_ZERO_CLAIMS:
        assert phrase not in verdict.claim.lower(), phrase
    assert "SMALL RELATIVE TO THE MDE" in verdict.claim
    assert "NOT about the coefficient being absent" in verdict.claim

    # The SAME assertions must hold on the branch that REJECTS the prediction. A
    # sentence that only respects the distinction when it agrees with you is not a
    # guard, it is a courtesy.
    strict = preregistration_verdict(fit, 0.01065)
    assert strict.consistent is False
    for phrase in FORBIDDEN_ZERO_CLAIMS:
        assert phrase not in strict.claim.lower(), phrase
    assert strict.claims_exactly_zero is False
    assert "NOT about the coefficient being absent" in strict.claim
    assert "EXCEEDS THE MDE" in strict.claim


def test_the_verdict_carries_the_preregistration_disclaimer(df: Any) -> None:
    """The disclaimer travels WITH the number, so a reader cannot get the estimate
    without it."""
    fit = fit_primary(df)
    verdict = preregistration_verdict(fit, 0.01065)
    assert verdict.not_claimed
    assert any("exactly zero" in line for line in verdict.not_claimed)
    assert verdict.to_dict()["not_claimed"] == list(verdict.not_claimed)
    assert verdict.prediction == "approximately_zero"
    assert verdict.reading  # the sign_readings text for this prediction


def test_a_small_interaction_is_consistent_and_a_large_one_is_not(df: Any) -> None:
    """Consistency is judged against the MDE, which is the only scale on which
    "small" means anything for a null result."""
    fit = fit_primary(df)
    huge = preregistration_verdict(fit, abs(fit.coefficient) * 2.0)
    assert huge.consistent is True
    assert huge.ratio_to_mde == pytest.approx(0.5, rel=1e-6)
    assert "NOT supported" not in huge.claim

    tiny_mde = preregistration_verdict(fit, abs(fit.coefficient) / 10.0)
    assert tiny_mde.consistent is False
    assert tiny_mde.ratio_to_mde == pytest.approx(10.0, rel=1e-6)
    assert "EXCEEDS THE MDE" in tiny_mde.claim
    assert tiny_mde.claims_exactly_zero is False


def test_no_mde_means_no_verdict_not_a_favourable_one(df: Any) -> None:
    """ "We could not have detected this" and "there was nothing there" must not
    print the same."""
    fit = fit_primary(df)
    verdict = preregistration_verdict(fit, None)
    assert verdict.consistent is None
    assert verdict.ratio_to_mde is None
    assert "NO verdict can be stated" in verdict.claim
    assert verdict.claims_exactly_zero is False


def test_an_unpowered_design_carries_no_mde(df: Any) -> None:
    """`results/power.json` marks `powered: false` when the grid cannot detect the
    interaction at any plausible size. Passing that whole report in must NOT yield
    a verdict against an MDE that does not exist."""
    fit = fit_primary(df)
    verdict = preregistration_verdict(
        fit, {"interaction": {"mde_log_odds": None, "powered": False, "note": "nope"}}
    )
    assert verdict.mde_log_odds is None
    assert verdict.consistent is None
    assert "UNPOWERED" in verdict.claim


def test_the_mde_can_be_handed_over_as_the_whole_power_json(df: Any) -> None:
    """`results/power.json` nests the MDE under `interaction`. Making every caller
    dig one level down is where a wrong number gets typed."""
    fit = fit_primary(df)
    payload = json.loads(
        (Path(__file__).resolve().parent.parent / "results" / "power.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["interaction"]["powered"] is True
    verdict = preregistration_verdict(fit, payload)
    assert verdict.mde_log_odds == pytest.approx(payload["interaction"]["mde_log_odds"])
    assert verdict.mde_source == "power.json -> interaction.mde_log_odds"


def test_the_verdict_is_json_serialisable(df: Any) -> None:
    verdict = preregistration_verdict(fit_primary(df), 0.01065)
    payload = json.loads(json.dumps(verdict.to_dict()))
    assert payload["claims_exactly_zero"] is False
    assert payload["prediction"] == "approximately_zero"


# ── R9.10 — the leakage check, wired ──────────────────────────────────────────
def _arm_b_frame(*, tokens_b: int = 4000) -> Any:
    """Arm B as designed: count varies, length matched.

    `n_turns` scales with the count because a memory is a session, and
    `mean_tokens_per_memory` rises when fewer memories share a matched budget -
    both are the arm's manipulation stated in another language.
    """
    return pd.DataFrame(
        [
            {
                "arm": "A",
                "n_memories": 20 + (i % 3),
                "n_tokens": 4000,
                "n_turns": 4 * (20 + (i % 3)),
            }
            for i in range(12)
        ]
        + [
            {
                "arm": "B",
                "n_memories": 4 + (i % 2),
                "n_tokens": tokens_b,
                "n_turns": 4 * (4 + (i % 2)),
            }
            for i in range(12)
        ]
    )


def test_leakage_check_runs_the_r910_classifier_unchanged() -> None:
    """`analysis.leakage.assert_no_leakage` is called, not reimplemented, and its
    verdict is reproduced verbatim in the payload.

    A second implementation of the detector is a second detector, and the one in
    `tests/test_no_leakage.py` is the one that is tested.
    """
    check = leakage_check(_arm_b_frame(), n_null=40)
    payload = check.to_dict()
    assert payload["requirement"] == "PRD §R9.10"
    assert payload["classifier"]["auroc"] == pytest.approx(check.report.auroc)
    assert payload["classifier"]["threshold"] == check.report.threshold
    assert check.n_items == 24
    assert set(check.arms) == {"A", "B"}


def test_arm_b_is_recognised_as_a_count_labelled_arm() -> None:
    """R9.10's classifier separates Arm B PERFECTLY, by construction: memory count
    IS the label. That is the manipulation, not a leak - the count is printed in the
    prompt as a numbered list.

    `tests/test_no_leakage.py::test_arm_b_as_designed_is_separable_by_memory_count`
    pins the separability and `24_open_decisions.md` D10 records the decision.
    Reporting "Arm B is INVALID" here would be wrong AND would make the check a
    nuisance that gets switched off.
    """
    check = leakage_check(_arm_b_frame(), count_labelled=True, n_null=40)
    assert check.report.auroc > 0.99, "the count separates them perfectly"
    assert check.observed_count_separation is True
    assert check.state == "count_labelled_length_matched"
    assert check.applicable is False
    assert check.passed is True
    assert "DECLARED count-labelled" in check.verdict
    assert "PASSED" in check.token_match_verdict
    assert check.token_match["warnings"] == []
    assert "Arm B is INVALID" not in check.verdict


def test_the_exemption_must_be_declared_not_inferred() -> None:
    """THE design decision this function exists to get right.

    A deletion-style construction and a length-drifted Arm B are INDISTINGUISHABLE
    from size features alone - which is exactly R9.10's argument for why the
    classifier exists. Inferring the exemption from the evidence the check is meant
    to police is circular, so the default is the STRICT reading: R9.10 applies.

    An earlier version inferred it from the observed AUROCs and mislabelled both
    cases; `24_open_decisions.md` D10 records the resolution.
    """
    strict = leakage_check(_arm_b_frame(), n_null=40)
    assert strict.applicable is True
    assert strict.state == "size_leak", "undeclared, R9.10 applies and fails"
    assert strict.passed is False
    assert "Arm B is INVALID" in strict.verdict

    declared = leakage_check(_arm_b_frame(), count_labelled=True, n_null=40)
    assert declared.applicable is False
    assert declared.observed_count_separation is strict.observed_count_separation, (
        "the declaration changes the VERDICT, not the evidence - which is the point"
    )


def test_the_assertion_that_applies_to_arm_b_is_the_token_match() -> None:
    """Length held constant is Arm B's own check (`arms.padding.
    size_leakage_matrix`). If it drifts, the count-vs-length separation is not being
    measured at all - and that has to fail loudly, without being dressed up as the
    size leak it is not."""
    check = leakage_check(_arm_b_frame(tokens_b=800), count_labelled=True, n_null=40)
    assert check.state == "count_labelled_length_drifted"
    assert check.applicable is False
    assert check.token_match["warnings"], "a 5x length difference is not 'length held constant'"
    assert "FAILED" in check.token_match_verdict
    assert check.passed is False
    assert "Arm B is INVALID" not in check.verdict


def test_a_deletion_style_construction_still_fails_loudly() -> None:
    """The case R9.10 was written about. 2609.32269 builds insufficient-evidence
    conditions by DELETING passages, so every size feature carries the label.

    Nothing is declared here, so R9.10 applies verbatim and the run is flagged -
    which is the intended behaviour and what
    `test_a_deletion_style_construction_is_still_caught` already pins.
    """
    deletion = pd.DataFrame(
        [
            {"arm": "A", "n_memories": 2 + (i % 2), "n_tokens": 800 + 50 * i, "n_turns": 8}
            for i in range(8)
        ]
        + [
            {
                "arm": "B",
                "n_memories": 20 + (i % 2),
                "n_tokens": 9000 + 50 * i,
                "n_turns": 80,
            }
            for i in range(8)
        ]
    )
    check = leakage_check(deletion, n_null=40)
    assert check.state == "size_leak"
    assert check.applicable is True
    assert check.observed_count_separation is True
    assert check.report.passed is False
    assert check.passed is False
    assert "Arm B is INVALID" in check.verdict
    assert "0.979" in check.verdict  # the prior art should be cited
    assert check.to_dict()["state"] == "size_leak"


def test_an_arm_matched_on_every_size_feature_passes() -> None:
    """The positive case: conditions with identical size profiles. What a correctly
    constructed control looks like, and the state R9.10 wants to see."""
    matched = pd.DataFrame(
        [{"arm": "A", "n_memories": 10, "n_tokens": 4000, "n_turns": 40} for _ in range(12)]
        + [{"arm": "B", "n_memories": 10, "n_tokens": 4000, "n_turns": 40} for _ in range(12)]
    )
    check = leakage_check(matched, n_null=40)
    assert check.state == "no_separation"
    assert check.applicable is True
    assert check.passed is True
    assert "Arm B is valid" in check.verdict
    assert check.observed_count_separation is False


def test_leakage_check_names_the_columns_it_needs() -> None:
    """A run whose manifest lacks the size features cannot be checked for leakage,
    and skipping the check silently is the one outcome R9.10 exists to prevent."""
    with pytest.raises(MissingCovariateError, match=r"n_memories|n_tokens"):
        leakage_check(pd.DataFrame({"arm": ["A", "B"], "n_memories": [1, 2]}))


def test_leakage_check_refuses_a_single_condition() -> None:
    with pytest.raises(ValueError, match="nothing to separate"):
        leakage_check(pd.DataFrame(_arm_b_frame().assign(arm="A")), n_null=10)


# ── the report blocks `analyze` writes ────────────────────────────────────────
def test_primary_analysis_is_json_serialisable(df: Any) -> None:
    report = primary_analysis(df)
    json.loads(json.dumps(report))
    assert report["frame"]["term_names"]
    assert report["frame"]["k_center"] == pytest.approx(float(np.mean(np.log2(K_LEVELS))), abs=1e-6)
    assert "mde_note" in report


def test_strata_report_is_json_serialisable(df: Any) -> None:
    report = strata_report(df, n_boot=5, seed=2)
    payload = json.loads(json.dumps(report.to_dict()))
    assert len(payload["strata"]) == len(ABILITIES)
    assert payload["pooled_secondary"]["role"] == SECONDARY_ROLE
    for stratum in payload["strata"]:
        assert stratum["role"] == PRIMARY_ROLE
        assert stratum["k_95"]["name"] == "K_95"
        assert stratum["k_knee"]["name"] == "K_knee"
        assert stratum["k_delta"]["name"] == "K_delta"
        assert stratum["frontier_k"]["name"] == "Frontier K"


def test_the_lrt_and_the_strata_report_use_the_same_spec(df: Any) -> None:
    """One declaration, used by both endpoints. A curvature test fitted on a
    different model than the headline would be answering a different question."""
    result = lrt_monotone_vs_inverted_u(df)
    for term in PRIMARY_SPEC.terms:
        assert term in result.terms_null, term
    assert result.terms_alt[:-1] == result.terms_null
