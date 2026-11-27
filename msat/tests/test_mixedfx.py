from __future__ import annotations

import ast
import json
import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from msat.analysis.mixedfx import (
    ALPHA,
    MIN_CLUSTERS,
    MissingCovariateError,
    NonBinaryOutcomeError,
    PrimaryFit,
    RankDeficientDesignError,
    TooFewClustersError,
    assert_binary_outcome,
    cluster_counts,
    fit_gee_binomial,
    logodds_to_pp,
    p_from_z,
    z_value,
)

INTERACTION = "log2K:log2B"
K_LEVELS = (1, 2, 4, 8, 16, 32)
B_LEVELS = (128.0, 256.0, 512.0)

DELTA_INTERACTION = 0.80
BETA_LOG2K = 0.30
BETA_LOG2B = 0.20

TERMS = ("log2K", "log2B", INTERACTION, "corpus_fraction", "gold_rank", "reader_strength")


def _synthetic(
    *,
    n_questions: int = 250,
    runs_per_cell: int = 2,
    delta: float = DELTA_INTERACTION,
    seed: int = 4,
) -> Any:
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
                + BETA_LOG2K * kc
                + BETA_LOG2B * bc
                + delta * kc * bc
                + 1.5 * strength
                + float(rng.normal(0.0, 0.25))
            )
            p = 1.0 / (1.0 + np.exp(-eta))
            for _ in range(runs_per_cell):
                rows.append(
                    {
                        "question": f"q{q:04d}",
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


def _design(df: Any) -> Any:
    from msat.design import design_matrix

    log2k = np.log2(df["K"].to_numpy(dtype=float))
    log2b = np.log2(df["B"].to_numpy(dtype=float))
    x = design_matrix(
        log2k,
        log2b,
        df["corpus_fraction"].to_numpy(dtype=float),
        df["gold_rank"].to_numpy(dtype=float),
        df["reader_strength"].to_numpy(dtype=float),
        True,
    )
    out = df.copy()
    out["log2K"] = x[:, 1]
    out["log2B"] = x[:, 2]
    out[INTERACTION] = x[:, 3]
    return out


@pytest.fixture(scope="module")
def frame() -> Any:
    return _design(_synthetic())


@pytest.fixture(scope="module")
def small_frame() -> Any:
    return _design(_synthetic(n_questions=60, seed=9))


def _imported_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_module_in_analysis_imports_mixedlm() -> None:
    analysis_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "msat" / "analysis"
    for module in ("mixedfx.py", "interaction.py"):
        names = _imported_names(analysis_dir / module)
        offenders = {n for n in names if "MixedLM" in n or "mixed_linear_model" in n}
        assert not offenders, f"{module} imports {offenders}"

    import sys

    assert "statsmodels.regression.mixed_linear_model" not in sys.modules


def test_the_result_names_the_estimator_it_used(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    estimator = fit.to_dict()["estimator"]
    assert "GEE" in estimator
    assert "Binomial" in estimator
    assert "MixedLM" not in estimator


def test_gee_keeps_fitted_probabilities_in_the_unit_interval(frame: Any) -> None:
    from statsmodels.genmod.cov_struct import Independence
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    x = np.column_stack([np.ones(len(frame))] + [frame[t].to_numpy() for t in TERMS])
    y = frame["correct"].to_numpy(dtype=float)
    groups = frame["question"].to_numpy()
    result = GEE(y, x, groups=groups, family=Binomial(), cov_struct=Independence()).fit()

    probs = result.fittedvalues
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION).converged is True


def test_matches_the_hand_written_sandwich(frame: Any) -> None:
    from msat.design import fit_logit_cluster_robust

    x = np.column_stack([np.ones(len(frame))] + [frame[t].to_numpy() for t in TERMS])
    y = frame["correct"].to_numpy(dtype=float)
    codes = pd.factorize(frame["question"])[0]
    beta, se = fit_logit_cluster_robust(y, x, codes)

    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    for i, name in enumerate(("intercept", *TERMS)):
        assert fit.coefficients[name] == pytest.approx(beta[i], abs=1e-4), name
        assert fit.std_errors[name] == pytest.approx(se[i], rel=5e-3), name


def test_a_non_binary_outcome_raises_rather_than_coercing(frame: Any) -> None:
    bad = frame.copy()
    bad["correct"] = bad["correct"].astype(float) + 2.0
    with pytest.raises(NonBinaryOutcomeError, match=r"must be 0/1"):
        fit_gee_binomial(bad, TERMS, "question")


def test_a_continuous_score_in_zero_one_is_still_refused(frame: Any) -> None:
    with pytest.raises(NonBinaryOutcomeError, match="must be 0/1"):
        assert_binary_outcome([0.0, 0.4, 1.0, 0.83])
    assert_binary_outcome([0.0, 1.0, 1.0, 0.0])
    assert_binary_outcome([True, False, True])
    with pytest.raises(NonBinaryOutcomeError, match="must be 0/1"):
        fit_gee_binomial(
            frame.assign(correct=frame["correct"].astype(float) * 0.5),
            TERMS,
            "question",
        )


def test_a_non_numeric_outcome_names_its_values(frame: Any) -> None:
    labels = frame.assign(correct="yes")
    with pytest.raises(NonBinaryOutcomeError, match="not numeric"):
        fit_gee_binomial(labels, TERMS, "question")


def test_a_nan_outcome_is_refused_not_dropped(frame: Any) -> None:
    with pytest.raises(NonBinaryOutcomeError, match="NaN/inf"):
        assert_binary_outcome([0.0, 1.0, float("nan")])


def test_one_cluster_is_refused(frame: Any) -> None:
    single = frame[frame["question"] == frame["question"].iloc[0]].copy()
    with pytest.raises(TooFewClustersError, match="cluster"):
        fit_gee_binomial(single, TERMS, "question")
    assert MIN_CLUSTERS == 2


def test_an_empty_frame_is_refused() -> None:
    empty = pd.DataFrame({"correct": [], "question": [], "x": []})
    with pytest.raises(ValueError, match="no usable rows"):
        fit_gee_binomial(empty, ["x"], "question")


def test_a_missing_column_raises_a_named_error(frame: Any) -> None:
    stripped = frame.drop(columns=["corpus_fraction"])
    with pytest.raises(MissingCovariateError, match="corpus_fraction"):
        fit_gee_binomial(stripped, TERMS, "question")


def test_a_rank_deficient_design_is_refused(frame: Any) -> None:
    singular = frame.assign(log2B_copy=frame["log2B"])
    with pytest.raises(RankDeficientDesignError, match="rank"):
        fit_gee_binomial(singular, (*TERMS, "log2B_copy"), "question")


def test_a_constant_covariate_is_refused(frame: Any) -> None:
    flat = frame.assign(reader_strength=0.78)
    with pytest.raises(RankDeficientDesignError):
        fit_gee_binomial(flat, TERMS, "question")


def test_recovers_a_planted_interaction(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    assert fit.coefficient == pytest.approx(DELTA_INTERACTION, abs=0.25)
    assert abs(fit.coefficient - DELTA_INTERACTION) < 2.5 * fit.std_error
    assert fit.p_value < 1e-3
    assert fit.n_obs == len(frame)
    assert fit.n_clusters == frame["question"].nunique()


def test_the_planted_interaction_is_recovered_on_average() -> None:
    estimates = [
        fit_gee_binomial(
            _design(_synthetic(n_questions=200, seed=seed)),
            TERMS,
            "question",
            focus=INTERACTION,
        ).coefficient
        for seed in range(5)
    ]
    assert all(e > 0 for e in estimates), estimates
    assert float(np.mean(estimates)) == pytest.approx(DELTA_INTERACTION, abs=0.08)


def test_recovers_the_planted_main_effects_too(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    assert fit.coefficients[INTERACTION] == pytest.approx(DELTA_INTERACTION, abs=0.25)
    assert fit.coefficients["log2K"] == pytest.approx(BETA_LOG2K, abs=0.15)
    assert fit.coefficients["log2B"] == pytest.approx(BETA_LOG2B, abs=0.15)
    assert set(fit.coefficients) == {"intercept", *TERMS}


def test_the_focused_term_is_the_named_one_not_its_neighbour(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    assert fit.coefficient == fit.coefficients[INTERACTION]
    assert fit.std_error == fit.std_errors[INTERACTION]
    assert fit.p_value == fit.p_values[INTERACTION]
    assert fit.ci_low == pytest.approx(fit.coefficient - 1.959963985 * fit.std_error)

    for term in TERMS:
        focused = fit_gee_binomial(frame, TERMS, "question", focus=term)
        assert focused.coefficient == fit.coefficients[term], term
        assert focused.std_error == fit.std_errors[term], term


def test_the_confidence_interval_brackets_the_estimate(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    assert fit.ci_low < fit.coefficient < fit.ci_high
    half = fit.ci_high - fit.coefficient
    assert half == pytest.approx(1.959963985 * fit.std_error, rel=1e-5)
    assert ALPHA == 0.05
    assert fit.z == pytest.approx(fit.coefficient / fit.std_error)


def test_clustering_widens_the_standard_error(frame: Any) -> None:
    clustered = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    per_row = fit_gee_binomial(frame, TERMS, "K", focus=INTERACTION)
    assert clustered.std_error > per_row.std_error


def test_nan_rows_are_dropped_and_counted(frame: Any) -> None:
    holed = frame.copy()
    holed.loc[holed.index[:100], "correct"] = np.nan
    fit = fit_gee_binomial(holed, TERMS, "question", focus=INTERACTION)
    assert fit.n_dropped == 100
    assert fit.n_obs == len(frame) - 100
    assert fit.to_dict()["n_dropped"] == 100


def test_term_fit_reads_a_different_term_from_the_same_fit(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    main_k = fit.term_fit("log2K")
    assert main_k.term == "log2K"
    assert main_k.coefficient == fit.coefficients["log2K"]
    assert main_k.n_obs == fit.n_obs
    assert main_k.log_likelihood == fit.log_likelihood
    assert main_k.cluster_note == fit.cluster_note
    with pytest.raises(KeyError):
        fit.term_fit("not_a_term")


def test_the_default_focus_is_the_interaction(frame: Any) -> None:
    assert fit_gee_binomial(frame, TERMS, "question").term == INTERACTION
    assert fit_gee_binomial(frame, ("log2K", "log2B"), "question").term == "log2B"


def test_an_explicit_focus_must_be_fitted(frame: Any) -> None:
    with pytest.raises(KeyError, match="not one of the fitted terms"):
        fit_gee_binomial(frame, TERMS, "question", focus="not_a_term")


def test_small_cluster_counts_are_flagged_not_hidden(frame: Any) -> None:
    by_reader = fit_gee_binomial(frame, TERMS, "reader", focus=INTERACTION)
    assert by_reader.n_clusters == 3
    assert "only 3 distinct" in by_reader.cluster_note
    assert by_reader.to_dict()["cluster_note"] == by_reader.cluster_note

    assert fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION).cluster_note == ""


def test_to_dict_is_json_serialisable_and_complete(frame: Any) -> None:
    payload = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION).to_dict()
    json.loads(json.dumps(payload))
    for key in (
        "term",
        "coefficient",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "n_obs",
        "n_clusters",
        "converged",
        "coefficients",
        "std_errors",
        "p_values",
        "estimator",
        "cluster_col",
        "cluster_note",
        "effect_pp_at_p50",
    ):
        assert key in payload, key


def test_logodds_to_pp_depends_on_the_operating_point() -> None:
    assert logodds_to_pp(0.5) == pytest.approx(100.0 * (1 / (1 + np.exp(-0.5)) - 0.5))
    assert logodds_to_pp(0.5, 0.5) == pytest.approx(12.25, abs=0.01)
    assert logodds_to_pp(0.5, 0.9) < logodds_to_pp(0.5, 0.5)
    assert logodds_to_pp(0.5, 0.2) < logodds_to_pp(0.5, 0.5)
    assert logodds_to_pp(-0.5, 0.2) == pytest.approx(-logodds_to_pp(0.5, 0.8))
    assert logodds_to_pp(0.0) == 0.0
    assert logodds_to_pp(-0.5, 0.5) < 0.0


def test_logodds_to_pp_rejects_an_impossible_baseline() -> None:
    for bad in (0.0, 1.0, -0.2, 1.4):
        with pytest.raises(ValueError, match="probability"):
            logodds_to_pp(0.1, bad)


def test_effect_pp_on_the_fit_agrees_with_the_free_function(frame: Any) -> None:
    fit = fit_gee_binomial(frame, TERMS, "question", focus=INTERACTION)
    assert fit.effect_pp_at(0.5) == pytest.approx(logodds_to_pp(fit.coefficient, 0.5))
    assert fit.effect_pp == pytest.approx(fit.effect_pp_at(0.5))


def test_cluster_counts_expose_the_imbalance() -> None:
    counts = cluster_counts(pd.DataFrame({"question": ["a"] * 3000 + ["b"] * 3}), "question")
    assert counts == {"a": 3000, "b": 3}
    with pytest.raises(MissingCovariateError):
        cluster_counts(pd.DataFrame({"question": ["a"]}), "nope")


def test_z_and_p_helpers_name_the_degenerate_case() -> None:
    assert z_value(0.5, 0.25) == pytest.approx(2.0)
    assert p_from_z(2.0) == pytest.approx(0.0455003, rel=1e-4)
    assert np.isnan(z_value(1.0, 0.0))
    assert np.isnan(z_value(1.0, float("nan")))
    assert np.isnan(p_from_z(float("nan")))


def test_primary_fit_defaults_are_not_shared_mutable_state() -> None:
    kwargs = {
        "coefficient": 0.0,
        "std_error": 1.0,
        "ci_low": -1.0,
        "ci_high": 1.0,
        "p_value": 0.5,
        "n_obs": 1,
        "n_clusters": 1,
        "converged": True,
    }
    a = PrimaryFit(term="x", **kwargs)
    b = PrimaryFit(term="y", **kwargs)
    assert a.coefficients == {} and b.coefficients == {}
    assert a.coefficients is not b.coefficients


def test_the_module_does_not_silence_statsmodels() -> None:
    before = list(warnings.filters)
    fit_gee_binomial(
        _design(_synthetic(n_questions=40, seed=2)), TERMS, "question", focus=INTERACTION
    )
    assert warnings.filters == before
