"""Tests for stored analysis workflow (Recommendation 05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from msat.analysis.analysis_workflow import (
    ConflictingMetadataError,
    DuplicateJoinError,
    IncompleteCovariatesError,
    MixedConfigHashError,
    run_analysis_workflow,
)
from msat.analysis.mixedfx import NonBinaryOutcomeError

K_LEVELS = (1, 2, 4, 8, 16, 32)
B_LEVELS = (128.0, 256.0, 512.0)
READERS = ("gemma", "qwen", "phi")
ABILITIES = ("single-session", "multi-session", "temporal-reasoning")


def _generate_synthetic_run(
    run_dir: Path,
    *,
    n_questions: int = 120,
    readers: tuple[str, ...] = READERS,
    delta: float = 0.80,
    seed: int = 42,
    include_official: bool = True,
    mixed_hash: bool = False,
    missing_cf: bool = False,
    arm: str = "A",
    auxiliary_arms: tuple[str, ...] = (),
    nullable_gold_rank: bool = False,
    missing_gold_rank: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    k_center = float(np.mean(np.log2(K_LEVELS)))
    b_center = float(np.mean(np.log2(B_LEVELS)))

    score_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    reader_strengths = {"gemma": 0.78, "qwen": 0.80, "phi": 0.72}

    for q in range(n_questions):
        qid = f"q{q:04d}"
        ability = ABILITIES[q % len(ABILITIES)]
        reader = readers[(q // len(ABILITIES)) % len(readers)]
        strength = reader_strengths[reader]
        n_sessions = 40.0 + (q % 15)

        for k in K_LEVELS:
            for b in B_LEVELS:
                cache_key = (
                    f"cache_{qid}_{k}_{int(b)}_{reader}"
                    if arm == "A"
                    else f"cache_{arm}_{qid}_{k}_{int(b)}_{reader}"
                )
                cond_key = f"{arm}|{k}|{b}|budget|S|{reader}"

                kc = float(np.log2(k)) - k_center
                bc = float(np.log2(b)) - b_center
                eta = (
                    0.10
                    + 0.30 * kc
                    + 0.20 * bc
                    + delta * kc * bc
                    + 1.5 * strength
                    + 0.8
                    * {"single-session": 1, "multi-session": 0, "temporal-reasoning": -1}[ability]
                    + float(rng.normal(0.0, 0.2))
                )
                p = 1.0 / (1.0 + np.exp(-eta))
                correct_em = float(rng.binomial(1, p))

                config_h = "hash_b" if (mixed_hash and q > 20) else "hash_a"

                # Row for EM
                score_rows.append(
                    {
                        "schema_version": 1,
                        "cache_key": cache_key,
                        "condition_key": cond_key,
                        "qid": qid,
                        "arm": arm,
                        "K": k,
                        "B": b,
                        "B_mode": "budget",
                        "scale": "S",
                        "reader": reader,
                        "replicate": 0,
                        "cached": False,
                        "metric": "em",
                        "value": correct_em,
                        "is_abstention": False,
                        "answerable": True,
                        "finish_reason": "stop",
                        "latency_ms": 20.0,
                        "ttft_ms": 10.0,
                        "queue_wait_ms": 1.0,
                        "config_hash": config_h,
                    }
                )

                # Row for F1 (continuous value in [0, 1])
                score_rows.append(
                    {
                        "schema_version": 1,
                        "cache_key": cache_key,
                        "condition_key": cond_key,
                        "qid": qid,
                        "arm": arm,
                        "K": k,
                        "B": b,
                        "B_mode": "budget",
                        "scale": "S",
                        "reader": reader,
                        "replicate": 0,
                        "cached": False,
                        "metric": "f1",
                        "value": 0.75 if correct_em else 0.25,
                        "is_abstention": False,
                        "answerable": True,
                        "finish_reason": "stop",
                        "latency_ms": 20.0,
                        "ttft_ms": 10.0,
                        "queue_wait_ms": 1.0,
                        "config_hash": config_h,
                    }
                )

                if include_official:
                    score_rows.append(
                        {
                            "schema_version": 1,
                            "cache_key": cache_key,
                            "condition_key": cond_key,
                            "qid": qid,
                            "arm": arm,
                            "K": k,
                            "B": b,
                            "B_mode": "budget",
                            "scale": "S",
                            "reader": reader,
                            "replicate": 0,
                            "cached": False,
                            "metric": "official",
                            "value": correct_em,
                            "is_abstention": False,
                            "answerable": True,
                            "finish_reason": "stop",
                            "latency_ms": 20.0,
                            "ttft_ms": 10.0,
                            "queue_wait_ms": 1.0,
                            "config_hash": config_h,
                        }
                    )

                # Metadata row
                cf = None if missing_cf else float(k / n_sessions)
                if missing_gold_rank or (nullable_gold_rank and (q % 2 == 0)):
                    gr = None
                else:
                    gr = float(rng.integers(1, 10))

                meta_dict: dict[str, Any] = {
                    "cache_key": cache_key,
                    "qid": qid,
                    "condition_key": cond_key,
                    "K": k,
                    "B": b,
                    "reader": reader,
                    "scale": "S",
                    "arm": arm,
                    "corpus_fraction": cf,
                    "ability": ability,
                    "reader_strength": strength,
                    "config_hash": config_h,
                }
                if not missing_gold_rank:
                    meta_dict["gold_rank"] = gr
                meta_rows.append(meta_dict)

                # Optional auxiliary arms (e.g. E, D, F, B, C)
                for aux_arm in auxiliary_arms:
                    aux_cache_key = f"cache_{aux_arm}_{qid}_{k}_{int(b)}_{reader}"
                    aux_cond_key = f"{aux_arm}|{k}|{b}|budget|S|{reader}"
                    score_rows.append(
                        {
                            "schema_version": 1,
                            "cache_key": aux_cache_key,
                            "condition_key": aux_cond_key,
                            "qid": qid,
                            "arm": aux_arm,
                            "K": k,
                            "B": b,
                            "B_mode": "budget",
                            "scale": "S",
                            "reader": reader,
                            "replicate": 0,
                            "cached": False,
                            "metric": "em",
                            "value": correct_em,
                            "is_abstention": False,
                            "answerable": True,
                            "finish_reason": "stop",
                            "latency_ms": 20.0,
                            "ttft_ms": 10.0,
                            "queue_wait_ms": 1.0,
                            "config_hash": config_h,
                        }
                    )
                    aux_meta_dict = dict(meta_dict)
                    aux_meta_dict["cache_key"] = aux_cache_key
                    aux_meta_dict["condition_key"] = aux_cond_key
                    aux_meta_dict["arm"] = aux_arm
                    meta_rows.append(aux_meta_dict)

    # Write scores.parquet
    scores_table = pa.Table.from_pylist(score_rows)
    pq.write_table(scores_table, run_dir / "scores.parquet")

    # Write metadata.parquet
    meta_table = pa.Table.from_pylist(meta_rows)
    pq.write_table(meta_table, run_dir / "metadata.parquet")

    # Write power.json
    power_payload = {
        "interaction": {
            "mde_log_odds": 0.05,
            "mde_pp_at_p50": 1.25,
            "powered": True,
        }
    }
    (run_dir / "power.json").write_text(json.dumps(power_payload), encoding="utf-8")


def test_analysis_workflow_end_to_end_known_interaction(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_analysis_success"
    _generate_synthetic_run(run_dir, n_questions=90, delta=0.85, seed=123)

    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=True,
    )

    assert result.status == "SUCCESS"
    assert result.analysis_json_path.is_file()
    assert result.results_md_path.is_file()
    assert result.n_observations > 0

    # Primary model verification
    assert result.primary_fit is not None
    assert result.primary_fit.converged is True
    assert result.primary_fit.term == "log2K:log2B"
    # Interaction coefficient should be positive given synthetic DGP
    assert result.primary_fit.coefficient > 0.0

    # Preregistration verdict
    assert result.preregistration is not None
    assert result.preregistration.mde_log_odds == 0.05
    assert result.preregistration.estimate_log_odds == result.primary_fit.coefficient

    # Strata report verification (Primary)
    assert result.strata_report is not None
    assert len(result.strata_report.strata) == 3
    assert result.strata_report.pooled is not None

    # LRT functional form verification
    assert result.lrt_result is not None
    assert result.lrt_result.converged is True

    # Reader agreement verification
    assert result.reader_agreement.get("available") is True
    assert "flip_rate" in result.reader_agreement

    # Determinism: rerun produces byte-identical files
    json_bytes_1 = result.analysis_json_path.read_bytes()
    md_bytes_1 = result.results_md_path.read_bytes()

    result2 = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=True,
    )
    assert result2.analysis_json_path.read_bytes() == json_bytes_1
    assert result2.results_md_path.read_bytes() == md_bytes_1


def test_validate_mixed_config_hash_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_mixed_hash"
    _generate_synthetic_run(run_dir, n_questions=30, mixed_hash=True)

    with pytest.raises(MixedConfigHashError, match="Found mixed config_hash"):
        run_analysis_workflow(run_dir, selected_outcome="em")


def test_validate_duplicate_joins_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dup_join"
    _generate_synthetic_run(run_dir, n_questions=10)

    # Corrupt metadata.parquet with duplicate rows
    meta_df = pd.read_parquet(run_dir / "metadata.parquet")
    meta_dup = pd.concat([meta_df, meta_df.iloc[:5]], ignore_index=True)
    meta_dup.to_parquet(run_dir / "metadata.parquet")

    with pytest.raises(DuplicateJoinError, match="duplicate"):
        run_analysis_workflow(run_dir, selected_outcome="em")


def test_validate_conflicting_metadata_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_conflict"
    _generate_synthetic_run(run_dir, n_questions=10)

    # Tamper with K in metadata to conflict with scores table
    meta_df = pd.read_parquet(run_dir / "metadata.parquet")
    meta_df.loc[0, "K"] = 999
    meta_df.to_parquet(run_dir / "metadata.parquet")

    with pytest.raises(
        ConflictingMetadataError, match="Metadata contradicts scores table on column 'K'"
    ):
        run_analysis_workflow(run_dir, selected_outcome="em")


def test_validate_incomplete_covariates_rejected(tmp_path: Path) -> None:
    # 1. Missing corpus_fraction
    run_dir_cf = tmp_path / "run_missing_cf"
    _generate_synthetic_run(run_dir_cf, n_questions=10, missing_cf=True)

    with pytest.raises(IncompleteCovariatesError, match="corpus_fraction"):
        run_analysis_workflow(run_dir_cf, selected_outcome="em")

    # 2. Missing gold_rank under strict=True
    run_dir_gr = tmp_path / "run_missing_gr"
    _generate_synthetic_run(run_dir_gr, n_questions=10)
    meta_df = pd.read_parquet(run_dir_gr / "metadata.parquet")
    meta_df_no_gr = meta_df.drop(columns=["gold_rank"])
    meta_df_no_gr.to_parquet(run_dir_gr / "metadata.parquet")

    with pytest.raises(IncompleteCovariatesError, match="gold_rank"):
        run_analysis_workflow(run_dir_gr, selected_outcome="em", strict=True)


def test_validate_nonbinary_outcome_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_nonbinary"
    _generate_synthetic_run(run_dir, n_questions=10)

    # Selecting F1 as correctness outcome must be rejected
    with pytest.raises(NonBinaryOutcomeError, match="F1 is a continuous token overlap score"):
        run_analysis_workflow(run_dir, selected_outcome="f1")


def test_single_reader_pilot_handling(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_single_reader"
    _generate_synthetic_run(run_dir, n_questions=60, readers=("gemma",))

    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=False,  # Single-reader pilot mode
    )

    assert result.status == "SUCCESS"
    assert result.primary_fit is not None

    # Reader clustered secondary should be marked unavailable (not fabricated)
    sec = result.primary_analysis.get("reader_clustered_secondary", {})
    assert sec.get("available") is False

    # Reader agreement comparison should be marked unavailable (not fabricated)
    assert result.reader_agreement.get("available") is False

    # Dropped constant note should be present
    assert any("Single-reader pilot run" in n for n in result.notes)
    assert any("reader_strength" in n for n in result.notes)


def test_unestimable_model_meaningful_error_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_unestimable"
    # Create tiny run with only 1 question cluster -> cluster-robust sandwich fails (< 2 clusters)
    _generate_synthetic_run(run_dir, n_questions=1, readers=("gemma",))

    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=False,
    )

    assert result.status == "UNESTIMABLE"
    assert result.primary_fit is None
    assert result.analysis_json_path.is_file()
    assert result.results_md_path.is_file()

    md_text = result.results_md_path.read_text(encoding="utf-8")
    assert "Primary Model Fitting Failure" in md_text
    assert "Coefficients are not fabricated" in md_text


def test_analyze_cli_reaches_stored_models_and_reproduces_reports(
    tmp_path: Path, no_network: None
) -> None:
    from typer.testing import CliRunner

    from msat.cli import app

    run_dir = tmp_path / "results" / "cli-analysis"
    _generate_synthetic_run(run_dir, n_questions=90, delta=0.85, seed=123)
    args = ["analyze", run_dir.name, "--results", str(run_dir.parent), "--bootstrap", "10"]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "not implemented" not in first.output
    report_bytes = (run_dir / "results.md").read_bytes()
    analysis_bytes = (run_dir / "analysis.json").read_bytes()
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0, second.output
    assert (run_dir / "results.md").read_bytes() == report_bytes
    assert (run_dir / "analysis.json").read_bytes() == analysis_bytes


def test_missing_requested_metadata_is_an_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-sidecar"
    _generate_synthetic_run(run_dir, n_questions=10)
    with pytest.raises(FileNotFoundError, match="Observation metadata"):
        run_analysis_workflow(run_dir, metadata_path=tmp_path / "absent.parquet")


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.1, float("inf")])
def test_invalid_corpus_fraction_is_rejected(tmp_path: Path, fraction: float) -> None:
    run_dir = tmp_path / "invalid-fraction"
    _generate_synthetic_run(run_dir, n_questions=10)
    metadata = pd.read_parquet(run_dir / "metadata.parquet")
    metadata.loc[0, "corpus_fraction"] = fraction
    metadata.to_parquet(run_dir / "metadata.parquet")
    with pytest.raises(IncompleteCovariatesError, match="corpus_fraction"):
        run_analysis_workflow(run_dir)
    assert not (run_dir / "analysis.json").exists()


def test_partial_metadata_does_not_silently_drop_observations(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial-sidecar"
    _generate_synthetic_run(run_dir, n_questions=10)
    metadata = pd.read_parquet(run_dir / "metadata.parquet")
    metadata.iloc[1:].to_parquet(run_dir / "metadata.parquet")
    with pytest.raises(IncompleteCovariatesError, match="every selected observation"):
        run_analysis_workflow(run_dir)


def test_primary_row_selection_excludes_auxiliary_arms(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_aux_exclusion"
    _generate_synthetic_run(
        run_dir,
        n_questions=30,
        arm="A+G",
        auxiliary_arms=("E", "D", "F", "B", "C"),
        seed=999,
    )

    scores_df = pd.read_parquet(run_dir / "scores.parquet")
    em_scores = scores_df[scores_df["metric"] == "em"]
    n_a_plus_g = int(em_scores["arm"].eq("A+G").sum())
    n_aux = int(em_scores["arm"].isin(["E", "D", "F", "B", "C"]).sum())
    assert n_aux > 0
    assert len(em_scores) == n_a_plus_g + n_aux

    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=True,
    )

    assert result.status == "SUCCESS"
    assert result.n_observations == n_a_plus_g
    assert any("Excluded" in n and "auxiliary-arm rows" in n for n in result.notes)
    expected_note = f"Excluded {n_aux} auxiliary-arm rows from the primary crossed-grid model."
    assert expected_note in result.notes

    # Verify report and JSON artifact both contain the exclusion note
    md_content = result.results_md_path.read_text(encoding="utf-8")
    assert expected_note in md_content
    assert "## 7. Notes & Dropped Terms" in md_content

    json_data = json.loads(result.analysis_json_path.read_text(encoding="utf-8"))
    assert expected_note in json_data["notes"]


def test_pilot_nullable_gold_rank_omitted_from_model_with_note(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_nullable_gold_rank"
    _generate_synthetic_run(
        run_dir,
        n_questions=30,
        readers=("gemma",),
        nullable_gold_rank=True,
        seed=101,
    )

    meta_df = pd.read_parquet(run_dir / "metadata.parquet")
    assert meta_df["gold_rank"].isna().any(), "Synthetic metadata must contain null gold_rank"
    assert meta_df["gold_rank"].notna().any(), "Synthetic metadata must contain non-null gold_rank"

    # 1. Strict full must reject missing/null gold_rank metadata
    with pytest.raises(IncompleteCovariatesError, match="gold_rank"):
        run_analysis_workflow(run_dir, selected_outcome="em", strict=True)

    # 2. Pilot mode (strict=False) must succeed, omit gold_rank without imputing, and record note
    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=False,
    )

    assert result.status == "SUCCESS"
    assert result.primary_fit is not None
    # gold_rank was omitted from model specification
    assert "gold_rank" not in result.primary_fit.coefficients
    assert "gold_rank" not in result.primary_fit.term_names
    assert "gold_rank" not in result.primary_analysis.get("primary", {}).get("term_names", [])

    # Dropped term note must be explicitly recorded
    expected_substr = "Omitted optional 'gold_rank' covariate from pilot model"
    assert any(expected_substr in n for n in result.notes)
    assert any(
        "nullable gold_rank metadata; gold absence is real, not imputed" in n for n in result.notes
    )

    # Note must appear in results.md and analysis.json
    md_content = result.results_md_path.read_text(encoding="utf-8")
    assert expected_substr in md_content
    json_data = json.loads(result.analysis_json_path.read_text(encoding="utf-8"))
    assert any(expected_substr in n for n in json_data.get("notes", []))


def test_pilot_completely_missing_gold_rank_omitted_with_note(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_absent_gold_rank"
    _generate_synthetic_run(
        run_dir,
        n_questions=30,
        readers=("gemma",),
        missing_gold_rank=True,
        seed=202,
    )

    meta_df = pd.read_parquet(run_dir / "metadata.parquet")
    assert "gold_rank" not in meta_df.columns

    # 1. Strict full must reject missing gold_rank column
    with pytest.raises(IncompleteCovariatesError, match="gold_rank"):
        run_analysis_workflow(run_dir, selected_outcome="em", strict=True)

    # 2. Pilot mode (strict=False) must omit unavailable gold_rank with note
    result = run_analysis_workflow(
        run_dir,
        selected_outcome="em",
        strict=False,
    )

    assert result.status == "SUCCESS"
    assert result.primary_fit is not None
    assert "gold_rank" not in result.primary_fit.term_names
    expected_substr = "Omitted optional 'gold_rank' covariate from pilot model"
    assert any(expected_substr in n for n in result.notes)
    assert any("gold_rank metadata unavailable" in n for n in result.notes)


def test_strict_full_rejects_missing_null_covariates(tmp_path: Path) -> None:
    # Test strict rejection of NULL reader_strength
    run_dir_rs = tmp_path / "run_null_rs"
    _generate_synthetic_run(run_dir_rs, n_questions=10)
    meta_df = pd.read_parquet(run_dir_rs / "metadata.parquet")
    meta_df.loc[0, "reader_strength"] = np.nan
    meta_df.to_parquet(run_dir_rs / "metadata.parquet")

    with pytest.raises(IncompleteCovariatesError, match="reader_strength"):
        run_analysis_workflow(run_dir_rs, selected_outcome="em", strict=True)

    # Test strict rejection of missing ability
    run_dir_ab = tmp_path / "run_missing_ab"
    _generate_synthetic_run(run_dir_ab, n_questions=10)
    meta_df = pd.read_parquet(run_dir_ab / "metadata.parquet")
    meta_df = meta_df.drop(columns=["ability"])
    meta_df.to_parquet(run_dir_ab / "metadata.parquet")

    with pytest.raises(IncompleteCovariatesError, match="ability"):
        run_analysis_workflow(run_dir_ab, selected_outcome="em", strict=True)


def test_analysis_workflow_supports_canonical_repetition_column(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_repetition"
    _generate_synthetic_run(run_dir, n_questions=30, readers=("gemma",), seed=42)

    # Rename replicate to repetition in both scores and metadata
    scores_df = pd.read_parquet(run_dir / "scores.parquet")
    scores_df = scores_df.rename(columns={"replicate": "repetition"})
    scores_df.to_parquet(run_dir / "scores.parquet")

    meta_df = pd.read_parquet(run_dir / "metadata.parquet")
    meta_df = meta_df.rename(columns={"replicate": "repetition"})
    meta_df.to_parquet(run_dir / "metadata.parquet")

    result = run_analysis_workflow(run_dir, selected_outcome="em", strict=False)
    assert result.status == "SUCCESS"
    assert result.primary_fit is not None
