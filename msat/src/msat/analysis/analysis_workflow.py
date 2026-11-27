"""Stored analysis workflow (Architecture Recommendation 05).

Owns:
- loading stored artifacts (scores.parquet, observation metadata, configs)
- metric/outcome selection and non-binary outcome validation
- validated joins between scores and metadata (duplicate and conflict detection)
- provenance, covariate completeness (corpus_fraction mandatory, no fabricated zeros)
- single-reader pilot handling (dropped constant, unavailable comparison reported)
- statistical estimation (reusing interaction.py, mixedfx.py, reader_agreement.py)
- meaningful GEE rank/constant/strata error reporting without fabricating numbers
- derived artifacts: analysis.json and results.md
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from msat.analysis.interaction import (
    PRIMARY_SPEC,
    LrtResult,
    PreregistrationVerdict,
    StrataReport,
    fit_primary,
    lrt_monotone_vs_inverted_u,
    preregistration_verdict,
    primary_analysis,
    strata_report,
)
from msat.analysis.mixedfx import (
    MissingCovariateError,
    NonBinaryOutcomeError,
    PrimaryFit,
    RankDeficientDesignError,
    TooFewClustersError,
    assert_binary_outcome,
)
from msat.analysis.reader_agreement import (
    ConditionScore,
    compare_readers,
    responses_from_conditions,
)


class MixedConfigHashError(ValueError):
    """Raised when score rows or metadata come from multiple distinct config hashes."""


class DuplicateJoinError(ValueError):
    """Raised when joining scores and metadata produces duplicate keys or row explosion."""


class ConflictingMetadataError(ValueError):
    """Raised when joined metadata contradicts the condition or replicate in scores."""


class IncompleteCovariatesError(MissingCovariateError):
    """Raised when declared covariates (especially mandatory corpus_fraction) are missing or NULL."""


def load_readers_config(config_root: Path | str | None = None) -> dict[str, float]:
    """Load declared reader strengths from readers.yaml."""
    root = Path(config_root) if config_root is not None else Path("configs")
    candidates = [root / "readers.yaml", root / "configs" / "readers.yaml"]
    p: Path | None = None
    for cand in candidates:
        if cand.is_file():
            p = cand
            break

    if p is None:
        return {}

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}

    strengths: dict[str, float] = {}
    for fam in data.get("families", []):
        if isinstance(fam, dict) and "id" in fam and "reader_strength" in fam:
            strengths[str(fam["id"])] = float(fam["reader_strength"])
    return strengths


def load_power_mde(
    run_dir: Path | str, config_root: Path | str | None = None
) -> tuple[float | None, str]:
    """Load interaction MDE from power.json."""
    rd = Path(run_dir)
    candidates = [
        rd / "power.json",
        rd.parent / "power.json",
        Path("results/power.json"),
    ]
    if config_root is not None:
        candidates.append(Path(config_root) / "power.json")

    for cand in candidates:
        if cand.is_file():
            try:
                raw = json.loads(cand.read_text(encoding="utf-8"))
                inter = raw.get("interaction", {})
                if isinstance(inter, dict) and inter.get("powered") is not False:
                    val = inter.get("mde_log_odds")
                    if val is not None:
                        return float(val), f"{cand} -> interaction.mde_log_odds"
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    return None, "power.json unavailable"


def load_metadata(
    metadata_path: Path | str | None = None,
    run_dir: Path | str | None = None,
) -> pd.DataFrame | None:
    """Load observation metadata from parquet, json, or csv."""
    p: Path | None = None
    if metadata_path is not None:
        p = Path(metadata_path)
    elif run_dir is not None:
        for cand_name in ("metadata.parquet", "metadata.json", "metadata.csv"):
            cand = Path(run_dir) / cand_name
            if cand.is_file():
                p = cand
                break

    if p is not None and metadata_path is not None and not p.is_file():
        raise FileNotFoundError(f"Observation metadata file does not exist: {p}")
    if p is None or not p.is_file():
        return None

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        return pd.DataFrame(raw)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported metadata extension {p.suffix}")


def _filter_primary_arm_rows(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Filter observation rows to the primary crossed-grid arms, excluding auxiliary arms."""
    if "arm" not in df.columns and "condition_key" in df.columns:
        df["arm"] = df["condition_key"].astype(str).str.split("|").str[0]
    if "arm" not in df.columns:
        return df

    has_a_plus_g = df["arm"].eq("A+G").any()
    primary_arms = {"A+G"} if has_a_plus_g else {"A", "G"}
    selected = df["arm"].isin(primary_arms)
    if (~selected).any():
        excluded_count = int((~selected).sum())
        note = f"Excluded {excluded_count} auxiliary-arm rows from the primary crossed-grid model."
        if note not in notes:
            notes.append(note)
        return df[selected].copy()
    return df


def build_validated_observation_frame(
    scores_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    reader_strengths: Mapping[str, float] | None = None,
    selected_outcome: str = "em",
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build and validate a single-metric analysis dataframe.

    Validates:
    - single homogeneous config_hash
    - exact outcome selection (one metric per observation)
    - non-binary outcome refusal (e.g. continuous F1)
    - join uniqueness and absence of conflicting conditions
    - complete covariates (corpus_fraction mandatory; no made-up zeros)
    - single-reader pilot handling
    """
    notes: list[str] = []

    # 1. Config hash validation
    for col in ("config_hash", "config_sha"):
        if col in scores_df.columns:
            hashes = scores_df[col].dropna().unique()
            if len(hashes) > 1:
                raise MixedConfigHashError(
                    f"Found mixed {col} values in scores: {list(hashes)}. "
                    "Analysis requires a single homogeneous run configuration."
                )

    if metadata_df is not None:
        for col in ("config_hash", "config_sha"):
            if col in metadata_df.columns:
                m_hashes = metadata_df[col].dropna().unique()
                if len(m_hashes) > 1:
                    raise MixedConfigHashError(
                        f"Found mixed {col} values in metadata: {list(m_hashes)}."
                    )
                if col in scores_df.columns:
                    s_hashes = scores_df[col].dropna().unique()
                    if len(s_hashes) == 1 and len(m_hashes) == 1 and s_hashes[0] != m_hashes[0]:
                        raise MixedConfigHashError(
                            f"Scores config_hash ({s_hashes[0]}) does not match metadata config_hash ({m_hashes[0]})."
                        )

    # 2. Metric / Outcome Selection (One metric per observation)
    clean_outcome = selected_outcome.lower().strip()
    if clean_outcome == "f1":
        raise NonBinaryOutcomeError(
            "F1 is a continuous token overlap score in [0, 1] and cannot be used as a "
            "binary correctness outcome without an explicit declared binarization threshold "
            "(PRD §R9.1, §6.1). Use 'em' or 'official'."
        )

    if "metric" not in scores_df.columns:
        raise ValueError("scores table missing 'metric' column")

    metric_mask = scores_df["metric"] == clean_outcome
    subset = scores_df[metric_mask].copy()
    subset = _filter_primary_arm_rows(subset, notes)
    if subset.empty:
        available = sorted(scores_df["metric"].unique())
        raise ValueError(
            f"Selected outcome {clean_outcome!r} not found in scores table. Available: {available}"
        )

    # Normalize repetition / replicate columns if either is present
    if "repetition" in subset.columns and "replicate" not in subset.columns:
        subset["replicate"] = subset["repetition"]
    elif "replicate" in subset.columns and "repetition" not in subset.columns:
        subset["repetition"] = subset["replicate"]
    if metadata_df is not None:
        if "repetition" in metadata_df.columns and "replicate" not in metadata_df.columns:
            metadata_df["replicate"] = metadata_df["repetition"]
        elif "replicate" in metadata_df.columns and "repetition" not in metadata_df.columns:
            metadata_df["repetition"] = metadata_df["replicate"]

    # Check for duplicate observations for the selected metric
    dup_keys: list[str] = []
    for k in ("cache_key",):
        if k in subset.columns:
            dup_keys.append(k)
            if "repetition" in subset.columns:
                dup_keys.append("repetition")
            elif "replicate" in subset.columns:
                dup_keys.append("replicate")
            break
    if not dup_keys and {"condition_key", "qid"}.issubset(subset.columns):
        dup_keys = ["condition_key", "qid"]
        if "repetition" in subset.columns:
            dup_keys.append("repetition")
        elif "replicate" in subset.columns:
            dup_keys.append("replicate")

    if dup_keys:
        if subset[dup_keys].isna().any().any():
            raise DuplicateJoinError("Observation identity contains NULL join keys.")
        dups = subset.duplicated(subset=dup_keys, keep=False)
        if dups.any():
            n_dups = int(dups.sum())
            raise DuplicateJoinError(
                f"Found {n_dups} duplicate observation rows in scores for metric {clean_outcome!r} "
                f"keyed on {dup_keys}."
            )

    # 3. Outcome Binary Validation
    # Exclude abstentions/NULL for binary outcome check
    non_abstention = subset[
        ~subset.get("is_abstention", pd.Series(False, index=subset.index))
    ].copy()
    valid_values = non_abstention["value"].dropna().to_numpy()
    assert_binary_outcome(valid_values, column="value")
    usable = subset["value"].notna() & ~subset.get(
        "is_abstention", pd.Series(False, index=subset.index)
    )
    if "finish_reason" in subset.columns:
        usable &= subset["finish_reason"].eq("stop")
    excluded = int((~usable).sum())
    if excluded:
        notes.append(
            f"Excluded {excluded} NULL, abstention, or unusable-output observations from the model."
        )
    subset = subset[usable].copy()
    if subset.empty:
        raise ValueError("No usable binary correctness observations remain for analysis.")
    subset["correct"] = subset["value"]

    # 4. Metadata Join
    if metadata_df is not None:
        join_key: str | list[str] | None = None
        if "cache_key" in subset.columns and "cache_key" in metadata_df.columns:
            join_key = (
                ["cache_key", "replicate"]
                if "replicate" in subset.columns and "replicate" in metadata_df.columns
                else "cache_key"
            )
        elif {"condition_key", "qid"}.issubset(subset.columns) and {
            "condition_key",
            "qid",
        }.issubset(metadata_df.columns):
            join_key = ["condition_key", "qid"]
            if "replicate" in subset.columns and "replicate" in metadata_df.columns:
                join_key.append("replicate")

        if join_key is None:
            raise DuplicateJoinError(
                "Cannot determine unambiguous join key between scores and metadata."
            )
        if metadata_df[join_key].isna().any().any():
            raise DuplicateJoinError("Metadata identity contains NULL join keys.")

        # Check for duplicates in metadata join keys
        if metadata_df.duplicated(subset=join_key).any():
            raise DuplicateJoinError(
                f"Metadata contains duplicate rows on join key(s) {join_key!r}."
            )

        merged = pd.merge(
            subset,
            metadata_df,
            on=join_key,
            how="left",
            suffixes=("", "_meta"),
            validate="many_to_one",
            indicator="_metadata_join",
        )
        if (merged["_metadata_join"] != "both").any():
            raise IncompleteCovariatesError("Metadata does not cover every selected observation.")
        merged = merged.drop(columns="_metadata_join")
        if len(merged) != len(subset):
            raise DuplicateJoinError(
                f"Metadata join expanded row count from {len(subset)} to {len(merged)}."
            )

        # Validate against conflicting condition attributes
        for check_col in (
            "K",
            "B",
            "B_mode",
            "reader",
            "scale",
            "arm",
            "qid",
            "replicate",
            "condition_key",
            "corpus_fraction",
            "gold_rank",
            "reader_strength",
            "ability",
        ):
            meta_col = f"{check_col}_meta"
            if check_col in merged.columns and meta_col in merged.columns:
                mismatches = merged[merged[check_col].notna() & merged[meta_col].notna()]
                if check_col in (
                    "K",
                    "B",
                    "replicate",
                    "corpus_fraction",
                    "gold_rank",
                    "reader_strength",
                ):
                    unequal = ~np.isclose(
                        pd.to_numeric(mismatches[check_col], errors="raise"),
                        pd.to_numeric(mismatches[meta_col], errors="raise"),
                    )
                else:
                    unequal = mismatches[check_col].astype(str) != mismatches[meta_col].astype(str)
                diff = mismatches[unequal]
                if not diff.empty:
                    raise ConflictingMetadataError(
                        f"Metadata contradicts scores table on column {check_col!r}: "
                        f"{len(diff)} mismatched row(s)."
                    )
        subset = merged
        subset = _filter_primary_arm_rows(subset, notes)

    # 5. Covariates Supply & Validation
    if strict and (
        "config_hash" not in subset.columns
        or subset["config_hash"].isna().any()
        or subset["config_hash"].astype(str).str.strip().eq("").any()
    ):
        raise IncompleteCovariatesError(
            "Confirmatory analysis requires config_hash on every observation."
        )
    if "scoring_config_hash" in subset.columns:
        hashes = subset["scoring_config_hash"].dropna().unique()
        if len(hashes) > 1:
            raise MixedConfigHashError("Analysis cannot mix different scoring configurations.")
    # Ensure question column
    if "question" not in subset.columns and "qid" in subset.columns:
        subset["question"] = subset["qid"]

    # corpus_fraction is MANDATORY
    if "corpus_fraction" not in subset.columns:
        raise IncompleteCovariatesError(
            "'corpus_fraction' is missing from observations. It is a declared mandatory "
            "covariate in PRD §R9.1 / §R2.8 and cannot be omitted or filled with zero."
        )

    cf_na = subset["corpus_fraction"].isna()
    if cf_na.any():
        raise IncompleteCovariatesError(
            f"{cf_na.sum()} observation(s) have NULL corpus_fraction. Never fill missing "
            "covariates with zero or arbitrary defaults."
        )
    fractions = pd.to_numeric(subset["corpus_fraction"], errors="raise")
    if (~np.isfinite(fractions) | (fractions <= 0) | (fractions > 1)).any():
        raise IncompleteCovariatesError("corpus_fraction must be finite and in (0, 1].")

    # reader_strength
    if "reader_strength" not in subset.columns:
        if reader_strengths and "reader" in subset.columns:
            subset["reader_strength"] = subset["reader"].map(reader_strengths)
        else:
            if strict:
                raise IncompleteCovariatesError(
                    "'reader_strength' is missing from observation data."
                )

    if "reader_strength" in subset.columns:
        rs_vals = pd.to_numeric(subset["reader_strength"], errors="coerce")
        if strict and (rs_vals.isna().any() or (~np.isfinite(rs_vals)).any()):
            missing_readers = (
                subset.loc[rs_vals.isna(), "reader"].unique().tolist()
                if "reader" in subset.columns
                else []
            )
            if missing_readers:
                raise IncompleteCovariatesError(
                    f"reader_strength is missing for reader(s): {missing_readers}."
                )
            raise IncompleteCovariatesError(
                "'reader_strength' contains missing or nonfinite values."
            )
        if not strict and not rs_vals.isna().any() and np.isfinite(rs_vals).all():
            subset["reader_strength"] = rs_vals

    # gold_rank validation and pilot handling
    if "gold_rank" not in subset.columns:
        if strict:
            raise IncompleteCovariatesError("'gold_rank' is missing from observation data.")
        notes.append(
            "Omitted optional 'gold_rank' covariate from pilot model (gold_rank metadata unavailable)."
        )
    else:
        gr_vals = pd.to_numeric(subset["gold_rank"], errors="coerce")
        has_invalid_or_missing = gr_vals.isna().any() or (~np.isfinite(gr_vals)).any()
        if has_invalid_or_missing:
            if strict:
                raise IncompleteCovariatesError("'gold_rank' contains missing or nonfinite values.")
            # Gold absence is real: do not impute ranks.
            # Explicitly omit optional unavailable rank from pilot model with a returned/report note.
            subset = subset.drop(columns=["gold_rank"])
            notes.append(
                "Omitted optional 'gold_rank' covariate from pilot model (nullable gold_rank metadata; gold absence is real, not imputed)."
            )
        else:
            subset["gold_rank"] = gr_vals

    # ability
    if "ability" not in subset.columns and strict:
        raise IncompleteCovariatesError("'ability' is missing from observation data.")
    if "ability" in subset.columns and (
        subset["ability"].isna().any() or subset["ability"].astype(str).str.strip().eq("").any()
    ):
        raise IncompleteCovariatesError("'ability' contains missing values.")

    # Single reader pilot note
    if "reader" in subset.columns and subset["reader"].nunique() == 1:
        reader_name = str(subset["reader"].iloc[0])
        notes.append(
            f"Single-reader pilot run with reader {reader_name!r}. Reader-clustered secondary "
            "and reader agreement comparisons are not available."
        )

    return subset, notes


def generate_results_md(analysis_data: dict[str, Any]) -> str:
    """Generate deterministic, publication-ready results.md (PRD §P10, R9.1-R9.12)."""
    p = analysis_data.get("primary", {})
    spec = analysis_data.get("specification", {})
    strata = analysis_data.get("strata", {})
    lrt = analysis_data.get("lrt", {})
    prereg = analysis_data.get("preregistration", {})
    rag = analysis_data.get("reader_agreement", {})
    sec = analysis_data.get("reader_clustered_secondary", {})
    prov = analysis_data.get("provenance", {})

    status = analysis_data.get("status", "SUCCESS")

    lines: list[str] = [
        "# MemSat — Analysis Results",
        "",
        "**Generated from stored run artifacts (offline).**",
        "",
        "## 1. Provenance & Dataset Summary",
        "",
        f"- **Outcome Metric**: `{prov.get('outcome', 'em')}` (binary correctness)",
        f"- **Total Observations**: {prov.get('n_obs', 0):,}",
        f"- **Unique Questions**: {prov.get('n_questions', 0):,}",
        f"- **Conditions**: {prov.get('n_conditions', 0)}",
        f"- **Readers**: {', '.join(prov.get('readers', []))}",
        "",
    ]

    if status != "SUCCESS":
        lines.extend(
            [
                "## ⚠️ Primary Model Fitting Failure",
                "",
                f"**Status**: `{status}`",
                f"**Error**: {analysis_data.get('error', 'Unestimable model')}",
                "",
                "The model could not be reliably estimated. Coefficients are not fabricated.",
                "",
            ]
        )
        return "\n".join(lines)

    # Primary interaction table
    lines.extend(
        [
            "## 2. Primary Model (GEE Binomial Interaction)",
            "",
            f"**Declared Specification**: `{spec.get('declared', '')}`",
            "",
            "| Term | Coef (log-odds) | Std Error | 95% CI | p-value | Effect at p50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    term_fits = p.get("coefficients", {})
    ses = p.get("std_errors", {})
    pvals = p.get("p_values", {})
    for term_name in p.get("term_names", []):
        coef = term_fits.get(term_name, 0.0)
        se = ses.get(term_name, 0.0)
        pval = pvals.get(term_name, 1.0)
        ci_low = coef - 1.95996 * se
        ci_high = coef + 1.95996 * se
        eff_p50 = (
            f"{((1.0 / (1.0 + np.exp(-(np.log(0.5 / (1 - 0.5)) + coef)))) - 0.5) * 100:+.2f} pp"
        )
        lines.append(
            f"| `{term_name}` | {coef:+.4f} | {se:.4f} | [{ci_low:+.4f}, {ci_high:+.4f}] | {pval:.4e} | {eff_p50} |"
        )

    lines.extend(
        [
            "",
            f"- **Convergence**: `{p.get('converged', True)}`",
            f"- **Number of Clusters**: {p.get('n_clusters', 0)}",
            f"- **Primary Cluster Column**: `{p.get('cluster_col', 'question')}`",
            "",
        ]
    )

    # Pre-registration verdict
    lines.extend(
        [
            "## 3. Pre-registration Verdict",
            "",
            f"- **Predicted Sign**: `{prereg.get('prediction', 'approximately_zero')}`",
            f"- **Estimate**: `{prereg.get('estimate_log_odds', 0.0):+.4f}` log-odds",
            f"- **MDE (log-odds)**: `{prereg.get('mde_log_odds', 'n/a')}`",
            f"- **Ratio to MDE**: `{prereg.get('ratio_to_mde', 'n/a')}`",
            f"- **Consistent with Prediction**: `{prereg.get('consistent', 'n/a')}`",
            "",
            f"> {prereg.get('claim', '')}",
            "",
        ]
    )

    # Ability strata
    strata_list = strata.get("strata", [])
    if strata_list:
        lines.extend(
            [
                "## 4. Stratified Analysis by Question Ability (Primary)",
                "",
                "| Ability Stratum | Obs | Questions | Saturation K (95%) | Knee K |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for s in strata_list:
            st = s.get("stats", {})
            k95 = st.get("k_95", {}).get("k", "n/a")
            knee = st.get("k_knee", {}).get("k", "n/a")
            lines.append(
                f"| `{s.get('label')}` | {s.get('n_obs'):,} | {s.get('n_questions'):,} | {k95} | {knee} |"
            )
        pooled = strata.get("pooled", {})
        if pooled:
            pst = pooled.get("stats", {})
            pk95 = pst.get("k_95", {}).get("k", "n/a")
            pknee = pst.get("k_knee", {}).get("k", "n/a")
            lines.append(
                f"| **Pooled (Descriptive)** | {pooled.get('n_obs'):,} | {pooled.get('n_questions'):,} | {pk95} | {pknee} |"
            )
        lines.append("")

    # Functional form (LRT)
    if lrt:
        lines.extend(
            [
                "## 5. Functional Form (LRT Monotone vs Inverted-U)",
                "",
                f"- **Likelihood Ratio Statistic**: `{lrt.get('statistic', 0.0):.4f}` (df = {lrt.get('df', 1)})",
                f"- **p-value**: `{lrt.get('p_value', 1.0):.4e}`",
                f"- **Quadratic Coefficient (log2K_sq)**: `{lrt.get('quadratic_coefficient', 0.0):+.4f}`",
                f"- **Verdict**: `{lrt.get('verdict', '')}`",
                "",
            ]
        )

    # Secondary fits & Reader Agreement
    lines.extend(
        [
            "## 6. Secondary Fits & Reader Agreement",
            "",
            f"- **Reader Clustered Secondary SE Available**: `{sec.get('available', False)}`",
        ]
    )
    if sec.get("available"):
        lines.append(
            f"- **Reader Clustered SE**: `{sec.get('std_error', 'n/a')}` (clusters = {sec.get('n_clusters', 0)})"
        )
    elif sec.get("note"):
        lines.append(f"- **Note**: {sec.get('note')}")

    lines.append(f"- **Reader Agreement Comparison Available**: `{rag.get('available', False)}`")
    if rag.get("available"):
        lines.append(f"- **Flip Rate**: `{rag.get('flip_rate', 'n/a')}`")
    elif rag.get("note"):
        lines.append(f"- **Note**: {rag.get('note')}")

    lines.append("")

    # Notes & dropped terms
    all_notes = analysis_data.get("notes", [])
    if all_notes:
        lines.extend(
            [
                "## 7. Notes & Dropped Terms",
                "",
            ]
        )
        for n in all_notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


@dataclass
class AnalysisWorkflowResult:
    """Public return contract for the stored analysis workflow."""

    run_dir: Path
    selected_outcome: str
    n_observations: int
    primary_fit: PrimaryFit | None
    primary_analysis: dict[str, Any]
    strata_report: StrataReport | None
    lrt_result: LrtResult | None
    preregistration: PreregistrationVerdict | None
    reader_agreement: dict[str, Any]
    analysis_json_path: Path
    results_md_path: Path
    notes: list[str] = field(default_factory=list)
    status: str = "SUCCESS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_dir": str(self.run_dir),
            "selected_outcome": self.selected_outcome,
            "n_observations": self.n_observations,
            "primary": self.primary_analysis.get("primary", {}),
            "specification": self.primary_analysis.get("specification", {}),
            "reader_clustered_secondary": self.primary_analysis.get(
                "reader_clustered_secondary", {}
            ),
            "interaction_pp": self.primary_analysis.get("interaction_pp", {}),
            "strata": self.strata_report.to_dict() if self.strata_report else {},
            "lrt": self.lrt_result.to_dict() if self.lrt_result else {},
            "preregistration": self.preregistration.to_dict() if self.preregistration else {},
            "reader_agreement": self.reader_agreement,
            "analysis_json_path": str(self.analysis_json_path),
            "results_md_path": str(self.results_md_path),
            "notes": self.notes,
        }


def run_analysis_workflow(
    run_dir: Path | str,
    *,
    config_root: Path | str | None = None,
    selected_outcome: str = "em",
    metadata_path: Path | str | None = None,
    strict: bool = True,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
    write_artifacts: bool = True,
) -> AnalysisWorkflowResult:
    """Deep module owning the complete stored analysis workflow (Recommendation 05).

    Offline-only: Never calls external services or models.
    """
    rd = Path(run_dir)
    scores_path = rd / "scores.parquet"
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Required scores file not found at {scores_path}. Run `msat score` first."
        )

    # 1. Load scores
    scores_df = pd.read_parquet(scores_path)

    # 2. Load configs & metadata
    reader_strengths = load_readers_config(config_root)
    mde_val, _mde_source = load_power_mde(rd, config_root)
    metadata_df = load_metadata(metadata_path, run_dir=rd)

    # 3. Validate observation frame & joins
    df, val_notes = build_validated_observation_frame(
        scores_df,
        metadata_df=metadata_df,
        reader_strengths=reader_strengths,
        selected_outcome=selected_outcome,
        strict=strict,
    )

    all_notes = list(val_notes)

    # 4. Fit primary model & interaction analysis
    primary_res: dict[str, Any] = {}
    prim_fit: PrimaryFit | None = None
    status = "SUCCESS"

    # Single-reader pilot constant strength handling
    effective_strict = strict

    try:
        primary_res = primary_analysis(df, spec=PRIMARY_SPEC, strict=effective_strict)
        prim_fit = fit_primary(df, spec=PRIMARY_SPEC, strict=effective_strict)
        fitted_values = (
            *prim_fit.coefficients.values(),
            *prim_fit.std_errors.values(),
            *prim_fit.p_values.values(),
        )
        if not prim_fit.converged or not np.isfinite(fitted_values).all():
            raise ValueError(
                "Primary model did not converge to finite coefficients and uncertainty."
            )
        if "frame" in primary_res and isinstance(primary_res["frame"], dict):
            for fn in primary_res["frame"].get("notes", []):
                if fn not in all_notes:
                    all_notes.append(fn)
    except (RankDeficientDesignError, TooFewClustersError, ValueError) as exc:
        prim_fit = None
        status = "UNESTIMABLE"
        all_notes.append(f"Primary model unestimable: {exc}")
        primary_res = {
            "specification": PRIMARY_SPEC.to_dict(),
            "status": "UNESTIMABLE",
            "error": str(exc),
            "primary": {"converged": False, "error": str(exc)},
        }

    # 5. Strata report (Ability strata primary, pooled descriptive)
    strata_rep: StrataReport | None = None
    if "ability" in df.columns and status == "SUCCESS":
        try:
            strata_rep = strata_report(
                df,
                strat_col="ability",
                k_col="K",
                y_col="correct",
                question_col="question",
                n_boot=n_boot,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - optional secondary report failure must be recorded, not crash workflow
            all_notes.append(f"Strata report failed: {exc}")

    # 6. LRT Monotone vs Inverted-U
    lrt_res: LrtResult | None = None
    if status == "SUCCESS":
        try:
            lrt_res = lrt_monotone_vs_inverted_u(df, spec=PRIMARY_SPEC, strict=effective_strict)
        except Exception as exc:  # noqa: BLE001 - optional secondary test failure must be recorded, not crash workflow
            all_notes.append(f"LRT test failed: {exc}")

    # 7. Preregistration verdict
    prereg_verdict: PreregistrationVerdict | None = None
    if prim_fit is not None and status == "SUCCESS":
        try:
            prereg_verdict = preregistration_verdict(
                prim_fit,
                mde=mde_val,
                config_root=config_root,
            )
        except Exception as exc:  # noqa: BLE001 - optional preregistration check failure must be recorded, not crash workflow
            all_notes.append(f"Preregistration verdict failed: {exc}")

    # 8. Reader agreement comparison
    rag_dict: dict[str, Any] = {}
    if "reader" in df.columns and df["reader"].nunique() >= 2:
        try:
            # Build ConditionScores
            conditions_scores: list[ConditionScore] = []
            for (reader, k, b, scale, arm), grp in df.groupby(["reader", "K", "B", "scale", "arm"]):
                conditions_scores.append(
                    ConditionScore(
                        reader=str(reader),
                        k=int(k),
                        b=float(b),
                        scale=str(scale),
                        arm=str(arm),
                        score=float(grp["correct"].mean()),
                        n_questions=int(grp["question"].nunique()),
                    )
                )
            responses = responses_from_conditions(conditions_scores)
            report_rag = compare_readers(responses)
            rag_dict = report_rag.as_dict()
            rag_dict["available"] = True
        except Exception as exc:  # noqa: BLE001 - optional reader agreement failure must be recorded, not crash workflow
            rag_dict = {"available": False, "note": f"Reader comparison failed: {exc}"}
    else:
        rag_dict = {
            "available": False,
            "note": "Single reader in run; cross-reader slope comparison requires >= 2 readers.",
        }

    # Provenance block
    prov_block = {
        "outcome": selected_outcome,
        "n_obs": len(df),
        "n_questions": int(df["question"].nunique()) if "question" in df.columns else 0,
        "n_conditions": int(df["condition_key"].nunique()) if "condition_key" in df.columns else 0,
        "readers": sorted(df["reader"].unique().tolist()) if "reader" in df.columns else [],
    }

    full_payload: dict[str, Any] = {
        "status": status,
        "provenance": prov_block,
        "specification": primary_res.get("specification", {}),
        "primary": primary_res.get("primary", {}),
        "reader_clustered_secondary": primary_res.get("reader_clustered_secondary", {}),
        "interaction_pp": primary_res.get("interaction_pp", {}),
        "strata": strata_rep.to_dict() if strata_rep else {},
        "lrt": lrt_res.to_dict() if lrt_res else {},
        "preregistration": prereg_verdict.to_dict() if prereg_verdict else {},
        "reader_agreement": rag_dict,
        "notes": all_notes,
    }
    if status != "SUCCESS":
        full_payload["error"] = primary_res.get("error", "Primary model is unestimable.")

    json_path = rd / "analysis.json"
    md_path = rd / "results.md"

    md_content = generate_results_md(full_payload)

    if write_artifacts:
        json_path.write_text(
            json.dumps(full_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(md_content + "\n", encoding="utf-8")

    return AnalysisWorkflowResult(
        run_dir=rd,
        selected_outcome=selected_outcome,
        n_observations=len(df),
        primary_fit=prim_fit,
        primary_analysis=primary_res,
        strata_report=strata_rep,
        lrt_result=lrt_res,
        preregistration=prereg_verdict,
        reader_agreement=rag_dict,
        analysis_json_path=json_path,
        results_md_path=md_path,
        notes=all_notes,
        status=status,
    )
