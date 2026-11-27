"""Stored scoring workflow (Architecture Recommendation 04).

Owns:
- validated mode, metric, and normalisation configuration loading
- gold loading supporting both LongMemEval oracle list and qid maps
- condition-aware official cached result selection and join validation
- response traversal across ResponseStore
- artifacts generation: scores.parquet, latency.json, disagreements.parquet
- observation metadata validation
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import yaml

from msat.metrics.latency import LatencySample, latency_report, summarise_latency
from msat.metrics.normalisation import Normaliser
from msat.metrics.score import (
    Observation,
    ScoreReport,
    _means,
    build_disagreements,
    condition_key_str,
    score_store,
    write_disagreements,
    write_scores,
)
from msat.store import ResponseStore

VALID_MODES: Final[frozenset[str]] = frozenset(
    {"fast", "full", "cached", "offline", "official", "all"}
)
VALID_METRICS: Final[frozenset[str]] = frozenset({"em", "f1", "official", "judge", "all"})


class AmbiguousOfficialJoinError(ValueError):
    """Raised when an official score mapping lacks condition identity across multiple conditions."""


class InvalidScoringConfigurationError(ValueError):
    """Raised when scoring mode or metrics are invalid or contradictory."""


def load_normaliser(
    config_path: Path | str | None = None,
    config_dir: Path | str | None = None,
    spec: dict[str, Any] | None = None,
) -> Normaliser:
    """Load and validate answer normalisation configuration.

    Order of resolution:
    1. Direct `spec` dictionary if supplied.
    2. Explicit `config_path` if supplied.
    3. `config_dir / "prompts" / "normalisation.yaml"` or `config_dir / "normalisation.yaml"`.
    4. CWD `configs/prompts/normalisation.yaml`.
    """
    if spec is not None:
        return Normaliser.from_config(spec)

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    elif config_dir is not None:
        cd = Path(config_dir)
        candidates.extend(
            [
                cd / "prompts" / "normalisation.yaml",
                cd / "normalisation.yaml",
            ]
        )
    else:
        candidates.append(Path("configs/prompts/normalisation.yaml"))

    resolved: Path | None = None
    for cand in candidates:
        if cand.is_file():
            resolved = cand
            break

    if resolved is None:
        searched = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Normalisation configuration not found. Searched: {searched}. "
            "R1.2 requires an explicit, documented normalisation configuration."
        )

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Normalisation file {resolved} must contain a YAML mapping.")
    return Normaliser.from_config(raw)


def validate_mode_and_metrics(
    mode: str = "fast",
    metric: str | Sequence[str] = "all",
    *,
    has_official: bool = False,
    has_verdicts: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Validate scoring mode and requested metric subset.

    Returns:
        (clean_mode, exact_metric_tuple)
    """
    clean_mode = str(mode).lower().strip()
    if clean_mode not in VALID_MODES:
        raise InvalidScoringConfigurationError(
            f"unknown mode {mode!r}; valid modes are {sorted(VALID_MODES)}"
        )

    metrics_set: tuple[str, ...]
    if isinstance(metric, str):
        m_str = metric.lower().strip()
        if m_str == "all":
            if clean_mode == "fast":
                metrics_set = ("em", "f1")
            else:
                metrics_set = ("em", "f1", "official") if has_official else ("em", "f1")
                if has_verdicts:
                    metrics_set += ("judge",)
        elif m_str == "judge":
            metrics_set = ("judge",)
        elif m_str in ("em", "f1", "official"):
            metrics_set = (m_str,)
        else:
            raise InvalidScoringConfigurationError(
                f"unknown metric(s) {[metric]}; use em, f1, official, judge or all"
            )
    else:
        clean_list: list[str] = []
        for m in metric:
            item = str(m).lower().strip()
            if item == "all":
                clean_list.extend(["em", "f1"] + (["official"] if has_official else []))
                if clean_mode != "fast" and has_verdicts:
                    clean_list.append("judge")
            elif item == "judge":
                clean_list.append("judge")
            elif item in ("em", "f1", "official"):
                clean_list.append(item)
            else:
                raise InvalidScoringConfigurationError(
                    f"unknown metric(s) {[m]}; use em, f1, official, judge or all"
                )
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for c in clean_list:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        metrics_set = tuple(deduped)

    if not metrics_set:
        raise InvalidScoringConfigurationError("At least one metric must be selected for scoring.")

    return clean_mode, metrics_set


def load_gold(
    gold_source: Path | str | Mapping[str, Any] | Sequence[Any],
) -> dict[str, str | tuple[str, ...]]:
    """Load gold answers supporting LongMemEval oracle lists and qid mappings.

    Never invents scores or placeholders.
    """
    raw_data: Any
    if isinstance(gold_source, (Path, str)):
        path = Path(gold_source)
        if not path.is_file():
            raise FileNotFoundError(f"Gold answer file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            raw_data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse gold file {path} as JSON: {exc}") from exc
    else:
        raw_data = gold_source

    gold_map: dict[str, str | tuple[str, ...]] = {}

    if isinstance(raw_data, Mapping):
        for k, v in raw_data.items():
            qid = str(k)
            if isinstance(v, (list, tuple)):
                gold_map[qid] = tuple(str(x) for x in v)
            elif v is not None:
                gold_map[qid] = str(v)
    elif isinstance(raw_data, Sequence):
        # Oracle list: e.g. [{"question_id": "...", "answer": ...}, ...]
        for i, item in enumerate(raw_data):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Oracle entry at index {i} must be a mapping, got {type(item).__name__}"
                )
            qid_val = item.get("question_id") or item.get("qid")
            if not qid_val:
                raise ValueError(f"Oracle entry at index {i} lacks 'question_id' or 'qid'")
            qid = str(qid_val)
            if qid in gold_map:
                raise ValueError(f"Duplicate gold question identity {qid!r} in oracle entries.")
            ans = item.get("answer")
            if ans is None:
                ans = item.get("gold_answers")
            if ans is None:
                ans = item.get("gold")

            if isinstance(ans, (list, tuple)):
                gold_map[qid] = tuple(str(x) for x in ans)
            elif ans is not None:
                gold_map[qid] = str(ans)
    else:
        raise ValueError(
            f"Unsupported gold source format {type(raw_data).__name__}; expected dict or list."
        )

    if not gold_map:
        raise ValueError(
            "Gold answer collection is empty; scoring requires at least one gold answer."
        )

    return gold_map


def load_official(
    source: Path | str | Mapping[str, Any] | Sequence[Any] | None = None,
    run_dir: Path | str | None = None,
) -> dict[Any, float] | None:
    """Load cached official model-based evaluation scores if available.

    Preserves condition identity when present:
    - `{(condition_key, qid): score}`
    - `{"condition_key|qid": score}`
    - `{cache_key: score}`
    - `{qid: score}`
    """
    raw: Any = None
    if source is not None:
        if isinstance(source, (Path, str)):
            p = Path(source)
            if not p.is_file():
                raise FileNotFoundError(f"Official score file does not exist: {p}")
            if p.suffix == ".parquet":
                import pyarrow.parquet as pq  # type: ignore[import-untyped]

                raw = pq.read_table(p).to_pylist()
            else:
                raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            raw = source
    elif run_dir is not None:
        p_json = Path(run_dir) / "official.json"
        p_parquet = Path(run_dir) / "official.parquet"
        if p_json.is_file():
            raw = json.loads(p_json.read_text(encoding="utf-8"))
        elif p_parquet.is_file():
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            tbl = pq.read_table(p_parquet)
            raw = tbl.to_pylist()

    if raw is None:
        return None

    out: dict[Any, float] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(k, tuple):
                out[k] = float(v)
            else:
                out[str(k)] = float(v)
    elif isinstance(raw, Sequence):
        for item in raw:
            if isinstance(item, Mapping):
                score_val = item.get("score", item.get("value", item.get("verdict")))
                if score_val is None:
                    continue
                score = float(score_val)
                qid = str(item.get("qid", item.get("question_id", "")))
                ck = str(item.get("condition_key", ""))
                cache_key = str(item.get("cache_key", ""))
                if ck and qid:
                    out[(ck, qid)] = score
                    out[f"{ck}|{qid}"] = score
                elif cache_key:
                    out[cache_key] = score
                elif qid:
                    out[qid] = score

    return out if out else None


def validate_official_join(
    official: Mapping[Any, float] | None,
    store: ResponseStore,
) -> Mapping[Any, float] | None:
    """Validate that official cached results do not ambiguously broadcast across conditions.

    If an official mapping is keyed only by bare qid, and that qid appears across multiple
    conditions in the store, raise AmbiguousOfficialJoinError.
    """
    if official is None:
        return None

    # Inventory conditions per qid in the store
    qid_conditions: dict[str, set[str]] = {}
    for record in store.iter_all():
        if not record.qid:
            continue
        ck = condition_key_str(record)
        qid_conditions.setdefault(record.qid, set()).add(ck)

    # Check for bare qid keys in official
    for key in official:
        if isinstance(key, tuple):
            continue
        key_str = str(key)
        conditions = qid_conditions.get(key_str, set())
        if len(conditions) > 1:
            raise AmbiguousOfficialJoinError(
                f"Official score mapping contains bare qid {key_str!r} without condition key, "
                f"but this question appears in {len(conditions)} distinct conditions: "
                f"{sorted(conditions)}. Refusing ambiguous broadcast across conditions."
            )

    return official


def _latency_samples(
    store: ResponseStore,
) -> tuple[list[LatencySample], dict[str, list[LatencySample]], int]:
    total: list[LatencySample] = []
    by_reader: dict[str, list[LatencySample]] = {}
    n_impossible = 0
    for record in store.iter_all():
        try:
            sample = LatencySample(
                latency_ms=record.latency_ms,
                queue_wait_ms=record.queue_wait_ms,
                ttft_ms=record.ttft_ms,
            )
        except ValueError:
            n_impossible += 1
            continue
        total.append(sample)
        reader = str(record.condition.get("reader", "unknown"))
        by_reader.setdefault(reader, []).append(sample)
    return total, by_reader, n_impossible


def _ttft_samples(samples: Sequence[LatencySample]) -> list[LatencySample]:
    return [
        LatencySample(
            latency_ms=s.ttft_ms,
            queue_wait_ms=s.queue_wait_ms,
            ttft_ms=s.ttft_ms,
        )
        for s in samples
        if s.ttft_ms is not None and s.ttft_ms > 0
    ]


def build_latency_artifact(store: ResponseStore) -> dict[str, Any]:
    """Summarise latency and TTFT metrics from the store (PRD §R8.5)."""
    total, by_reader, n_impossible = _latency_samples(store)
    ttft = _ttft_samples(total)
    return {
        "requirement": "PRD §R8.5",
        "total_latency": latency_report(total, by_label=dict(by_reader)),
        "ttft": summarise_latency(ttft, label="ttft").as_dict(),
        "ttft_is_separate_because": (
            "ttft_ms is queue_wait -> first token and latency_ms is queue_wait -> last "
            "token. They measure different things, so they are summarised separately; "
            "the total is a property of the generator's tail, the TTFT is a property "
            "of the reader (PRD §R5.3)."
        ),
        "n_impossible_records": n_impossible,
        "n_latency_samples": len(total),
        "no_bootstrap_ci": (
            "Deliberately absent. A bootstrap CI on a heavy-tailed mean reports "
            "uncertainty about a statistic nobody should quote (PRD §R8.5)."
        ),
    }


def load_verdicts(
    verdicts_path: Path | str | None = None,
    run_dir: Path | str | None = None,
) -> list[dict[str, Any]] | None:
    """Load judge verdict records if present."""
    p: Path | None = None
    if verdicts_path is not None:
        p = Path(verdicts_path)
    elif run_dir is not None:
        p = Path(run_dir) / "verdicts.json"

    if p is not None and verdicts_path is not None and not p.is_file():
        raise FileNotFoundError(f"Verdict file does not exist: {p}")
    if p is None or not p.is_file():
        return None

    if p.suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        raw = pq.read_table(p).to_pylist()
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Verdict file {p} must contain a list of verdict records.")
    return raw


def build_disagreements_from_verdicts(
    verdicts: list[dict[str, Any]],
    store: ResponseStore,
    gold: Mapping[str, Any],
    norm: Normaliser,
) -> tuple[list[dict[str, Any]], int]:
    """Build pairwise judge disagreements preserving condition identity.

    Returns:
        (disagreements_rows, judge_count)
    """
    by_judge: dict[str, dict[tuple[str, str], Any]] = {}
    texts: dict[tuple[str, str], str] = {}
    from msat.judges.verdicts import as_verdict

    for r in verdicts:
        if int(r.get("pass_index", 1)) != 1:
            continue
        ck = str(r.get("condition_key", ""))
        qid = str(r.get("qid", ""))
        if not ck or not qid:
            continue
        key = (ck, qid)
        judge_id = str(r.get("judge_id", ""))
        verdict = as_verdict(r.get("verdict"))
        by_judge.setdefault(judge_id, {})[key] = verdict

        if key not in texts:
            cache_key = r.get("cache_key")
            if cache_key:
                rec = store.get(str(cache_key))
                if rec is not None:
                    texts[key] = rec.text
            else:
                # Find matching record in store
                for record in store.iter_all():
                    if record.qid == qid and condition_key_str(record) == ck:
                        texts[key] = record.text
                        break

    gold_single: dict[str, str] = {}
    for k, v in gold.items():
        gold_single[k] = v[0] if isinstance(v, (list, tuple)) and v else str(v)

    disagreements = build_disagreements(by_judge, texts, gold_single, norm)
    return disagreements, len(by_judge)


def score_cached_judges(
    templates: Sequence[Observation], verdicts: Sequence[dict[str, Any]]
) -> list[Observation]:
    """Join first-pass cached verdicts to stored answers without pooling judges."""
    from msat.judges.verdicts import Verdict, as_verdict

    by_cache = {row.cache_key: row for row in templates}
    by_item: dict[tuple[str, str], set[str]] = {}
    for row in templates:
        by_item.setdefault((row.condition_key, row.qid), set()).add(row.cache_key)
    values: dict[tuple[str, str], float] = {}
    judges: set[str] = set()
    for record in verdicts:
        if int(record.get("pass_index", 1)) != 1:
            continue
        judge = str(record.get("judge_id", "")).strip()
        if not judge:
            raise InvalidScoringConfigurationError("A cached judge verdict lacks judge_id.")
        item = (str(record.get("condition_key", "")), str(record.get("qid", "")))
        key = str(record.get("cache_key", ""))
        if key:
            template = by_cache.get(key)
            if template is None or (template.condition_key, template.qid) != item:
                raise InvalidScoringConfigurationError(
                    "Cached judge verdict identity does not match its stored answer."
                )
        else:
            candidates = by_item.get(item, set())
            if len(candidates) != 1:
                raise InvalidScoringConfigurationError(
                    "Cached judge verdict requires a unique stored answer or an explicit cache_key."
                )
            key = next(iter(candidates))
        value = float(as_verdict(record.get("verdict")) == Verdict.CORRECT)
        identity = (key, judge)
        if identity in values and values[identity] != value:
            raise InvalidScoringConfigurationError(
                "Conflicting cached verdicts for one answer and judge."
            )
        values[identity] = value
        judges.add(judge)
    if not judges:
        raise InvalidScoringConfigurationError(
            "The judge metric requires cached first-pass verdicts."
        )
    return [
        replace(row, metric=f"judge:{judge}", value=values.get((row.cache_key, judge)))
        for row in templates
        for judge in sorted(judges)
    ]


def load_and_validate_observation_metadata(
    metadata_path: Path | str | None = None,
    run_dir: Path | str | None = None,
) -> Any | None:
    """Validate observation metadata sidecar if supplied or present."""
    import pandas as pd  # type: ignore[import-untyped]

    p: Path | None = None
    if metadata_path is not None:
        p = Path(metadata_path)
    elif run_dir is not None:
        candidates = [
            Path(run_dir) / "metadata.parquet",
            Path(run_dir) / "metadata.json",
        ]
        for c in candidates:
            if c.is_file():
                p = c
                break

    if p is not None and metadata_path is not None and not p.is_file():
        raise FileNotFoundError(f"Observation metadata file does not exist: {p}")
    if p is None or not p.is_file():
        return None

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported metadata format for {p}; use parquet, json, or csv")

    # Validate observation uniqueness if key column exists
    if "cache_key" in df.columns:
        keys = ["cache_key"] + (["replicate"] if "replicate" in df.columns else [])
        if df[keys].isna().any().any() or df.duplicated(subset=keys).any():
            raise ValueError(
                "Observation metadata contains missing or duplicate observation identity."
            )

    return df


@dataclass
class ScoringWorkflowResult:
    """Public return contract for the stored scoring workflow."""

    observations: list[Observation] = field(default_factory=list)
    report: ScoreReport = field(default_factory=ScoreReport)
    scores_path: Path = field(default_factory=Path)
    latency_path: Path = field(default_factory=Path)
    latency_data: dict[str, Any] = field(default_factory=dict)
    disagreements_path: Path | None = None
    disagreements_count: int = 0
    verdicts_missing: bool = False
    metadata_path: Path | None = None
    notes: list[str] = field(default_factory=list)
    rows: list[Observation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.observations and self.rows:
            self.observations = self.rows
        elif not self.rows and self.observations:
            self.rows = self.observations

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_observations": len(self.observations),
            "n_rows": len(self.observations),
            "report": self.report.as_dict(),
            "scores_path": str(self.scores_path),
            "latency_path": str(self.latency_path),
            "latency": self.latency_data,
            "disagreements_path": str(self.disagreements_path) if self.disagreements_path else None,
            "disagreements_count": self.disagreements_count,
            "verdicts_missing": self.verdicts_missing,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "notes": self.notes,
        }


def run_scoring_workflow(
    run_dir: Path | str,
    *,
    gold: Path | str | Mapping[str, Any] | Sequence[Any],
    config_dir: Path | str | None = None,
    mode: str = "fast",
    metric: str | Sequence[str] = "all",
    official: Path | str | Mapping[str, Any] | Sequence[Any] | None = None,
    verdicts_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    write_artifacts: bool = True,
) -> ScoringWorkflowResult:
    """Deep module owning the complete stored scoring workflow (Recommendation 04).

    Offline-only: Never invokes live models.
    """
    run_path = Path(run_dir)
    validate_mode_and_metrics(mode, metric)
    responses_dir = run_path / "responses"
    if not responses_dir.is_dir():
        raise FileNotFoundError(f"No responses directory found at {responses_dir}")

    store = ResponseStore(responses_dir)
    if len(store) == 0:
        raise ValueError(f"{responses_dir} holds zero responses; nothing to score.")

    # 1. Normaliser
    norm = load_normaliser(config_dir=config_dir)

    # 2. Gold Answers
    gold_map = load_gold(gold)

    # 3. Official evaluations
    official_data = load_official(official, run_dir=run_path)
    verdict_records = load_verdicts(verdicts_path, run_dir=run_path)

    # 4. Mode & Metrics validation
    _clean_mode, metrics_subset = validate_mode_and_metrics(
        mode=mode,
        metric=metric,
        has_official=official_data is not None,
        has_verdicts=verdict_records is not None,
    )
    official_validated = (
        validate_official_join(official_data, store) if "official" in metrics_subset else None
    )
    if "official" in metrics_subset and official_validated is None:
        raise InvalidScoringConfigurationError(
            "The official metric requires cached official scores; no model is called by score."
        )

    # 5. Score Store
    rows, report = score_store(
        store,
        gold_map,
        norm,
        metrics=metrics_subset,
        official=official_validated,
        expand_replicates=True,
    )
    if "judge" in metrics_subset:
        if verdict_records is None:
            raise InvalidScoringConfigurationError(
                "The judge metric requires cached verdicts; no model is called by score."
            )
        templates, _ = score_store(store, gold_map, norm, metrics=("em",), expand_replicates=True)
        rows.extend(score_cached_judges(templates, verdict_records))
        report.n_rows = len(rows)
        report.metrics_written = tuple(dict.fromkeys(row.metric for row in rows))
        report.means = _means(rows)

    notes = list(report.notes)

    # 6. Artifact paths
    scores_dest = run_path / "scores.parquet"
    latency_dest = run_path / "latency.json"
    dis_dest = run_path / "disagreements.parquet"

    # Latency artifact
    latency_data = build_latency_artifact(store)

    # Verdicts & disagreements
    disagreements_rows: list[dict[str, Any]] = []
    verdicts_missing = verdict_records is None
    dis_path_result: Path | None = None

    if verdict_records is not None:
        disagreements_rows, n_judges = build_disagreements_from_verdicts(
            verdict_records, store, gold_map, norm
        )
        if n_judges < 2:
            notes.append(
                f"Only {n_judges} judge present; >= 2 judges required for disagreement analysis (PRD §R8.2)."
            )
        dis_path_result = dis_dest
    else:
        notes.append(
            "no verdicts.json; disagreements.parquet was not written. A missing file is not evidence that the judges agreed."
        )

    # Observation metadata validation
    _obs_meta = load_and_validate_observation_metadata(metadata_path, run_dir=run_path)
    meta_path_result = Path(metadata_path) if metadata_path else None

    from msat.config import compute_config_hash

    scoring_hash = compute_config_hash(
        {
            "normalisation": {
                "steps": [{"id": step.id, "op": step.op, **step.params} for step in norm.steps],
                "apply_to": norm.apply_to,
            },
            "gold": gold_map,
            "metrics": metrics_subset,
            "mode": _clean_mode,
            "official": sorted(
                (str(key), value) for key, value in (official_validated or {}).items()
            ),
            "judges": verdict_records if "judge" in metrics_subset else None,
        }
    )
    metadata_index: dict[Any, Any] = {}
    repeated_metadata = _obs_meta is not None and (
        "repetition" in _obs_meta.columns or "replicate" in _obs_meta.columns
    )
    if _obs_meta is not None and "cache_key" in _obs_meta.columns:
        rep_col = "repetition" if "repetition" in _obs_meta.columns else "replicate"
        for entry in _obs_meta.to_dict(orient="records"):
            key = (entry["cache_key"], entry[rep_col]) if repeated_metadata else entry["cache_key"]
            metadata_index[key] = entry
    for row in rows:
        key = (row.cache_key, row.repetition) if repeated_metadata else row.cache_key
        entry = metadata_index.get(key, {})
        recorded_hash = entry.get("config_hash")
        if recorded_hash and row.config_hash and recorded_hash != row.config_hash:
            raise ValueError(
                "Observation metadata and stored response configuration hashes disagree."
            )
        row.config_hash = row.config_hash or recorded_hash
        row.scoring_config_hash = scoring_hash
    unattributed_config = sum(row.config_hash is None for row in rows)
    if unattributed_config:
        notes.append(
            f"{unattributed_config} legacy score row(s) have no recorded generation config_hash."
        )

    # Write artifacts if requested
    if write_artifacts:
        write_scores(rows, scores_dest)
        report.path = str(scores_dest)
        report.n_bytes = scores_dest.stat().st_size

        latency_dest.write_text(
            json.dumps(latency_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if verdict_records is not None:
            write_disagreements(disagreements_rows, dis_dest)

    return ScoringWorkflowResult(
        rows=rows,
        report=report,
        scores_path=scores_dest,
        latency_path=latency_dest,
        latency_data=latency_data,
        disagreements_path=dis_path_result if write_artifacts else None,
        disagreements_count=len(disagreements_rows),
        verdicts_missing=verdicts_missing,
        metadata_path=meta_path_result,
        notes=notes,
    )
