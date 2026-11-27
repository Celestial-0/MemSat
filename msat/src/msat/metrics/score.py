"""Per-item scores from the store, with the network closed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.arms.evidence import ABSTENTION_SUFFIX
from msat.metrics.normalisation import Normaliser, exact_match, token_f1
from msat.store import Condition, Observation, ResponseStore, StoredResponse

SCORES_SCHEMA_VERSION = 1

METRICS: tuple[str, ...] = ("em", "f1", "official")


def scores_schema() -> Any:
    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("cache_key", pa.string(), nullable=False),
            pa.field("condition_key", pa.string(), nullable=False),
            pa.field("qid", pa.string(), nullable=False),
            pa.field("arm", pa.string(), nullable=False),
            pa.field("K", pa.int32(), nullable=False),
            pa.field("B", pa.float64(), nullable=False),
            pa.field("B_mode", pa.string(), nullable=False),
            pa.field("scale", pa.string(), nullable=False),
            pa.field("reader", pa.string(), nullable=False),
            pa.field("repetition", pa.int32(), nullable=True),
            pa.field("cached", pa.bool_(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
            pa.field("is_abstention", pa.bool_(), nullable=False),
            pa.field("answerable", pa.bool_(), nullable=False),
            pa.field("finish_reason", pa.string(), nullable=False),
            pa.field("latency_ms", pa.float64(), nullable=True),
            pa.field("ttft_ms", pa.float64(), nullable=True),
            pa.field("queue_wait_ms", pa.float64(), nullable=True),
            pa.field("config_hash", pa.string(), nullable=True),
            pa.field("scoring_config_hash", pa.string(), nullable=True),
        ]
    )


def condition_key_str(record: StoredResponse) -> str:
    try:
        cond = Condition.from_dict(record.condition)
    except (KeyError, TypeError, ValueError):
        return "|".join(
            str(record.condition.get(k, "")) for k in ("arm", "K", "B", "B_mode", "scale", "reader")
        )
    return "|".join(str(x) for x in cond.key)


@dataclass
class ScoreReport:
    n_rows: int = 0
    n_items: int = 0
    n_attributable: int = 0
    n_unattributable: int = 0
    n_abstention: int = 0
    n_missing_gold: int = 0
    metrics_written: tuple[str, ...] = ()
    means: dict[str, dict[str, float | None]] = field(default_factory=dict)
    path: str = ""
    n_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_items": self.n_items,
            "n_attributable": self.n_attributable,
            "n_unattributable": self.n_unattributable,
            "n_abstention": self.n_abstention,
            "n_missing_gold": self.n_missing_gold,
            "metrics_written": list(self.metrics_written),
            "means": self.means,
            "path": self.path,
            "n_bytes": self.n_bytes,
            "notes": self.notes,
        }


def _is_abstention(qid: str) -> bool:
    return qid.endswith(ABSTENTION_SUFFIX)


def score_store(
    store: ResponseStore,
    gold: Mapping[str, str | Sequence[str]],
    norm: Normaliser,
    *,
    metrics: Sequence[str] = ("em", "f1"),
    official: Mapping[Any, float] | None = None,
    expand_replicates: bool = False,
    expand_repetitions: bool | None = None,
) -> tuple[list[Observation], ScoreReport]:
    report = ScoreReport()
    rows: list[Observation] = []
    wanted = tuple(m for m in metrics if m in ("em", "f1"))
    if "official" in metrics:
        available = list(wanted) + (["official"] if official is not None else [])
    elif official is not None and set(metrics) == {"em", "f1"}:
        available = [*list(wanted), "official"]
    else:
        available = list(wanted)

    expand_rep = expand_repetitions if expand_repetitions is not None else expand_replicates
    for record in store.iter_all():
        qid = record.qid
        if not qid:
            report.n_unattributable += 1
            continue
        reps = record.repetitions
        repeats = tuple(sorted(set(reps))) if expand_rep else ()
        n_observations = len(repeats) or 1
        report.n_attributable += n_observations
        abstention = _is_abstention(qid)
        if abstention:
            report.n_abstention += n_observations
        g = gold.get(qid)
        if g is None and not abstention:
            report.n_missing_gold += n_observations

        ck = condition_key_str(record)
        common: dict[str, Any] = {
            "cache_key": record.key,
            "condition_key": ck,
            "qid": qid,
            "arm": str(record.condition.get("arm", "")),
            "k": int(record.condition.get("K", 0) or 0),
            "b": float(record.condition.get("B", 0.0) or 0.0),
            "b_mode": str(record.condition.get("B_mode", "")),
            "scale": str(record.condition.get("scale", "")),
            "reader": str(record.condition.get("reader", "")),
            "repetition": max(reps) if reps else None,
            "cached": len(reps) > 1,
            "is_abstention": abstention,
            "answerable": not abstention,
            "finish_reason": record.finish_reason,
            "latency_ms": record.latency_ms,
            "ttft_ms": record.ttft_ms,
            "queue_wait_ms": record.queue_wait_ms,
            "config_hash": record.raw.get("config_hash") or record.condition.get("config_hash"),
        }

        for metric in available:
            value: float | None
            if metric == "official":
                if official is None:
                    value = None
                elif (ck, qid) in official:
                    value = official[(ck, qid)]
                elif f"{ck}|{qid}" in official:
                    value = official[f"{ck}|{qid}"]
                elif record.key in official:
                    value = official[record.key]
                elif qid in official:
                    value = official[qid]
                else:
                    value = None
            elif abstention or g is None:
                value = None
            elif metric == "em":
                if isinstance(g, (list, tuple)):
                    value = max(
                        (exact_match(record.text, str(ans), norm) for ans in g), default=0.0
                    )
                else:
                    value = exact_match(record.text, str(g), norm)
            else:
                if isinstance(g, (list, tuple)):
                    value = max((token_f1(record.text, str(ans), norm) for ans in g), default=0.0)
                else:
                    value = token_f1(record.text, str(g), norm)
            for repetition in repeats or (common["repetition"],):
                rows.append(
                    Observation(metric=metric, value=value, **{**common, "repetition": repetition})
                )

    report.n_rows = len(rows)
    report.n_items = report.n_attributable
    report.metrics_written = tuple(available)
    report.means = _means(rows)
    if official is None:
        report.notes.append(
            "The official model-based evaluator was not supplied. R8.1 requires "
            "its cached results alongside string metrics; no official score was invented."
        )
    if report.n_missing_gold:
        report.notes.append(
            f"{report.n_missing_gold} item(s) had no gold answer and scored NULL. They "
            f"are retained as rows so the denominator is visible."
        )
    return rows, report


def _means(rows: Sequence[Observation]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for metric in sorted({r.metric for r in rows}):
        vals = [
            r.value
            for r in rows
            if r.metric == metric and r.value is not None and not r.is_abstention
        ]
        out[metric] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n_scored": len(vals),
            "n_abstention_excluded": sum(1 for r in rows if r.metric == metric and r.is_abstention),
        }
    return out


def write_scores(rows: Sequence[Observation], path: Path, *, compression: str = "zstd") -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([r.as_dict() for r in rows], schema=scores_schema())
    pq.write_table(table, path, compression=compression)


def read_scores(path: Path) -> Any:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return pq.read_table(path)


def read_observations(path: Path) -> list[Observation]:
    table = read_scores(path)
    return [Observation.from_dict(row) for row in table.to_pylist()]


write_observations = write_scores


def disagreements_schema() -> Any:
    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("condition_key", pa.string(), nullable=False),
            pa.field("qid", pa.string(), nullable=False),
            pa.field("judge_a", pa.string(), nullable=False),
            pa.field("judge_b", pa.string(), nullable=False),
            pa.field("verdict_a", pa.string(), nullable=False),
            pa.field("verdict_b", pa.string(), nullable=False),
            pa.field("em", pa.float64(), nullable=True),
            pa.field("f1", pa.float64(), nullable=True),
            pa.field("is_abstention", pa.bool_(), nullable=False),
            pa.field("answer", pa.string(), nullable=False),
            pa.field("gold", pa.string(), nullable=False),
        ]
    )


def build_disagreements(
    verdicts: Mapping[str, Mapping[tuple[str, str], Any]],
    texts: Mapping[tuple[str, str], str],
    gold: Mapping[str, str],
    norm: Normaliser,
) -> list[dict[str, Any]]:
    labels = sorted(verdicts)
    rows: list[dict[str, Any]] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            a, b = verdicts[la], verdicts[lb]
            for key in sorted(set(a) & set(b)):
                if a[key] is b[key] or a[key] == b[key]:
                    continue
                answer = texts.get(key, "")
                g = gold.get(key[1], "")
                abstention = _is_abstention(key[1])
                rows.append(
                    {
                        "schema_version": SCORES_SCHEMA_VERSION,
                        "condition_key": key[0],
                        "qid": key[1],
                        "judge_a": la,
                        "judge_b": lb,
                        "verdict_a": str(a[key]),
                        "verdict_b": str(b[key]),
                        "em": None if (abstention or not g) else exact_match(answer, g, norm),
                        "f1": None if (abstention or not g) else token_f1(answer, g, norm),
                        "is_abstention": abstention,
                        "answer": answer,
                        "gold": g,
                    }
                )
    return rows


def write_disagreements(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=disagreements_schema())
    pq.write_table(table, path, compression="zstd")


def read_disagreements(path: Path) -> Any:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return pq.read_table(path)
