"""`msat score` — per-item scores from the store, with the network closed (R8.1).

PRD §R8.1 and G4: three metrics side by side, ALWAYS - EM, token-F1 under our own
documented normalisation, and the official model-based evaluator. Never one alone.

WHY "ALWAYS" AND NOT "WHICHEVER IS AVAILABLE"
---------------------------------------------
A single number cannot be sanity-checked by anything, and the three fail differently.
EM is unforgiving of paraphrase, which is the right behaviour for a benchmark and
the wrong behaviour for a memory paper - a reader that retrieves the right session
and phrases the answer correctly scores 0. F1 is forgiving in the other direction
and will reward a long answer containing the right span. Official-eval is the
community's number and exists so the results can be compared to prior work. A
reader who is shown EM alone concludes the system is worse than it is; one shown F1
alone concludes it is better. Neither conclusion is supportable from one metric, and
the disagreement between them is itself the finding.

ONE ROW PER ITEM, NEVER AN AGGREGATE
------------------------------------
R8.3. `scores.parquet` is per `(condition, qid, metric)`. Aggregates are computed
from it at analysis time, never stored. A stored aggregate cannot be re-derived
under a different metric, a different normalisation, or a different exclusion
rule, and the metric code WILL change - that is the reason the raw provider JSON is
kept (R6.2).

NORMALISATION COMES FROM CONFIG, NOT FROM CODE
----------------------------------------------
`configs/prompts/normalisation.yaml` is the single definition (R1.2). This module
reads it. A `strip()` and a `lower()` written here would be a second definition,
and the two would agree until someone edited the config.

ABSTENTION IS A COLUMN, NOT A FILTER (R8.4)
-------------------------------------------
`_abs` items get `em = NULL` and `is_abstention = true`. They are not dropped: they
are scored by their own declared rule, and dropping them would shrink the
denominator of every accuracy number without saying so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.arms.evidence import ABSTENTION_SUFFIX
from msat.metrics.normalisation import Normaliser, exact_match, token_f1
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse

SCORES_SCHEMA_VERSION = 1

METRICS: tuple[str, ...] = ("em", "f1", "official")


def scores_schema() -> Any:
    """Declared, not inferred. See `arms/conditions.py` for why."""
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
            pa.field("replicate", pa.int32(), nullable=True),
            pa.field("cached", pa.bool_(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
            pa.field("is_abstention", pa.bool_(), nullable=False),
            pa.field("answerable", pa.bool_(), nullable=False),
            pa.field("finish_reason", pa.string(), nullable=False),
            pa.field("latency_ms", pa.float64(), nullable=True),
            pa.field("ttft_ms", pa.float64(), nullable=True),
            pa.field("queue_wait_ms", pa.float64(), nullable=True),
        ]
    )


def condition_key_str(record: StoredResponse) -> str:
    """A stable string form of the condition, so scores join on one key."""
    try:
        cond = ConditionKey.from_dict(record.condition)
    except (KeyError, TypeError, ValueError):
        return "|".join(
            str(record.condition.get(k, "")) for k in ("arm", "K", "B", "B_mode", "scale", "reader")
        )
    return "|".join(str(x) for x in cond.key)


@dataclass
class ScoreRow:
    cache_key: str
    condition_key: str
    qid: str
    metric: str
    value: float | None
    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str
    replicate: int | None = None
    cached: bool = False
    is_abstention: bool = False
    answerable: bool = True
    finish_reason: str = ""
    latency_ms: float | None = None
    ttft_ms: float | None = None
    queue_wait_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        # `K` and `B` are mapped EXPLICITLY. The dataclass fields are `k` and `b`
        # (python style) while the column names are `K` and `B`, matching every
        # other manifest in the project. A silent mismatch here fills a
        # non-nullable column with nulls, which pyarrow rejects - and if the column
        # had been nullable it would have been a scores.parquet full of holes.
        return {
            "schema_version": SCORES_SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "condition_key": self.condition_key,
            "qid": self.qid,
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "scale": self.scale,
            "reader": self.reader,
            "replicate": self.replicate,
            "cached": self.cached,
            "metric": self.metric,
            "value": self.value,
            "is_abstention": self.is_abstention,
            "answerable": self.answerable,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "queue_wait_ms": self.queue_wait_ms,
        }


@dataclass
class ScoreReport:
    """What the scoring run produced, including what it could NOT score."""

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
    gold: Mapping[str, str],
    norm: Normaliser,
    *,
    metrics: Sequence[str] = ("em", "f1"),
    official: Mapping[str, float] | None = None,
) -> tuple[list[ScoreRow], ScoreReport]:
    """Score every stored response. Returns rows and a report.

    `gold` maps qid -> gold answer. `official` maps qid -> official-evaluator score,
    supplied rather than computed here: that evaluator is a model-based service and
    R8.1 requires it SIDE BY SIDE with EM and F1, not instead of them. When it is
    absent the metric is left out and the report says so - a silent two-metric run
    would look like a three-metric run in the artifact.

    Responses with no recoverable qid are COUNTED and reported, never scored. A
    store whose entries cannot be attributed to a question cannot produce a
    per-item metric, and inventing a qid would put numbers in a table with no way to
    trace them back.
    """
    report = ScoreReport()
    rows: list[ScoreRow] = []
    wanted = tuple(m for m in metrics if m in ("em", "f1"))
    available = list(wanted) + (["official"] if official is not None else [])

    for record in store.iter_all():
        qid = record.qid
        if not qid:
            report.n_unattributable += 1
            continue
        report.n_attributable += 1
        abstention = _is_abstention(qid)
        if abstention:
            report.n_abstention += 1
        g = gold.get(qid)
        if g is None and not abstention:
            report.n_missing_gold += 1

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
            "replicate": max(record.replicates) if record.replicates else None,
            "cached": len(record.replicates) > 1,
            "is_abstention": abstention,
            "answerable": not abstention,
            "finish_reason": record.finish_reason,
            "latency_ms": record.latency_ms,
            "ttft_ms": record.ttft_ms,
            "queue_wait_ms": record.queue_wait_ms,
        }

        for metric in available:
            value: float | None
            if metric == "official":
                value = None if official is None else official.get(qid)
            elif abstention or g is None:
                # R8.4: abstention items get NULL, never 0.0, and an item with no
                # gold is not a miss.
                value = None
            elif metric == "em":
                value = exact_match(record.text, g, norm)
            else:
                value = token_f1(record.text, g, norm)
            rows.append(ScoreRow(metric=metric, value=value, **common))

    report.n_rows = len(rows)
    report.n_items = report.n_attributable
    report.metrics_written = tuple(available)
    report.means = _means(rows)
    if official is None:
        report.notes.append(
            "The official model-based evaluator was not supplied, so only EM and "
            "token-F1 were written. R8.1 requires all three side by side; a "
            "two-metric table looks identical to a three-metric one unless this is said."
        )
    if report.n_missing_gold:
        report.notes.append(
            f"{report.n_missing_gold} item(s) had no gold answer and scored NULL. They "
            f"are retained as rows so the denominator is visible."
        )
    return rows, report


def _means(rows: Sequence[ScoreRow]) -> dict[str, dict[str, float | None]]:
    """Mean per metric, over ANSWERABLE items only, plus the answerable count.

    The count is part of the summary because a mean without its n is not
    interpretable - 0.42 over 12 items and over 12,000 are different claims.
    """
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


def write_scores(rows: Sequence[ScoreRow], path: Path, *, compression: str = "zstd") -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([r.as_dict() for r in rows], schema=scores_schema())
    pq.write_table(table, path, compression=compression)


def read_scores(path: Path) -> Any:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return pq.read_table(path)


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
    """One row per DISAGREEING item, with its content (R8.3).

    Per item, never an aggregate. A disagreement count says the panel is unstable;
    the rows say on what, and a panel that disagrees only on abstention items is a
    different problem from one that disagrees on scored ones.
    """
    labels = sorted(verdicts)
    rows: list[dict[str, Any]] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            a, b = verdicts[la], verdicts[lb]
            for key in sorted(set(a) & set(b)):
                if a[key] is b[key]:
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
