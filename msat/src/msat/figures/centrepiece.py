"""The centrepiece figure: measured K-curves against the analytical ceiling.

PRD §R9.6, §G9, §G12. Two design rules, both asserted in `tests/test_centrepiece.py`
and both stated here because they are the whole point of the figure:

**RULE 1 - the ceiling is a bound LINE, never a shaded region.**
Shading the area under the measured curve up to the ceiling reads as "the area
the method achieved", which recasts the result as a fraction of a bound and
makes the crossover - the one thing this figure exists to show - unlocatable.
So the ceiling is drawn dashed and never filled. Dashed for a second reason:
a solid line invites the reader to compare its slope with the measured curves'
slopes, and the ceiling is not a measurement.

**RULE 2 - one line per B, never one line per scale.**
Averaging over B collapses the K x B plane and produces a figure that looks like
it is showing an interaction and is not. The interaction (R9.1) is the headline
result, so the figure must not average away the thing it is about.

The x axis is log2(K), because the primary model is fitted on `log2(K)*log2(B)`.
Drawing in raw K moves the crossing point - the same category of error as fitting
the interaction on raw K.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .svg import PALETTE, Panel, Series, Sheet, fmt, iter_texts

#: Dash pattern for the bound. A named constant because a test asserts the
#: literal string appears in the output, and because "is this line the ceiling?"
#: should be answerable from the data model rather than by matching a label.
CEILING_DASH = "7,5"

CEILING_LABEL = "ceiling(K)"

PREVIEW_WARNING = "PREVIEW - SYNTHETIC VALUES, NOT RESULTS"

SCALES = ("S", "M")

#: Mean sessions per question, MEASURED in P1 (`results/dataset_report.json`,
#: carried into `configs/pricing/latest.yaml`). Used only to state the
#: corpus_fraction RANGE on the figure, which R9.8 requires to be annotated
#: rather than axised. Not a substitute for `corpus_fraction.json` - that
#: artifact is the measured per-question distribution and is what analysis reads.
_SESSIONS_MEAN = {"S": 47.73, "M": 475.31}

#: Budget grid used by the synthetic fixture. Three levels, which is what makes
#: `test_element_census_reports_what_was_drawn` assert 4 polylines per panel
#: (3 budgets + 1 ceiling) and therefore catch a regression back to averaging.
_SYNTHETIC_B = (128.0, 512.0, 2048.0)


class FigureInputMissing(Exception):  # noqa: N818 - public name kept for compatibility
    """A required artifact is absent, unparseable, or out of range.

    Named rather than returning a partial figure: the figure is the artefact a
    reader opens, and a figure with a quietly missing bound looks like a finding.
    """


@dataclass(frozen=True)
class Cell:
    """One (K, B, scale) measurement. `accuracy` and `recall` are Optional
    independently - a cell may carry only one metric, and dropping it would turn
    a gap into a measurement of zero."""

    k: int
    b: float
    accuracy: float | None = None
    recall: float | None = None
    scale: str = "S"
    n_questions: int = 0

    @property
    def key(self) -> tuple[str, int, float]:
        return (self.scale, self.k, self.b)


# ── the x axis: log2(K), exactly ───────────────────────────────────────────────


def x_of(k: float) -> float:
    """log2(k), exact on the power-of-two grid.

    `math.log2` returns exactly the right value for powers of two, but float
    drift on any other k would put a label a fraction of a pixel from where the
    model put it. `bit_length` is exact for integers, which is what K is.
    """
    ik = int(k)
    if ik == k and ik > 0 and (ik & (ik - 1)) == 0:
        return float(ik.bit_length() - 1)
    return math.log2(k)


_x_of = x_of


# ── ceiling: an INPUT, never computed here ─────────────────────────────────────


def ceiling_from_artifact(doc: dict[str, Any], scale: str) -> dict[int, float]:
    """Read `{K: ceiling}` for one scale out of `results/ceiling.json`.

    Three refusal modes, each because the alternative renders as a finding:
    a value outside [0,1] draws a line above the axis, which reads as "accuracy
    exceeded the ceiling"; a NaN parses cleanly and plots as a line that silently
    vanishes, leaving a legend entry and no bound; and a missing scale must name
    itself rather than produce an empty bound.
    """
    try:
        raw = doc["scales"][scale]["ceiling"]
    except (KeyError, TypeError) as exc:
        raise FigureInputMissing(
            f"no scale {scale!r} with a 'ceiling' block in the ceiling artifact "
            f"(scales present: {sorted(doc.get('scales') or {})})"
        ) from exc

    out: dict[int, float] = {}
    for k_key, v in raw.items():
        try:
            k = int(k_key)
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise FigureInputMissing(
                f"ceiling[{scale}][{k_key!r}] = {v!r} is not numeric; refused rather than coerced"
            ) from exc
        if math.isnan(value):
            raise FigureInputMissing(
                f"ceiling[{scale}][{k}] is NaN; it would plot as a line that silently vanishes"
            )
        if not 0.0 <= value <= 1.0:
            raise FigureInputMissing(
                f"ceiling[{scale}][{k}] = {value} is outside [0, 1]; plotted above 1 "
                f"it would read as accuracy exceeding the ceiling"
            )
        out[k] = value
    return out


def ceiling_curve(n_gold: Sequence[int], ceiling: dict[int, float]) -> list[tuple[float, float]]:
    """Project a ceiling lookup into (log2(K), value) points.

    Interpolates on the n_gold grid so the drawn bound is defined between the K
    values the artifact sampled, and returns ONLY the K values it was given - a
    figure that invented bound points would be comparing the measured curve
    against an artifact of itself.
    """
    ks = sorted(n_gold)
    points: list[tuple[float, float]] = []
    for k in ks:
        exact = ceiling.get(k)
        if exact is not None:
            points.append((_x_of(k), exact))
            continue
        lo = next((j for j in ks if j < k), None)
        hi = next((j for j in ks if j > k), None)
        if lo is None or hi is None or lo not in ceiling or hi not in ceiling:
            continue
        w = (k - lo) / (hi - lo)
        points.append((_x_of(k), ceiling[lo] + w * (ceiling[hi] - ceiling[lo])))
    return points


# ── loaders: refuse rather than invent ─────────────────────────────────────────


def cells_from_scores(rows: Iterable[Any], *, metric: str) -> list[Cell]:
    """Aggregate per-item scores to one cell per (K, B, scale).

    `metric` is a parameter because `em` and `official` are different numbers
    (R8.1) and pooling them produces a number that is neither.
    """
    buckets: dict[tuple[str, int, float], list[float]] = {}
    seen: dict[tuple[str, int, float], set[str]] = {}
    for row in rows:
        if getattr(row, "metric", None) != metric:
            continue
        value = getattr(row, "value", None)
        if value is None:
            # Missing data is not zero. A cell nobody scored must not put a point
            # on the floor of the figure.
            continue
        if not getattr(row, "answerable", True):
            # An abstention scores 0 by the scoring rule but is not a wrong
            # answer; pooling it in makes the curve fall for reasons unrelated
            # to retrieval (R8.4).
            continue
        k, b = int(row.k), float(row.b)
        key = (str(row.scale), k, b)
        qid = str(row.qid)
        if qid in seen.setdefault(key, set()):
            raise ValueError(
                f"duplicate qid {qid!r} in cell {key}; a silent mean would weight a "
                f"cached cell extra, and a cached cell is exactly the reused one"
            )
        seen[key].add(qid)
        buckets.setdefault(key, []).append(float(value))

    out = [
        Cell(k=k, b=b, accuracy=sum(v) / len(v), scale=scale, n_questions=len(v))
        for (scale, k, b), v in buckets.items()
    ]
    return sorted(out, key=lambda c: (c.scale, c.k, c.b))


def cells_from_recall(summaries: Iterable[Any]) -> list[Cell]:
    """Aggregate `RecallSummary` buckets.

    Uses each bucket's own `mean_recall`. Averaging means-of-means would weight a
    12-item bucket like a 500-item one, because the buckets do not have equal
    item counts.
    """
    out = [
        Cell(
            k=int(s.k),
            b=float(s.b),
            recall=float(s.mean_recall),
            scale=str(s.scale),
            n_questions=len(s.answerable),
        )
        for s in summaries
        if len(s.answerable) > 0
    ]
    return sorted(out, key=lambda c: (c.scale, c.k, c.b))


def merge_cells(*groups: Sequence[Cell]) -> list[Cell]:
    """Union cells, first non-None wins per field.

    Union, not intersect: a cell present in only one metric must stay in the
    figure, because the reader cannot distinguish a gap from a measured zero.
    """
    merged: dict[tuple[str, int, float], Cell] = {}
    for group in groups:
        for cell in group:
            existing = merged.get(cell.key)
            if existing is None:
                merged[cell.key] = cell
            else:
                merged[cell.key] = Cell(
                    k=cell.k,
                    b=cell.b,
                    scale=cell.scale,
                    accuracy=existing.accuracy if existing.accuracy is not None else cell.accuracy,
                    recall=existing.recall if existing.recall is not None else cell.recall,
                )
    return sorted(merged.values(), key=lambda c: (c.scale, c.k, c.b))


# ── the figure ────────────────────────────────────────────────────────────────


def _corpus_fraction_range(cells: Sequence[Cell], scale: str) -> tuple[float, float] | None:
    """The range of `K / |corpus|` over the K values this panel actually shows.

    A RANGE, not a point: corpus_fraction moves with K within a scale, so one
    number would be a value that has drifted from what it describes. S and M sit
    at ~10x different fractions, which is exactly why R9.8 annotates rather than
    axes - a third axis compresses the K x B plane into a corner.
    """
    mean = _SESSIONS_MEAN.get(scale)
    if mean is None or not cells:
        return None
    fracs = sorted({min(c.k, mean) / mean for c in cells})
    return (fracs[0], fracs[-1]) if fracs else None


def _annotations(cells: Sequence[Cell], has_ceiling: bool, scale: str) -> list[str]:
    notes: list[str] = []
    span = _corpus_fraction_range(cells, scale)
    if span is not None:
        lo, hi = span
        notes.append(
            f"corpus_fraction {fmt(lo)}-{fmt(hi)} across the K values shown at scale "
            f"{scale} - annotated, not a third axis: S and M sit ~10x apart and a "
            f"third axis would compress the K x B plane into a corner"
        )
    if has_ceiling:
        notes.append(
            f"{CEILING_LABEL} is a BOUND on the analytic ceiling, not a measurement - "
            f"drawn dashed and never shaded"
        )
    notes.append(
        "read for the crossing: where a B line passes the ceiling, more budget at "
        "that K cannot help (G9 interaction)"
    )
    return notes


def centrepiece(
    cells: Sequence[Cell],
    ceiling: dict[int, float],
    *,
    scale: str | None = None,
    metric_name: str | None = None,
) -> Sheet:
    """Build the sheet: an accuracy panel and a recall@K panel."""
    if not cells:
        raise ValueError("no cells to plot")

    picked = [c for c in cells if scale is None or c.scale == scale]
    used_scale = scale or picked[0].scale
    picked = [c for c in picked if c.scale == used_scale]
    budgets = sorted({c.b for c in picked})

    scale_ceiling = {k: v for k, v in ceiling.items() if k in {c.k for c in picked}}
    ys = [v for c in picked for v in (c.accuracy, c.recall) if v is not None] + list(
        scale_ceiling.values()
    )
    top = max(ys) if ys else 1.0
    y_range = (0.0, min(1.0, max(1.02, top * 1.05)))
    # Both panels share one y range: if each auto-scaled, a 5pp gap between the
    # metrics would render as a full-height gap in one panel and a sliver in the
    # other, and the reader would be comparing shapes instead of levels.

    def panel(title: str, attr: str, subtitle: str) -> Panel:
        series: list[Series] = []
        for i, b in enumerate(budgets):
            points = [
                (_x_of(c.k), getattr(c, attr))
                for c in sorted(picked, key=lambda c: c.k)
                if c.b == b and getattr(c, attr) is not None
            ]
            if len(points) < 2:
                continue
            series.append(
                Series(label=f"B={fmt(b)}", points=points, colour=PALETTE[i % len(PALETTE)])
            )
        if scale_ceiling:
            points = ceiling_curve([c.k for c in picked], scale_ceiling)
            if len(points) >= 2:
                series.append(
                    Series(
                        label=CEILING_LABEL,
                        points=points,
                        colour="#333333",
                        dash=CEILING_DASH,
                        marker=False,
                    )
                )
        return Panel(
            panel_title=title,
            series=series,
            subtitle=subtitle,
            xlabel="memories retrieved K (log2 scale)",
            ylabel="proportion",
            annotations=_annotations(picked, bool(scale_ceiling), used_scale),
            y_range=y_range,
        )

    sub_acc = f"accuracy = {metric_name}" if metric_name else ""
    sub_rec = "recall = gold sessions among the K rendered"
    return Sheet(
        title=f"K x B against the analytical ceiling - scale {used_scale}",
        subtitle="one line per budget; the K x B plane is never averaged away",
        panels=[panel("accuracy", "accuracy", sub_acc), panel("recall@K", "recall", sub_rec)],
    )


# ── synthetic fixture, for `--preview` only ───────────────────────────────────


def synthetic_cells() -> list[Cell]:
    """Fixture values. ONLY reachable through `preview_report`, and every file it
    writes carries `PREVIEW_WARNING` in its own bytes - not only in the console
    report, because whoever opens the .svg months later is not the console."""
    cells: list[Cell] = []
    for scale in SCALES:
        ks = (1, 2, 4, 8, 16, 32, 64)
        for bi, b in enumerate(_SYNTHETIC_B):
            for ki, k in enumerate(ks):
                base = 0.30 + 0.05 * bi + 0.04 * ki
                acc = min(0.97, base + (0.04 if scale == "S" else -0.03) * ki / 6)
                rec = min(0.99, 0.25 + 0.09 * ki - 0.03 * bi)
                cells.append(
                    Cell(k=k, b=b, accuracy=round(acc, 4), recall=round(rec, 4), scale=scale)
                )
    return cells


def synthetic_ceiling() -> dict[int, float]:
    return {1: 0.32, 2: 0.55, 4: 0.74, 8: 0.86, 16: 0.94, 32: 0.98, 64: 1.0}


# ── disk ──────────────────────────────────────────────────────────────────────


def _write_one(
    cells: Sequence[Cell],
    ceiling: dict[int, float],
    dest: Path,
    *,
    metric_name: str | None,
) -> dict[str, Any]:
    """Write one file per scale, with the scale IN THE NAME.

    A single figure carrying both scales would need a third encoding on top of
    colour-for-B, and the two sit at ~10x different corpus_fraction so they are
    not comparable on one axis regardless.
    """
    written: dict[str, Any] = {}
    for scale in SCALES:
        subset = [c for c in cells if c.scale == scale]
        if not subset:
            continue
        path = dest.with_name(f"{dest.stem}_{scale}{dest.suffix or '.svg'}")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = centrepiece(subset, ceiling, scale=scale, metric_name=metric_name).render()
        path.write_text(text, encoding="utf-8", newline="\n")
        written[scale] = {
            "path": str(path),
            "n_bytes": path.stat().st_size,
            "n_cells": len(subset),
            "annotations": list(_annotations(subset, bool(ceiling), scale)),
            "visible_text_head": " ".join(iter_texts(text))[:200],
        }
    return written


def write_centrepiece(
    cells: Sequence[Cell],
    ceiling: dict[int, float],
    dest: Path,
    *,
    metric_name: str | None = None,
) -> dict[str, Any]:
    if not cells:
        raise ValueError("no cells to plot")
    dest.parent.mkdir(parents=True, exist_ok=True)
    scale_reports = _write_one(cells, ceiling, dest, metric_name=metric_name)
    return {
        "is_preview": False,
        "source": "scores.parquet + ceiling.json",
        "metric": metric_name,
        "scale_reports": scale_reports,
    }


def preview_report(out_dir: Path) -> dict[str, Any]:
    """Render the fixture, stamped in the FILE.

    The stamp goes into the SVG bytes, not only into the returned report. An
    earlier version patched the report after writing, which satisfied a test
    while leaving the artifact clean - the report and the file disagreeing, which
    is the drift this project keeps having to catch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_cells = "\n".join(
        f'<text x="14" y="{18 + i * 14}" font-size="11" fill="#b00020">{PREVIEW_WARNING}</text>'
        for i in range(3)
    )
    seed = centrepiece(synthetic_cells(), synthetic_ceiling()).render()
    staged = seed.replace("</svg>", text_cells + "\n</svg>")

    reports = _write_one(
        synthetic_cells(),
        synthetic_ceiling(),
        out_dir / "fig_centrepiece.svg",
        metric_name=None,
    )
    for entry in reports.values():
        p = Path(entry["path"])
        p.write_text(
            staged.replace('viewBox="0 0 ', 'viewBox="0 0 ', 1), encoding="utf-8", newline="\n"
        )
        entry["n_bytes"] = p.stat().st_size
    return {
        "is_preview": True,
        "source": "synthetic",
        "metric": None,
        "warning": PREVIEW_WARNING,
        "scale_reports": reports,
    }


def build_from_disk(
    *,
    scores_path: Path,
    ceiling_path: Path,
    metric: str,
    out_dir: Path,
    scales: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load both artifacts and render. Refuses by NAME on anything absent."""
    scores_path, ceiling_path = Path(scores_path), Path(ceiling_path)
    missing = []
    if not scores_path.exists():
        missing.append(f"scores: {scores_path} (run `msat score <run_id>`)")
    if not ceiling_path.exists():
        missing.append(f"ceiling: {ceiling_path} (run `msat ceiling`)")
    if missing:
        # Name every missing input at once: a cold checkout has neither, and
        # failing on the first turns a two-second fix into rounds of one error.
        raise FigureInputMissing(
            "cannot build the figure - missing "
            + "; ".join(missing)
            + ". Use `--preview` to check the layout against synthetic values."
        )

    doc = json.loads(ceiling_path.read_text(encoding="utf-8"))
    cells = _cells_from_parquet(scores_path, metric=metric)
    if scales:
        cells = [c for c in cells if c.scale in set(scales)]

    return write_centrepiece(
        cells,
        ceiling_from_artifact(doc, cells[0].scale),
        out_dir / "fig_centrepiece.svg",
        metric_name=metric,
    )


def _cells_from_parquet(path: Path, *, metric: str) -> list[Cell]:
    """Read scores through DuckDB. Never `pandas.read_parquet` into a frame of
    500k rows here - the aggregate is pushed down and only cells come back."""
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT k, b, scale, metric, value, qid, answerable
            FROM read_parquet(?)
            WHERE metric = ? AND value IS NOT NULL AND answerable
            """,
            [str(path), metric],
        ).fetchall()
    finally:
        con.close()

    class _Row:
        __slots__ = ("answerable", "b", "k", "metric", "qid", "scale", "value")

        def __init__(self, r: tuple[Any, ...]) -> None:
            (self.k, self.b, self.scale, self.metric, self.value, self.qid, self.answerable) = r

    return cells_from_scores((_Row(r) for r in rows), metric=metric)
