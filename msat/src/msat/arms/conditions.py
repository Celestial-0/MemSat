"""`results/<run_id>/conditions.parquet` — the R4.2 manifest, on disk.

PRD §R4.2: *"Every condition emits a manifest: ordered memory IDs + ranks per
position variant, plus K, B, B_mode, corpus_fraction. This is the ground truth for
recall@K and gold-rank analysis - computed from the actual pipeline, never a
reimplementation."*

ONE ROW PER (condition, question, position variant)
---------------------------------------------------
Not one row per condition. A condition is rendered once per question, and the
question is what carries the gold, the scale and the answer. Collapsing them makes
`recall@K` impossible to compute without re-running the pipeline, which is the thing
R4.2 exists to prevent.

WHY PARQUET AND NOT JSON
------------------------
The manifest is one row per rendered prompt. For the full grid that is
108 cells x 500 questions x 3 replicates = **162,000 rows**, each carrying an
ordered list of memory ids. As JSON that is a file no analysis tool will open and
no human will read; as parquet it is a column store that `msat analyze` filters by
`(K, B, arm)` without loading the arms it is not asking about.

THE THREE THINGS THIS FILE MUST NEVER LOSE
-------------------------------------------
1. `session_ids` alongside `memory_ids`. Gold is a SESSION; a memory is a chunk of
   one. Drop the session column and recall has to be reconstructed by parsing
   memory ids, which is a guess.
2. `qid`. Without it the row cannot be joined to a gold answer.
3. `evidence_survival`. An arm that discarded context must say so per row, or a
   `full_context_max` score is indistinguishable from a `full_context` one.

NULLS ARE NOT DEFAULTS
----------------------
A missing `evidence_survival` and a survival of 1.0 are different facts. Arrow
distinguishes null from 1.0, and the writer refuses to coerce.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from msat.arms.render import RenderedPrompt

CONDITIONS_SCHEMA_VERSION = 1

CONDITION_COLUMNS: tuple[str, ...] = (
    "arm",
    "K",
    "B",
    "B_mode",
    "scale",
    "reader",
    "corpus_fraction",
)


def arrow_schema() -> Any:
    """The manifest schema, declared explicitly.

    Explicit rather than inferred from the first batch of rows: an inferred schema
    silently takes the type of whatever happened to be in row zero, so a K that is
    always an int in the first chunk and sometimes a float in a later one produces
    two incompatible files with the same name.
    """
    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("arm", pa.string(), nullable=False),
            pa.field("qid", pa.string(), nullable=False),
            pa.field("K", pa.int32(), nullable=False),
            pa.field("B", pa.float64(), nullable=False),
            pa.field("B_mode", pa.string(), nullable=False),
            pa.field("scale", pa.string(), nullable=False),
            pa.field("reader", pa.string(), nullable=False),
            pa.field("position_variant", pa.string(), nullable=False),
            pa.field("corpus_fraction", pa.float64(), nullable=False),
            pa.field("rendered_k", pa.int32(), nullable=False),
            pa.field("evidence_survival", pa.float64(), nullable=True),
            pa.field("memory_ids", pa.list_(pa.string()), nullable=False),
            pa.field("session_ids", pa.list_(pa.string()), nullable=False),
            pa.field("ranks", pa.list_(pa.int32()), nullable=False),
            pa.field("prompt_sha256", pa.string(), nullable=False),
            pa.field("n_prompt_chars", pa.int32(), nullable=False),
        ]
    )


def prompt_digest(prompt: RenderedPrompt) -> str:
    """Stable hash of the rendered prompt TEXT.

    Not the cache key. The cache key covers the condition and the request; this
    covers what the reader will actually see, which is what makes it possible to
    detect two cells that rendered identically - the exact failure that left 78 of
    126 grid cells rendering identical prompts under the old absolute-token budget.
    """
    import hashlib

    body = json.dumps(
        {"system": prompt.system, "user": prompt.user}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def row_for(prompt: RenderedPrompt) -> dict[str, Any]:
    cond = prompt.condition
    return {
        "schema_version": CONDITIONS_SCHEMA_VERSION,
        "arm": prompt.arm,
        "qid": prompt.qid,
        "K": cond.k,
        "B": cond.b,
        "B_mode": cond.b_mode,
        "scale": cond.scale,
        "reader": cond.reader,
        "position_variant": prompt.position_variant,
        "corpus_fraction": prompt.corpus_fraction,
        "rendered_k": prompt.k,
        # None stays None. A missing survival and a survival of 1.0 are different
        # facts, and coercing the first into the second is how an arm that
        # discarded context becomes indistinguishable from one that did not.
        "evidence_survival": prompt.evidence_survival,
        "memory_ids": list(prompt.memory_ids),
        "session_ids": list(prompt.session_ids),
        "ranks": list(prompt.ranks),
        "prompt_sha256": prompt_digest(prompt),
        "n_prompt_chars": len(prompt.system) + len(prompt.user),
    }


def write_conditions(
    prompts: Iterable[RenderedPrompt], path: Path, *, compression: str = "zstd"
) -> dict[str, Any]:
    """Write the manifest. Returns a report, because a manifest's own health matters.

    `identical_prompt_cells` is the headline of the report: cells that rendered the
    same bytes are indistinguishable conditions, and the row count alone would not
    reveal them.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    rows = [row_for(p) for p in prompts]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=arrow_schema())
    pq.write_table(table, path, compression=compression)

    digests: dict[str, set[tuple[Any, ...]]] = {}
    counts: dict[tuple[Any, ...], int] = {}
    for r in rows:
        key = (r["arm"], r["K"], r["B"], r["B_mode"], r["scale"], r["reader"])
        digests.setdefault(r["prompt_sha256"], set()).add(key)
        counts[key] = counts.get(key, 0) + 1

    # THE CHECK IS ACROSS CELLS, NOT WITHIN ONE. Two DIFFERENT conditions that
    # render the same bytes are indistinguishable, and that is the failure the
    # absolute-token budget produced when 78 of 126 cells rendered identical
    # prompts. Within a single cell, prompts are SUPPOSED to vary across questions -
    # counting that as degenerate would flag every healthy manifest.
    collisions = sorted(
        f"{sorted('/'.join(str(x) for x in k) for k in cells)} render identical bytes"
        for cells in digests.values()
        if len(cells) > 1
    )
    return {
        "path": str(path),
        "n_rows": table.num_rows,
        "n_cells": len(counts),
        "schema_version": CONDITIONS_SCHEMA_VERSION,
        "compression": compression,
        "n_bytes": path.stat().st_size,
        "indistinguishable_cells": collisions,
        "n_distinct_prompts": len(digests),
        "note": (
            "Two different conditions rendering identical bytes are "
            "indistinguishable - the same failure the absolute-token budget produced "
            "when 78 of 126 cells rendered identical prompts. Empty list is the "
            "expected state."
        ),
    }


def read_conditions(path: Path) -> Any:
    """Read the manifest back. `msat analyze` reads ONLY this and the store."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return pq.read_table(path)


def cells_from(path: Path) -> list[tuple[Any, ...]]:
    """The distinct conditions in a manifest, for the gate and the report."""
    table = read_conditions(path)
    cols = ["arm", "K", "B", "B_mode", "scale", "reader"]
    seen = {tuple(row) for row in zip(*(table[c].to_pylist() for c in cols), strict=True)}
    return sorted(seen, key=lambda t: tuple(str(x) for x in t))


def verify_manifest(
    path: Path, *, expected_k: Sequence[int], expected_b: Sequence[float]
) -> dict[str, Any]:
    """Does the manifest on disk cover the full crossing, per scale and per reader?

    R4.4 read from the ARTIFACT rather than from the config, so a hand-edited
    manifest cannot pass and a plan that dropped cells is visible here.
    """
    table = read_conditions(path)
    rows = table.to_pylist()
    expected = {(int(k), round(float(b), 6)) for k in expected_k for b in expected_b}
    by_sr: dict[str, set[tuple[int, float]]] = {}
    for r in rows:
        by_sr.setdefault(f"{r['scale']}/{r['reader']}", set()).add(
            (int(r["K"]), round(float(r["B"]), 6))
        )
    problems = [
        f"{sr}: {len(expected - got)} of {len(expected)} (K,B) cells missing"
        for sr, got in by_sr.items()
        if expected - got
    ]
    grids = {
        reader: {(int(r["K"]), round(float(r["B"]), 6)) for r in rows if r["reader"] == reader}
        for reader in {r["reader"] for r in rows}
    }
    return {
        "n_rows": len(rows),
        "crossed": not problems,
        "reader_balanced": len({tuple(sorted(g)) for g in grids.values()}) <= 1,
        "problems": problems,
    }
