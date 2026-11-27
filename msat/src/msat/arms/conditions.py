"""The per-run condition manifest, on disk."""

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
        "evidence_survival": prompt.evidence_survival,
        "memory_ids": list(prompt.memory_ids),
        "session_ids": list(prompt.session_ids),
        "ranks": list(prompt.ranks),
        "prompt_sha256": prompt_digest(prompt),
        "n_prompt_chars": len(prompt.system) + len(prompt.user),
    }


observation_for = row_for


def write_conditions(
    prompts: Iterable[RenderedPrompt], path: Path, *, compression: str = "zstd"
) -> dict[str, Any]:
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

    collisions = sorted(
        f"{sorted('/'.join(str(x) for x in k) for k in cells)} render identical bytes"
        for cells in digests.values()
        if len(cells) > 1
    )
    return {
        "path": str(path),
        "n_rows": table.num_rows,
        "n_conditions": len(counts),
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
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return pq.read_table(path)


def conditions_from(path: Path) -> list[tuple[Any, ...]]:
    table = read_conditions(path)
    cols = ["arm", "K", "B", "B_mode", "scale", "reader"]
    seen = {tuple(row) for row in zip(*(table[c].to_pylist() for c in cols), strict=True)}
    return sorted(seen, key=lambda t: tuple(str(x) for x in t))


cells_from = conditions_from


def verify_manifest(
    path: Path, *, expected_k: Sequence[int], expected_b: Sequence[float]
) -> dict[str, Any]:
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
