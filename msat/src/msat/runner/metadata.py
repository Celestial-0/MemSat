"""Observation metadata persistence sidecar for scoring and analysis."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

METADATA_SCHEMA_VERSION = 1


def write_bytes_artifact(path: Path | str, data: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".artifact-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def write_json_artifact(path: Path | str, payload: Any) -> Path:
    return write_bytes_artifact(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """Metadata recorded per execution unit for downstream analysis."""

    config_hash: str
    condition_key: str
    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str
    qid: str
    memory_ranks: list[int]
    gold_ranks: list[int]
    corpus_fraction: float
    ability: str
    reader_strength: float
    prompt_sha256: str
    cache_key: str
    gold_rank: int | None = None
    question_type: str = ""
    canonical_config_hash: str = ""
    replicate: int = 0
    repetition: int = 0

    def __post_init__(self) -> None:
        if self.repetition == 0 and self.replicate != 0:
            object.__setattr__(self, "repetition", self.replicate)
        elif self.replicate == 0 and self.repetition != 0:
            object.__setattr__(self, "replicate", self.repetition)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = METADATA_SCHEMA_VERSION
        d["K"] = self.k
        d["B"] = self.b
        d["B_mode"] = self.b_mode
        d["replicate"] = self.repetition
        d["repetition"] = self.repetition
        if not d.get("question_type"):
            d["question_type"] = self.ability
        if not d.get("canonical_config_hash"):
            d["canonical_config_hash"] = self.config_hash
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationMetadata:
        k = int(d.get("k", d.get("K", 0)))
        b = float(d.get("b", d.get("B", 0.0)))
        b_mode = str(d.get("b_mode", d.get("B_mode", "")))
        ability = str(d.get("ability", d.get("question_type", "unknown")))
        question_type = str(d.get("question_type", ability))
        rep = int(d.get("repetition", d.get("replicate", 0)))
        return cls(
            config_hash=str(d.get("config_hash", "")),
            condition_key=str(d.get("condition_key", "")),
            arm=str(d.get("arm", "")),
            k=k,
            b=b,
            b_mode=b_mode,
            scale=str(d.get("scale", "")),
            reader=str(d.get("reader", "")),
            qid=str(d.get("qid", "")),
            repetition=rep,
            memory_ranks=[int(x) for x in d.get("memory_ranks", [])],
            gold_ranks=[int(x) for x in d.get("gold_ranks", [])],
            corpus_fraction=float(d.get("corpus_fraction", 0.0)),
            ability=ability,
            reader_strength=float(d.get("reader_strength", 1.0)),
            prompt_sha256=str(d.get("prompt_sha256", "")),
            cache_key=str(d.get("cache_key", "")),
            gold_rank=int(d["gold_rank"]) if d.get("gold_rank") is not None else None,
            question_type=question_type,
            canonical_config_hash=str(d.get("canonical_config_hash", "")),
            replicate=rep,
        )


def read_observation_metadata(path: Path | str) -> list[ObservationMetadata]:
    """Read observation metadata from json or parquet if present."""
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        return [ObservationMetadata.from_dict(row) for row in pq.read_table(p).to_pylist()]
    json_path = p.with_suffix(".json")
    if json_path.is_file():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Observation metadata must be a list of recorded observations.")
        return [ObservationMetadata.from_dict(r) for r in raw]
    return []


def write_observation_metadata(
    observations: Iterable[ObservationMetadata],
    path: Path | str,
) -> Path:
    """Write observations metadata sidecar (both observations and metadata, JSON and Parquet)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [obs.as_dict() for obs in observations]

    # Always write JSON sidecars
    json_text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    json_path = path.with_suffix(".json")
    write_bytes_artifact(json_path, json_text.encode("utf-8"))

    # If writing observations, mirror to metadata sidecar as well
    target_names = [path.name]
    if path.name == "observations.parquet":
        target_names.append("metadata.parquet")
        write_bytes_artifact(path.parent / "metadata.json", json_text.encode("utf-8"))
    elif path.name == "metadata.parquet":
        target_names.append("observations.parquet")
        write_bytes_artifact(path.parent / "observations.json", json_text.encode("utf-8"))

    # If pyarrow is available, write parquet for all targets
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    schema = pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("config_hash", pa.string(), nullable=False),
            pa.field("canonical_config_hash", pa.string(), nullable=False),
            pa.field("condition_key", pa.string(), nullable=False),
            pa.field("arm", pa.string(), nullable=False),
            pa.field("k", pa.int32(), nullable=False),
            pa.field("K", pa.int32(), nullable=False),
            pa.field("b", pa.float64(), nullable=False),
            pa.field("B", pa.float64(), nullable=False),
            pa.field("b_mode", pa.string(), nullable=False),
            pa.field("B_mode", pa.string(), nullable=False),
            pa.field("scale", pa.string(), nullable=False),
            pa.field("reader", pa.string(), nullable=False),
            pa.field("qid", pa.string(), nullable=False),
            pa.field("replicate", pa.int32(), nullable=False),
            pa.field("memory_ranks", pa.list_(pa.int32()), nullable=False),
            pa.field("gold_ranks", pa.list_(pa.int32()), nullable=False),
            pa.field("gold_rank", pa.int32(), nullable=True),
            pa.field("corpus_fraction", pa.float64(), nullable=False),
            pa.field("ability", pa.string(), nullable=False),
            pa.field("question_type", pa.string(), nullable=False),
            pa.field("reader_strength", pa.float64(), nullable=False),
            pa.field("prompt_sha256", pa.string(), nullable=False),
            pa.field("cache_key", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    for name in target_names:
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer, compression="zstd")
        write_bytes_artifact(path.parent / name, buffer.getvalue().to_pybytes())

    return path
