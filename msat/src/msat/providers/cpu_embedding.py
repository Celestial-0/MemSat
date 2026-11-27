"""Read-only checkpoint embeddings with bounded CPU lookups."""

from __future__ import annotations

import json
import mmap
import struct
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


class CpuMappedEmbedding(nn.Module):
    """Read selected checkpoint rows on CPU, then transfer only those rows."""

    def __init__(
        self, path: Path, key: str, shape: tuple[int, int], *, scale: float, dtype: torch.dtype
    ) -> None:
        super().__init__()
        with path.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_size))
            entry = header[key]
            if tuple(entry["shape"]) != shape:
                raise ValueError(
                    f"Checkpoint embedding shape differs from configured shape {shape}."
                )
            source_dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}.get(
                entry["dtype"]
            )
            if source_dtype is None:
                raise ValueError(f"Unsupported checkpoint embedding dtype {entry['dtype']}.")
            self._offset = 8 + header_size + entry["data_offsets"][0]
            self._mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        self._source_dtype = source_dtype
        self._source_tensor: Any = None
        self._target_dtype = dtype
        self.num_embeddings, self.embedding_dim = shape
        self.scalar_embed_scale = scale
        self.source_info = {
            "file": str(path),
            "key": key,
            "shape": list(shape),
            "bytes": entry["data_offsets"][1] - entry["data_offsets"][0],
            "placement": "read_only_cpu_lookup",
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._source_tensor is None:
            with warnings.catch_warnings(), torch.device("cpu"):
                # The buffer is deliberately read-only; embedding() never writes weights.
                warnings.filterwarnings(
                    "ignore", message="The given buffer is not writable.*", category=UserWarning
                )
                self._source_tensor = torch.frombuffer(
                    self._mapping,
                    dtype=self._source_dtype,
                    count=self.num_embeddings * self.embedding_dim,
                    offset=self._offset,
                ).reshape(self.num_embeddings, self.embedding_dim)
        rows = functional.embedding(input_ids.to("cpu"), self._source_tensor).to(self._target_dtype)
        rows = rows * rows.new_tensor(self.scalar_embed_scale)
        return rows.to(input_ids.device)

    def close(self) -> None:
        self._source_tensor = None
        self._mapping.close()


def checkpoint_file(model: str, revision: str | None, *, local_files_only: bool = False) -> Path:
    """Resolve the exact existing single-file Gemma checkpoint."""
    if Path(model).is_dir():
        return Path(model) / "model.safetensors"
    from transformers.utils.hub import cached_file

    path = cached_file(
        model, "model.safetensors", revision=revision, local_files_only=local_files_only
    )
    if path is None:
        raise FileNotFoundError(f"No checkpoint weights for {model}@{revision}.")
    return Path(path)


def gemma_cpu_embedding_class(
    path: Path, registry: list[CpuMappedEmbedding], *, dtype: torch.dtype
) -> Any:
    """Keep Gemma's exact language weights while serving PLE lookups on CPU."""
    from transformers import Gemma4ForCausalLM

    class Gemma4CpuEmbeddingForCausalLM(Gemma4ForCausalLM):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            ignored_keys = set(self._keys_to_ignore_on_load_unexpected or ())
            ignored_keys.update(
                {
                    r"model\.audio_tower\..*",
                    r"model\.vision_tower\..*",
                    r"model\.embed_audio\..*",
                    r"model\.embed_vision\..*",
                    r"model\.embed_tokens_per_layer\.weight",
                }
            )
            self._keys_to_ignore_on_load_unexpected = ignored_keys
            if config.hidden_size_per_layer_input:
                embedding = CpuMappedEmbedding(
                    path,
                    "model.language_model.embed_tokens_per_layer.weight",
                    (
                        config.vocab_size_per_layer_input,
                        config.num_hidden_layers * config.hidden_size_per_layer_input,
                    ),
                    scale=config.hidden_size_per_layer_input**0.5,
                    dtype=dtype,
                )
                self.model.embed_tokens_per_layer = embedding
                registry.append(embedding)

    return Gemma4CpuEmbeddingForCausalLM
