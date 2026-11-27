"""Deterministic, seeded, swappable embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class HashEmbedder:
    dim: int = 256
    name: str = "hash-lexical-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _WORD.findall(text.lower())
        if not words:
            return vec
        counts = Counter(words)
        for word, n in counts.items():
            idx = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big")
            vec[idx % self.dim] += 1.0 + math.log(n)
        for a, b in pairwise(words):
            pair = f"{a}_{b}".encode()
            idx = int.from_bytes(hashlib.blake2b(pair, digest_size=4).digest(), "big")
            vec[idx % self.dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": device}
        if revision:
            kwargs["revision"] = revision
        self._model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"{model_name}@{(revision or 'main')} reported no embedding dimension. "
                "That usually means the checkpoint is not a sentence-embedding "
                "model - check the repo before assuming a dim."
            )
        self.dim = int(dim)
        self.name = f"sentence-transformers:{model_name}@{revision or 'main'}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
