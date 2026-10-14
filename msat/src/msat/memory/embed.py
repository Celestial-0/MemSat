"""Embedding (Architecture §P3.1) — deterministic, seeded, swappable.

The embedder is an INJECTED DEPENDENCY, not an import. Three reasons:

1. The baseline must run in tests, in `make selftest`, and on machines with no GPU.
   A hard `from sentence_transformers import ...` makes the whole write path
   untestable off the execution box, which is where a bug in it would be found
   last and cost most.
2. Swapping the embedder changes recall. If it is baked in, that swap is a code
   change; if it is a constructor argument, it is a recorded config value, and the
   retrieval number stays attributable (Architecture §P3.2).
3. A weak embedder is acceptable. A CHANGING one is not. Everything here is
   deterministic by construction, and `seeded` is reported so a caller can assert
   it rather than trust it.

`HashEmbedder` is the default for tests and for `make selftest`. It is a real
embedding - lexical, deterministic, no model - not a stub that returns zeros. It
retrieves sensibly enough to exercise the pipeline end to end, which a stub would
not.
"""

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
    """What the write path needs from an embedding model."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        """One vector per input, L2-normalised so cosine is a dot product."""
        ...


@dataclass(frozen=True, slots=True)
class HashEmbedder:
    """Deterministic lexical embedder. No model, no network, no GPU.

    Tokens are hashed into `dim` buckets and weighted by term frequency, with a
    second pass over adjacent token pairs so that word ORDER carries some signal.
    L2-normalised, so similarity is a plain dot product in [-1, 1].

    This is deliberately NOT a random projection of zeros. It retrieves real
    lexical overlap, which means a test that asserts "the right session comes back
    first" is testing the retrieval code rather than the luck of the hash.
    """

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
        # Bigrams carry a little word order, which makes the embedder sensitive to
        # phrasing rather than only to vocabulary.
        for a, b in pairwise(words):
            pair = f"{a}_{b}".encode()
            idx = int.from_bytes(hashlib.blake2b(pair, digest_size=4).digest(), "big")
            vec[idx % self.dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """The real thing. Imported lazily so the dependency stays optional.

    A plain class, not a frozen dataclass: it holds a live model handle plus two
    attributes derived from that handle at construction. `uv sync --extra ...`
    provides the model — this is the only class here that needs anything on disk,
    and it records which checkpoint at which revision so the provenance stays
    attributable (Architecture §P3.2).
    """

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
    """Dot product of already-normalised vectors.

    Assumes normalisation rather than recomputing it: at 189,000 generations the
    retriever is called far more often than the embedder, and recomputing two norms
    per comparison is a cost with no benefit when both sides came from the same
    encoder.
    """
    return sum(x * y for x, y in zip(a, b, strict=True))
