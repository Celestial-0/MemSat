"""A reader family bound to a local backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from msat.hardware.vram import DEFAULT_HEADROOM_GB, ModelShape, kv_reader_capacity_tokens
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    LocalBackend,
    select_backend,
)


class ContextOverflowError(RuntimeError):
    """A prompt exceeds the reader's servable ceiling on this card."""


class UnknownReaderFamilyError(KeyError):
    """A reader family was requested that `configs/readers.yaml` does not declare."""


@dataclass(frozen=True, slots=True)
class ReaderVramSpec:
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_param: float = 0.55

    def to_shape(self, *, name: str, dtype: str = "bfloat16") -> ModelShape:
        n_params = int(self.params_b * 1_000_000_000)
        return ModelShape(
            name=name,
            n_params=n_params,
            n_layers=self.layers,
            n_attention_heads=self.kv_heads,
            n_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            n_params_is_exact=False,
        )


@dataclass(frozen=True, slots=True)
class LocalReader:
    id: str
    model: str
    version: str
    released: str
    role: str
    reader_strength: float
    declared_context_window: int
    vram: ReaderVramSpec
    backend_name: str = "transformers"
    quant_bits: float | None = 4.0
    card_gb: float = 8.0
    headroom_gb: float = DEFAULT_HEADROOM_GB
    exact_checkpoint_confirmed: bool = False
    rationale: str = ""
    _backend: LocalBackend | None = field(default=None, repr=False, compare=False)

    @property
    def shape(self) -> ModelShape:
        return self.vram.to_shape(name=self.model)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.shape.kv_bytes_per_token()

    @property
    def effective_quant_bits(self) -> float:
        return self.vram.bytes_per_param * 8.0

    @property
    def weights_gb(self) -> float:
        return self.shape.weights_gb(quant_bits=self.effective_quant_bits)

    def vram_capacity_tokens(self, *, card_gb: float | None = None) -> int:
        card = self.card_gb if card_gb is None else card_gb
        return kv_reader_capacity_tokens(
            self.shape,
            card,
            quant_bits=self.effective_quant_bits,
            headroom_gb=self.headroom_gb,
        )

    @property
    def reader_capacity_tokens(self) -> int:
        return min(self.declared_context_window, self.vram_capacity_tokens())

    @property
    def binding_constraint(self) -> str:
        v = self.vram_capacity_tokens()
        if v < self.declared_context_window:
            return "vram"
        return "declared_window"

    def vram_capacity_at_headroom(self, headroom_gb: float) -> int:
        return kv_reader_capacity_tokens(
            self.shape,
            self.card_gb,
            quant_bits=self.effective_quant_bits,
            headroom_gb=headroom_gb,
        )

    def capacity_sensitivity(self) -> dict[str, int]:
        return {
            "tight_0.8gb": self.vram_capacity_at_headroom(0.8),
            "default": self.vram_capacity_tokens(),
            "loose_5.0gb": self.vram_capacity_at_headroom(5.0),
        }

    def check_capacity(self, n_tokens: int, *, card_gb: float | None = None) -> int:
        effective_card = self.card_gb if card_gb is None else card_gb
        vram_capacity = self.vram_capacity_tokens(card_gb=effective_card)
        limit = min(self.declared_context_window, vram_capacity)
        if n_tokens > limit:
            binding = "vram" if vram_capacity < self.declared_context_window else "declared_window"
            raise ContextOverflowError(
                f"reader {self.id!r} cannot serve {n_tokens:,} tokens: its capacity on "
                f"{effective_card} GB is {limit:,} "
                f"(VRAM {vram_capacity:,}, declared window {self.declared_context_window:,}, "
                f"bound by {binding}). Truncating would leave the "
                f"prompt able to contain its own answer while appearing not to, and "
                f"the resulting curve would measure nothing (R3.2, "
                f"never_truncate_silently)."
            )
        return limit

    def supports(self, n_tokens: int) -> bool:
        return n_tokens <= self.reader_capacity_tokens

    def coverage_over(
        self,
        cells: Sequence[tuple[int, float, str]],
        *,
        card_gb: float | None = None,
        token_basis: Mapping[tuple[int, float, str], int]
        | Callable[[int, float, str], int]
        | None = None,
    ) -> dict[str, Any]:
        from msat.hardware.capacity import MissingAnchorError

        if cells and token_basis is None:
            raise MissingAnchorError(
                "Coverage requires modeled or measured prompt tokens; K is a memory count."
            )
        effective_card = self.card_gb if card_gb is None else card_gb
        vram_capacity = self.vram_capacity_tokens(card_gb=effective_card)
        limit = min(self.declared_context_window, vram_capacity)
        supported: list[tuple[int, float, str]] = []
        unsupported: list[dict[str, Any]] = []
        for k, b, scale in cells:
            entry = {"k": k, "b": b, "scale": scale}
            if token_basis is not None:
                if callable(token_basis):
                    tok = int(token_basis(k, b, scale))
                elif isinstance(token_basis, Mapping):
                    if (k, b, scale) not in token_basis:
                        raise MissingAnchorError(f"No prompt-token basis for {(k, b, scale)!r}.")
                    tok = int(token_basis[(k, b, scale)])
                else:
                    tok = int(k)
            else:
                raise MissingAnchorError("Coverage requires an explicit prompt-token basis.")
            if tok <= limit:
                supported.append((k, b, scale))
            else:
                reason_tok = f"prompt ({tok:,} tokens)" if token_basis is not None else f"K={k}"
                unsupported.append(
                    {**entry, "reason": f"{reason_tok} exceeds the {limit:,}-token reader capacity"}
                )
        total = len(supported) + len(unsupported)
        binding = "vram" if vram_capacity < self.declared_context_window else "declared_window"
        return {
            "reader": self.id,
            "reader_capacity_tokens": limit,
            "declared_context_window": self.declared_context_window,
            "vram_capacity_tokens": vram_capacity,
            "binding_constraint": binding,
            "n_cells": total,
            "n_supported": len(supported),
            "n_unsupported": len(unsupported),
            "coverage": len(supported) / total if total else 0.0,
            "unsupported": unsupported,
            "note": (
                "Coverage compares the explicit modeled or measured prompt-token basis against the reader capacity."
            ),
        }

    def reader_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "version": self.version,
            "released": self.released,
            "role": self.role,
            "reader_strength": self.reader_strength,
            "backend": self.backend_name,
            "quant_bits": self.quant_bits,
            "bytes_per_param": self.vram.bytes_per_param,
            "declared_context_window": self.declared_context_window,
            "vram_capacity_tokens": self.vram_capacity_tokens(),
            "reader_capacity_tokens": self.reader_capacity_tokens,
            "binding_constraint": self.binding_constraint,
            "exact_checkpoint_confirmed": self.exact_checkpoint_confirmed,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.reader_metadata(),
            "vram": {
                "params_b": self.vram.params_b,
                "layers": self.vram.layers,
                "kv_heads": self.vram.kv_heads,
                "head_dim": self.vram.head_dim,
                "bytes_per_param": self.vram.bytes_per_param,
            },
            "rationale": self.rationale,
        }

    def backend(self) -> LocalBackend:
        if self._backend is None:
            adapter = select_backend(self.backend_name)
            if self.backend_name == "transformers":
                from msat.providers.transformers_backend import TransformersBackend

                if isinstance(adapter, TransformersBackend):
                    adapter.bind_revision(self.model, self.version)
            object.__setattr__(self, "_backend", adapter)
        return self._backend  # type: ignore[return-value]

    def health(self) -> dict[str, Any]:
        try:
            engine = self.backend().health()
        except BackendUnavailableError as exc:
            engine = {"ready": False, "reason": str(exc)}
        except ValueError as exc:
            engine = {"ready": False, "reason": str(exc)}
        return {
            **self.reader_metadata(),
            "engine": engine,
            "ready": bool(engine.get("ready")),
            "can_serve_deepest": self.supports(self.declared_context_window),
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = self.backend().generate(request)
        if not result.model_version:
            object.__setattr__(result, "model_version", self.version)
        return result

    def context_window(self, model: str | None = None) -> int:
        return self.reader_capacity_tokens


def reader_from_family(
    family: Mapping[str, Any],
    *,
    card_gb: float = 8.0,
    backend: str = "transformers",
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> LocalReader:
    missing = [
        k for k in ("id", "model", "version", "released", "reader_strength") if k not in family
    ]
    if missing:
        raise ValueError(
            f"reader family is missing {missing}; R7.6 requires identity, version "
            f"and date to be recorded, and a default would be a guess"
        )
    raw_vram = family.get("vram") or {}
    return LocalReader(
        id=str(family["id"]),
        model=str(family["model"]),
        version=str(family["version"]),
        released=str(family["released"]),
        role=str(family.get("role", "secondary")),
        reader_strength=float(family["reader_strength"]),
        declared_context_window=int(family.get("context_window", 0)),
        vram=ReaderVramSpec(
            params_b=float(raw_vram.get("params_b", 0.0)),
            layers=int(raw_vram.get("layers", 0)),
            kv_heads=int(raw_vram.get("kv_heads", 0)),
            head_dim=int(raw_vram.get("head_dim", 0)),
            bytes_per_param=float(raw_vram.get("bytes_per_param", 0.55)),
        ),
        backend_name=backend,
        card_gb=card_gb,
        headroom_gb=headroom_gb,
        exact_checkpoint_confirmed=bool(family.get("exact_checkpoint_confirmed", False)),
        rationale=str(family.get("rationale", "")).strip(),
    )


def readers_from_config(
    config: Mapping[str, Any],
    *,
    card_gb: float = 8.0,
    backend: str = "transformers",
    allow_single: bool = False,
) -> list[LocalReader]:
    families = config.get("families") or []
    policy = config.get("context_policy") or {}
    headroom = float(policy.get("headroom_gb", DEFAULT_HEADROOM_GB))
    out = [
        reader_from_family(f, card_gb=card_gb, backend=backend, headroom_gb=headroom)
        for f in families
    ]
    if not allow_single and len(out) < 2:
        raise ValueError(
            f"readers.yaml declares {len(out)} family/families; R7.5 requires at "
            f"least 2, and R9.12 needs a pair to compare"
        )
    return out


def panel_coverage(
    readers: Sequence[LocalReader],
    cells: Sequence[tuple[int, float, str]],
    *,
    card_gb: float | None = None,
    token_basis: Mapping[tuple[int, float, str], int]
    | Callable[[int, float, str], int]
    | None = None,
) -> dict[str, Any]:
    per_reader = [r.coverage_over(cells, card_gb=card_gb, token_basis=token_basis) for r in readers]
    supports = {r["reader"]: r["n_supported"] for r in per_reader}
    supported_counts = set(supports.values())
    return {
        "readers": per_reader,
        "n_cells": len(cells),
        "supported_per_reader": supports,
        "uniform_coverage": len(supported_counts) == 1,
        "common_support": min(supported_counts) if supported_counts else 0,
        "note": (
            "Non-uniform coverage is a hardware fact, not a defect in the panel "
            "(readers.yaml: gemma ~66k, phi ~44k, qwen ~21k tokens on 8 GB). R9.12 "
            "refuses to compare readers over unequal K support for this reason."
        ),
    }


def load_readers(
    path: Path, *, card_gb: float = 8.0, backend: str = "transformers"
) -> list[LocalReader]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return readers_from_config(raw, card_gb=card_gb, backend=backend)


class _FamilySource(Protocol):
    families: Sequence[Mapping[str, Any]]
