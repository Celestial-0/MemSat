"""Compute the VRAM a checkpoint actually needs, from its own config."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GB = 1024**3

DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
    "fp8": 1.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
    "nf4": 0.5,
}

DEFAULT_HEADROOM_GB = 3.0


class ConfigMissingError(ValueError):
    """The checkpoint config lacks fields the VRAM arithmetic needs."""


@dataclass(frozen=True, slots=True)
class ModelShape:
    name: str
    n_params: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_attention_heads: int
    dtype: str
    is_moe: bool = False
    n_experts: int | None = None
    n_active_experts: int | None = None
    n_params_is_exact: bool = False

    @property
    def dtype_bytes(self) -> float:
        return DTYPE_BYTES.get(self.dtype.lower(), 2.0)

    @property
    def gqa_factor(self) -> int:
        return max(1, self.n_attention_heads // max(1, self.n_kv_heads))

    def kv_bytes_per_token(self) -> int:
        return int(2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes)

    def weights_gb(self, *, quant_bits: float | None = None) -> float:
        bits = quant_bits if quant_bits is not None else self.dtype_bytes * 8
        return self.n_params * bits / 8 / GB

    def kv_gb(self, n_tokens: int) -> float:
        return self.kv_bytes_per_token() * n_tokens / GB


def _first(config: Mapping[str, Any], *names: str) -> Any:
    for n in names:
        if n in config and config[n] is not None:
            return config[n]
    return None


def shape_from_config(config: Mapping[str, Any], *, name: str = "model") -> ModelShape:
    text_cfg = config.get("text_config") or config
    num_layers = _first(text_cfg, "num_hidden_layers", "n_layer", "n_layers")
    if num_layers is None:
        raise ConfigMissingError(f"{name}: no num_hidden_layers in config.json")

    n_heads = _first(text_cfg, "num_attention_heads", "n_head", "n_heads")
    kv_heads = _first(text_cfg, "num_key_value_heads", "num_kv_heads", "n_head_kv")
    hidden = _first(text_cfg, "hidden_size", "n_embd", "d_model")
    head_dim = _first(text_cfg, "head_dim")

    if kv_heads is None:
        raise ConfigMissingError(
            f"{name}: config.json declares no num_key_value_heads. Assuming "
            f"num_attention_heads would model a model WITHOUT grouped-query attention "
            f"and overstate the KV cache by the GQA factor - commonly 8x - which "
            f"would demand a card this checkpoint does not need. Read the config "
            f"rather than guessing it."
        )
    if head_dim is None:
        if hidden is None or n_heads is None:
            raise ConfigMissingError(
                f"{name}: no head_dim, and not enough to derive it (need hidden_size "
                f"and num_attention_heads)"
            )
        head_dim = hidden // n_heads

    declared_params = _first(config, "num_parameters", "n_params")
    exact = declared_params is not None
    n_params = declared_params or _estimate_params(
        num_layers, hidden or 0, kv_heads, head_dim, n_heads, config
    )

    return ModelShape(
        name=name,
        n_params=int(n_params),
        n_layers=int(num_layers),
        n_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        n_attention_heads=int(n_heads or kv_heads),
        dtype=str(_first(text_cfg, "torch_dtype", "dtype") or "bfloat16"),
        is_moe=bool(_first(config, "num_local_experts", "num_experts", "n_routed_experts")),
        n_experts=_first(config, "num_local_experts", "num_experts", "n_routed_experts"),
        n_active_experts=_first(config, "num_experts_per_tok", "n_experts_per_token"),
        n_params_is_exact=exact,
    )


def _estimate_params(
    layers: int,
    hidden: int,
    kv_heads: int,
    head_dim: int,
    n_heads: Any,
    config: Mapping[str, Any],
) -> int:
    vocab = int(_first(config, "vocab_size") or 32000)
    dense_inter = int(_first(config, "intermediate_size") or 4 * (hidden or 0))
    moe_inter = _first(config, "moe_intermediate_size", "expert_intermediate_size")
    n_experts = _first(config, "num_local_experts", "num_experts", "n_routed_experts")
    expert_inter = int(moe_inter) if moe_inter else dense_inter
    h = hidden or 0
    nh = int(n_heads or kv_heads)
    attn = int(layers * h * (nh + 2 * kv_heads) * head_dim)
    dense_mlp = int(layers * 3 * h * dense_inter)
    expert_mlp = int(layers * 3 * h * expert_inter * int(n_experts)) if n_experts else 0
    return int(vocab * h + attn + dense_mlp + expert_mlp)


def load_config(model_path: Path) -> Mapping[str, Any]:
    p = Path(model_path)
    f = p if p.is_file() else p / "config.json"
    if not f.exists():
        raise ConfigMissingError(
            f"{f} not found. Pass a local checkpoint directory (downloaded via "
            f"`huggingface-cli download <repo>`), not a repo id - this does no network "
            f"access, by design."
        )
    parsed: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
    return parsed


@dataclass(frozen=True, slots=True)
class VramPlan:
    model: str
    n_tokens: int
    quant_bits: float | None
    weights_gb: float
    kv_gb: float
    headroom_gb: float
    omitted: tuple[str, ...]

    @property
    def floor_gb(self) -> float:
        return self.weights_gb + self.kv_gb

    @property
    def total_gb(self) -> float:
        return self.floor_gb + self.headroom_gb

    def smallest_card(self) -> str:
        gb = self.total_gb
        for card in (
            "24 GB (RTX 4090 / L4 / A5000)",
            "40 GB (A100-40GB / L40S)",
            "48 GB (A6000 / A40-48)",
            "80 GB (A100-80GB / H100-80GB)",
        ):
            if gb <= float(card.split()[0]):
                return card
        return "NO CARD IN THIS LIST - needs a bigger SXM part"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_tokens": self.n_tokens,
            "quant_bits": self.quant_bits,
            "weights_gb": round(self.weights_gb, 2),
            "kv_gb": round(self.kv_gb, 2),
            "floor_gb": round(self.floor_gb, 2),
            "headroom_gb": self.headroom_gb,
            "total_gb": round(self.total_gb, 2),
            "smallest_standard_card": self.smallest_card(),
            "omitted": list(self.omitted),
            "note": (
                "weights + KV is the FLOOR. Activation memory, the CUDA context, "
                "allocator fragmentation and any multimodal tower are in headroom, "
                "not here. A plan that says 'fits' means 'fits with the stated "
                "headroom', not 'fits exactly'."
            ),
        }


def plan_vram(
    shape: ModelShape,
    *,
    n_tokens: int,
    quant_bits: float | None = None,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> VramPlan:
    omitted = [
        "activation memory for one decode step",
        "CUDA context (~0.5 GB)",
        "caching-allocator fragmentation",
        "batch size beyond 1 (this is single-stream)",
    ]
    if shape.is_moe:
        omitted.append("expert-routing workspace (MoE models need extra scratch)")
    return VramPlan(
        model=shape.name,
        n_tokens=n_tokens,
        quant_bits=quant_bits,
        weights_gb=shape.weights_gb(quant_bits=quant_bits),
        kv_gb=shape.kv_gb(n_tokens),
        headroom_gb=headroom_gb,
        omitted=tuple(omitted),
    )


def kv_reader_capacity_tokens(
    shape: ModelShape,
    vram_gb: float,
    *,
    quant_bits: float | None = None,
    reserve_tokens: int = 512,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> int:
    per_token = shape.kv_bytes_per_token()
    if per_token <= 0:
        return 0
    budget = (vram_gb - headroom_gb) * GB
    usable = budget - shape.weights_gb(quant_bits=quant_bits) * GB
    if usable <= 0:
        return 0
    return max(0, int(usable / per_token) - reserve_tokens)
