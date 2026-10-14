"""Compute the VRAM a checkpoint actually needs, from its own config.

WHY THIS EXISTS
---------------
Every VRAM figure in this project so far has been a NUMBER SOMEONE TYPED. The
ceilings in `readers.yaml` (66,160 / 44,318 / 20,889 tokens) were estimated by hand
and then checked against the corpus - which caught that Arm D could not run, but
only after the fact, and only for the three models already chosen.

Two open decisions (D1: the judge engine, D4: Arm D) both turn on the same
question - *which card does this checkpoint need?* - and the answer is arithmetic on
fields that are already in `config.json`:

    KV bytes per token = 2 x n_layers x n_kv_heads x head_dim x dtype_bytes

`2` is K and V. `n_kv_heads` is the GQA factor and it is the whole reason a 70B
model can serve long context at all - a model with 8 KV heads uses 8x less KV
cache than its parameter count suggests.

READ THE CONFIG, DO NOT GUESS THE ARCHITECTURE
----------------------------------------------
The KV cost of a mixture-of-experts model depends only on its ATTENTION layers, not
on its parameter count. `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` has 30B
parameters but if most of its layers are experts rather than attention, its KV
footprint is far below what "30B" implies. Computing it is a ten-line arithmetic
operation; reasoning about it from the name is not.

WHAT THIS DOES NOT DO
---------------------
It does not include activations, the CUDA graph, fragmentation, or a vision tower.
It returns the weights + KV floor and applies an explicit headroom factor, and it
says which factors it omitted. A number presented as "the requirement" when it is
the floor is the same class of error as every hand-typed ceiling in this project.
"""

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

#: Activation graph, CUDA context, allocator slack and fragmentation.
DEFAULT_HEADROOM_GB = 3.0
"""Fixed, and stated rather than hidden.

2-3 GB is what a real vLLM/transformers process carries beyond weights and KV:
the CUDA context alone is ~500 MB, activations for one token scale with the hidden
size and batch, and the caching allocator fragments. Anyone who has watched a
process OOM at 97% of capacity already believes this.
"""


class ConfigMissingError(ValueError):
    """The config does not carry the fields the arithmetic needs.

    Raised rather than defaulted. A default for `num_key_value_heads` is the single
    most dangerous guess in this project, because assuming `num_attention_heads`
    (i.e. no GQA) overstates the KV requirement by exactly the GQA factor - often
    8x - and would demand a card the model does not need.
    """


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The fields that determine the VRAM floor."""

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
    """Whether `n_params` came from the config or from `_estimate_params`.

    Recorded rather than inferred. An estimate presented as a count is how a 24 GB
    card gets mistaken for a 48 GB one, and no downstream number should be allowed
    to quietly present one as the other.
    """

    @property
    def dtype_bytes(self) -> float:
        return DTYPE_BYTES.get(self.dtype.lower(), 2.0)

    @property
    def gqa_factor(self) -> int:
        """How much smaller the KV cache is than a full MHA model of this depth."""
        return max(1, self.n_attention_heads // max(1, self.n_kv_heads))

    def kv_bytes_per_token(self) -> int:
        """2 (K and V) x layers x kv_heads x head_dim x bytes."""
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
    """Read a HF `config.json` into a `ModelShape`.

    Accepts the several spellings transformers has used for the same field over the
    years (`n_layer`/`num_hidden_layers`, `num_kv_heads`/`num_key_value_heads`).
    """
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

    # Parameter count: only if the config carries it. Otherwise estimate from the
    # known fields and SAY that, because an estimate presented as a count is how a
    # 24 GB card gets mistaken for a 48 GB one.
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
    """Rough parameter count from the config. ALWAYS reported as an estimate.

    Expert weights ARE counted when the config declares them, because for a
    mixture-of-experts model they are the overwhelming majority of the parameters
    and ignoring them understates the weights requirement by roughly the expert
    count. A tool that reports 10 GB for a 30B-A3B model is not approximate, it is
    wrong in exactly the direction that gets someone to under-buy a card.

    `moe_intermediate_size` is preferred over `intermediate_size` when present.
    MoE models routinely give experts a SMALLER feed-forward width than the dense
    layers, and using the dense width for every expert over-counts by the ratio -
    which on a 60-expert model is a factor of several and produces an absurd
    number. When neither field exists the result is an over-estimate and the caller
    is told so via `n_params_is_exact`.
    """
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
    """Read `config.json` from a local checkpoint directory."""
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
    """What a given (model, context, quantisation) combination needs."""

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


def kv_ceiling_tokens(
    shape: ModelShape,
    vram_gb: float,
    *,
    quant_bits: float | None = None,
    reserve_tokens: int = 512,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> int:
    """How many prompt tokens fit in `vram_gb`.

    This is the arithmetic behind the ceilings in `readers.yaml`. Those were typed;
    this is computed from the checkpoint's own `config.json`, so when a model or a
    card changes the ceiling moves with it instead of being re-guessed.
    """
    per_token = shape.kv_bytes_per_token()
    if per_token <= 0:
        return 0
    budget = (vram_gb - headroom_gb) * GB
    usable = budget - shape.weights_gb(quant_bits=quant_bits) * GB
    if usable <= 0:
        return 0
    return max(0, int(usable / per_token) - reserve_tokens)
