"""Hardware capacity modeling and context limit enforcement.

Consolidates prompt token modeling, reader limits across cards/headroom,
cell capacity verdicts, and panel coverage into a single deep module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from msat.hardware.vram import (
    DEFAULT_HEADROOM_GB,
    ModelShape,
    kv_reader_capacity_tokens,
)
from msat.store import Condition

if TYPE_CHECKING:
    from msat.providers.local import LocalReader


class ContextOverflowError(RuntimeError):
    """A prompt exceeds the reader's servable reader capacity on this card."""


class MissingAnchorError(KeyError, ValueError):
    """Required token anchors for a scale are missing from pricing configuration."""


class MissingReaderError(KeyError, ValueError):
    """A reader required by a cell was not found in the panel or configuration."""


def _interp_tokens(k: int, k_lo: int, p_lo: float, k_hi: int, p_hi: float) -> float:
    if k <= k_lo:
        return p_lo
    if k >= k_hi:
        return p_hi
    frac = math.log2(k / k_lo) / math.log2(k_hi / k_lo)
    return float(p_lo * (p_hi / p_lo) ** frac)


@dataclass(frozen=True, slots=True)
class ReaderLimit:
    """The servable context limit of a reader family on a card allowance."""

    id: str
    context_window: int
    max_prompt_tokens: int
    vram_capacity_tokens: int = 0
    card_gb: float = 8.0
    headroom_gb: float = DEFAULT_HEADROOM_GB
    weight_gb: float | None = None
    kv_mb_per_1k: float | None = None

    @property
    def reader_capacity_tokens(self) -> int:
        return min(self.context_window, self.max_prompt_tokens)

    @property
    def binding_constraint(self) -> str:
        return "vram" if self.max_prompt_tokens < self.context_window else "window"

    @property
    def binding(self) -> str:
        return self.binding_constraint

    @property
    def vram_gb(self) -> float:
        return self.card_gb

    @classmethod
    def from_window(cls, reader_id: str, window: int) -> ReaderLimit:
        return cls(
            id=reader_id,
            context_window=window,
            max_prompt_tokens=window,
            vram_capacity_tokens=window,
            card_gb=8.0,
            headroom_gb=DEFAULT_HEADROOM_GB,
        )

    @classmethod
    def from_vram(
        cls,
        reader_id: str,
        *,
        context_window: int,
        weight_gb: float,
        kv_mb_per_1k: float,
        vram_gb: float = 8.0,
        overhead_gb: float = DEFAULT_HEADROOM_GB,
    ) -> ReaderLimit:
        from msat.design import vram_max_prompt_tokens

        max_prompt = vram_max_prompt_tokens(
            weight_gb=weight_gb,
            kv_mb_per_1k=kv_mb_per_1k,
            vram_gb=vram_gb,
            overhead_gb=overhead_gb,
        )
        return cls(
            id=reader_id,
            context_window=context_window,
            max_prompt_tokens=max_prompt,
            vram_capacity_tokens=max_prompt,
            card_gb=vram_gb,
            headroom_gb=overhead_gb,
            weight_gb=weight_gb,
            kv_mb_per_1k=kv_mb_per_1k,
        )


@dataclass(frozen=True, slots=True)
class CapacityVerdict:
    """Verdict on whether a single condition fits a reader's hardware capacity."""

    condition: Condition
    prompt_tokens: int
    reader_capacity_tokens: int
    vram_capacity_tokens: int
    declared_context_window: int
    binding_constraint: str
    card_gb: float
    headroom_gb: float
    status: str
    reason: str = ""
    modeled_prompt_tokens: int | None = None
    rendered_prompt_tokens: int | None = None

    @property
    def scale(self) -> str:
        return self.condition.scale

    @property
    def reader(self) -> str:
        return self.condition.reader

    @property
    def k(self) -> int:
        return self.condition.k

    @property
    def b(self) -> float:
        return self.condition.b

    @property
    def b_mode(self) -> str:
        return self.condition.b_mode

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.condition.scale,
            "reader": self.condition.reader,
            "K": self.condition.k,
            "B": self.condition.b,
            "B_mode": self.condition.b_mode,
            "prompt_tokens": self.prompt_tokens,
            "reader_capacity_tokens": self.reader_capacity_tokens,
            "vram_capacity_tokens": self.vram_capacity_tokens,
            "overflow_by": max(0, self.prompt_tokens - self.reader_capacity_tokens),
            "label": f"{self.condition.scale}/{self.condition.reader}/K={self.condition.k}/B={int(self.condition.b)}",
            "declared_context_window": self.declared_context_window,
            "binding_constraint": self.binding_constraint,
            "card_gb": self.card_gb,
            "headroom_gb": self.headroom_gb,
            "status": self.status,
            "supported": self.supported,
            "reason": self.reason,
            "modeled_prompt_tokens": self.modeled_prompt_tokens,
            "rendered_prompt_tokens": self.rendered_prompt_tokens,
        }


def model_prompt_tokens(
    k: int,
    b: float | None,
    anchors: Mapping[str, Any],
    *,
    overhead: float = 500.0,
    b_mode: str = "per_memory_tokens",
    corpus_tokens: float | None = None,
) -> int:
    """Model the expected prompt tokens for a given (K, B) cell under the declared anchors.

    Fails explicitly if anchors are missing required fields.
    """
    if not anchors:
        raise MissingAnchorError("token anchor dictionary is empty or missing")
    if k <= 0 or not math.isfinite(overhead) or overhead < 0:
        raise ValueError(
            "Retrieval depth must be positive and prompt overhead finite and nonnegative."
        )
    if b is not None and (not math.isfinite(float(b)) or b <= 0):
        raise ValueError("Prompt budget must be finite and positive.")
    if b is not None and b_mode == "corpus_fraction":
        total = corpus_tokens if corpus_tokens is not None else anchors.get("corpus_tokens")
        if total is None:
            raise MissingAnchorError("corpus_fraction requires the corpus token count.")
        if b > 1:
            raise ValueError("Corpus-fraction budget must be in (0, 1].")
        return model_prompt_tokens(
            k,
            b * float(total),
            anchors,
            overhead=overhead,
            b_mode="absolute_tokens",
            corpus_tokens=float(total),
        )

    sessions = anchors.get("sessions_mean")
    per_session = anchors.get("tokens_per_session")

    if sessions is not None and per_session is not None:
        sessions_f = float(sessions)
        per_session_f = float(per_session)
        if not all(math.isfinite(v) and v > 0 for v in (sessions_f, per_session_f)):
            raise MissingAnchorError("Session count and token anchors must be finite and positive.")

        if b is None:
            content = min(float(k), sessions_f) * per_session_f
            return int(content + overhead)

        if b_mode == "per_memory_tokens":
            n = min(float(k), sessions_f)
            content = n * min(per_session_f, float(b))
            return int(content + overhead)

        if b_mode == "absolute_tokens":
            content = min(float(k), sessions_f) * per_session_f
            c_toks = float(
                corpus_tokens
                if corpus_tokens is not None
                else anchors.get("corpus_tokens", content)
            )
            budget = min(float(b), c_toks)
            return int(min(content, budget) + overhead)

        raise ValueError(f"Unsupported b_mode: {b_mode!r}")

    # If b_mode is per_memory_tokens, measured sessions and tokens per session are strictly required
    if b_mode == "per_memory_tokens":
        raise MissingAnchorError(
            f"b_mode='per_memory_tokens' requires 'sessions_mean' and 'tokens_per_session' anchors; got {list(anchors.keys())}"
        )

    # Legacy anchor interpolation fallback for absolute_tokens / None budgets in legacy fixtures
    k1 = anchors.get("prompt_tokens_at_K1")
    k64 = anchors.get("prompt_tokens_at_K64")
    if k1 is None or k64 is None:
        raise MissingAnchorError(
            f"Anchors missing required fields ('sessions_mean', 'tokens_per_session' or 'prompt_tokens_at_K1', 'prompt_tokens_at_K64'); got {list(anchors.keys())}"
        )

    content = _interp_tokens(k, 1, float(k1), 64, float(k64))
    if b is None:
        return int(content + overhead)

    if b_mode == "absolute_tokens":
        c_toks = float(
            corpus_tokens if corpus_tokens is not None else anchors.get("corpus_tokens", content)
        )
        budget = min(float(b), c_toks)
        return int(min(content, budget) + overhead)

    raise ValueError(f"Unsupported b_mode: {b_mode!r}")


def compute_reader_limit(
    reader_or_family: Any,
    *,
    card_gb: float = 8.0,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> ReaderLimit:
    """Compute the ReaderLimit for a LocalReader, ReaderFamily, or raw mapping."""
    if isinstance(reader_or_family, ReaderLimit):
        return reader_or_family

    if hasattr(reader_or_family, "shape") and hasattr(reader_or_family, "effective_quant_bits"):
        vram_capacity = kv_reader_capacity_tokens(
            reader_or_family.shape,
            card_gb,
            quant_bits=reader_or_family.effective_quant_bits,
            headroom_gb=headroom_gb,
        )
        declared = reader_or_family.declared_context_window
        return ReaderLimit(
            id=reader_or_family.id,
            context_window=declared,
            max_prompt_tokens=vram_capacity,
            vram_capacity_tokens=vram_capacity,
            card_gb=card_gb,
            headroom_gb=headroom_gb,
            weight_gb=reader_or_family.weights_gb,
            kv_mb_per_1k=reader_or_family.kv_bytes_per_token * 1000.0 / 1048576.0,
        )

    def value(obj: Any, name: str, default: Any = None) -> Any:
        return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)

    vram = value(reader_or_family, "vram")
    declared = int(value(reader_or_family, "context_window", 0))
    reader_id = str(value(reader_or_family, "id", ""))
    if not reader_id or declared <= 0:
        raise MissingReaderError("A reader must declare its identity and context window.")

    if vram is None:
        return ReaderLimit.from_window(reader_id, declared)

    params_b = float(value(vram, "params_b", 0.0))
    layers = int(value(vram, "layers", 0))
    kv_heads = int(value(vram, "kv_heads", 0))
    head_dim = int(value(vram, "head_dim", 0))
    bytes_per_param = float(value(vram, "bytes_per_param", 0.55))

    shape = ModelShape(
        name=str(getattr(reader_or_family, "model", reader_id)),
        n_params=int(params_b * 1_000_000_000),
        n_layers=layers,
        n_attention_heads=kv_heads,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype="bfloat16",
        n_params_is_exact=False,
    )
    quant_bits = bytes_per_param * 8.0
    vram_capacity = kv_reader_capacity_tokens(
        shape, card_gb, quant_bits=quant_bits, reserve_tokens=512, headroom_gb=headroom_gb
    )
    return ReaderLimit(
        id=reader_id,
        context_window=declared,
        max_prompt_tokens=vram_capacity,
        vram_capacity_tokens=vram_capacity,
        card_gb=card_gb,
        headroom_gb=headroom_gb,
        weight_gb=shape.weights_gb(quant_bits=quant_bits),
        kv_mb_per_1k=shape.kv_bytes_per_token() * 1000.0 / 1048576.0,
    )


def check_cell_capacity(
    reader_or_limit: LocalReader | ReaderLimit | Any,
    prompt_tokens: int,
    *,
    scale: str = "",
    k: int = 0,
    b: float = 0.0,
    b_mode: str = "per_memory_tokens",
    card_gb: float = 8.0,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    is_rendered: bool = False,
) -> CapacityVerdict:
    """Evaluate capacity for a given prompt token count.

    Equality (prompt_tokens == reader_capacity_tokens) is supported.
    prompt_tokens == reader_capacity_tokens + 1 is unsupported.
    """
    limit = (
        reader_or_limit
        if isinstance(reader_or_limit, ReaderLimit)
        else compute_reader_limit(reader_or_limit, card_gb=card_gb, headroom_gb=headroom_gb)
    )
    capacity_tokens = limit.reader_capacity_tokens
    supported = prompt_tokens <= capacity_tokens
    status = "supported" if supported else "unsupported"
    reason = ""
    if not supported:
        reason = (
            f"prompt ({prompt_tokens:,} tokens) exceeds {limit.id} reader capacity ({capacity_tokens:,} "
            f"tokens, bound by {limit.binding_constraint} on {card_gb} GB, headroom {headroom_gb} GB)"
        )

    condition = Condition(
        arm="A+G",
        k=k,
        b=b,
        b_mode=b_mode,
        scale=scale,
        reader=limit.id,
    )

    return CapacityVerdict(
        condition=condition,
        prompt_tokens=prompt_tokens,
        reader_capacity_tokens=capacity_tokens,
        vram_capacity_tokens=limit.vram_capacity_tokens,
        declared_context_window=limit.context_window,
        binding_constraint=limit.binding_constraint,
        card_gb=card_gb,
        headroom_gb=headroom_gb,
        status=status,
        reason=reason,
        modeled_prompt_tokens=None if is_rendered else prompt_tokens,
        rendered_prompt_tokens=prompt_tokens if is_rendered else None,
    )


def evaluate_cell_capacity(
    reader: LocalReader | ReaderLimit | Any,
    k: int,
    b: float,
    pricing: Mapping[str, Any],
    *,
    scale: str = "S",
    b_mode: str = "per_memory_tokens",
    card_gb: float = 8.0,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    overhead: float = 500.0,
) -> CapacityVerdict:
    """Model prompt tokens and evaluate cell capacity for a reader against pricing anchors."""
    estimates = pricing.get("token_estimates", {})
    anchors = estimates.get(scale)
    if not anchors:
        raise MissingAnchorError(f"Scale {scale!r} not found in pricing token_estimates")

    prompt_tokens = model_prompt_tokens(k, b, anchors, overhead=overhead, b_mode=b_mode)
    return check_cell_capacity(
        reader,
        prompt_tokens,
        scale=scale,
        k=k,
        b=b,
        b_mode=b_mode,
        card_gb=card_gb,
        headroom_gb=headroom_gb,
        is_rendered=False,
    )


def _as_condition(cell: Condition | Any) -> Condition:
    if isinstance(cell, Condition):
        return cell
    cond = getattr(cell, "condition", None)
    if isinstance(cond, Condition):
        return cond
    raise TypeError(
        f"Expected Condition or object with condition attribute, got {type(cell).__name__}"
    )


def evaluate_panel_capacity(
    panel: Sequence[LocalReader],
    cells: Sequence[Any],
    pricing: Mapping[str, Any],
    *,
    card_gb: float = 8.0,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    overhead: float | None = None,
) -> dict[str, Any]:
    """Evaluate panel coverage over a sequence of cells using modeled prompt tokens.

    This shares the exact verdict basis with preflight capacity checks.
    """
    ovh = float(pricing.get("prompt_overhead_tokens", 500)) if overhead is None else float(overhead)
    by_reader_id = {r.id: r for r in panel}

    verdicts_per_reader: dict[str, list[CapacityVerdict]] = {r.id: [] for r in panel}
    for cell in cells:
        cond = _as_condition(cell)
        scale = cond.scale
        k = cond.k
        b = cond.b
        b_mode = cond.b_mode
        target_reader = cond.reader

        if target_reader is not None and target_reader not in by_reader_id:
            raise MissingReaderError(f"Reader {target_reader!r} not found in the selected panel.")
        eval_readers = [by_reader_id[target_reader]] if target_reader is not None else panel
        for r in eval_readers:
            v = evaluate_cell_capacity(
                r,
                k,
                b,
                pricing,
                scale=scale,
                b_mode=b_mode,
                card_gb=card_gb,
                headroom_gb=headroom_gb,
                overhead=ovh,
            )
            verdicts_per_reader[r.id].append(v)

    per_reader_reports: list[dict[str, Any]] = []
    for r in panel:
        rv = verdicts_per_reader[r.id]
        limit = compute_reader_limit(r, card_gb=card_gb, headroom_gb=headroom_gb)
        supported = [v for v in rv if v.supported]
        unsupported = [v.as_dict() for v in rv if not v.supported]
        tot = len(rv)
        cov = len(supported) / tot if tot else 0.0
        per_reader_reports.append(
            {
                "reader": r.id,
                "reader_capacity_tokens": limit.reader_capacity_tokens,
                "declared_context_window": limit.context_window,
                "vram_capacity_tokens": limit.vram_capacity_tokens,
                "binding_constraint": limit.binding_constraint,
                "n_cells": tot,
                "n_supported": len(supported),
                "n_unsupported": len(unsupported),
                "coverage": cov,
                "supported": [v.as_dict() for v in supported],
                "unsupported": unsupported,
                "note": (
                    "Coverage computed via deep capacity modeling against modeled prompt tokens. "
                    "Truncation is refused rather than silent."
                ),
            }
        )

    supports = {rep["reader"]: rep["n_supported"] for rep in per_reader_reports}
    support_sets = {
        reader_id: {(v.scale, v.k, round(v.b, 6), v.b_mode) for v in verdicts if v.supported}
        for reader_id, verdicts in verdicts_per_reader.items()
    }
    common = set.intersection(*support_sets.values()) if support_sets else set()

    return {
        "readers": per_reader_reports,
        "n_cells": len(cells),
        "supported_per_reader": supports,
        "uniform_coverage": len({frozenset(s) for s in support_sets.values()}) == 1,
        "common_support": len(common),
        "common_support_cells": [list(cell) for cell in sorted(common)],
        "note": (
            "Non-uniform coverage is a hardware fact, not a defect in the panel "
            "(readers.yaml: gemma ~66k, phi ~44k, qwen ~21k tokens on 8 GB). R9.12 "
            "refuses to compare readers over unequal K support for this reason."
        ),
    }


def preflight_capacity_report(
    bundle: Any,
    plan: Any,
    panel: Sequence[LocalReader] | None = None,
    *,
    card_gb: float = 8.0,
    headroom_gb: float | None = None,
) -> dict[str, Any]:
    """Execute preflight capacity evaluation across a plan and reader panel."""
    from msat.providers.local import readers_from_config

    pricing = getattr(bundle, "pricing", {}) or {}
    estimates = pricing.get("token_estimates", {})
    overhead = float(pricing.get("prompt_overhead_tokens", 500))
    chars_per_token = float(estimates.get("chars_per_token", 4.0))

    if panel is None:
        raw_readers = getattr(bundle, "raw", {}).get("readers.yaml", {})
        headroom_val = (
            headroom_gb
            if headroom_gb is not None
            else float(
                raw_readers.get("context_policy", {}).get("headroom_gb", DEFAULT_HEADROOM_GB)
            )
        )
        panel = readers_from_config(raw_readers, card_gb=card_gb)
    else:
        headroom_val = (
            headroom_gb
            if headroom_gb is not None
            else float(
                getattr(bundle, "raw", {})
                .get("readers.yaml", {})
                .get("context_policy", {})
                .get("headroom_gb", DEFAULT_HEADROOM_GB)
            )
        )

    by_id = {r.id: r for r in panel}

    raw_conditions = getattr(plan, "conditions", None)
    plan_conditions: Sequence[Any] = (
        raw_conditions if raw_conditions is not None else (getattr(plan, "cells", ()) or ())
    )

    verdicts: list[CapacityVerdict] = []
    for cell in plan_conditions:
        reader_id = cell.condition.reader
        reader = by_id.get(reader_id)
        if reader is None:
            raise MissingReaderError(f"Reader {reader_id!r} not found in reader panel")

        scale = cell.condition.scale
        anchors = estimates.get(scale)
        if not anchors:
            raise MissingAnchorError(f"Scale {scale!r} not found in token_estimates")

        prompt = model_prompt_tokens(
            cell.condition.k,
            cell.condition.b,
            anchors,
            overhead=overhead,
            b_mode=cell.condition.b_mode,
            corpus_tokens=anchors.get("corpus_tokens"),
        )
        verdicts.append(
            check_cell_capacity(
                reader,
                prompt,
                scale=scale,
                k=cell.condition.k,
                b=cell.condition.b,
                b_mode=cell.condition.b_mode,
                card_gb=card_gb,
                headroom_gb=headroom_val,
                is_rendered=False,
            )
        )

    cov = evaluate_panel_capacity(
        list(panel), plan_conditions, pricing, card_gb=card_gb, headroom_gb=headroom_val
    )
    limits = {
        r.id: compute_reader_limit(r, card_gb=card_gb, headroom_gb=headroom_val) for r in panel
    }
    headroom_sens = {
        r.id: {
            label: compute_reader_limit(
                r, card_gb=card_gb, headroom_gb=allowance
            ).vram_capacity_tokens
            for label, allowance in (
                ("tight_0.8gb", 0.8),
                ("default", headroom_val),
                ("loose_5.0gb", 5.0),
            )
        }
        for r in panel
    }

    per_reader = {
        r.id: {
            "reader_capacity_tokens": limits[r.id].reader_capacity_tokens,
            "vram_capacity_tokens": limits[r.id].vram_capacity_tokens,
            "declared_context_window": limits[r.id].context_window,
            "binding_constraint": limits[r.id].binding_constraint,
            "n_supported": sum(1 for v in verdicts if v.reader == r.id and v.supported),
            "n_unsupported": sum(1 for v in verdicts if v.reader == r.id and not v.supported),
        }
        for r in panel
    }

    n_supported = sum(1 for v in verdicts if v.supported)
    n_unsupported = sum(1 for v in verdicts if not v.supported)

    return {
        "requirement": "PRD §R3.2",
        "on_overflow": "reject_cell",
        "never_truncate_silently": True,
        "n_checked": len(verdicts),
        "n_cells": len(verdicts),
        "n_supported": n_supported,
        "n_unsupported": n_unsupported,
        "supported": [v.as_dict() for v in verdicts if v.supported],
        "unsupported": [v.as_dict() for v in verdicts if not v.supported],
        "n_accepted": n_supported,
        "n_rejected": n_unsupported,
        "rejected": [v.as_dict() for v in verdicts if not v.supported],
        "per_reader": per_reader,
        "panel_coverage": cov,
        "headroom_sensitivity": headroom_sens,
        "prompt_tokens_model": (
            f"declared design model, +{overhead:,.0f} tokens overhead, chars_per_token="
            f"{chars_per_token} (ASSUMED). The RENDERED prompt is measured instead at "
            f"dispatch time."
        ),
        "headroom_gb_is_open_decision": (
            "D11. Every ceiling above moves with it: qwen's servable depth on 8 GB is "
            "20,889 tokens at 0.8 GB of headroom and 2,682 at the 3.0 GB default. The "
            "number nobody decided is the number that decides which cells run."
        ),
        "note": (
            "A rejected cell is reported, not repaired. Trimming the prompt leaves it "
            "able to contain its own answer while appearing not to, and the curve that "
            "follows measures nothing (PRD §R0.3 — propose, never truncate)."
        ),
    }
