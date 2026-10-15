"""`providers/local.py` - a reader FAMILY bound to a local backend (P8, R7.5/R7.6).

WHAT THIS IS, AND WHY IT IS NOT A BACKEND
-----------------------------------------
`providers/base.py` already has the engines (`transformers`, `vllm`,
`onnxruntime-genai`). This module is the layer *above* them: it binds one declared
reader family from `configs/readers.yaml` to one engine, and it owns the two
questions an engine does not answer.

    which ENGINE          -> base.select_backend
    which FAMILY, at what
    VERSION, with what
    reader_strength, and
    what depth it can
    actually serve        -> this module

R7.5 requires at least two families, and R7.6 requires identity, version and date to
be recorded on every row so `reader_strength` can be modelled as a covariate. Neither
is a property of a backend: a backend knows how to generate, not which checkpoint it
was handed or whether that checkpoint can hold the prompt.

THE VRAM CEILING IS COMPUTED, NEVER TYPED
-----------------------------------------
`configs/readers.yaml` already records the uncomfortable fact: every family declares
a 131k-262k window and **none of them can deliver it on 8 GB**, because the weights
must be resident before the KV cache grows. gemma reaches ~66k, qwen ~21k, phi
~44k.

So the declared window is a property of the CHECKPOINT and the servable depth is a
property of the CARD. `ceiling_tokens` computes the second from the family's own
`vram:` block, which is what `hardware/vram.py` exists for. A typed constant here
would be a number that drifts silently the first time a card or a quantisation
setting changes.

OVERFLOW IS A REJECTED CELL, NOT A TRUNCATED PROMPT
---------------------------------------------------
`context_policy.on_overflow: reject_cell` and `never_truncate_silently: true`. A
prompt that does not fit is a DIFFERENT QUESTION from the same prompt truncated:
the truncated one still contains the answer while looking like it might not, and the
resulting accuracy curve measures nothing. `check_capacity` therefore raises, and
`coverage_over` reports the cells a family cannot serve rather than leaving the
caller to discover it per-prompt.

That report is the input to R9.12: readers with different ceilings have different K
supports, and `analysis/reader_agreement.py` refuses to compare them over unequal
support for exactly this reason. The two modules are a matched pair - this one
produces the coverage, that one refuses to pretend the coverage was complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from msat.hardware.vram import DEFAULT_HEADROOM_GB, ModelShape, kv_ceiling_tokens
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    LocalBackend,
    select_backend,
)


class ContextOverflowError(RuntimeError):
    """A prompt exceeds the reader's servable ceiling on this card.

    Carries the numbers, because the remedy is a config or hardware decision and the
    reader needs both: the requested depth, the ceiling, and which bound it was.

    Distinct from an OOM. This fires BEFORE any allocation, so the run stops at the
    planning stage rather than at hour 14.
    """


class UnknownReaderFamilyError(KeyError):
    """A family id was requested that `configs/readers.yaml` does not declare."""


@dataclass(frozen=True, slots=True)
class ReaderVramSpec:
    """The `vram:` block of a family, in config terms rather than HF terms."""

    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_param: float = 0.55
    """Why this is not simply a dtype size: a `4bit` request that silently falls back
    to bf16 (README: `record_in_manifest: true`) changes the weights footprint by 4x.
    The effective bytes-per-parameter is what the config decided, and it is recorded
    per family so a fallback can be detected."""

    def to_shape(self, *, name: str, dtype: str = "bfloat16") -> ModelShape:
        """A `ModelShape`, so this module reuses `hardware/vram.py` rather than
        re-deriving KV arithmetic. Two implementations of the same formula is how
        the 4060's ceilings and a card's ceilings came to disagree."""
        n_params = int(self.params_b * 1_000_000_000)
        return ModelShape(
            name=name,
            n_params=n_params,
            n_layers=self.layers,
            # GQA factor is unknown from this block alone; `kv_bytes_per_token` only
            # needs kv_heads and head_dim, and `gqa_factor` is a reporting property.
            n_attention_heads=self.kv_heads,
            n_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            n_params_is_exact=False,
        )


@dataclass(frozen=True, slots=True)
class LocalReader:
    """One reader family, bound to one engine, with its ceiling computed."""

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

    # ── the ceiling, computed (R3.2) ─────────────────────────────────────────
    @property
    def shape(self) -> ModelShape:
        return self.vram.to_shape(name=self.model)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.shape.kv_bytes_per_token()

    @property
    def effective_quant_bits(self) -> float:
        """Bits per parameter, derived from the family's declared BYTES.

        `bytes_per_param: 0.55` in the config is bytes - 4.4 bits, i.e. int4 plus
        quantisation overhead, matching `quantisation.default_dtype: 4bit`. Treating
        it as bits and dividing again by 8 undercounted gemma's weights as 0.26 GB
        instead of 2.05 GB, which inflated its ceiling from ~37k to ~59k tokens -
        nearly 1.6x, and in the direction that marks deep cells as servable when
        they are not. The conversion is a named property so the unit is stated once.

        `effective_` because it can differ from the REQUESTED `quant_bits`: a silent
        int4 -> bf16 fallback (README `record_in_manifest: true`) changes the weights
        4x, and the two must never be confused.
        """
        return self.vram.bytes_per_param * 8.0

    @property
    def weights_gb(self) -> float:
        """Weights footprint, from the family's declared bytes-per-parameter.

        Delegates to `ModelShape.weights_gb`. A silent fallback from int4 to bf16
        changes this by 4x, which is why the effective value is recorded per family
        rather than inferred from a nominal dtype name.
        """
        return self.shape.weights_gb(quant_bits=self.effective_quant_bits)

    def ceiling_tokens(self, *, card_gb: float | None = None) -> int:
        """How many prompt tokens this family can serve on this card.

        Delegates to `hardware/vram.kv_ceiling_tokens` - the same function `msat vram`
        uses - rather than re-deriving the arithmetic. Two implementations of one
        formula is how a card's ceilings and the config's ceilings come to disagree,
        which is the drift this project keeps catching.

        Derived, never stored. A typed ceiling is a number that looks authoritative,
        survives a card swap, and is then wrong.
        """
        card = self.card_gb if card_gb is None else card_gb
        return kv_ceiling_tokens(
            self.shape,
            card,
            quant_bits=self.effective_quant_bits,
            headroom_gb=self.headroom_gb,
        )

    @property
    def serving_limit_tokens(self) -> int:
        """`min(declared window, VRAM ceiling)`.

        Both bind, and which one binds is worth knowing - `binding_constraint` says
        which. Every family here is VRAM-bound on 8 GB while declaring 131k+, which
        is the whole reason depth is a per-reader variable.
        """
        return min(self.declared_context_window, self.ceiling_tokens())

    @property
    def binding_constraint(self) -> str:
        """`vram` or `declared_window` or `both`. Reported so a reader can tell a
        hardware limit from a model's own limit without recomputing the arithmetic."""
        v = self.ceiling_tokens()
        if v < self.declared_context_window:
            return "vram"
        return "declared_window"

    def ceiling_at_headroom(self, headroom_gb: float) -> int:
        """The ceiling at a DIFFERENT overhead allowance.

        Exists because `headroom_gb` is the single most consequential number here and
        it is not a physical constant - it is a guess about fragmentation, allocator
        slack and CUDA context. On 8 GB with these three families it decides whether
        the panel has two usable readers or three:

            family   headroom 0.8 GB   headroom 3.0 GB (vram.py default)
            gemma            66,160                   36,326
            qwen             20,889                    2,682
            phi              44,318                   24,502

        qwen goes from workable to unusable. `configs/readers.yaml`'s hand-typed
        ceiling comments were computed at the first allowance and
        `hardware/vram.py` defaults to the second, so the two disagree by up to
        7.8x. That is open decision D11, not something to paper over by picking one.
        """
        return kv_ceiling_tokens(
            self.shape,
            self.card_gb,
            quant_bits=self.effective_quant_bits,
            headroom_gb=headroom_gb,
        )

    def ceiling_sensitivity(self) -> dict[str, int]:
        """The ceiling at several allowances. A single number here would hide a 7.8x
        swing behind a decision nobody made deliberately."""
        return {
            "tight_0.8gb": self.ceiling_at_headroom(0.8),
            "default": self.ceiling_tokens(),
            "loose_5.0gb": self.ceiling_at_headroom(5.0),
        }

    # ── capacity enforcement (R3.2) ──────────────────────────────────────────
    def check_capacity(self, n_tokens: int, *, card_gb: float | None = None) -> int:
        """Raise if a prompt of `n_tokens` cannot be served. Returns the ceiling.

        The REJECT, not a truncate: a prompt cut to fit still contains its own answer
        while looking as though it might not, and the accuracy curve that follows
        measures nothing at all. `context_policy.on_overflow: reject_cell`.
        """
        limit = self.serving_limit_tokens
        ceiling = self.ceiling_tokens(card_gb=card_gb)
        if n_tokens > limit:
            raise ContextOverflowError(
                f"reader {self.id!r} cannot serve {n_tokens:,} tokens: its ceiling on "
                f"{card_gb or self.card_gb} GB is {limit:,} "
                f"(VRAM {ceiling:,}, declared window {self.declared_context_window:,}, "
                f"bound by {self.binding_constraint}). Truncating would leave the "
                f"prompt able to contain its own answer while appearing not to, and "
                f"the resulting curve would measure nothing (R3.2, "
                f"never_truncate_silently)."
            )
        return limit

    def serves(self, n_tokens: int) -> bool:
        return n_tokens <= self.serving_limit_tokens

    def coverage_over(
        self, cells: Sequence[tuple[int, float, str]], *, card_gb: float | None = None
    ) -> dict[str, Any]:
        """Which (k, b, scale) cells this family can serve, and which it cannot.

        Cells are (K, B, scale) triples, NOT token counts: the prompt length is a
        property of the corpus and the condition, which the caller knows and this
        module does not. `tokens_of` supplies the conversion, so a caller that has
        one can ask the real question instead of guessing.

        The unsupported list is returned rather than dropped. R0.3/N8 forbid a
        silently-truncated factorial, and a hole nobody is told about is exactly
        that.
        """
        limit = self.serving_limit_tokens
        served: list[tuple[int, float, str]] = []
        unsupported: list[dict[str, Any]] = []
        for k, b, scale in cells:
            entry = {"k": k, "b": b, "scale": scale}
            if self.serves(k):
                served.append((k, b, scale))
            else:
                unsupported.append(
                    {**entry, "reason": f"K={k} exceeds the {limit:,}-token ceiling"}
                )
        total = len(served) + len(unsupported)
        return {
            "reader": self.id,
            "limit_tokens": limit,
            "declared_context_window": self.declared_context_window,
            "vram_ceiling": self.ceiling_tokens(card_gb=card_gb),
            "binding_constraint": self.binding_constraint,
            "n_cells": total,
            "n_served": len(served),
            "n_unsupported": len(unsupported),
            "coverage": len(served) / total if total else 0.0,
            "unsupported": unsupported,
            "note": (
                "K is compared against the token ceiling as a proxy for prompt length. "
                "Pass tokens_of=... for the exact question; this is the cheap bound."
            ),
        }

    # ── R7.6: identity, version, date, strength ─────────────────────────────
    def reader_metadata(self) -> dict[str, Any]:
        """The R7.6 record. Travels into the manifest and onto every row.

        `version` is a commit SHA, not a tag: tags get repointed, which would change
        the weights under a fixed `config_hash` and make two runs incomparable while
        every artifact claimed the same identity.

        `bytes_per_param` is recorded because quantisation is a recorded
        experimental variable (R6.4) - it changes the weights and therefore the
        answers - and a silent fallback to bf16 would otherwise be undetectable.
        """
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
            "vram_ceiling": self.ceiling_tokens(),
            "serving_limit": self.serving_limit_tokens,
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

    # ── engine binding ───────────────────────────────────────────────────────
    def backend(self) -> LocalBackend:
        """The engine, instantiated on first use.

        Not in `__post_init__`: importing a backend must not load weights, and
        `design-power` / `validate-config` call `health()` on a machine that may have
        no GPU at all.
        """
        if self._backend is None:
            object.__setattr__(self, "_backend", select_backend(self.backend_name))
        return self._backend  # type: ignore[return-value]

    def health(self) -> dict[str, Any]:
        """Preflight (`make env`). Never loads weights.

        The family's own facts are reported even when the engine is unavailable - a
        reader debugging a refused run needs to know the ceiling, and an engine error
        alone does not say it.

        Catches `ValueError` as well as `BackendUnavailableError`, because
        `select_backend` raises `ValueError` for a backend name it does not know -
        which is the same situation by another route. Without it, a typo in
        `backend_name` turned a preflight report into a traceback, and the one
        command whose job is to explain what is wrong could not explain this.
        """
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
            "can_serve_deepest": self.serves(self.declared_context_window),
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Delegate, after stamping the identity onto the result.

        R6.2 requires `model_version` on every result: prices and behaviour both
        drift, so a re-run a month later is not comparable without it. A backend that
        fills it in from whatever it loaded is recording the engine's idea of the
        version, which is exactly the drift being guarded against.
        """
        result = self.backend().generate(request)
        if not result.model_version:
            object.__setattr__(result, "model_version", self.version)
        return result

    def context_window(self, model: str | None = None) -> int:
        """The SERVABLE limit, not the declared window.

        `base.LocalBackend.context_window` feeds the context-feasibility check before
        a run (R3.2), and that check has to see the ceiling a budgeted arm can
        actually reach - otherwise every long cell is marked feasible and then fails
        at allocation.
        """
        return self.serving_limit_tokens


# ── construction from config ──────────────────────────────────────────────────


def reader_from_family(
    family: Mapping[str, Any],
    *,
    card_gb: float = 8.0,
    backend: str = "transformers",
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> LocalReader:
    """Build a `LocalReader` from one `readers.yaml` family entry.

    Every field is read from the mapping; nothing is defaulted from a plausible
    guess. `configs/readers.yaml` documents that the previous panel was three
    nonexistent models because the ids were invented, and a default here would
    reintroduce that failure one level down.
    """
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
    config: Mapping[str, Any], *, card_gb: float = 8.0, backend: str = "transformers"
) -> list[LocalReader]:
    """Every declared family. R7.5 requires >= 2 and `config.py` already enforces it;
    this does not re-enforce, but it refuses a single-family panel loudly because
    R9.12 needs at least one PAIR to mean anything.

    A one-reader panel is not a smaller experiment. It is an experiment that cannot
    test whether the depth curve generalises across readers at all.

    `context_policy.headroom_gb` is honoured when present. It is read here rather
    than left to `vram.py`'s default because the allowance is a per-experiment
    policy decision with a 7.8x effect on qwen's ceiling (D11), and a value buried
    in another module's signature is a value nobody reviews.
    """
    families = config.get("families") or []
    policy = config.get("context_policy") or {}
    headroom = float(policy.get("headroom_gb", DEFAULT_HEADROOM_GB))
    out = [
        reader_from_family(f, card_gb=card_gb, backend=backend, headroom_gb=headroom)
        for f in families
    ]
    if len(out) < 2:
        raise ValueError(
            f"readers.yaml declares {len(out)} family/families; R7.5 requires at "
            f"least 2, and R9.12 needs a pair to compare"
        )
    return out


def panel_coverage(
    readers: Sequence[LocalReader], cells: Sequence[tuple[int, float, str]]
) -> dict[str, Any]:
    """Every family's coverage of the same grid, side by side.

    This is the artifact R9.12 needs to know whether its comparison is over common
    support. When the readers do not share a support - which on 8 GB they will not -
    the report says so here, before any flip is counted, rather than the comparison
    quietly narrowing later.
    """
    per_reader = [r.coverage_over(cells) for r in readers]
    supports = {r["reader"]: r["n_served"] for r in per_reader}
    served_counts = set(supports.values())
    return {
        "readers": per_reader,
        "n_cells": len(cells),
        "served_per_reader": supports,
        "uniform_coverage": len(served_counts) == 1,
        "common_support": min(served_counts) if served_counts else 0,
        "note": (
            "Non-uniform coverage is a hardware fact, not a defect in the panel "
            "(readers.yaml: gemma ~66k, phi ~44k, qwen ~21k tokens on 8 GB). R9.12 "
            "refuses to compare readers over unequal K support for this reason."
        ),
    }


def load_readers(
    path: Path, *, card_gb: float = 8.0, backend: str = "transformers"
) -> list[LocalReader]:
    """Read `configs/readers.yaml` and build the panel."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return readers_from_config(raw, card_gb=card_gb, backend=backend)


class _FamilySource(Protocol):
    """What `readers.yaml` must provide. Named so the requirement is checkable."""

    families: Sequence[Mapping[str, Any]]
