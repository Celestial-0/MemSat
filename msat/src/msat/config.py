"""Configuration is the single source of truth (PRD §R1).

`A(K, B)` is a property of a six-tuple `(K, B, E, X, M, P, order)`, not of K.
Five of those six are *confounders*: they are not swept, so nothing forces them
to be written down. If they are not pinned, named, and hashed into every
artifact, the curve is uninterpretable and unattributable — a retriever
specified after the sweep has run invalidates every figure, on a budget already
declared an order of magnitude short.

This module makes that mechanical rather than aspirational:

- every model here is **strict** — an unknown or missing key is an error, not a
  default. `extra="forbid"` is what turns "someone forgot `tie_break_rule`" into
  a non-zero exit instead of a silent default that quietly becomes a confound;
- **no field may be null** (PRD §R1.1), enforced by a post-validation sweep, so
  a YAML `null` cannot masquerade as a value;
- `config_hash` is SHA-256 over the *resolved* bundle, stable under key
  reordering and sensitive to a single-character prompt edit.

Problems carry a severity. `error` aborts the command; `warn` is printed loudly
and does not. The split exists because some decisions are genuinely blocked on
the guide (reader models, budget ceiling) and a harness that refuses to compute
anything until a human replies is not a harness, it is a mailbox.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BUNDLE_VERSION = 1

#: Filenames that make up the resolved bundle, in hash order.
BUNDLE_FILES: tuple[str, ...] = (
    "design.yaml",
    "scales.yaml",
    "retrieval.yaml",
    "decoding.yaml",
    "readers.yaml",
    "budget.yaml",
    "arms.yaml",
    "budget_guard.yaml",
    "preregistration.yaml",
    "prompts/abstention.yaml",
    "prompts/normalisation.yaml",
    "prompts/arm_a.j2",
    "prompts/arm_b.j2",
    "prompts/arm_c.j2",
    "prompts/arm_d.j2",
    "prompts/arm_e.j2",
    "prompts/arm_f.j2",
    "prompts/arm_g.j2",
    "pricing/latest.yaml",
)


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"


class Problem(BaseModel):
    """One validation finding."""

    model_config = ConfigDict(frozen=True)

    severity: Severity
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        location = f" [{self.where}]" if self.where else ""
        return f"{self.severity.value.upper():5s} {self.code}: {self.message}{location}"


class Strict(BaseModel):
    """Base for every config model: no unknown keys, no silent defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


# ── retrieval.yaml (PRD §R1.1) ─────────────────────────────────────────────────
class EmbedderConfig(Strict):
    id: str = Field(min_length=1, description="Model ID. An unnamed embedder is a confound.")
    version: str = Field(min_length=1, description="Exact revision / commit, not a tag.")
    dimensionality: int = Field(gt=0)
    normalisation: Literal["l2", "none"]
    batch_size: int = Field(gt=0)
    device: Literal["cuda", "cpu"]
    max_input_tokens: int = Field(gt=0)

    @field_validator("id", "version")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class ChunkerConfig(Strict):
    id: str = Field(min_length=1, description="Chunker ID.")
    scheme: Literal["session", "chunk", "fact"]
    chunk_size_units: int = Field(gt=0)
    overlap_units: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> ChunkerConfig:
        if self.overlap_units >= self.chunk_size_units:
            raise ValueError(
                f"overlap_units ({self.overlap_units}) must be < chunk_size_units "
                f"({self.chunk_size_units}); otherwise a unit is fully duplicated"
            )
        return self


class SimilarityConfig(Strict):
    metric: Literal["cosine", "dot", "euclidean"]
    top_k_tie_break: Literal[
        "lowest_session_id_ascending",
        "highest_session_id_descending",
        "earliest_date_first",
        "lowest_similarity_id_ascending",
    ]
    """Deterministic tie-break. Scores tie constantly at small K, and an
    unspecified tie-break silently reorders the exact cells under study."""


class TokenizerConfig(Strict):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    """Pinned here because `tokens(corpus)` feeds the B-guard (PRD §R3.4)."""
    truncation: bool
    max_length: int = Field(gt=0)


class RetrievalConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    embedder: EmbedderConfig
    chunker: ChunkerConfig
    similarity: SimilarityConfig
    tokenizer: TokenizerConfig
    unit: Literal["session", "chunk", "fact"] = Field(
        description="PRD §R1.6. Every metrics output reports items AND tokens with this label."
    )

    @model_validator(mode="after")
    def _unit_matches_chunker(self) -> RetrievalConfig:
        if self.unit != self.chunker.scheme:
            raise ValueError(
                f"unit ({self.unit!r}) must equal chunker.scheme ({self.chunker.scheme!r}); "
                "a mismatch makes the reported item count uninterpretable (PRD §R1.6)"
            )
        return self


# ── design.yaml (PRD §R0.1) ───────────────────────────────────────────────────
class BLevel(Strict):
    """One budget level. The MODE travels with the value (PRD §R1.7).

    Mixing absolute and relative budgets silently invalidates the interaction:
    `4096` tokens is a different fraction of a 3 M-character corpus than of a
    300 K-character one, so `log2(B)` stops meaning one thing across the grid.

    `per_memory_tokens` is the mode the grid now uses, and the reason is a hard
    result rather than a preference. Under `absolute_tokens` the budget is a
    CUTOFF ON K expressed in tokens, so the number of active budget levels at
    depth K is monotone in K — a staircase that provably cannot exceed 2 distinct
    outcomes at K=1 and K=2 for ANY choice of values. An axis that cannot vary
    independently cannot carry an interaction.

    Under `per_memory_tokens` the prompt is `min(K, sessions) x min(mean_memory,
    B)`, which scales WITH K: doubling K doubles the prompt at any budget, and
    changing B changes density without changing how many memories were retrieved.
    The crossing is then real.

    It is also the more meaningful manipulation — "16 memories, first 512 tokens of
    each" is a different retrieval strategy from "4 memories in full", and only the
    per-memory form can express that difference at all.
    """

    value: float = Field(gt=0)
    mode: Literal["absolute_tokens", "corpus_fraction", "per_memory_tokens"]


class ItemsConfig(Strict):
    """Which items enter the grid."""

    include_abstention: bool = True
    """`_abs` items stay in end-to-end accuracy and are scored by the benchmark's
    own rule; they are excluded from `recall@K` (PRD §R8.4)."""
    limit: int | None = Field(default=None, ge=1)
    """`null` = all. This is NOT a PRD §R1.1 null — that rule governs retrieval.yaml,
    where a null would mean "unnamed" rather than "no limit"."""


class ReferenceSpec(Strict):
    """The one (K, B) a marginal single-factor sweep would centre on.

    Recorded so a reader can see what the four papers that hold depth fixed
    (k=5, k=12, k=15, "unspecified") are implicitly running, and therefore what the
    interaction is being measured against.
    """

    K: int = Field(gt=0)
    B: float = Field(gt=0)


class DesignConfig(Strict):
    """`configs/design.yaml` — THE single source of the grid (PRD §R0.1)."""

    schema_version: int = BUNDLE_VERSION
    K_grid: list[int] = Field(min_length=2)
    B_grid: list[BLevel] = Field(min_length=2)
    scales: list[Literal["S", "M"]] = Field(min_length=1)
    readers: list[str] = Field(min_length=2, description="PRD §R7.5: >=2 families, enforced.")
    items: ItemsConfig
    runs_per_cell: int = Field(ge=1)
    seed: int
    reference: ReferenceSpec

    @field_validator("K_grid")
    @classmethod
    def _strictly_increasing(cls, v: list[int]) -> list[int]:
        if any(k <= 0 for k in v):
            raise ValueError("every K must be > 0")
        if sorted(set(v)) != v:
            raise ValueError("K_grid must be sorted and free of duplicates")
        return v

    @model_validator(mode="after")
    def _internally_consistent(self) -> DesignConfig:
        if self.reference.K not in self.K_grid:
            raise ValueError(f"reference.K {self.reference.K} is not in K_grid {self.K_grid}")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("scales contains duplicates")
        if len(set(self.readers)) != len(self.readers):
            raise ValueError("readers contains duplicates")
        return self

    @property
    def b_values(self) -> list[float]:
        return [b.value for b in self.B_grid]

    @property
    def n_cells(self) -> int:
        """Every cell in `K_grid x B_grid` is a distinct condition (PRD §R4.4)."""
        return len(self.K_grid) * len(self.B_grid) * len(self.scales) * len(self.readers)


# ── decoding.yaml (PRD §R7.1-R7.4) ────────────────────────────────────────────
class DecodingConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)
    stop_sequences: list[str]
    provider_seed: int
    reasoning_budget_tokens: int = Field(ge=0)
    """PRD §R7.3. Explicit because it moves cost ~10x; an unstated default makes
    the cost model unsound."""
    reasoning_budget_rationale: str = Field(min_length=1)
    prompt_nonce: bool = False
    """A per-request nonce in the prompt. Belt-and-braces with `temperature > 0`."""

    @model_validator(mode="after")
    def _seeds_must_vary_something(self) -> DecodingConfig:
        """PRD §R7.1.

        At `temperature: 0` with a fixed prompt, repeat runs are byte-identical,
        and the content-addressed store (PRD §R6.1) dedupes them to ONE file.
        Seed-variance estimation then becomes impossible and silently so — the
        data collapses without any error. Either knob makes repeats distinct.
        """
        if self.temperature == 0.0 and not self.prompt_nonce:
            raise ValueError(
                "temperature is 0 and prompt_nonce is false: repeat runs would be "
                "byte-identical and PRD §R6.1 would dedupe them to one cached file, "
                "making seed-variance estimation unachievable. Set temperature > 0 "
                "or enable prompt_nonce (PRD §R7.1)."
            )
        return self


# ── readers.yaml (PRD §R7.5, §R7.6) ───────────────────────────────────────────
class ReaderVram(Strict):
    """Published architecture facts, from which the prompt ceiling is DERIVED.

    These are FACTS read from each repo's `config.json`, not tuning knobs. They
    exist because the number that decides which cells can run is not the model's
    declared context window - it is how many prompt tokens fit in VRAM once the
    weights are resident:

        weight_gb     = params_b * bytes_per_param
        kv_mb_per_1k = 2 (K and V) * layers * kv_heads * head_dim * 2 bytes / 1000

    `head_dim` deserves attention because it is the term people forget. Gemma 4
    12B uses `head_dim: 256` across 48 layers with 8 KV heads, giving 375 MB per
    1k tokens - 4.6x the KV footprint of a 9B model with `head_dim: 128`. It fits
    on a card by weight and then cannot hold a 2.5k-token prompt. Reading the
    declared `max_position_embeddings` (262k for that model) would have called
    every cell feasible.
    """

    params_b: float = Field(gt=0, description="Total parameters, MoE included.")
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    bytes_per_param: float = Field(default=0.55, gt=0)
    """0.55 for int4 with quantisation metadata. Recorded because quantisation is
    a recorded experimental variable (PRD §R6.4), not a free choice."""

    @property
    def weight_gb(self) -> float:
        return self.params_b * self.bytes_per_param

    @property
    def kv_mb_per_1k(self) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * 2 * 1000 / 1048576


class ReaderFamily(Strict):
    id: str = Field(min_length=1)
    provider: Literal["local", "anthropic", "openai"]
    model: str = Field(min_length=1)
    version: str = Field(min_length=1, description="Exact snapshot / revision, not a tag.")
    released: str = Field(min_length=1, description="ISO date.")
    role: Literal["primary", "secondary"]
    reader_strength: float = Field(ge=0.0, le=1.0)
    """Emitted as a covariate so R9 can model it (PRD §R7.6). 2606.21807 found
    compression gain DECREASES with reader baseline and flips 31% of pairwise
    rankings, so `U` is reader-strength-dependent, not just prompt-dependent."""
    context_window: int = Field(gt=0)
    rationale: str = Field(min_length=1)
    vram: ReaderVram | None = None
    """Architecture facts for deriving the VRAM ceiling. Optional so a family can
    be declared before its architecture has been measured, but then
    `context_window` is the ONLY ceiling and feasibility will be optimistic."""
    exact_checkpoint_confirmed: bool = False
    """Whether `model` / `version` / `released` are the REAL identifiers.

    Defaults to `False`, so a newly added family cannot ship unnamed by forgetting
    a field. An unconfirmed checkpoint is an ERROR, not a warning (PRD §R7.6): an
    unnamed model makes every number in the paper unattributable, which is the
    exact failure this config layer exists to prevent — and a warning is a thing
    you can ship straight past.
    """

    @model_validator(mode="after")
    def _identifiers_are_real(self) -> ReaderFamily:
        if not self.exact_checkpoint_confirmed:
            raise ValueError(
                f"reader family {self.id!r} has an UNCONFIRMED checkpoint "
                f"(model={self.model!r}, version={self.version!r}). PRD §R7.6 "
                "requires the exact repo id, revision commit, and release date — a "
                "tag can be repointed, which would silently change the weights under "
                "a fixed config_hash. Set the real values and "
                "exact_checkpoint_confirmed: true."
            )
        for name in ("model", "version", "released"):
            value = getattr(self, name).upper()
            if any(token in value for token in ("UNCONFIRMED", "TODO", "TBD", "XXX")):
                raise ValueError(
                    f"reader family {self.id!r}: {name} is the placeholder "
                    f"{getattr(self, name)!r}. PRD §R7.6 requires the real value."
                )
        return self


class ContextPolicy(Strict):
    """What happens when a cell's prompt exceeds a reader's context window.

    PRD §R3.2: a degenerate cell is REJECTED, never silently truncated. A prompt
    that overflows the window means the reader answers a different question than
    the one asked, so the cell is invalid and the grid must be trimmed.
    """

    on_overflow: Literal["reject_cell"]
    never_truncate_silently: bool
    require_context_window_ge_prompt: bool


class Quantisation(Strict):
    """Recorded as an experimental variable, not a free choice.

    A 9B model in bf16 is ~18 GB of weights against 8 GB of VRAM, so bf16 needs
    CPU offload and runs several times slower. Quantisation changes the weights,
    and therefore the answers (PRD §R6.4).
    """

    default_dtype: Literal["4bit", "8bit", "bfloat16", "float16"]
    fallback_dtype: Literal["4bit", "8bit", "bfloat16", "float16"]
    offload_to_cpu: bool
    record_in_manifest: bool


class ReadersConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    families: list[ReaderFamily] = Field(min_length=1)
    context_policy: ContextPolicy
    quantisation: Quantisation

    @field_validator("families")
    @classmethod
    def _distinct_ids(cls, v: list[ReaderFamily]) -> list[ReaderFamily]:
        ids = [f.id for f in v]
        if len(set(ids)) != len(ids):
            raise ValueError("reader family ids must be unique")
        return v

    @model_validator(mode="after")
    def _exactly_one_primary(self) -> ReadersConfig:
        primaries = [f.id for f in self.families if f.role == "primary"]
        if len(primaries) != 1:
            raise ValueError(f"exactly one family must have role 'primary', found {primaries}")
        return self

    @property
    def n_families(self) -> int:
        return len(self.families)


# ── budget.yaml (PRD §R1.3, §R1.7) ────────────────────────────────────────────
class BudgetEnforcement(Strict):
    reject_cell_if_b_ge_corpus: bool = True
    """PRD §R3.4 B-guard: `B >= tokens(corpus)` for a question is degenerate.
    Per-question, never corpus-wide - a corpus-wide percentile passes B on a
    small conversation while the condition is already full-context."""

    reject_cell_if_prompt_exceeds_reader_capacity: bool = True
    """The corpus check above is NOT sufficient on its own.

    A cell can satisfy `B < corpus` and still not fit, because the reader's VRAM
    ceiling for the prompt is smaller than the corpus for every reader on an 8 GB
    card - the weights are resident first, and what is left is 2.5-5.5 GB of KV
    cache. Checking only the corpus marked all 126 cells valid on 2026-10-13 and
    would have failed 14 of them during P5.

    Kept as a declared rule rather than an implicit behaviour so that turning it
    off is a visible, hashed change - and because "which ceiling did you check"
    is the first question anyone should ask of a feasibility report."""


class ScaleSpec(Strict):
    """One corpus scale, with the numbers P1 measured for it (PRD §R1.3)."""

    id: Literal["S", "M"]
    label: str = Field(min_length=1)
    corpus_fraction_target: float = Field(gt=0.0, le=1.0)
    sessions_mean: float = Field(gt=0)
    sessions_median: float = Field(gt=0)
    sessions_min: int = Field(gt=0)
    sessions_max: int = Field(gt=0)
    chars_per_question: int = Field(gt=0)
    factor_vs_reference: float = Field(gt=0)

    @property
    def chars_per_session(self) -> float:
        return self.chars_per_question / self.sessions_mean

    @property
    def sessions_span(self) -> tuple[int, int]:
        return (self.sessions_min, self.sessions_max)


class ScaleRegistry(Strict):
    """The ONE ScaleProfile registry (Architecture §P3.3, PRD §R1.3).

    `factor_vs_reference` is CHECKED against the measured `sessions_mean`s on
    load. Without that check, editing a `sessions_mean` and forgetting the factor
    produces a registry that reads correctly and disagrees with itself — the same
    drift as a stale constant, and the whole reason this file exists.

    The thing being protected is small and easy to state: **the two scales differ
    by an order of magnitude in HOW MANY sessions, and by less than 1% in HOW BIG
    each one is.** So an absolute token budget does not transfer between them, but
    a per-memory budget does — which is why `B` is `per_memory_tokens`.
    """

    schema_version: int = BUNDLE_VERSION
    reference_scale: Literal["S", "M"]
    scales: list[ScaleSpec] = Field(min_length=2)
    reporting: dict[str, Any]

    def get(self, scale_id: str) -> ScaleSpec:
        for s in self.scales:
            if s.id == scale_id:
                return s
        raise KeyError(
            f"scale {scale_id!r} is not registered; known: {[s.id for s in self.scales]}"
        )

    @property
    def scale_factor(self) -> float:
        """How many times bigger the non-reference scale is, in sessions."""
        others = [s for s in self.scales if s.id != self.reference_scale]
        return others[0].factor_vs_reference if others else 1.0

    @model_validator(mode="after")
    def _factor_matches_the_measured_means(self) -> ScaleRegistry:
        ref = next((s for s in self.scales if s.id == self.reference_scale), None)
        if ref is None:
            raise ValueError(f"reference_scale {self.reference_scale!r} is not in `scales`")
        for s in self.scales:
            if s.id == self.reference_scale and s.factor_vs_reference != 1.0:
                raise ValueError(
                    f"reference scale {s.id!r} must have factor_vs_reference = 1.0, "
                    f"got {s.factor_vs_reference}"
                )
            expected = s.sessions_mean / ref.sessions_mean
            if abs(s.factor_vs_reference - expected) > 0.02:
                raise ValueError(
                    f"scale {s.id}: factor_vs_reference={s.factor_vs_reference} does not "
                    f"match sessions_mean / reference ({s.sessions_mean} / "
                    f"{ref.sessions_mean} = {expected:.3f}). Update both together - a "
                    f"registry that disagrees with itself is worse than none, because "
                    f"it reads correctly."
                )
        return self


class BudgetProfile(Strict):
    """The ONE `BudgetProfile` in the repo (PRD §R1.3) — the PROFILE, not the grid.

    It declares the mode vocabulary and the enforcement rule that apply to every
    budget level. It does **NOT** declare the levels themselves: those live in
    `configs/design.yaml` alongside `K_grid`, because the B axis and the K axis are
    one experimental design and belong in one file.

    This file used to carry a `levels:` list as well. It was the same three values
    in a different mode from the design's, and for a while the repo held two
    different B axes simultaneously - `design.yaml` saying `per_memory_tokens`,
    `budget.yaml` saying `absolute_tokens`. The duplicate-grid guard did not catch
    it because it grepped for the key `B_grid` and this file used `levels`. The
    guard is now structural, and this model has no field to restate the grid in.
    """

    schema_version: int = BUNDLE_VERSION
    default_mode: Literal["absolute_tokens", "corpus_fraction", "per_memory_tokens"]
    """Mode applied to any level that does not carry its own. Every level in
    `design.yaml` states its mode explicitly, so this is a fallback rather than a
    default anyone relies on - and `default_mode` here is deliberately NOT a
    statement about the design's B axis."""
    budget_source: Literal["design.yaml"]
    """Explicit pointer, so a reader looking for the B levels in this file is
    directed instead of left to infer."""
    enforcement: BudgetEnforcement


# ── budget_guard.yaml (PRD §R0.5, §N4, §N8) ───────────────────────────────────
class BudgetGuardConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    max_usd: float | None = Field(default=None, ge=0)
    max_usd_status: Literal["set", "pending_guide_signoff"] = "pending_guide_signoff"
    disk_budget_bytes: int = Field(gt=0)
    disk_budget_status: Literal["estimated", "confirmed"] = "estimated"
    max_concurrency: int = Field(gt=0)
    rate_limit_rpm: int = Field(gt=0)
    retry_max_attempts: int = Field(gt=0)

    @model_validator(mode="after")
    def _usd_presence_matches_status(self) -> BudgetGuardConfig:
        if self.max_usd_status == "set" and self.max_usd is None:
            raise ValueError("max_usd_status is 'set' but max_usd is null")
        if self.max_usd_status == "pending_guide_signoff" and self.max_usd is not None:
            raise ValueError(
                "max_usd_status is 'pending_guide_signoff' but max_usd is set; "
                "an estimate must not masquerade as an approved ceiling"
            )
        return self


# ── arms.yaml (PRD §R4) ───────────────────────────────────────────────────────
class ArmSpec(Strict):
    id: str = Field(min_length=1)
    isolates: str = Field(min_length=1)
    enabled: bool = True
    prompt_template: str = Field(min_length=1)
    notes: str = ""


class CrossedGridSpec(Strict):
    """PRD §R4.4 — arms A and G are ONE crossed grid, not two sweeps.

    Declared in config so the crossed design is *data*, not an emergent property
    of calling two functions in a loop. The condition key carries `B_mode`
    because a prefix cached at B=4K must never be served at B=16K (PRD §R6.5,
    hazard H4).
    """

    arms: list[str] = Field(min_length=2)
    key: list[str] = Field(min_length=1)
    notes: str = ""


class ArmsConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    arms: list[ArmSpec] = Field(min_length=1)
    crossed_grid: CrossedGridSpec

    @field_validator("arms")
    @classmethod
    def _unique_ids(cls, v: list[ArmSpec]) -> list[ArmSpec]:
        ids = [a.id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError("arm ids must be unique")
        return v

    @model_validator(mode="after")
    def _crossed_grid_members_exist(self) -> ArmsConfig:
        known = {a.id for a in self.arms}
        unknown = [a for a in self.crossed_grid.arms if a not in known]
        if unknown:
            raise ValueError(
                f"crossed_grid references undeclared arms {unknown}; declared: {sorted(known)}"
            )
        if set(self.crossed_grid.arms) != {"A", "G"}:
            raise ValueError(
                f"crossed_grid must join arms A and G, got {self.crossed_grid.arms} (PRD §R4.4)"
            )
        return self


# ── the bundle ────────────────────────────────────────────────────────────────
class PreregistrationConfig(Strict):
    """The pre-registered hypothesis (PRD §R0.4).

    The sign prediction lives here and its SHA-256 is stored in `power.json`, so
    the claim "this was registered in advance" rests on a hash pinned into the
    same artifact that reports the estimate — not on a promise.
    """

    schema_version: int = BUNDLE_VERSION
    registered: str = Field(min_length=1, description="ISO date, before data collection.")
    registered_by: str = Field(min_length=1)
    status: Literal["pending_guide_signoff", "signed"]
    primary_endpoint: dict[str, Any]
    interaction_sign_prediction: Literal["approximately_zero", "negative", "positive"]
    direction: float
    sign_readings: dict[str, str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    not_claimed: list[str] = Field(default_factory=list)
    secondary_endpoints: list[str] = Field(default_factory=list)
    multiplicity: dict[str, Any]


class ConfigBundle(BaseModel):
    """The resolved, validated configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    design: DesignConfig
    retrieval: RetrievalConfig
    decoding: DecodingConfig
    readers: ReadersConfig
    scales: ScaleRegistry
    budget: BudgetProfile
    arms: ArmsConfig
    budget_guard: BudgetGuardConfig
    preregistration: PreregistrationConfig
    prompts: dict[str, str]
    pricing: dict[str, Any]
    raw: dict[str, Any]
    config_hash: str
    preregistration_hash: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "K_grid": self.design.K_grid,
            "B_grid": [(b.value, b.mode) for b in self.design.B_grid],
            "scales": self.design.scales,
            "readers": self.readers.families and [f.id for f in self.readers.families],
            "unit": self.retrieval.unit,
            "embedder": self.retrieval.embedder.id,
            "n_cells": self.design.n_cells,
            "runs_per_cell": self.design.runs_per_cell,
        }


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(raw: dict[str, Any]) -> str:
    """SHA-256 over the canonicalised resolved bundle (PRD §R1.4).

    Key order in the YAML does not change the hash — reordering keys is not a
    configuration change. A single-character edit to any prompt template *is*.
    """
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_bundle(root: Path) -> ConfigBundle:
    """Load and validate every config file. Raises on a hard structural error."""
    import pydantic

    raw = _collect_raw(root)
    try:
        bundle = ConfigBundle(
            root=root,
            design=DesignConfig.model_validate(raw["design.yaml"]),
            retrieval=RetrievalConfig.model_validate(raw["retrieval.yaml"]),
            decoding=DecodingConfig.model_validate(raw["decoding.yaml"]),
            readers=ReadersConfig.model_validate(raw["readers.yaml"]),
            scales=ScaleRegistry.model_validate(raw["scales.yaml"]),
            budget=BudgetProfile.model_validate(raw["budget.yaml"]),
            arms=ArmsConfig.model_validate(raw["arms.yaml"]),
            budget_guard=BudgetGuardConfig.model_validate(raw["budget_guard.yaml"]),
            preregistration=PreregistrationConfig.model_validate(raw["preregistration.yaml"]),
            prompts={k: v for k, v in raw.items() if k.endswith(".j2")},
            pricing=raw["pricing/latest.yaml"],
            raw=raw,
            config_hash=compute_config_hash(raw),
        )
    except pydantic.ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc
    # PRD §R0.4: the pre-registration gets its OWN hash, separate from the bundle,
    # so "the prediction was registered in advance" stays verifiable even if an
    # unrelated config file is edited afterwards.
    object.__setattr__(
        bundle, "preregistration_hash", compute_config_hash(raw["preregistration.yaml"])
    )
    return bundle


def _collect_raw(root: Path) -> dict[str, Any]:
    """Read every bundle file into a plain dict, keyed by bundle-relative path."""
    raw: dict[str, Any] = {}
    for name in BUNDLE_FILES:
        path = root / name
        if not path.is_file():
            raise ConfigError(f"missing config file: {name}")
        raw[name] = (
            _read_yaml(path)
            if path.suffix in (".yaml", ".yml")
            else path.read_text(encoding="utf-8")
        )
    return raw


class ConfigError(RuntimeError):
    """The configuration is unusable — a structural error, not a warning."""


def format_validation_error(exc: Any) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "configuration is invalid:\n" + "\n".join(lines)


# ── validate-config (PRD §R1.5, §R7.5) ────────────────────────────────────────
def validate(root: Path) -> list[Problem]:
    """Full validation. Never raises — returns every problem it can find.

    Reporting *all* problems in one pass matters: a harness that stops at the
    first failure turns a ten-minute fix into ten round-trips.
    """
    problems: list[Problem] = []
    try:
        raw = _collect_raw(root)
    except ConfigError as exc:
        return [Problem(severity=Severity.ERROR, code="CFG_MISSING_FILE", message=str(exc))]

    models: dict[str, tuple[type[Strict], str]] = {
        "design.yaml": (DesignConfig, "R0.1/R1.3"),
        "retrieval.yaml": (RetrievalConfig, "R1.1"),
        "decoding.yaml": (DecodingConfig, "R7.1-R7.4"),
        "readers.yaml": (ReadersConfig, "R7.5"),
        "budget.yaml": (BudgetProfile, "R1.7"),
        "scales.yaml": (ScaleRegistry, "R1.3/P3.3"),
        "arms.yaml": (ArmsConfig, "R4"),
        "budget_guard.yaml": (BudgetGuardConfig, "N4/N8"),
        "preregistration.yaml": (PreregistrationConfig, "R0.4"),
    }
    parsed: dict[str, Any] = {}
    for name, (model, reqs) in models.items():
        try:
            parsed[name] = model.model_validate(raw[name])
        except Exception as exc:  # noqa: BLE001 - pydantic.ValidationError and its subclasses
            # Deliberately broad: a validator that only catches the exception type it
            # expected would abort on the first surprise instead of reporting every
            # problem, which is the behaviour that turns a ten-minute fix into ten
            # round-trips. `format_validation_error` reads `.errors()`, so a
            # non-pydantic exception degrades to a one-line message rather than a
            # traceback.
            for line in format_validation_error(exc).splitlines():
                problems.append(
                    Problem(
                        severity=Severity.ERROR,
                        code="CFG_INVALID",
                        message=line.strip(),
                        where=f"{name} ({reqs})",
                    )
                )

    problems.extend(_check_nulls(root, problems))
    if "readers.yaml" in parsed:
        problems.extend(_check_readers(parsed["readers.yaml"]))
    if "design.yaml" in parsed:
        problems.extend(_check_design(parsed["design.yaml"]))
    if "budget_guard.yaml" in parsed:
        problems.extend(_check_budget_guard(parsed["budget_guard.yaml"]))
    problems.extend(_check_prompts(raw))
    problems.extend(_check_single_grid_definition(root))
    return problems


def _check_nulls(root: Path, already: list[Problem]) -> list[Problem]:
    """PRD §R1.1: no field in `retrieval.yaml` may be null.

    Scanned structurally rather than via the model, because the model's own
    validators cannot distinguish "field absent" from "field present and null"
    once a default exists.
    """
    out: list[Problem] = []
    path = root / "retrieval.yaml"
    if not path.is_file():
        return out

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{trail}[{i}]")
        elif node is None:
            out.append(
                Problem(
                    severity=Severity.ERROR,
                    code="CFG_NULL_FIELD",
                    message="field is null; PRD §R1.1 forbids it",
                    where=f"retrieval.yaml:{trail}",
                )
            )

    walk(yaml.safe_load(path.read_text(encoding="utf-8")), "")
    return out


def _check_readers(readers: ReadersConfig) -> list[Problem]:
    out: list[Problem] = []
    if readers.n_families < 2:
        out.append(
            Problem(
                severity=Severity.ERROR,
                code="CFG_TOO_FEW_READERS",
                message=(
                    f"{readers.n_families} reader family declared; PRD §R7.5 requires >= 2. "
                    "2606.21807 found compression gain falls with reader baseline and "
                    "flips 31% of pairwise rankings, so a depth curve at one reader "
                    "is not attributable."
                ),
                where="readers.yaml (R7.5)",
            )
        )
    strengths = [f.reader_strength for f in readers.families]
    if strengths and max(strengths) - min(strengths) < 0.05:
        out.append(
            Problem(
                severity=Severity.WARN,
                code="CFG_READER_STRENGTH_FLAT",
                message=(
                    f"reader_strength spans only {max(strengths) - min(strengths):.3f}. "
                    "PRD §R7.6 makes it a modelled covariate, and 2606.21807's finding "
                    "is that compression gain shrinks as the reader gets stronger — a "
                    "panel of near-identical readers cannot test reader-strength "
                    "dependence, only seed noise."
                ),
                where="readers.yaml (R7.6)",
            )
        )
    return out


def _check_design(design: DesignConfig) -> list[Problem]:
    out: list[Problem] = []
    modes = {b.mode for b in design.B_grid}
    if len(modes) > 1:
        out.append(
            Problem(
                severity=Severity.WARN,
                code="CFG_MIXED_B_MODES",
                message=(
                    f"B_grid mixes {sorted(modes)}. PRD §R1.7 requires the mode to be "
                    "recorded per condition; mixing modes silently invalidates the "
                    "interaction, because `log2(B)` stops meaning one thing across the grid."
                ),
                where="design.yaml (R1.7)",
            )
        )
    if max(design.K_grid) > 64:
        out.append(
            Problem(
                severity=Severity.WARN,
                code="CFG_LARGE_K",
                message=f"K_grid reaches {max(design.K_grid)}; the per-question K-guard (P3) "
                "may reject whole cells on the S split",
                where="design.yaml (R3.1)",
            )
        )
    return out


def _check_budget_guard(guard: BudgetGuardConfig) -> list[Problem]:
    if guard.max_usd_status != "set":
        return [
            Problem(
                severity=Severity.WARN,
                code="CFG_BUDGET_UNSET",
                message=(
                    "budget.max_usd is not set (blocker B7). `design-power` will still emit a "
                    "cost ESTIMATE — that estimate is the input to the budget conversation — "
                    "but the hard cost guard cannot be enforced and no run may be launched."
                ),
                where="budget_guard.yaml (R0.5/N8)",
            )
        ]
    return []


def _check_prompts(raw: dict[str, Any]) -> list[Problem]:
    """PRD §R1.2: prompts hold the templates, the abstention rule, and the
    answer-normalisation function — not code."""
    out: list[Problem] = []
    for key in ("prompts/abstention.yaml", "prompts/normalisation.yaml"):
        if key not in raw:
            out.append(
                Problem(
                    severity=Severity.ERROR,
                    code="CFG_MISSING_PROMPT_RULE",
                    message=f"{key} is required; PRD §R1.2 keeps scoring rules in config",
                    where="prompts/",
                )
            )

    norm = (raw.get("prompts/normalisation.yaml") or {}).get("normalisation") or {}
    if not norm.get("steps"):
        out.append(
            Problem(
                severity=Severity.ERROR,
                code="CFG_NO_NORMALISATION",
                message="normalisation.steps must be a non-empty ordered list; token-F1 "
                "depends on a documented, reproducible normalisation (PRD §R8.1)",
                where="prompts/normalisation.yaml (R1.2)",
            )
        )

    abst = (raw.get("prompts/abstention.yaml") or {}).get("abstention") or {}
    if not abst.get("scoring", {}).get("decline_markers"):
        out.append(
            Problem(
                severity=Severity.ERROR,
                code="CFG_NO_ABSTENTION_RULE",
                message="abstention.scoring.decline_markers is empty; the `_abs` rule must "
                "be declared in config, not improvised in code (PRD §R1.2, §R8.4)",
                where="prompts/abstention.yaml (R1.2)",
            )
        )

    for key, text in raw.items():
        if key.endswith(".j2") and "{{" not in str(text):
            out.append(
                Problem(
                    severity=Severity.WARN,
                    code="CFG_TEMPLATE_NO_VARS",
                    message=f"{key} has no Jinja placeholders; it may be a constant prompt",
                    where=key,
                )
            )
    return out


def _declares_budget_levels(node: object) -> bool:
    """True if this subtree contains a `{value, mode}` pair - i.e. a B level.

    Detected STRUCTURALLY, on the shape of the data, not on the key name.

    The previous version of this check grepped for the literal key `B_grid`. That
    caught `design.yaml` being edited twice and caught nothing else: `budget.yaml`
    declared the same budget levels under the key `levels`, sailed past the guard,
    and the repo ended up with two different B axes in two different modes at the
    same time. A guard that only recognises the name of the thing it forbids is a
    guard that gets bypassed the first time someone picks a different name.

    `{value, mode}` is specific enough to be safe: it is the `BLevel` shape, and
    nothing else in `configs/` uses both keys together.
    """
    if isinstance(node, dict):
        if "value" in node and "mode" in node:
            return True
        return any(_declares_budget_levels(v) for v in node.values())
    if isinstance(node, list):
        return any(_declares_budget_levels(v) for v in node)
    return False


#: Keys that name a design GRID. A grid is a list of SCALARS - the values the
#: experiment runs. A list of OBJECTS under the same key is a PROFILE: it describes
#: what a scale is, not which scales run. `scales.yaml` is the latter and must not
#: be flagged; conflating the two is how a correct guard turns into a nuisance and
#: gets disabled.
_GRID_KEYS = ("K_grid", "B_grid", "scales", "readers")


def _declares_grid(node: object, key: str) -> bool:
    """Does this mapping declare the design grid under `key`?"""
    if not isinstance(node, dict):
        return False
    value = node.get(key)
    if not isinstance(value, list) or not value:
        return False
    if key in ("K_grid", "readers"):
        return all(isinstance(v, (int, str)) for v in value)
    if key == "scales":
        return all(isinstance(v, str) for v in value)
    return False


def _check_single_grid_definition(root: Path) -> list[Problem]:
    """PRD §R0.1: `design.yaml` is the *only* place a grid may be declared.

    Enforced on the SHAPE of the data, not the name of the key. Key-name matching
    both under- and over-fires: it missed `budget.yaml`'s `levels:`, and it would
    flag `scales.yaml`'s registry of scale DESCRIPTIONS, which is a profile and
    not a second grid.
    """
    import yaml

    offenders: list[str] = []
    for path in sorted(root.rglob("*.yaml")):
        if path.name == "design.yaml" and path.parent == root:
            continue
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        rel = str(path.relative_to(root))
        for key in _GRID_KEYS:
            if _declares_grid(parsed, key):
                offenders.append(f"{rel}: declares the grid under `{key}`")
        if _declares_budget_levels(parsed):
            offenders.append(f"{rel}: declares budget levels ({{value, mode}})")
    if offenders:
        return [
            Problem(
                severity=Severity.ERROR,
                code="CFG_DUPLICATE_GRID",
                message=(
                    "grid declared outside design.yaml: "
                    + ", ".join(sorted(set(offenders)))
                    + ". PRD §R0.1 makes configs/design.yaml the single source; the "
                    "levels live there and other files declare only PROFILES "
                    "(descriptions of a thing, not the list of things run)."
                ),
                where="configs/",
            )
        ]
    return []


def errors_only(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity is Severity.ERROR]
