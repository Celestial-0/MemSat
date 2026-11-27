"""Configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BUNDLE_VERSION = 1

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
    model_config = ConfigDict(frozen=True)

    severity: Severity
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        location = f" [{self.where}]" if self.where else ""
        return f"{self.severity.value.upper():5s} {self.code}: {self.message}{location}"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


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


class TokenizerConfig(Strict):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
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


class BLevel(Strict):
    value: float = Field(gt=0)
    mode: Literal["absolute_tokens", "corpus_fraction", "per_memory_tokens"]


class ItemsConfig(Strict):
    include_abstention: bool = True
    limit: int | None = Field(default=None, ge=1)


class ReferenceSpec(Strict):
    K: int = Field(gt=0)
    B: float = Field(gt=0)


class DesignConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    K_grid: list[int] = Field(min_length=2)
    B_grid: list[BLevel] = Field(min_length=2)
    scales: list[Literal["S", "M"]] = Field(min_length=1)
    readers: list[str] = Field(min_length=2, description="PRD §R7.5: >=2 families, enforced.")
    items: ItemsConfig
    repetition: int = Field(ge=1)
    seed: int
    reference: ReferenceSpec

    @property
    def runs_per_cell(self) -> int:
        return self.repetition

    @model_validator(mode="before")
    @classmethod
    def _sync_repetition_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "repetition" not in data and "runs_per_cell" in data:
                data["repetition"] = data.pop("runs_per_cell")
            elif "runs_per_cell" in data:
                data.pop("runs_per_cell")
        return data

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
    def n_conditions(self) -> int:
        return len(self.K_grid) * len(self.B_grid) * len(self.scales) * len(self.readers)

    @property
    def n_cells(self) -> int:
        return self.n_conditions


class DecodingConfig(Strict):
    schema_version: int = BUNDLE_VERSION
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)
    stop_sequences: list[str]
    provider_seed: int
    reasoning_budget_tokens: int = Field(ge=0)
    reasoning_budget_rationale: str = Field(min_length=1)
    prompt_nonce: bool = False

    @model_validator(mode="after")
    def _seeds_must_vary_something(self) -> DecodingConfig:
        if self.temperature == 0.0 and not self.prompt_nonce:
            raise ValueError(
                "temperature is 0 and prompt_nonce is false: repeat runs would be "
                "byte-identical and PRD §R6.1 would dedupe them to one cached file, "
                "making seed-variance estimation unachievable. Set temperature > 0 "
                "or enable prompt_nonce (PRD §R7.1)."
            )
        return self


class ReaderVram(Strict):
    params_b: float = Field(gt=0, description="Total parameters, MoE included.")
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    bytes_per_param: float = Field(default=0.55, gt=0)

    @property
    def weight_gb(self) -> float:
        return self.params_b * self.bytes_per_param * 1e9 / (1024**3)

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
    context_window: int = Field(gt=0)
    rationale: str = Field(min_length=1)
    vram: ReaderVram | None = None
    exact_checkpoint_confirmed: bool = False

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
    on_overflow: Literal["reject_cell"]
    never_truncate_silently: bool
    require_context_window_ge_prompt: bool


class Quantisation(Strict):
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


class BudgetEnforcement(Strict):
    reject_cell_if_b_ge_corpus: bool = True

    reject_cell_if_prompt_exceeds_reader_capacity: bool = True


class ScaleSpec(Strict):
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
    schema_version: int = BUNDLE_VERSION
    default_mode: Literal["absolute_tokens", "corpus_fraction", "per_memory_tokens"]
    budget_source: Literal["design.yaml"]
    enforcement: BudgetEnforcement


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


class ArmSpec(Strict):
    id: str = Field(min_length=1)
    isolates: str = Field(min_length=1)
    enabled: bool = True
    prompt_template: str = Field(min_length=1)
    notes: str = ""


class CrossedGridSpec(Strict):
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


class PreregistrationConfig(Strict):
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
    canonical_config_hash: str | None = None
    effective_config_hash: str | None = None
    mode: str = "full"

    def summary(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "K_grid": self.design.K_grid,
            "B_grid": [(b.value, b.mode) for b in self.design.B_grid],
            "scales": self.design.scales,
            "readers": self.readers.families and [f.id for f in self.readers.families],
            "unit": self.retrieval.unit,
            "embedder": self.retrieval.embedder.id,
            "n_conditions": self.design.n_conditions,
            "n_cells": self.design.n_cells,
            "repetition": self.design.repetition,
            "runs_per_cell": self.design.runs_per_cell,
        }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(raw: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def deep_merge(base: Any, override: Any) -> Any:
    """Deep merge dictionaries and list-of-dicts keyed by 'id'."""
    if isinstance(base, dict) and isinstance(override, dict):
        res = dict(base)
        for k, v in override.items():
            if k in res:
                res[k] = deep_merge(res[k], v)
            else:
                res[k] = v
        return res
    if isinstance(base, list) and isinstance(override, list):
        if base and isinstance(base[0], dict) and "id" in base[0]:
            by_id = {item["id"]: item for item in base}
            for item in override:
                if isinstance(item, dict) and "id" in item and item["id"] in by_id:
                    by_id[item["id"]] = deep_merge(by_id[item["id"]], item)
                else:
                    target_id = item.get("id", len(by_id)) if isinstance(item, dict) else len(by_id)
                    by_id[target_id] = item
            return list(by_id.values())
        return override
    return override


def _find_canonical_root(root: Path) -> Path:
    candidates = [
        root.parent.parent / "configs",
        root.parent.parent.parent / "configs",
        root.parent / "configs",
        Path("configs"),
        Path(__file__).resolve().parent.parent.parent / "configs",
    ]
    for c in candidates:
        if c.is_dir() and (c / "design.yaml").is_file():
            return c
    raise ConfigError(f"could not locate canonical configs directory for profile {root}")


def _resolve_profile_dir(root: Path) -> Path | None:
    if root.is_file() and root.name in ("overrides.yaml", "overrides.yml"):
        return root.parent
    if root.is_dir() and (
        (root / "overrides.yaml").is_file() or (root / "overrides.yml").is_file()
    ):
        return root
    if root.name == "configs" and (
        (root.parent / "overrides.yaml").is_file() or (root.parent / "overrides.yml").is_file()
    ):
        return root.parent
    return None


def load_bundle(root: Path) -> ConfigBundle:
    import pydantic

    raw = _collect_raw(root)
    profile_dir = _resolve_profile_dir(root)
    if profile_dir is not None:
        canonical_dir = _find_canonical_root(profile_dir)
        canonical_raw = _collect_raw(canonical_dir)
        canonical_hash = compute_config_hash(canonical_raw)
    else:
        canonical_hash = compute_config_hash(raw)

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
    object.__setattr__(
        bundle, "preregistration_hash", compute_config_hash(raw["preregistration.yaml"])
    )
    object.__setattr__(bundle, "canonical_config_hash", canonical_hash)
    object.__setattr__(bundle, "effective_config_hash", bundle.config_hash)
    return bundle


def _collect_raw(root: Path) -> dict[str, Any]:
    profile_dir = _resolve_profile_dir(root)
    if profile_dir is not None:
        canonical_dir = _find_canonical_root(profile_dir)
        raw_base = _collect_raw(canonical_dir)
        overrides_path = profile_dir / "overrides.yaml"
        if not overrides_path.is_file():
            overrides_path = profile_dir / "overrides.yml"
        overrides = _read_yaml(overrides_path) or {}
        return cast(dict[str, Any], deep_merge(raw_base, overrides))

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
    """The configuration is unusable: a structural error, not a warning."""


def format_validation_error(exc: Any) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "configuration is invalid:\n" + "\n".join(lines)


def validate(root: Path) -> list[Problem]:
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
                    "budget.max_usd is not set (blocker B7). `estimate-power` will still emit a "
                    "cost ESTIMATE — that estimate is the input to the budget conversation — "
                    "but the hard cost guard cannot be enforced and no run may be launched."
                ),
                where="budget_guard.yaml (R0.5/N8)",
            )
        ]
    return []


def _check_prompts(raw: dict[str, Any]) -> list[Problem]:
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


def declares_budget_levels(node: object) -> bool:
    if isinstance(node, dict):
        if "value" in node and "mode" in node:
            return True
        return any(declares_budget_levels(v) for v in node.values())
    if isinstance(node, list):
        return any(declares_budget_levels(v) for v in node)
    return False


_GRID_KEYS = ("K_grid", "B_grid", "scales", "readers")


def _declares_grid(node: object, key: str) -> bool:
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
        if declares_budget_levels(parsed):
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
