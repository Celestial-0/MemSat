"""The judge panel."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from msat.arms.evidence import ABSTENTION_SUFFIX
from msat.arms.render import render_template
from msat.hardware.vram import ModelShape
from msat.judges.kappa import CAUTIONARY_KAPPA, panel_agreement
from msat.judges.verdicts import (
    ItemKey,
    JudgeVerdict,
    UnscorableError,
    Verdict,
    VerdictSet,
    as_verdict,
)
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    LocalBackend,
    select_backend,
)
from msat.providers.local import ReaderVramSpec

JUDGE_PROMPT_TEMPLATE = "prompts/judge.yaml"

JUDGE_ABSTENTION_PROMPT_TEMPLATE = "prompts/judge_abstention.yaml"

ABSTENTION_RULE_KEY = "abstention"

CARD_GB_DEFAULT = 8.0

JUDGE_TEMPERATURE = 0.0

JUDGE_TOP_P = 1.0

JUDGE_SEED = 20261012

JUDGE_STOP: tuple[str, ...] = ("\n\n",)

JUDGE_INPUT_TOKENS_APPROX = 4000

HEADROOM_GB = 3.0

UNRECOGNISED_LINEAGE = "unrecognised"

MIN_JUDGES = 2

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

_PLACEHOLDERS = ("UNCONFIRMED", "TODO", "TBD", "XXX")

MODEL_FAMILIES: dict[str, str] = {
    "gemma": "gemma",
    "qwen": "qwen",
    "phi": "phi",
    "mistral": "mistral",
    "mixtral": "mistral",
    "granite": "granite",
    "nemotron": "nemotron",
    "llama": "llama",
    "gpt": "gpt",
}


class PanelSpecError(ValueError):
    """A judge is declared with something missing, null, or unconfirmable."""


class JudgeVramError(PanelSpecError):
    """A judge cannot fit the card even on its own."""


class JudgeIndependenceError(ValueError):
    """A judge cannot independently adjudicate a panel it shares weights with."""


class JudgeSequentialError(RuntimeError):
    """Two judges were resident at once; the panel must load them sequentially."""


class JudgeBackendUnavailableError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reasons: Mapping[str, str] | None = None,
        run: PanelRun | None = None,
        readiness: Sequence[JudgeReadiness] | None = None,
    ) -> None:
        super().__init__(message)
        self.reasons: dict[str, str] = dict(reasons or {})
        self.run = run
        self.readiness: list[JudgeReadiness] = list(readiness or [])


class PanelKappaBelowBarError(RuntimeError):
    def __init__(self, message: str, *, agreement: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.agreement: dict[str, Any] = dict(agreement)
        self.bar: float = CAUTIONARY_KAPPA


@dataclass(frozen=True, slots=True)
class JudgeDecoding:
    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int

    def seed_for_pass(self, pass_index: int) -> int:
        if pass_index < 1:
            raise ValueError(f"pass_index must be >= 1, got {pass_index}")
        return self.seed + (pass_index - 1)

    def request(
        self,
        model: str,
        system: str,
        prompt: str,
        *,
        pass_index: int,
        request_id: str,
    ) -> GenerationRequest:
        return GenerationRequest(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
            seed=self.seed_for_pass(pass_index),
            request_id=request_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    id: str
    model: str
    version: str
    released: str
    prompt_template: str
    abstention_prompt_template: str
    decoding: JudgeDecoding
    vram: ReaderVramSpec
    rationale: str
    exact_checkpoint_confirmed: bool
    backend_name: str = "transformers"

    @property
    def family(self) -> str | None:
        return model_family(self.model)

    @property
    def shape(self) -> ModelShape:
        return self.vram.to_shape(name=self.model)

    @property
    def weights_gb(self) -> float:
        return self.shape.weights_gb(quant_bits=self.vram.bytes_per_param * 8.0)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.shape.kv_bytes_per_token()

    def kv_gb(self, n_tokens: int) -> float:
        return self.kv_bytes_per_token * n_tokens / (1024**3)

    def footprint_gb(self, n_tokens: int = JUDGE_INPUT_TOKENS_APPROX) -> float:
        return self.weights_gb + self.kv_gb(n_tokens)

    def template_for(self, item: JudgeItem) -> str:
        return self.abstention_prompt_template if item.is_abstention else self.prompt_template

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "version": self.version,
            "released": self.released,
            "family": self.family or UNRECOGNISED_LINEAGE,
            "prompt_template": self.prompt_template,
            "abstention_prompt_template": self.abstention_prompt_template,
            "decoding": self.decoding.as_dict(),
            "backend": self.backend_name,
            "weights_gb": round(self.weights_gb, 3),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "footprint_gb": round(self.footprint_gb(), 3),
            "exact_checkpoint_confirmed": self.exact_checkpoint_confirmed,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class JudgeItem:
    condition_key: str
    qid: str
    question: str
    answer: str
    gold: str = ""

    @property
    def key(self) -> ItemKey:
        return (self.condition_key, self.qid)

    @property
    def is_abstention(self) -> bool:
        return self.qid.endswith(ABSTENTION_SUFFIX)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> JudgeItem:
        return cls(
            condition_key=str(row["condition_key"]),
            qid=str(row["qid"]),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            gold=str(row.get("gold", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "qid": self.qid,
            "is_abstention": self.is_abstention,
        }


@dataclass(frozen=True, slots=True)
class UnscoredRow:
    condition_key: str
    qid: str
    judge_id: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "qid": self.qid,
            "judge_id": self.judge_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DisagreementRow:
    condition_key: str
    qid: str
    judge_a: str
    verdict_a: Verdict
    judge_b: str
    verdict_b: Verdict
    is_abstention: bool
    answer: str
    gold: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "qid": self.qid,
            "judge_a": self.judge_a,
            "verdict_a": str(self.verdict_a),
            "judge_b": self.judge_b,
            "verdict_b": str(self.verdict_b),
            "is_abstention": self.is_abstention,
            "answer": self.answer,
            "gold": self.gold,
        }


@dataclass(frozen=True, slots=True)
class UnavailableRow:
    judge_id: str
    model: str
    reason: str
    n_items_affected: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "model": self.model,
            "reason": self.reason,
            "n_items_affected": self.n_items_affected,
        }


@dataclass
class PanelRun:
    verdicts: VerdictSet = field(default_factory=VerdictSet)
    disagreements: list[DisagreementRow] = field(default_factory=list)
    unscored: list[UnscoredRow] = field(default_factory=list)
    unavailable: list[UnavailableRow] = field(default_factory=list)
    n_items: int = 0
    pass_index: int = 1
    execution_order: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    resident_peak_gb: float = 0.0

    @property
    def n_judges(self) -> int:
        return len(self.verdicts.judges())

    @property
    def n_abstention(self) -> int:
        return len(
            {
                (v.condition_key, v.qid)
                for v in self.verdicts.verdicts
                if v.qid.endswith(ABSTENTION_SUFFIX)
            }
        )

    def verdicts_by_judge(self) -> dict[str, dict[ItemKey, Verdict]]:
        return {
            j: self.verdicts.for_judge(j, pass_index=self.pass_index)
            for j in self.verdicts.judges()
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "n_items": self.n_items,
            "n_judges": self.n_judges,
            "n_verdicts": len(self.verdicts.verdicts),
            "execution_order": list(self.execution_order),
            "verdicts": self.verdicts.as_dicts(),
            "disagreements": [d.as_dict() for d in self.disagreements],
            "unscored": [u.as_dict() for u in self.unscored],
            "unavailable": [u.as_dict() for u in self.unavailable],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_seconds": round(self.wall_seconds, 3),
            "resident_peak_gb": round(self.resident_peak_gb, 3),
            "retention_note": (
                "One row per (item, judge, pass). No accuracy scalar is returned on "
                "purpose: PRD R8.3 retains verdicts per item, and an aggregate is "
                "recomputed from these rows under any metric, exclusion rule or "
                "normalisation - a stored one cannot be."
            ),
        }


@dataclass(frozen=True, slots=True)
class JudgeReadiness:
    judge_id: str
    model: str
    version: str
    family: str
    weights_gb: float
    footprint_gb: float
    fits_alone: bool
    ready: bool
    reason: str
    remedy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "model": self.model,
            "version": self.version,
            "family": self.family,
            "weights_gb": round(self.weights_gb, 3),
            "footprint_gb": round(self.footprint_gb, 3),
            "fits_alone": self.fits_alone,
            "ready": self.ready,
            "reason": self.reason,
            "remedy": self.remedy,
        }


@dataclass
class PanelHealth:
    readiness: list[JudgeReadiness] = field(default_factory=list)
    agreement: dict[str, Any] = field(default_factory=dict)
    n_items: int = 0
    pass_index: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.agreement.get("panel_usable", False))

    @property
    def kappa_min(self) -> float | None:
        value = self.agreement.get("kappa_min")
        return None if value is None else float(value)

    @property
    def blocked(self) -> list[JudgeReadiness]:
        return [r for r in self.readiness if not r.ready or not r.fits_alone]

    def headline(self) -> dict[str, Any]:
        return {
            "panel_usable": self.usable,
            "kappa_min": self.kappa_min,
            "cautionary_bar": CAUTIONARY_KAPPA,
            "n_judges": len(self.readiness),
            "n_items": self.n_items,
            "n_blocked_judges": len(self.blocked),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline(),
            "judges": [r.as_dict() for r in self.readiness],
            "panel_agreement": self.agreement,
            "bar_source": "2511.02246, carried as kappa.CAUTIONARY_KAPPA",
            "notes": list(self.notes),
        }


def _default_loader(spec: JudgeSpec) -> LocalBackend:
    return select_backend(spec.backend_name)


@dataclass(frozen=True, slots=True)
class JudgePanel:
    judges: tuple[JudgeSpec, ...]
    templates: Mapping[str, str]
    abstention_rule: Mapping[str, Any]
    reader_families: tuple[str, ...]
    load: Callable[[JudgeSpec], LocalBackend] = _default_loader
    card_gb: float = CARD_GB_DEFAULT
    headroom_gb: float = HEADROOM_GB
    _resident: list[JudgeSpec] = field(default_factory=list, repr=False, compare=False)

    @property
    def judge_ids(self) -> tuple[str, ...]:
        return tuple(j.id for j in self.judges)

    @property
    def n_judges(self) -> int:
        return len(self.judges)

    @property
    def weights_gb(self) -> float:
        return sum(j.weights_gb for j in self.judges)

    @property
    def resident_peak_gb(self) -> float:
        return max((j.footprint_gb() for j in self.judges), default=0.0)

    @property
    def fits_concurrently(self) -> bool:
        return bool(self.judges) and self.weights_gb + self.headroom_gb <= self.card_gb

    def why_sequential(self) -> str:
        if self.fits_concurrently:
            return (
                f"The declared weights total {self.weights_gb:.2f} GB, which would fit "
                f"{self.card_gb:.1f} GB alongside {self.headroom_gb:.1f} GB of headroom. "
                f"This panel still runs SEQUENTIALLY: the invariant is structural, and "
                f"relaxing it is a hardware decision that must be made with a measured "
                f"free-VRAM reading (D11), not inferred from a config."
            )
        return (
            f"The declared weights total {self.weights_gb:.2f} GB against a "
            f"{self.card_gb:.1f} GB card with {self.headroom_gb:.1f} GB of headroom, "
            f"before any KV cache. The panel therefore runs ONE judge at a time and "
            f"swaps; peak residency is {self.resident_peak_gb:.2f} GB. Concurrent "
            f"loading is not a slower path here, it is an OOM."
        )

    def get(self, judge_id: str) -> JudgeSpec:
        for j in self.judges:
            if j.id == judge_id:
                return j
        raise KeyError(f"unknown judge {judge_id!r}; the panel is {list(self.judge_ids)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "judges": [j.as_dict() for j in self.judges],
            "reader_families": list(self.reader_families),
            "card_gb": self.card_gb,
            "headroom_gb": self.headroom_gb,
            "weights_gb_total": round(self.weights_gb, 3),
            "resident_peak_gb": round(self.resident_peak_gb, 3),
            "fits_concurrently": self.fits_concurrently,
            "execution": "sequential: one checkpoint resident at a time",
            "why_sequential": self.why_sequential(),
            "abstention_rule": dict(self.abstention_rule),
        }

    def acquire(self, spec: JudgeSpec) -> LocalBackend:
        if self._resident:
            live = self._resident[0]
            raise JudgeSequentialError(
                f"refusing to load judge {spec.id!r} ({spec.model}) while {live.id!r} "
                f"({live.model}) is still resident. {self.why_sequential()} Call "
                f"release() for the outgoing judge before acquiring the next one."
            )
        engine = self.load(spec)
        self._resident.append(spec)
        return engine

    def release(self, spec: JudgeSpec, engine: LocalBackend) -> None:
        try:
            engine.close()
        finally:
            if spec in self._resident:
                self._resident.remove(spec)

    def template_for(self, spec: JudgeSpec, item: JudgeItem) -> str:
        return spec.template_for(item)

    def source_for(self, spec: JudgeSpec, item: JudgeItem) -> str:
        name = self.template_for(spec, item)
        try:
            return self.templates[name]
        except KeyError:
            raise PanelSpecError(
                f"judge {spec.id!r} routes {item.qid!r} to template {name!r}, which "
                f"was not supplied. Known templates: {sorted(self.templates)}. A "
                f"missing rubric renders as a prompt with no scoring rule, and the "
                f"resulting verdicts look like judge noise rather than a missing "
                f"file."
            ) from None

    def render(self, spec: JudgeSpec, item: JudgeItem) -> tuple[str, str]:
        source = self.source_for(spec, item)
        context: dict[str, Any] = {
            "question": item.question,
            "answer": item.answer,
            "is_abstention": item.is_abstention,
            "condition_key": item.condition_key,
            "qid": item.qid,
        }
        if item.is_abstention:
            context["gold"] = ""
            context["abstention"] = dict(self.abstention_rule)
            context["decline_markers"] = list(self.abstention_rule.get("decline_markers", []))
            context["guess_markers"] = list(self.abstention_rule.get("guess_markers", []))
        else:
            context["gold"] = item.gold
        return render_template(source, context)


def model_family(model: str) -> str | None:
    lowered = model.strip().lower()
    segments = [s for s in re.split(r"[/_.\-]", lowered) if s]
    best: tuple[int, str] | None = None
    for seg in segments:
        for key, family in MODEL_FAMILIES.items():
            if (seg == key or seg.startswith(key)) and (best is None or len(key) > best[0]):
                best = (len(key), family)
    return best[1] if best else None


def _require(entry: Mapping[str, Any], key: str, judge_label: str) -> Any:
    if key not in entry:
        raise PanelSpecError(
            f"judge {judge_label} is missing {key!r}; every field is required "
            f"(R1.1). A default here is a guess, and a guess about a checkpoint "
            f"makes the verdicts unattributable."
        )
    value = entry[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PanelSpecError(
            f"judge {judge_label}: {key!r} is null or blank (R1.1). Null is not a "
            f"value here; it is an unnamed decision."
        )
    return value


def judge_from_entry(
    entry: Mapping[str, Any],
    *,
    max_tokens: int,
    prompt_template: str = JUDGE_PROMPT_TEMPLATE,
    abstention_prompt_template: str = JUDGE_ABSTENTION_PROMPT_TEMPLATE,
    backend_name: str = "transformers",
) -> JudgeSpec:
    label = str(entry.get("id", "<unnamed>"))
    model = str(_require(entry, "model", label))
    version = str(_require(entry, "version", label))
    released = str(_require(entry, "released", label))
    confirmed = bool(entry.get("exact_checkpoint_confirmed", False))

    if not confirmed:
        raise PanelSpecError(
            f"judge {label} ({model}) has an UNCONFIRMED checkpoint. PRD R7.6's "
            f"reason applies to judges unchanged: an unnamed model makes every "
            f"verdict unattributable, and `configs/pricing/latest.yaml` records "
            f"that the previous panel shipped three nonexistent repo ids."
        )
    if not _COMMIT_SHA.match(version):
        raise PanelSpecError(
            f"judge {label} ({model}) declares version {version!r}, which is not a "
            f"40-character revision commit. A tag can be repointed, which changes "
            f"the weights under a fixed config_hash and makes two runs incomparable "
            f"while every artifact claims the same identity."
        )
    if any(token in f"{model} {version} {released}".upper() for token in _PLACEHOLDERS):
        raise PanelSpecError(
            f"judge {label}: model/version/released contains a placeholder. The real "
            f"identifier is required, not a TODO."
        )
    if max_tokens <= 0:
        raise PanelSpecError(
            f"judge {label}: judge.output_tokens must be > 0, got {max_tokens}. A cap "
            f"of zero truncates every verdict to nothing and the panel reports "
            f"unanimous abstention."
        )

    raw_vram = entry.get("vram")
    if not isinstance(raw_vram, Mapping):
        raise PanelSpecError(
            f"judge {label} declares no `vram:` block. The weights footprint is what "
            f"decides the panel runs sequentially, so a judge without one cannot be "
            f"scheduled on the card."
        )

    return JudgeSpec(
        id=label,
        model=model,
        version=version,
        released=released,
        prompt_template=prompt_template,
        abstention_prompt_template=abstention_prompt_template,
        decoding=JudgeDecoding(
            max_tokens=max_tokens,
            temperature=JUDGE_TEMPERATURE,
            top_p=JUDGE_TOP_P,
            stop=JUDGE_STOP,
            seed=JUDGE_SEED,
        ),
        vram=ReaderVramSpec(
            params_b=float(_require(raw_vram, "params_b", label)),
            layers=int(_require(raw_vram, "layers", label)),
            kv_heads=int(_require(raw_vram, "kv_heads", label)),
            head_dim=int(_require(raw_vram, "head_dim", label)),
            bytes_per_param=float(raw_vram.get("bytes_per_param", 0.55)),
        ),
        rationale=str(entry.get("rationale", "")).strip(),
        exact_checkpoint_confirmed=confirmed,
        backend_name=backend_name,
    )


def judge_panel_from_config(
    pricing: Mapping[str, Any],
    *,
    templates: Mapping[str, str],
    abstention_rule: Mapping[str, Any],
    reader_families: Sequence[str],
    card_gb: float = CARD_GB_DEFAULT,
    headroom_gb: float = HEADROOM_GB,
    load: Callable[[JudgeSpec], LocalBackend] = _default_loader,
) -> JudgePanel:
    entries = pricing.get("judges_local")
    if not entries:
        raise PanelSpecError(
            "`judges_local` is empty or absent from the pricing config. PRD R8.2 "
            "requires at least 2 judges on all outputs; a panel with none is not a "
            "smaller experiment, it is an experiment that cannot be graded."
        )
    judge_block = pricing.get("judge") or {}
    if "output_tokens" not in judge_block:
        raise PanelSpecError(
            "`judge.output_tokens` is missing from the pricing config. The judge's "
            "decoding cap is a config value (R1.2); restating it here would give "
            "two answers."
        )

    specs: list[JudgeSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PanelSpecError(f"judges_local entry is not a mapping: {entry!r}")
        specs.append(
            judge_from_entry(
                entry,
                max_tokens=int(judge_block["output_tokens"]),
                backend_name=str(judge_block.get("backend", "transformers")),
            )
        )

    if len(specs) < MIN_JUDGES:
        raise PanelSpecError(
            f"the panel declares {len(specs)} judge(s); PRD R8.2 requires at least "
            f"{MIN_JUDGES}. Cohen's kappa is defined for two raters — with one, "
            f"'agreement' is a constant and the bar cannot be measured."
        )

    ids = [s.id for s in specs]
    if len(set(ids)) != len(ids):
        raise PanelSpecError(f"judge ids must be unique; got {ids}")

    readers = {str(f) for f in reader_families}
    seen: dict[str, str] = {}
    for spec in specs:
        family = spec.family
        if family is None:
            raise JudgeIndependenceError(
                f"judge {spec.id!r} ({spec.model}) has no recognisable lineage in "
                f"MODEL_FAMILIES, so it cannot be shown to be absent from the reader "
                f"panel ({sorted(readers)}). Independence is REFUSED rather than "
                f"assumed: an unnameable lineage is not a new family, it is an "
                f"unverified one."
            )
        if family in readers:
            raise JudgeIndependenceError(
                f"judge {spec.id!r} ({spec.model}) is the {family!r} lineage, which "
                f"is also a READER family ({sorted(readers)}). A judge that shares "
                f"weights with a reader cannot independently adjudicate it — it can "
                f"only reproduce that reader's blind spots — and R8.2's kappa would "
                f"then be measuring self-agreement."
            )
        if family in seen:
            raise JudgeIndependenceError(
                f"judges {seen[family]!r} and {spec.id!r} are both {family!r} "
                f"({spec.model}). Two entries naming the same weights are one rater "
                f"entered twice; kappa between identical deterministic outputs is a "
                f"tautology, not a reliability estimate."
            )
        seen[family] = spec.id

    for spec in specs:
        if spec.weights_gb >= card_gb:
            raise JudgeVramError(
                f"judge {spec.id!r} ({spec.model}) needs {spec.weights_gb:.2f} GB of "
                f"weights on a {card_gb:.1f} GB card. No swap schedule puts that in "
                f"VRAM; the judge has to be quantised harder or replaced, and both "
                f"are config changes that must be visible in the config_hash."
            )

    declared = {s.prompt_template for s in specs} | {s.abstention_prompt_template for s in specs}
    missing = sorted(declared - set(templates))
    if missing:
        raise PanelSpecError(
            f"the panel declares template(s) {missing} that were not supplied. A "
            f"judge with no rubric grades on its own pretraining, which is exactly "
            f"the unrecorded condition R1.1 exists to prevent."
        )
    if not abstention_rule:
        raise PanelSpecError(
            "the abstention scoring rule is empty. PRD R8.4 gives `_abs` items an "
            "explicit rule declared in `configs/prompts/abstention.yaml`; an empty "
            "one leaves abstention to the judge's improvisation, which is a "
            "convention nobody reviewed."
        )

    return JudgePanel(
        judges=tuple(specs),
        templates=dict(templates),
        abstention_rule=dict(abstention_rule),
        reader_families=tuple(sorted(readers)),
        load=load,
        card_gb=card_gb,
        headroom_gb=headroom_gb,
    )


def read_templates(root: Path | str, names: Sequence[str]) -> dict[str, str]:
    base = Path(root)
    out: dict[str, str] = {}
    for name in names:
        path = base / name
        if not path.is_file():
            continue
        out[name] = path.read_text(encoding="utf-8")
    return out


def _first_line_verdict(text: str) -> Verdict:
    for line in text.strip().splitlines():
        candidate = line.strip().strip(".,:;*`\"'")
        if candidate:
            return as_verdict(candidate)
    raise UnscorableError("the judge returned no output at all")


def _verdict_row(
    spec: JudgeSpec, item: JudgeItem, verdict: Verdict, raw: str, *, pass_index: int
) -> JudgeVerdict:
    return JudgeVerdict(
        qid=item.qid,
        condition_key=item.condition_key,
        judge_id=spec.id,
        verdict=verdict,
        pass_index=pass_index,
        raw=raw,
    )


def _judge_with(
    panel: JudgePanel,
    spec: JudgeSpec,
    items: Sequence[JudgeItem],
    run: PanelRun,
    *,
    pass_index: int,
) -> None:
    run.execution_order.append(spec.id)
    try:
        engine = panel.acquire(spec)
    except JudgeSequentialError:
        raise
    except BackendUnavailableError as exc:
        run.unavailable.append(
            UnavailableRow(
                judge_id=spec.id,
                model=spec.model,
                reason=str(exc),
                n_items_affected=len(items),
            )
        )
        return
    except Exception as exc:  # noqa: BLE001 - a loader failure must be reportable, not fatal
        run.unavailable.append(
            UnavailableRow(
                judge_id=spec.id,
                model=spec.model,
                reason=f"{type(exc).__name__}: {exc}",
                n_items_affected=len(items),
            )
        )
        return

    try:
        for index, item in enumerate(items):
            system, user = panel.render(spec, item)
            request = spec.decoding.request(
                spec.model,
                system,
                user,
                pass_index=pass_index,
                request_id=f"{spec.id}|{item.condition_key}|{item.qid}|p{pass_index}",
            )
            try:
                result = engine.generate(request)
            except BackendUnavailableError as exc:
                run.unavailable.append(
                    UnavailableRow(
                        judge_id=spec.id,
                        model=spec.model,
                        reason=str(exc),
                        n_items_affected=len(items) - index,
                    )
                )
                break

            run.prompt_tokens += result.prompt_tokens
            run.completion_tokens += result.completion_tokens

            if result.finish_reason != "stop":
                run.unscored.append(
                    UnscoredRow(
                        condition_key=item.condition_key,
                        qid=item.qid,
                        judge_id=spec.id,
                        reason=f"finish_reason={result.finish_reason}",
                    )
                )
                continue
            try:
                verdict = _first_line_verdict(result.text)
            except UnscorableError as exc:
                run.unscored.append(
                    UnscoredRow(
                        condition_key=item.condition_key,
                        qid=item.qid,
                        judge_id=spec.id,
                        reason=str(exc),
                    )
                )
                continue

            run.verdicts.add(_verdict_row(spec, item, verdict, result.text, pass_index=pass_index))
    finally:
        panel.release(spec, engine)


def _disagreements(
    verdicts: VerdictSet, by_key: Mapping[ItemKey, JudgeItem], *, pass_index: int
) -> list[DisagreementRow]:
    rows: list[DisagreementRow] = []
    ids = verdicts.judges()
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            va = verdicts.for_judge(a, pass_index=pass_index)
            vb = verdicts.for_judge(b, pass_index=pass_index)
            for key in verdicts.disagreements(a, b):
                item = by_key.get(key)
                rows.append(
                    DisagreementRow(
                        condition_key=key[0],
                        qid=key[1],
                        judge_a=a,
                        verdict_a=va[key],
                        judge_b=b,
                        verdict_b=vb[key],
                        is_abstention=(
                            item.is_abstention
                            if item is not None
                            else key[1].endswith(ABSTENTION_SUFFIX)
                        ),
                        answer="" if item is None else item.answer,
                        gold="" if item is None else item.gold,
                    )
                )
    return rows


def run_panel(
    panel: JudgePanel,
    items: Sequence[JudgeItem],
    *,
    pass_index: int = 1,
    on_backend_unavailable: Literal["raise", "exclude"] = "raise",
) -> PanelRun:
    started = time.perf_counter()
    run = PanelRun(n_items=len(items), pass_index=pass_index)
    by_key = {item.key: item for item in items}
    run.resident_peak_gb = panel.resident_peak_gb

    for spec in panel.judges:
        _judge_with(panel, spec, items, run, pass_index=pass_index)

    run.wall_seconds = time.perf_counter() - started
    run.disagreements = _disagreements(run.verdicts, by_key, pass_index=pass_index)

    if run.unavailable and on_backend_unavailable == "raise":
        reasons = {u.judge_id: u.reason for u in run.unavailable}
        detail = "; ".join(f"{j}: {r}" for j, r in sorted(reasons.items()))
        raise JudgeBackendUnavailableError(
            f"{len(run.unavailable)} of {panel.n_judges} judge(s) could not be run: "
            f"{detail}. No verdict was recorded for them and none was invented — a "
            f"judge scored INCORRECT for being unreachable would drag kappa down for "
            f"a reason that has nothing to do with the panel.",
            reasons=reasons,
            run=run,
        )
    return run


def judge_item(panel: JudgePanel, item: JudgeItem, *, pass_index: int = 1) -> list[JudgeVerdict]:
    return run_panel(panel, [item], pass_index=pass_index).verdicts.verdicts


def panel_readiness(panel: JudgePanel) -> list[JudgeReadiness]:
    out: list[JudgeReadiness] = []
    for spec in panel.judges:
        fits_alone = spec.weights_gb < panel.card_gb
        ready, reason, remedy = False, "", ""
        try:
            engine = panel.acquire(spec)
            try:
                report = engine.health()
            finally:
                panel.release(spec, engine)
            ready = bool(report.get("ready"))
            reason = str(report.get("reason", ""))
            remedy = str(report.get("remedy", ""))
        except JudgeSequentialError:
            raise
        except BackendUnavailableError as exc:
            reason, remedy = str(exc), ""
        except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
            reason = f"{type(exc).__name__}: {exc}"
        if not fits_alone and ready:
            ready = False
            reason = reason or (
                f"{spec.weights_gb:.2f} GB of weights does not fit a "
                f"{panel.card_gb:.1f} GB card, whatever the engine says."
            )
        out.append(
            JudgeReadiness(
                judge_id=spec.id,
                model=spec.model,
                version=spec.version,
                family=spec.family or UNRECOGNISED_LINEAGE,
                weights_gb=spec.weights_gb,
                footprint_gb=spec.footprint_gb(),
                fits_alone=fits_alone,
                ready=ready,
                reason=reason,
                remedy=remedy,
            )
        )
    return out


def _bar_message(agreement: Mapping[str, Any]) -> str:
    kappa_min = agreement.get("kappa_min")
    measured = "undefined" if kappa_min is None else f"{float(kappa_min):.4f}"
    lines = [
        f"judge panel is NOT usable: kappa_min={measured} against the "
        f"{CAUTIONARY_KAPPA} bar (2511.02246). Agreement no better than "
        f"chance-corrected cannot support a headline number — this is the "
        f"preregistered floor, not a target to argue with."
    ]
    for entry in agreement.get("verdicts", []):
        lines.append(f"  {entry.get('pair', '?')}: {entry.get('verdict', '')}")
    lines.append(
        "Run `panel_health` on a PILOT before the sweep: a noisy judge found at "
        "hour 2 costs an afternoon, and the same judge found at hour 150 costs "
        "41% of the project's compute budget and cannot be acted on."
    )
    return "\n".join(lines)


def panel_health(
    panel: JudgePanel,
    pilot_items: Sequence[JudgeItem],
    *,
    on_unusable: Literal["raise", "report"] = "raise",
    pass_index: int = 1,
) -> PanelHealth:
    if on_unusable not in ("raise", "report"):
        raise ValueError(f"on_unusable must be raise|report, got {on_unusable!r}")

    health = PanelHealth(n_items=len(pilot_items), pass_index=pass_index)
    health.readiness = panel_readiness(panel)

    run = run_panel(
        panel,
        pilot_items,
        pass_index=pass_index,
        on_backend_unavailable="exclude",
    )
    blocked = {r.judge_id: r.reason for r in health.blocked}
    for row in run.unavailable:
        blocked.setdefault(row.judge_id, row.reason)

    by_judge = run.verdicts_by_judge()
    health.agreement = panel_agreement(by_judge)

    if blocked:
        detail = "; ".join(f"{j}: {r}" for j, r in sorted(blocked.items()))
        health.notes.append(
            f"{len(blocked)} judge(s) could not be reached: {detail}. Kappa below is "
            f"computed over the judges that ran and is NOT a statement about the "
            f"panel."
        )
        if on_unusable == "raise":
            raise JudgeBackendUnavailableError(
                f"judge panel cannot be measured: {detail}. Readiness and the "
                f"partial verdicts are attached.",
                reasons=blocked,
                run=run,
                readiness=health.readiness,
            )

    if run.unscored:
        by_judge_id: dict[str, int] = {}
        for miss in run.unscored:
            by_judge_id[miss.judge_id] = by_judge_id.get(miss.judge_id, 0) + 1
        health.notes.append(
            f"{len(run.unscored)} item/judge pair(s) produced no usable verdict and "
            f"are excluded pairwise: {dict(sorted(by_judge_id.items()))}. Kappa is "
            f"corrected for nothing here — dropping refusals silently makes it look "
            f"BETTER as refusals rise, so the count travels with the figure."
        )
    if len(health.readiness) < MIN_JUDGES:
        health.notes.append(
            f"only {len(health.readiness)} judge; PRD R8.2 requires {MIN_JUDGES} and "
            f"kappa does not exist with one rater."
        )

    if not health.usable:
        message = _bar_message(health.agreement)
        if on_unusable == "raise" and not blocked:
            raise PanelKappaBelowBarError(message, agreement=health.agreement)
        health.notes.append(message)

    return health


def format_health(health: PanelHealth) -> str:
    lines = [
        "",
        f"  judges              {len(health.readiness)}",
        f"  pilot items         {health.n_items}",
        f"  panel usable        {'YES' if health.usable else 'NO'}",
        f"  kappa min           {health.kappa_min}  (bar {CAUTIONARY_KAPPA})",
        "",
        f"  {'judge':<10}{'family':<11}{'weights':>9}{'fits':>6}{'ready':>7}",
    ]
    for r in health.readiness:
        lines.append(
            f"  {r.judge_id:<10}{r.family:<11}{r.weights_gb:>8.2f}G"
            f"{('yes' if r.fits_alone else 'NO'):>6}{('yes' if r.ready else 'NO'):>7}"
        )
        if r.reason:
            lines.append(f"    {r.judge_id}: {r.reason}")
    if health.notes:
        lines.append("")
        lines.append("  notes")
        lines += [f"    - {n}" for n in health.notes]
    return "\n".join(lines)
