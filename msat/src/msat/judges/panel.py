"""The judge PANEL — the 75-hour line item, built so a bad judge is found at hour 2.

    `configs/pricing/latest.yaml` -> `judges_local`:
        judge_1  mistralai/Mistral-7B-Instruct-v0.3      c170c708...
        judge_2  ibm-granite/granite-3.3-8b-instruct   51dd4bc2...

Three properties of that panel are HARDWARE and CONFIG facts, and every design
decision below follows from one of them rather than from taste.

1. THE PANEL IS SEQUENTIAL. IT IS NOT A STYLE CHOICE
---------------------------------------------------
The two checkpoints' weights together are ~8.6 GB before any KV cache, and the
card is 8 GB. They cannot both be resident, so `JudgePanel.acquire` REFUSES a
second judge while one is loaded. That is a type-level invariant rather than a
convention: a future edit that reached for `generate_batch` across two models
would hit `JudgeSequentialError` at the first call instead of OOMing at hour 60.
`JudgeSpec.weights_gb` and `panel.resident_peak_gb` are derived from the declared
architecture through `hardware/vram.py`, so the refusal's arithmetic is the same
one `msat vram` prints — two implementations of one formula is how two ceilings
came to disagree by 7.8x in this repo before.

This is the same shape as the bugs this project keeps catching: a value that
means something else than it appears to. "Two judges" reads like concurrency and
means a swap.

2. A JUDGE THAT SHARES WEIGHTS WITH A READER IS NOT A JUDGE
-----------------------------------------------------------
`configs/pricing/latest.yaml` chooses mistral and granite precisely because those
families appear NOWHERE in `readers.yaml` (gemma / qwen / phi). A judge that
shares a reader's weights cannot independently adjudicate it, and R8.2's
kappa would then be measuring self-agreement. `judges_from_config` takes
`reader_families` and REFUSES the panel with `JudgeIndependenceError` if any
judge's lineage is in that set, if two judges are the same weights, or if a
lineage cannot be identified at all — an unidentifiable lineage cannot be shown
independent, and the direction that fails closed is the one that matters.

3. κ >= 0.118 IS A BAR, AND CROSSING IT IS NOT OPTIONAL
------------------------------------------------------
`2511.02246` reports LLM-judge agreement at kappa = 0.118, and `kappa.py` already
carries that as `CAUTIONARY_KAPPA`. `panel_health` measures it with
`panel_agreement` and raises `PanelKappaBelowBarError` when the panel cannot
clear it. The point of the error is TIMING: `panel_health` is runnable on a
PILOT, before the 75-hour sweep, so a judge that is noisy rather than broken is
discovered at hour 2 rather than at hour 150 — where the finding costs 41% of the
project's compute budget and cannot be acted on.

PER-ITEM, NEVER AN AGGREGATE (R8.3)
-----------------------------------
`run_panel` returns a `PanelRun` carrying a `VerdictSet` — one `JudgeVerdict` per
(item, judge, pass) — plus the disagreeing items with their content. There is no
accuracy scalar on the return type, because a function that returns 0.62 can be
read without opening anything, and 0.62 cannot say WHICH items the panel is wrong
about. `metrics/score.build_disagreements` wants the same shape and is left to
its owner; this module produces the verdicts it consumes.

VerdictSet was dead code until now. It is the right abstraction — it is what
`kappa.py`, `audit.py` and `report.py` all index by — so `PanelRun` uses it
rather than parallel lists, and the three modules now share one definition of
"a verdict set".

ABSTENTION IS A DIFFERENT PROMPT, NOT A DIFFERENT SCORE (R8.4)
-------------------------------------------------------------
An `_abs` item is defined by the ABSENCE of evidence, so a judge shown its gold
learns the answer to the very item it is being asked to grade. `render` refuses
to put gold in the abstention context at all and routes the item to the
abstention rubric, which receives the declared decline/guess markers from
`configs/prompts/abstention.yaml` instead. Normal items never see that rule: a
declining answer to an answerable question is wrong, and a rubric that had seen
the markers would call it right.

THE SECOND PASS MUST NOT BE A CACHE HIT (R8.6)
----------------------------------------------
R8.6 runs the same deterministic system twice under the same judge and counts the
rows whose verdict changed. `GenerationRequest.canonical_body()` includes
`seed`, so a second pass that reuses pass 1's seed is a byte-identical request —
and the content-addressed store (R6.1) would serve the cached verdict, producing
a flip count of 0 by CONSTRUCTION rather than by determinism. `seed_for_pass`
offsets the seed per pass so the two passes are genuinely two passes.

NO RESULTS EXIST
----------------
Nothing in this module has been run against real weights. Every test fixture is
synthetic and says so. The only thing measured anywhere in the repo is kappa on
a synthetic table, and kappa on a synthetic table is arithmetic, not a finding.
"""

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

# ── declared constants. No unnamed numbers. (R1.1) ─────────────────────────────

JUDGE_PROMPT_TEMPLATE = "prompts/judge.yaml"
"""Rubric for an ANSWERABLE item. A path, not a string, so the prompt that graded
a verdict is the prompt that appears in the artifact.

⚠️ This file DOES NOT EXIST in `configs/` yet and this module does not create it:
config files are outside this module's ownership. `judges_from_config` therefore
takes the template TEXT as an injected mapping keyed by these paths, and refuses
to build a panel whose declared template was not supplied. Creating the rubric is
the one piece of this work left undone, and the report says so rather than
silently shipping a default prompt that would differ from the shipped one."""

JUDGE_ABSTENTION_PROMPT_TEMPLATE = "prompts/judge_abstention.yaml"
"""Rubric for an `_abs` item (R8.4). Same status as above: declared, not shipped."""

ABSTENTION_RULE_KEY = "abstention"
"""Key inside `configs/prompts/abstention.yaml` holding the scoring rule. Named so
the lookup is checkable — a wrong key would yield an empty rule and a judge that
had been told nothing."""

CARD_GB_DEFAULT = 8.0
"""The RTX 4060 Laptop, 8 GB (PRD §N8). The panel's weights are ~8.6 GB together,
which is the whole reason for the sequential invariant above."""

JUDGE_TEMPERATURE = 0.0
"""Greedy, and deliberately so.

R8.6 measures the judge's own instability by running the same item twice. A
sampling judge cannot distinguish "the judge changed its mind" from "the decoder
drew another token", so a non-zero temperature makes `n_flips` uninterpretable —
it would measure entropy, not reliability.

This does NOT collide with R7.1's "seeds must vary something": that rule exists
because the content-addressed store DEDUPES byte-identical requests, which would
collapse the reader's replicate grid. Here the dedupe is the intent on pass 1, and
`seed_for_pass` deliberately breaks it on pass 2 so pass 2 is a real second pass
rather than a cache read."""

JUDGE_TOP_P = 1.0
"""Unused at temperature 0, and stated rather than left implicit so a reader
checking the two numbers together is not left wondering which one binds."""

JUDGE_SEED = 20261012
"""Matches `decoding.yaml: provider_seed`, so one seed convention governs the
repo. Changing it is a config change, not an edit here."""

JUDGE_STOP: tuple[str, ...] = ("\n\n",)
"""A judge that emits a paragraph after its verdict has already answered a
different question. Stopping at the first blank line keeps the verdict token
isolated from commentary, which `as_verdict` would otherwise have to refuse."""

JUDGE_INPUT_TOKENS_APPROX = 4000
"""`pricing.judge.input_tokens_approx`. Used only for the KV reservation reported
in the readiness table; the real prompt is what gets sent."""

HEADROOM_GB = 3.0
"""Matches `hardware.vram.DEFAULT_HEADROOM_GB`. Reported, not applied to a
ceiling: the panel's constraint is weights, not a servable prompt length."""

UNRECOGNISED_LINEAGE = "unrecognised"
"""Spelled out rather than left `None`, because a null in a readiness row reads as
a value the reader failed to fill in — and this project's whole complaint is
about numbers that mean something other than they appear to. It cannot occur in a
constructed panel: `judges_from_config` refuses unidentifiable lineages outright."""

MIN_JUDGES = 2
"""R8.2 requires >= 2 judges on all outputs. One judge has no kappa: the statistic
needs two raters, and with one rater "agreement" is a constant."""

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
"""A revision commit. Checked because the whole point of the field is that a tag
can be repointed: `configs/pricing/latest.yaml` records that `judge_2` was
replaced partly BECAUSE a gated repo's revision could not be verified. A `version`
that is not a commit is the claim that could not be checked."""

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
"""Model id -> lineage token, matched on repo/name SEGMENTS.

Matching whole segments rather than substrings is the point: `phi` is a three-
letter token that occurs inside a dozen unrelated words, and a substring match
would file a judge under a reader's lineage on a coin flip. `Qwen3.5-9B` has no
segment equal to `qwen`, so a segment also matches when it STARTS WITH the key —
stated here because it is a prefix match and therefore can still over-fire on an
exotic name, which is why `judge_panel_from_config` treats an unmatched model as a
refusal rather than as a new family."""


# ── errors. Each one is a distinct, named, actionable failure. ─────────────────


class PanelSpecError(ValueError):
    """A judge is declared with something missing, null, or unconfirmable.

    Raised rather than defaulted (R1.1). A default for `version` would produce a
    panel whose verdicts cannot be attributed to a checkpoint — the exact failure
    `configs/readers.yaml` documents paying for three times with invented repo
    ids that failed silently, at load, hours into a run.
    """


class JudgeVramError(PanelSpecError):
    """A judge cannot fit the card even on its own.

    Distinct from "the panel is too big", which is answered by sequencing. This
    is the unrecoverable direction: no swap schedule puts a 10 GB checkpoint in
    8 GB of VRAM.
    """


class JudgeIndependenceError(ValueError):
    """A judge cannot independently adjudicate the panel it is on.

    R8.2's kappa is a statement about two INDEPENDENT raters. A judge sharing
    weights with a reader can only confirm that reader's own blind spots, and two
    entries naming the same weights are one rater entered twice. Both directions
    produce a healthy-looking kappa over a panel that measured nothing.
    """


class JudgeSequentialError(RuntimeError):
    """Two judges were resident at once.

    On an 8 GB card the panel's weights are ~8.6 GB together. This is the
    invariant stated in the module docstring, enforced by `JudgePanel.acquire`
    rather than by a comment, so the failure arrives at the first offending call
    with both model ids in the message instead of as an OOM 60 hours in.
    """


class JudgeBackendUnavailableError(RuntimeError):
    """A judge could not be reached, and the reason is carried.

    Never swallowed into a verdict. A missing judge scored as INCORRECT would
    charge the reader for the harness being down and would drag kappa DOWN for a
    reason that has nothing to do with the panel — which is precisely the kind of
    figure that gets "fixed" by changing the rubric.
    """

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
        """judge id -> the backend's own message. The remediation text travels with
        it, because the answer is always a config or environment change and a bare
        ImportError makes the reader work that out from scratch."""
        self.run = run
        """The partial run, so a caller that wanted the exception still has every
        verdict collected before the backend died."""
        self.readiness: list[JudgeReadiness] = list(readiness or [])


class PanelKappaBelowBarError(RuntimeError):
    """The panel cannot support a headline number.

    R8.2 requires judges that agree; `2511.02246` puts the bar at kappa = 0.118
    and `kappa.py` carries it as `CAUTIONARY_KAPPA`. Below it, agreement is no
    better than chance-corrected, and a headline judged by a panel that cannot
    clear the bar is a number about the panel rather than about memory
    saturation.

    Carries the full agreement block so the caller can still WRITE the artifact
    (R8.7 makes the audit standalone and it must survive a zero interaction) while
    refusing to promote it to a result.
    """

    def __init__(self, message: str, *, agreement: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.agreement: dict[str, Any] = dict(agreement)
        self.bar: float = CAUTIONARY_KAPPA


# ── the declaration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class JudgeDecoding:
    """A judge's decoding parameters, as data.

    Not arguments to `generate` scattered across call sites: these are recorded
    experimental variables, and the judge half of the study is 41% of the compute
    budget, so the parameters that produced a verdict travel with it.
    """

    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int

    def seed_for_pass(self, pass_index: int) -> int:
        """A DIFFERENT seed per pass, on purpose. See the module docstring."""
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
    """One judge: WHO it is, at WHICH weights, reading WHICH prompt.

    Every field is required and none may be null. `version` is a revision COMMIT,
    checked as one — see `_COMMIT_SHA`.
    """

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

    # ── lineage (R8.2 independence) ───────────────────────────────────────────
    @property
    def family(self) -> str | None:
        return model_family(self.model)

    # ── VRAM, derived from the declared architecture ──────────────────────────
    @property
    def shape(self) -> ModelShape:
        return self.vram.to_shape(name=self.model)

    @property
    def weights_gb(self) -> float:
        """Weights footprint, in GB.

        `bytes_per_param` is BYTES (int4 plus quantisation metadata), so the quant
        level is derived from it rather than read from a nominal dtype. Treating
        the number as bits and dividing again by 8 is the unit bug that once
        understated gemma's weights as 0.26 GB instead of 2.05 GB and inflated its
        ceiling by 1.6x — the same class of error, in a module whose whole job is
        to be the thing that refuses when the card is full.
        """
        return self.shape.weights_gb(quant_bits=self.vram.bytes_per_param * 8.0)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.shape.kv_bytes_per_token()

    def kv_gb(self, n_tokens: int) -> float:
        return self.kv_bytes_per_token * n_tokens / (1024**3)

    def footprint_gb(self, n_tokens: int = JUDGE_INPUT_TOKENS_APPROX) -> float:
        """Weights plus the KV reservation for a judge-sized prompt."""
        return self.weights_gb + self.kv_gb(n_tokens)

    # ── prompt routing (R8.4) ─────────────────────────────────────────────────
    def template_for(self, item: JudgeItem) -> str:
        """Which rubric this item is graded by.

        Abstention items are routed by the `_abs` SUFFIX, the same detection
        `metrics/score.py` and `judges/human_sample.py` use — one suffix, one
        place the benchmark's own convention is written down.
        """
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
    """One output to be judged.

    Deliberately not a StoredResponse: judging must be replayable from a verdict
    record without the reader's weights, and coupling the two would make a judge
    panel that cannot be re-run offline.
    """

    condition_key: str
    qid: str
    question: str
    answer: str
    gold: str = ""
    """Empty for abstention items, and `render` refuses to pass it to the
    abstention rubric even if a caller supplies one — see the module docstring."""

    @property
    def key(self) -> ItemKey:
        return (self.condition_key, self.qid)

    @property
    def is_abstention(self) -> bool:
        return self.qid.endswith(ABSTENTION_SUFFIX)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> JudgeItem:
        """Build from a store row or a human-sample row.

        `gold` is optional in both and absent in neither meaning: an abstention row
        has none by construction, and an answerable row with no gold is a data
        problem the judge rubric will surface rather than hide.
        """
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


# ── what a run produced (per item, never an aggregate) ─────────────────────────


@dataclass(frozen=True, slots=True)
class UnscoredRow:
    """An item a judge did not produce a usable verdict for.

    Excluded PAIRWISE and counted, for the reason `kappa.py` already documents:
    dropping refusals silently makes kappa look better as refusals rise, so the
    count of what is missing travels with the statistic.
    """

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
    """One item the panel split on, WITH its content (R8.3).

    The count says the panel is unstable; the rows say on what. A panel that
    disagrees only on abstention items is a different problem from one that
    disagrees on answerable ones, and the pair of verdicts alone cannot tell them
    apart.
    """

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
    """A judge that could not be run, and what it cost."""

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
    """Every verdict, every disagreement, and everything that went missing.

    Deliberately carries NO accuracy scalar. `verdicts.as_dicts()` is the per-item
    record (R8.3) and `verdicts_by_judge()` is the index `kappa.py` consumes;
    an aggregate would be a second, easier-to-misread representation of the same
    rows, and the one that gets quoted.
    """

    verdicts: VerdictSet = field(default_factory=VerdictSet)
    disagreements: list[DisagreementRow] = field(default_factory=list)
    unscored: list[UnscoredRow] = field(default_factory=list)
    unavailable: list[UnavailableRow] = field(default_factory=list)
    n_items: int = 0
    pass_index: int = 1
    execution_order: list[str] = field(default_factory=list)
    """Judge ids in the order they were loaded. Recorded because the order is a
    consequence of the sequential invariant, and a reader comparing two runs needs
    to know whether the swap order changed."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    resident_peak_gb: float = 0.0

    @property
    def n_judges(self) -> int:
        return len(self.verdicts.judges())

    @property
    def n_abstention(self) -> int:
        """Abstention ITEMS judged, by the `_abs` suffix (R8.4).

        Counted off the verdict rows rather than off the disagreement list: R8.4
        requires abstention reported SEPARATELY from recall@K, and "3 of the 2,000
        items the panel split on were abstention" is a statistic about disagreement
        wearing a statistic about coverage's name. Both are worth having, and they
        are not the same number.
        """
        return len(
            {
                (v.condition_key, v.qid)
                for v in self.verdicts.verdicts
                if v.qid.endswith(ABSTENTION_SUFFIX)
            }
        )

    def verdicts_by_judge(self) -> dict[str, dict[ItemKey, Verdict]]:
        """`judge -> (item -> verdict)`, the shape `cohen_kappa` takes."""
        return {j: self.verdicts.for_judge(j, pass_index=self.pass_index) for j in self.verdicts.judges()}

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


# ── readiness and health ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class JudgeReadiness:
    """One judge, before it has judged anything.

    Readiness is asked WITHOUT loading weights: `LocalBackend.health` is the
    project's existing weight-free probe, and paying 30 seconds and 4 GB to learn
    the answer is "no" is exactly the cost the interface was designed to avoid.
    """

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
    """Can this panel support a headline number, and why not if it cannot.

    The deliverable is the conjunction: per-judge readiness AND measured kappa.
    A panel whose judges are all loadable and which agrees at kappa = 0.04 is
    worse than useless, and a panel that cannot load is not a reliability
    question at all — the two failures need different responses and they arrive
    together in practice.
    """

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


# ── the panel ──────────────────────────────────────────────────────────────────


def _default_loader(spec: JudgeSpec) -> LocalBackend:
    """The production loader. INJECTED everywhere else.

    A module-level `select_backend` call inside `run_panel` would make the panel
    untestable off the execution box, which is the same mistake `runner/executor`
    avoids with its injected `Generator`: the loop that spends 75 GPU-hours is the
    loop most in need of being testable with no GPU.
    """
    return select_backend(spec.backend_name)


@dataclass(frozen=True, slots=True)
class JudgePanel:
    """The judges, their rubrics, and the discipline that keeps them off each
    other's VRAM.

    `load` is injected for the same reason as `Executor.generator`. `templates` is
    a mapping from DECLARED PATH to template text, so a rubric that is not shipped
    (see `JUDGE_PROMPT_TEMPLATE`) is a loud build error rather than a default
    prompt that silently differs from the reviewed one.
    """

    judges: tuple[JudgeSpec, ...]
    templates: Mapping[str, str]
    abstention_rule: Mapping[str, Any]
    reader_families: tuple[str, ...]
    load: Callable[[JudgeSpec], LocalBackend] = _default_loader
    card_gb: float = CARD_GB_DEFAULT
    headroom_gb: float = HEADROOM_GB
    _resident: list[JudgeSpec] = field(
        default_factory=list, repr=False, compare=False
    )
    """Which checkpoints are loaded RIGHT NOW. A frozen dataclass can still mutate
    a list it owns, so the invariant below cannot be bypassed by assigning to the
    panel."""

    # ── shape ─────────────────────────────────────────────────────────────────
    @property
    def judge_ids(self) -> tuple[str, ...]:
        return tuple(j.id for j in self.judges)

    @property
    def n_judges(self) -> int:
        return len(self.judges)

    @property
    def weights_gb(self) -> float:
        """Every judge's weights at once."""
        return sum(j.weights_gb for j in self.judges)

    @property
    def resident_peak_gb(self) -> float:
        """The largest single judge, weights plus its KV reservation.

        This is what the card actually has to hold, because the panel is
        sequential. Quoting the SUM here would overstate the requirement and invite
        the wrong fix ("buy a bigger card") when the panel already runs.
        """
        return max((j.footprint_gb() for j in self.judges), default=0.0)

    @property
    def fits_concurrently(self) -> bool:
        return bool(self.judges) and self.weights_gb + self.headroom_gb <= self.card_gb

    def why_sequential(self) -> str:
        """The arithmetic, in one sentence, with the numbers in it."""
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

    # ── the sequential invariant ──────────────────────────────────────────────
    def acquire(self, spec: JudgeSpec) -> LocalBackend:
        """Load ONE judge, and refuse to have two.

        The refusal is here rather than in the run loop because this is the only
        place that can know what is already resident. It costs a list append.
        """
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
        """Unload a judge. Always safe to call twice.

        Double-close is tolerated because the `finally` that guarantees a swap on
        an error path and the happy path can both reach it; making that idempotent
        is cheaper than making every caller track it. The backend owns actually
        returning the VRAM — `TransformersBackend.close` drops its model and
        tokenizer caches — which is why the panel swaps rather than overlaps.
        """
        try:
            engine.close()
        finally:
            if spec in self._resident:
                self._resident.remove(spec)

    # ── prompt rendering ──────────────────────────────────────────────────────
    def template_for(self, spec: JudgeSpec, item: JudgeItem) -> str:
        """The rubric path this item is graded by (R8.4)."""
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
        """`(system, user)` for one item under one judge.

        ABSTENTION GOLD IS NEVER PASSED. An `_abs` item is defined by the absence
        of evidence; handing the rubric a gold string teaches it the answer to the
        item it is grading, and every abstention verdict after that is measuring
        the prompt. The rule travels instead, verbatim from
        `configs/prompts/abstention.yaml`, so the judge and the metric apply the
        SAME convention (R8.4 / R1.2) rather than two that agree until one is
        edited.
        """
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


# ── construction from config ───────────────────────────────────────────────────


def model_family(model: str) -> str | None:
    """Lineage token for a model id, or `None` when it cannot be identified.

    Segment-prefix match against `MODEL_FAMILIES`; see that mapping for why a
    substring match would be unsafe. `None` is a REFUSAL, not a new family: a
    lineage nobody can name cannot be shown to be absent from the reader panel,
    and the reader panel is the thing independence is defined against.
    """
    lowered = model.strip().lower()
    segments = [s for s in re.split(r"[/_.\-]", lowered) if s]
    best: tuple[int, str] | None = None
    for seg in segments:
        for key, family in MODEL_FAMILIES.items():
            if (seg == key or seg.startswith(key)) and (best is None or len(key) > best[0]):
                best = (len(key), family)
    return best[1] if best else None


def _require(entry: Mapping[str, Any], key: str, judge_label: str) -> Any:
    """One field, present and non-null. (R1.1)

    YAML `null` and "absent" are indistinguishable to a `.get(key, default)`, and
    both are the failure this project keeps paying for: a field that looks
    populated and names nothing.
    """
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
    """One `judges_local` entry -> one `JudgeSpec`.

    `max_tokens` comes from `judge.output_tokens` in the same file rather than
    being restated: a rubric capped at 100 tokens in one place and 512 in another
    is a truncation that only shows up as judge noise.
    """
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
    """The whole panel, from `configs/pricing/latest.yaml`.

    `reader_families` is `[f.id for f in bundle.readers.families]` — the same
    vocabulary as `MODEL_FAMILIES`' values, which is what makes the independence
    check a set lookup rather than a judgement call.

    Refuses, in this order, because each refusal makes the next one moot:

      * a malformed declaration      -> `PanelSpecError`
      * a judge that will not fit    -> `JudgeVramError`
      * a judge sharing a lineage
        with a reader, or with
        another judge                -> `JudgeIndependenceError`
      * fewer than two judges        -> `PanelSpecError`
    """
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
    """Template text by declared path, from a config root.

    Separate from the panel so a missing rubric is a read-time fact with a path in
    the message, rather than a KeyError raised from inside a 75-hour loop.
    """
    base = Path(root)
    out: dict[str, str] = {}
    for name in names:
        path = base / name
        if not path.is_file():
            continue
        out[name] = path.read_text(encoding="utf-8")
    return out


# ── running the panel ──────────────────────────────────────────────────────────


def _first_line_verdict(text: str) -> Verdict:
    """A judge's raw output -> a verdict, or a refusal.

    `as_verdict` stays exactly as strict as `verdicts.py` made it. What this adds
    is the *shape* of the reduction: a judge emits prose and the verdict is its
    first non-empty line, so "The answer is correct because..." is read as
    correct while "I'm not sure" is still REFUSED. Narrowing the vocabulary to
    reach for meaning inside a sentence is the change that manufactures agreement.
    """
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
    """Run ONE judge over every item, then swap it out.

    The outer loop is over judges precisely so residency is bounded by one
    checkpoint. An item-major loop would have to re-load a 7 GB model for every
    one of ~162,000 items, which is not a performance problem but a schedule
    problem: the swap cost would dominate the panel's 75 hours.
    """
    run.execution_order.append(spec.id)
    try:
        engine = panel.acquire(spec)
    except JudgeSequentialError:
        # A concurrency violation is a BUG in this module, not a missing engine, and
        # it must never be filed as an `UnavailableRow`. Two judges' weights total
        # ~8.6 GB against an 8 GB card, so the one way to reach this is an edit that
        # loads a second checkpoint - and the reader who hits it needs the invariant,
        # not a row saying judge_2 "could not be reached".
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
                # EVERY item from here on is unjudged, not just this one. Reporting
                # `1` here would understate the gap by whatever is left in the
                # queue, and an understated gap is the number that lets a run
                # claim it judged the full set when it did not.
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

            # R5.5: `refusal`, `length` and `error` are EXPLICIT and counted. An
            # empty or truncated judge output is missing data, and scoring it INCORRECT
            # would charge the reader for the harness's own failure.
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

            run.verdicts.add(
                _verdict_row(spec, item, verdict, result.text, pass_index=pass_index)
            )
    finally:
        panel.release(spec, engine)


def _disagreements(
    verdicts: VerdictSet, by_key: Mapping[ItemKey, JudgeItem], *, pass_index: int
) -> list[DisagreementRow]:
    """Every item the panel split on, per item (R8.3).

    Built from `VerdictSet.disagreements` rather than a second comparison, so the
    disagreement set and the kappa matrix cannot come from two implementations that
    agree until a verdict is missing from one of them.
    """
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
    """Every judge on every item, SEQUENTIALLY, per item.

    `on_backend_unavailable` is `"raise"` by default and `"exclude"` by choice:

    * `"raise"` — the default, because a judge that cannot load must not be
      discovered at hour 150. The exception carries the partial `PanelRun`, so
      nothing judged so far is lost.
    * `"exclude"` — for a pilot, where a broken engine should not discard the
      verdicts the working judge did produce. The judge is still NAMED in
      `run.unavailable` with its own message, and `panel_health` refuses the panel
      on the strength of it. Nothing is scored zero either way: an unreachable
      judge produces no verdict, and an unscorable item produces an `UnscoredRow`.
    """
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


def judge_item(
    panel: JudgePanel, item: JudgeItem, *, pass_index: int = 1
) -> list[JudgeVerdict]:
    """One item, every judge, in the same sequential order as the sweep.

    Provided because a single item is the natural unit of debugging and of the
    pilot that `panel_health` runs. It pays a full load/unload per judge, which is
    why the 75-hour sweep uses `run_panel`: swapping per item would put the swap
    cost above the generation cost. The discipline is identical either way —
    `run_panel` on a one-item list, not a second code path that could drift.
    """
    return run_panel(panel, [item], pass_index=pass_index).verdicts.verdicts


# ── health, and the pilot that must precede the sweep ──────────────────────────


def panel_readiness(panel: JudgePanel) -> list[JudgeReadiness]:
    """Can each judge be loaded on this card? Never loads weights.

    `LocalBackend.health()` is the project's existing weight-free probe; asking it
    per judge is what turns "the panel will not start" into a per-model row at hour
    0 instead of a stack trace from `transformers` at hour 0 of the sweep.
    """
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
    """The refusal, in words, naming the bar and the pairs that failed it."""
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
    """Readiness + MEASURED kappa, and a refusal when the panel cannot be used.

    `pilot_items` is the whole point. The judge panel is 75 GPU-hours and 41% of
    the budget, and every reason it might be wrong — a judge that hedges, a rubric
    that reads as prose, a model that cannot load — is discoverable on a few
    hundred items in an afternoon. This function is what makes that afternoon
    happen deliberately rather than as a crisis at hour 150.

    `on_unusable="report"` returns the health block instead of raising. It exists
    for the R8.7 standalone audit, which must be WRITTEN whether or not the panel
    passed — "we found nothing" is exactly the case where an unreliable panel is
    the reason and nobody would otherwise know.

    A blocked judge is reported BEFORE the kappa is judged, because kappa computed
    over a surviving judge is not weak, it is *wrong*: it measures one rater.
    """
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
    """Fixed-width summary for the CLI. Column widths are fixed because a number
    whose position moves between rows is a number nobody reads twice."""
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