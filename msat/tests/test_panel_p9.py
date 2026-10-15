"""R8.2 / R8.3 / R8.4 — the judge panel, on an injected fake backend.

WHAT IS REAL AND WHAT IS SYNTHETIC
----------------------------------
Real: `configs/pricing/latest.yaml` (the pinned judge ids, models and SHAs),
`configs/readers.yaml` (the reader families independence is defined against),
`configs/prompts/abstention.yaml` (the decline/guess marker sets), and `kappa.py`
(the 0.118 bar and the arithmetic it uses).

Synthetic, and labelled as such at the point of use: the two judge rubrics, every
item, and every verdict any judge returns. `configs/prompts/judge.yaml` DOES NOT
EXIST — see `test_the_declared_judge_rubrics_are_declared_but_not_shipped` — so a
rubric that looked like the shipped one would be a fabricated result. NO judge has
been run against real weights anywhere in this repo, and a kappa measured on a
scripted fixture is arithmetic, not a finding.

THE FAKE ENFORCES THE HARDWARE CONSTRAINT, IT DOES NOT JUST OBSERVE IT
----------------------------------------------------------------------
The panel's two checkpoints are 7.98 GiB of weights against an 8 GiB card, so it
runs one judge at a time and swaps. `_Residency` below is the SECOND, independent
check on that: it is wired into the fake loader and raises the moment a second
checkpoint is constructed while one is live. If someone deletes the refusal in
`JudgePanel.acquire`, `test_the_panel_never_holds_two_checkpoints_at_once` still
fails — which is the only way "the panel is sequential" is a property rather than
a claim.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat.judges import panel as panel_mod
from msat.judges.kappa import CAUTIONARY_KAPPA, cohen_kappa
from msat.judges.panel import (
    JUDGE_ABSTENTION_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SEED,
    JudgeBackendUnavailableError,
    JudgeIndependenceError,
    JudgeItem,
    JudgePanel,
    JudgeSequentialError,
    JudgeSpec,
    JudgeVramError,
    PanelKappaBelowBarError,
    PanelSpecError,
    format_health,
    judge_from_entry,
    judge_item,
    judge_panel_from_config,
    model_family,
    panel_health,
    panel_readiness,
    read_templates,
    run_panel,
)
from msat.judges.verdicts import Verdict, VerdictSet
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)

CONFIG = Path("configs")
PRICING = yaml.safe_load((CONFIG / "pricing" / "latest.yaml").read_text(encoding="utf-8"))
READERS = yaml.safe_load((CONFIG / "readers.yaml").read_text(encoding="utf-8"))
ABSTENTION_YAML = yaml.safe_load(
    (CONFIG / "prompts" / "abstention.yaml").read_text(encoding="utf-8")
)
ABSTENTION_RULE = ABSTENTION_YAML["abstention"]["scoring"]
READER_FAMILIES = [str(f["id"]) for f in READERS["families"]]

# ── synthetic rubrics ─────────────────────────────────────────────────────────
# The real judge rubrics are not in `configs/` yet (see `test_the_declared_judge
# _rubrics_are_declared_but_not_shipped`). These two are the TEST's own rubrics,
# each carrying a banner line so a rendered prompt says which rubric produced it.
# That is what lets the R8.4 routing assertion be made about the text that reached
# the backend, rather than about a return value that could be right while the
# prompt is wrong.

NORMAL_RUBRIC = """\
system: |
  RUBRIC: standard-answerable-v1
  You are a strict grader. Reply with exactly one word: correct or incorrect.

user: |
  QID: {{ qid }}
  CONDITION: {{ condition_key }}
  QUESTION: {{ question }}
  PROPOSED ANSWER: {{ answer }}
  GOLD: {{ gold }}
  Verdict:
"""

ABSTENTION_RUBRIC = """\
system: |
  RUBRIC: abstention-v1
  You are a strict grader of abstention answers. Reply with exactly one word:
  correct or incorrect.

user: |
  QID: {{ qid }}
  CONDITION: {{ condition_key }}
  QUESTION: {{ question }}
  PROPOSED ANSWER: {{ answer }}
  GOLD: {{ gold }}
  A decline counts as CORRECT only if it is a clean decline. Declines:
  {% for marker in decline_markers %}- {{ marker }}
  {% endfor %}
  A decline that also volunteers a guess is INCORRECT. Guesses:
  {% for marker in guess_markers %}- {{ marker }}
  {% endfor %}
  Verdict:
"""

SYNTHETIC_TEMPLATES = {
    JUDGE_PROMPT_TEMPLATE: NORMAL_RUBRIC,
    JUDGE_ABSTENTION_PROMPT_TEMPLATE: ABSTENTION_RUBRIC,
}

STANDARD_BANNER = "RUBRIC: standard-answerable-v1"
ABSTENTION_BANNER = "RUBRIC: abstention-v1"

QID_MARKER = re.compile(r"^QID: (\S+)$", re.MULTILINE)


def _qid_of(request: GenerationRequest) -> str:
    """The qid as it appears in a RENDERED prompt, read back out of what was sent."""
    match = QID_MARKER.search(request.prompt)
    assert match is not None, f"rendered prompt carries no QID line:\n{request.prompt}"
    return match.group(1)


# ── the fake backend ──────────────────────────────────────────────────────────


class _Residency:
    """Which checkpoints are open on the card RIGHT NOW.

    Deliberately NOT `panel._resident`: that is the thing under test, and a test
    that only reads the code it is testing proves nothing.
    """

    def __init__(self) -> None:
        self.live: list[str] = []
        self.max_live = 0
        self.events: list[str] = []
        self.loads: list[str] = []

    def open(self, model: str) -> None:
        if self.live:
            raise AssertionError(
                f"{model!r} was loaded while {self.live!r} was still resident. The two "
                f"judge checkpoints are 7.98 GiB of weights against an 8 GiB card, so "
                f"this is an OOM, not a slower path."
            )
        self.live.append(model)
        self.loads.append(model)
        self.max_live = max(self.max_live, len(self.live))
        self.events.append(f"load {model}")

    def close(self, model: str) -> None:
        assert self.live and self.live[-1] == model, (
            f"close({model!r}) with {self.live!r} open; the panel must release the "
            f"outgoing judge before acquiring the next one"
        )
        self.live.pop()
        self.events.append(f"close {model}")


class FakeJudgeBackend:
    """A `LocalBackend` that returns scripted verdicts and tracks residency.

    `decide` receives the FULL request, so a test can key on anything the panel
    actually sent — the rubric banner, the seed, the qid — rather than on something
    the panel reported out of band.
    """

    name = "fake-judge"

    def __init__(
        self,
        spec: JudgeSpec,
        residency: _Residency,
        decide: Callable[[GenerationRequest], GenerationResult],
        *,
        log: list[GenerationRequest] | None = None,
    ) -> None:
        self.spec = spec
        self._residency = residency
        self._decide = decide
        self.requests: list[GenerationRequest] = []
        self.log: list[GenerationRequest] = [] if log is None else log
        self.closed = False
        self.name = f"fake-judge:{spec.id}"
        residency.open(spec.model)

    def health(self) -> dict[str, Any]:
        return {"ready": True, "backend": "fake-judge", "model": self.spec.model}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        self.log.append(request)
        return self._decide(request)

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [self.generate(r) for r in requests]

    def context_window(self, model: str) -> int:
        return 32768

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._residency.close(self.spec.model)


def _ok(text: str, *, finish_reason: str = "stop") -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=1200,
        completion_tokens=2,
        latency_ms=41.0,
        backend="fake-judge",
        model_version="synthetic",
    )


def _verdicts_for(
    mapping: Mapping[str, Verdict],
) -> Callable[[GenerationRequest], GenerationResult]:
    """One judge, one qid -> verdict. A missing qid is a fixture bug, not a refusal."""

    def decide(request: GenerationRequest) -> GenerationResult:
        qid = _qid_of(request)
        assert qid in mapping, f"no scripted verdict for {qid!r}; scripted {sorted(mapping)}"
        return _ok(str(mapping[qid]))

    return decide


def _repeat(text: str, *, finish_reason: str = "stop") -> Callable[..., GenerationResult]:
    def decide(request: GenerationRequest) -> GenerationResult:
        return _ok(text, finish_reason=finish_reason)

    return decide


# ── panel builders ────────────────────────────────────────────────────────────


def _panel(
    scripts: Mapping[str, Mapping[str, Verdict]],
    residency: _Residency | None = None,
    *,
    pricing: Mapping[str, Any] | None = None,
    templates: Mapping[str, str] | None = None,
    abstention_rule: Mapping[str, Any] | None = None,
    reader_families: Sequence[str] | None = None,
    unloadable: Mapping[str, str] | None = None,
    log: list[GenerationRequest] | None = None,
    decide: Callable[[GenerationRequest], GenerationResult] | None = None,
) -> JudgePanel:
    """A real panel from the real config, with only the ENGINE swapped for a fake.

    The model ids, SHAs, VRAM arithmetic and independence check are the shipped
    ones, because those are precisely what a hand-written spec would let a bug
    through.
    """
    residency = residency if residency is not None else _Residency()
    unavailable = dict(unloadable or {})

    def load(spec: JudgeSpec) -> FakeJudgeBackend:
        if spec.id in unavailable:
            raise BackendUnavailableError(unavailable[spec.id])
        answer = decide if decide is not None else _verdicts_for(scripts.get(spec.id, {}))
        return FakeJudgeBackend(spec, residency, answer, log=log)

    return judge_panel_from_config(
        copy.deepcopy(pricing if pricing is not None else PRICING),
        templates=dict(templates if templates is not None else SYNTHETIC_TEMPLATES),
        abstention_rule=dict(
            ABSTENTION_RULE if abstention_rule is None else abstention_rule
        ),
        reader_families=list(reader_families if reader_families is not None else READER_FAMILIES),
        load=load,
    )


def _items(n: int = 4) -> list[JudgeItem]:
    """`n` answerable items plus one abstention item. Synthetic questions."""
    out = [
        JudgeItem(
            condition_key="A|1|512",
            qid=f"q{i}",
            question=f"synthetic question {i}",
            answer=f"synthetic answer {i}",
            gold=f"synthetic gold {i}",
        )
        for i in range(n)
    ]
    out.append(
        JudgeItem(
            condition_key="A|1|512",
            qid="q0_abs",
            question="synthetic abstention question",
            answer="I don't know.",
            gold="",
        )
    )
    return out


def _pilot_n(n: int) -> list[JudgeItem]:
    return [
        JudgeItem(
            condition_key="A|1|512",
            qid=f"p{i}",
            question=f"synthetic pilot item {i}",
            answer=f"synthetic pilot answer {i}",
            gold=f"synthetic pilot gold {i}",
        )
        for i in range(n)
    ]


def _pilot_with(
    judge_1: Sequence[Verdict], judge_2: Sequence[Verdict]
) -> tuple[JudgePanel, list[JudgeItem]]:
    """A pilot whose two judges have scripted verdicts. n == 100 for even margins."""
    items = _pilot_n(len(judge_1))
    panel = _panel(
        {
            "judge_1": {i.qid: v for i, v in zip(items, judge_1, strict=True)},
            "judge_2": {i.qid: v for i, v in zip(items, judge_2, strict=True)},
        }
    )
    return panel, items


def _alternating(n: int) -> list[Verdict]:
    """Balanced CORRECT/INCORRECT. A 95%-INCORRECT pilot reports a high kappa for two
    judges that are both always wrong, which is why these fixtures are balanced."""
    return [Verdict.CORRECT if i % 2 == 0 else Verdict.INCORRECT for i in range(n)]


def _all(n: int, verdict: Verdict) -> list[Verdict]:
    return [verdict] * n


# ═══ the panel builds from the shipped config ════════════════════════════════


def test_the_panel_builds_from_the_shipped_pricing_config() -> None:
    panel = _panel({"judge_1": {}, "judge_2": {}})

    assert panel.judge_ids == ("judge_1", "judge_2")
    assert panel.get("judge_1").model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert panel.get("judge_1").version == "c170c708c41dac9275d15a8fff4eca08d52bab71"
    assert panel.get("judge_2").model == "ibm-granite/granite-3.3-8b-instruct"
    assert panel.get("judge_2").version == "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
    assert panel.n_judges >= 2, "R8.2 requires >= 2 judges; kappa is undefined with one"


def test_a_version_is_never_null_or_missing_r1_1() -> None:
    """R1.1: a field that looks populated and names nothing is the failure mode."""
    base = dict(PRICING["judges_local"][0])

    with pytest.raises(PanelSpecError, match="version"):
        judge_from_entry({**base, "version": None}, max_tokens=100)

    with pytest.raises(PanelSpecError, match="version"):
        judge_from_entry({k: v for k, v in base.items() if k != "version"}, max_tokens=100)

    with pytest.raises(PanelSpecError, match="version"):
        judge_from_entry({**base, "version": "   "}, max_tokens=100)


def test_the_refusal_on_a_null_version_names_the_judge_and_r1_1() -> None:
    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"][0]["version"] = None

    with pytest.raises(PanelSpecError) as excinfo:
        _panel({"judge_1": {}, "judge_2": {}}, pricing=pricing)

    message = str(excinfo.value)
    assert "judge_1" in message
    assert "version" in message
    assert "R1.1" in message


def test_a_version_must_be_a_revision_commit_not_a_tag() -> None:
    """A repointable tag changes the weights under a fixed `config_hash`."""
    base = dict(PRICING["judges_local"][0])

    with pytest.raises(PanelSpecError, match="40-character revision commit"):
        judge_from_entry({**base, "version": "v0.3"}, max_tokens=100)


def test_an_unconfirmed_checkpoint_is_refused() -> None:
    base = dict(PRICING["judges_local"][0])

    with pytest.raises(PanelSpecError, match="UNCONFIRMED checkpoint"):
        judge_from_entry({**base, "exact_checkpoint_confirmed": False}, max_tokens=100)


@pytest.mark.parametrize("token", ["UNCONFIRMED", "TODO", "TBD", "XXX"])
def test_a_placeholder_identifier_is_refused(token: str) -> None:
    base = dict(PRICING["judges_local"][0])

    with pytest.raises(PanelSpecError, match="placeholder"):
        judge_from_entry({**base, "released": token}, max_tokens=100)


def test_a_zero_output_cap_is_refused() -> None:
    """A cap of zero truncates every verdict to nothing and the panel reports
    unanimous abstention — which looks like a finding."""
    base = dict(PRICING["judges_local"][0])

    with pytest.raises(PanelSpecError, match="must be > 0"):
        judge_from_entry(base, max_tokens=0)


def test_a_judge_without_a_vram_block_is_refused() -> None:
    """The weights footprint is what makes the panel sequential; no block, no schedule."""
    base = {k: v for k, v in PRICING["judges_local"][0].items() if k != "vram"}

    with pytest.raises(PanelSpecError, match="no `vram:` block"):
        judge_from_entry(base, max_tokens=100)


def test_a_judge_too_large_for_the_card_alone_is_refused() -> None:
    """No swap schedule puts a 10 GB checkpoint in 8 GB of VRAM."""
    base = copy.deepcopy(PRICING["judges_local"][0])
    base["vram"] = {**base["vram"], "params_b": 20.0}
    spec = judge_from_entry(base, max_tokens=100)
    assert spec.weights_gb > 8.0

    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"] = [base, pricing["judges_local"][1]]

    with pytest.raises(JudgeVramError, match="weights"):
        _panel({"judge_1": {}, "judge_2": {}}, pricing=pricing)


def test_a_single_judge_panel_is_refused() -> None:
    """With one rater 'agreement' is a constant and the bar cannot be measured."""
    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"] = pricing["judges_local"][:1]

    with pytest.raises(PanelSpecError, match="at least 2"):
        _panel({"judge_1": {}}, pricing=pricing)


def test_duplicate_judge_ids_are_refused() -> None:
    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"][1] = {**pricing["judges_local"][1], "id": "judge_1"}

    with pytest.raises(PanelSpecError, match="unique"):
        _panel({"judge_1": {}}, pricing=pricing)


def test_an_undeclared_rubric_is_a_loud_build_error() -> None:
    """A judge with no rubric grades on its own pretraining."""
    partial = {JUDGE_PROMPT_TEMPLATE: NORMAL_RUBRIC}

    with pytest.raises(PanelSpecError, match=JUDGE_ABSTENTION_PROMPT_TEMPLATE):
        _panel({"judge_1": {}, "judge_2": {}}, templates=partial)


def test_an_empty_abstention_rule_is_refused() -> None:
    """R8.4 gives `_abs` items an explicit rule; an empty one is improvisation."""
    with pytest.raises(PanelSpecError, match="abstention scoring rule is empty"):
        _panel({"judge_1": {}, "judge_2": {}}, abstention_rule={})


def test_the_declared_judge_rubrics_are_declared_but_not_shipped() -> None:
    """PINS THE KNOWN GAP rather than letting a default prompt paper over it.

    `configs/prompts/judge.yaml` and `configs/prompts/judge_abstention.yaml` do not
    exist. `read_templates` therefore returns nothing for them and
    `judge_panel_from_config` refuses the panel by name. Config files are outside
    this module's ownership, so the gap is recorded here and reported rather than
    closed with a default prompt that would differ from a reviewed one.
    """
    declared = [JUDGE_PROMPT_TEMPLATE, JUDGE_ABSTENTION_PROMPT_TEMPLATE]
    found = read_templates(CONFIG, declared)

    assert JUDGE_PROMPT_TEMPLATE not in found
    assert JUDGE_ABSTENTION_PROMPT_TEMPLATE not in found
    with pytest.raises(PanelSpecError, match="were not supplied"):
        _panel({"judge_1": {}, "judge_2": {}}, templates=found)


# ═══ R8.2: a judge sharing weights with a reader is not a judge ═════════════


def test_the_shipped_panel_families_appear_nowhere_in_the_reader_panel() -> None:
    """The reason mistral and granite were chosen. Re-derived here, not asserted
    once in a comment."""
    assert set(READER_FAMILIES) == {"gemma", "qwen", "phi"}

    panel = _panel({"judge_1": {}, "judge_2": {}})
    families = {spec.id: spec.family for spec in panel.judges}

    assert families == {"judge_1": "mistral", "judge_2": "granite"}
    for judge_id, family in families.items():
        assert family not in READER_FAMILIES, f"{judge_id} shares weights with a reader"


def test_a_judge_sharing_a_reader_family_is_refused() -> None:
    """The check the pinned SHAs alone cannot make: a correct, existing checkpoint
    can still be the wrong one."""
    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"][0] = {
        **pricing["judges_local"][0],
        "model": "Qwen/Qwen3.5-9B",
        "version": "0" * 40,
    }

    with pytest.raises(JudgeIndependenceError) as excinfo:
        _panel({"judge_1": {}, "judge_2": {}}, pricing=pricing)

    message = str(excinfo.value)
    assert "judge_1" in message
    assert "qwen" in message
    assert "independently adjudicate" in message


def test_two_judges_of_the_same_family_are_refused() -> None:
    """Two entries naming the same weights are one rater entered twice; kappa
    between identical deterministic outputs is a tautology."""
    pricing = copy.deepcopy(PRICING)
    granite = pricing["judges_local"][1]
    pricing["judges_local"] = [dict(granite, id="judge_1"), dict(granite, id="judge_2")]

    with pytest.raises(JudgeIndependenceError, match="one rater entered twice"):
        _panel({"judge_1": {}, "judge_2": {}}, pricing=pricing)


def test_an_unidentifiable_lineage_is_refused_not_assumed_independent() -> None:
    """A lineage nobody can name cannot be shown absent from the reader panel."""
    pricing = copy.deepcopy(PRICING)
    pricing["judges_local"][0] = {
        **pricing["judges_local"][0],
        "model": "acme-labs/house-model-9b",
    }

    with pytest.raises(JudgeIndependenceError, match="no recognisable lineage"):
        _panel({"judge_1": {}, "judge_2": {}}, pricing=pricing)


def test_lineage_matching_is_on_segments_not_substrings() -> None:
    """`phi` is a three-letter token that occurs inside a dozen unrelated words."""
    assert model_family("microsoft/Phi-4-mini-instruct") == "phi"
    assert model_family("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
    assert model_family("ibm-granite/granite-3.3-8b-instruct") == "granite"
    assert model_family("acme-labs/house-model-9b") is None


# ═══ the hardware constraint: 8 GB, two checkpoints, one at a time ══════════


def test_the_two_checkpoints_do_not_fit_together_on_the_card() -> None:
    """7.98 GiB of weights against an 8 GiB card, before any KV cache.

    `fits_concurrently` is the operational fact, and it is driven by the headroom
    policy rather than by the raw total — which is the point: 7.98 GiB leaves
    0.02 GiB, which is less than a KV cache and far less than allocator slack.
    """
    panel = _panel({"judge_1": {}, "judge_2": {}})

    assert panel.fits_concurrently is False
    assert panel.weights_gb + panel.headroom_gb > panel.card_gb
    assert panel.card_gb - panel.weights_gb < 1.0, "there is no room for a second checkpoint"
    why = panel.why_sequential()
    assert "ONE judge at a time" in why
    assert f"{panel.weights_gb:.2f} GB" in why


def test_every_judge_fits_alone_and_peak_residency_is_one_of_them() -> None:
    """Quoting the SUM as the requirement would overstate it and invite the wrong
    fix ('buy a bigger card') for a panel that already runs."""
    panel = _panel({"judge_1": {}, "judge_2": {}})

    for spec in panel.judges:
        assert spec.weights_gb < panel.card_gb, f"{spec.id} cannot be scheduled at all"
        assert spec.kv_bytes_per_token > 0, f"{spec.id} has no KV arithmetic to reserve"

    assert panel.resident_peak_gb == max(s.footprint_gb() for s in panel.judges)
    assert panel.resident_peak_gb < panel.card_gb
    assert panel.resident_peak_gb < panel.weights_gb


def test_acquire_refuses_a_second_resident_judge() -> None:
    panel = _panel({"judge_1": {}, "judge_2": {}})
    first, second = panel.judges

    engine = panel.acquire(first)
    try:
        with pytest.raises(JudgeSequentialError) as excinfo:
            panel.acquire(second)
        assert first.id in str(excinfo.value)
        assert second.id in str(excinfo.value)
    finally:
        panel.release(first, engine)

    engine2 = panel.acquire(second)
    panel.release(second, engine2)


def test_release_is_idempotent() -> None:
    """The `finally` that guarantees a swap and the happy path can both reach it."""
    panel = _panel({"judge_1": {}, "judge_2": {}})
    spec = panel.judges[0]

    engine = panel.acquire(spec)
    panel.release(spec, engine)
    panel.release(spec, engine)

    assert panel.acquire(panel.judges[1]) is not None


def test_the_panel_never_holds_two_checkpoints_at_once() -> None:
    """THE TEST THAT FAILS IF SOMEONE LOADS BOTH JUDGES.

    `_Residency.open` raises on a second concurrent load, independently of
    `JudgePanel.acquire`. Deleting the refusal in `acquire` therefore breaks this
    test rather than producing a run that OOMs at hour 60 of 75.
    """
    items = _items(4)
    residency = _Residency()
    panel = _panel(
        {
            "judge_1": {i.qid: Verdict.CORRECT for i in items},
            "judge_2": {i.qid: Verdict.CORRECT for i in items},
        },
        residency,
    )

    run = run_panel(panel, items)

    assert residency.max_live == 1
    assert residency.live == [], "every checkpoint must be released when the run returns"
    assert residency.loads == [s.model for s in panel.judges], (
        "each judge is loaded once for the whole item list. An item-major loop would "
        "re-load a ~4 GB checkpoint per item and put the swap cost above the "
        "generation cost for all ~162,000 items."
    )
    assert run.execution_order == ["judge_1", "judge_2"]
    assert [e.split()[0] for e in residency.events] == ["load", "close", "load", "close"]


def test_the_fake_backend_itself_would_catch_a_concurrent_load() -> None:
    """Guards the guard.

    Without this, a future change that made `_Residency.open` permissive would turn
    the test above into a tautology that passes while the panel loads both judges.
    """
    residency = _Residency()
    panel = _panel({"judge_1": {}, "judge_2": {}}, residency)

    engine = panel.acquire(panel.judges[0])
    with pytest.raises(AssertionError, match="still resident"):
        FakeJudgeBackend(panel.judges[1], residency, _verdicts_for({}))
    panel.release(panel.judges[0], engine)


def test_a_concurrency_violation_is_not_filed_as_an_unavailable_judge() -> None:
    """Two checkpoints will not load is a bug in this module, not a dead engine.

    If it were swallowed into an `UnavailableRow`, `run_panel(...,
    on_backend_unavailable="exclude")` would quietly return a one-judge run and the
    invariant would be reported as a scheduling problem.
    """
    panel = _panel({"judge_1": {}, "judge_2": {}})
    first, second = panel.judges
    engine = panel.acquire(first)
    try:
        with pytest.raises(JudgeSequentialError):
            run_panel(panel, _items(1), on_backend_unavailable="exclude")
        assert second.id
    finally:
        panel.release(first, engine)


def test_readiness_also_respects_the_sequential_invariant() -> None:
    """`panel_health` probes every judge before the pilot; it must not overlap them."""
    residency = _Residency()
    panel = _panel({"judge_1": {}, "judge_2": {}}, residency)

    rows = panel_readiness(panel)

    assert [r.judge_id for r in rows] == ["judge_1", "judge_2"]
    assert all(r.ready for r in rows)
    assert residency.max_live == 1
    assert residency.live == []


def test_a_checkpoint_is_released_even_when_generation_raises() -> None:
    """A swap guaranteed only on the happy path leaks VRAM for the rest of the run."""
    residency = _Residency()
    panel = _panel(
        {}, residency, decide=_repeat("correct", finish_reason="stop")
    )

    def explode(request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("CUDA out of memory")

    panel = _panel({}, residency, decide=explode)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_panel(panel, _items(2))

    assert residency.live == []
    assert residency.max_live == 1


# ═══ R8.4: abstention is a different prompt, not a different score ═══════════


def test_abstention_items_route_to_the_abstention_rubric() -> None:
    panel = _panel({"judge_1": {}, "judge_2": {}})
    spec = panel.judges[0]

    abstention = JudgeItem(condition_key="A|1|512", qid="q7_abs", question="q", answer="a")
    answerable = JudgeItem(condition_key="A|1|512", qid="q7", question="q", answer="a")

    assert spec.template_for(abstention) == JUDGE_ABSTENTION_PROMPT_TEMPLATE
    assert spec.template_for(answerable) == JUDGE_PROMPT_TEMPLATE


def test_routing_uses_the_suffix_the_benchmark_uses() -> None:
    """`_abs` is `arms.evidence.ABSTENTION_SUFFIX`, the same constant the metrics read."""
    from msat.arms.evidence import ABSTENTION_SUFFIX

    assert ABSTENTION_SUFFIX == "_abs"

    def item(qid: str) -> JudgeItem:
        return JudgeItem(
            condition_key="c", qid=qid, question="q", answer="a", gold="g"
        )

    assert item(f"q1{ABSTENTION_SUFFIX}").is_abstention is True
    assert item("q1").is_abstention is False


def test_every_rendered_prompt_carries_its_routed_rubric_banner() -> None:
    """Asserted on the text the backend RECEIVED, so a correct return value over a
    wrong prompt cannot pass.

    The banner lives in the rubric's `system:` block, so the assertion reads
    `request.system`. Checking `request.prompt` would pass against an empty
    banner while the judge was handed the standard rubric for an abstention item.
    """
    items = _items(4)
    log: list[GenerationRequest] = []
    panel = _panel(
        {
            "judge_1": {i.qid: Verdict.CORRECT for i in items},
            "judge_2": {i.qid: Verdict.CORRECT for i in items},
        },
        log=log,
    )

    run_panel(panel, items)

    assert len(log) == len(items) * 2
    for request in log:
        if _qid_of(request).endswith("_abs"):
            assert ABSTENTION_BANNER in request.system
            assert STANDARD_BANNER not in request.system
        else:
            assert STANDARD_BANNER in request.system
            assert ABSTENTION_BANNER not in request.system


def test_the_abstention_rubric_never_receives_gold() -> None:
    """An `_abs` item is defined by the ABSENCE of evidence.

    Handing the rubric a gold string teaches it the answer to the item it is
    grading, and every abstention verdict after that measures the prompt. The item
    below deliberately CARRIES a gold, because the guarantee has to survive a
    caller that supplies one.
    """
    panel = _panel({"judge_1": {}, "judge_2": {}})
    spec = panel.judges[0]
    leaky = JudgeItem(
        condition_key="A|1|512", qid="q0_abs", question="q", answer="a", gold="LEAKED-GOLD"
    )

    _, user = panel.render(spec, leaky)

    assert "LEAKED-GOLD" not in user
    assert re.search(r"^GOLD:\s*$", user, re.MULTILINE)


def test_the_standard_rubric_receives_the_gold() -> None:
    panel = _panel({"judge_1": {}, "judge_2": {}})
    spec = panel.judges[0]
    item = JudgeItem(
        condition_key="A|1|512", qid="q0", question="q", answer="a", gold="the-gold"
    )

    _, user = panel.render(spec, item)

    assert "GOLD: the-gold" in user


def test_the_abstention_rubric_carries_the_declared_markers() -> None:
    """R1.2: the rule lives in config, so the judge and the metric apply ONE rule."""
    panel = _panel({"judge_1": {}, "judge_2": {}})
    spec = panel.judges[0]
    item = JudgeItem(condition_key="A|1|512", qid="q0_abs", question="q", answer="a")

    _, user = panel.render(spec, item)

    for marker in ABSTENTION_RULE["decline_markers"]:
        assert f"- {marker}" in user
    for marker in ABSTENTION_RULE["guess_markers"]:
        assert f"- {marker}" in user


def test_a_template_routed_to_but_not_supplied_is_a_loud_error() -> None:
    """A missing rubric renders as a prompt with no scoring rule, and the resulting
    verdicts look like judge noise rather than a missing file."""
    panel = _panel({"judge_1": {}, "judge_2": {}})
    bare = JudgePanel(
        judges=panel.judges,
        templates={k: v for k, v in panel.templates.items() if "abstention" not in k},
        abstention_rule=panel.abstention_rule,
        reader_families=panel.reader_families,
        load=panel.load,
    )
    item = JudgeItem(condition_key="c", qid="q0_abs", question="q", answer="a")

    with pytest.raises(PanelSpecError, match="was not supplied"):
        bare.source_for(bare.judges[0], item)


# ═══ R8.3: per item, never a single aggregate ═══════════════════════════════


def _split_scripts() -> tuple[dict[str, Verdict], dict[str, Verdict]]:
    """Two judges agreeing on 3 of 5 items, splitting on 2 — one of them abstention."""
    items = _items(4)
    a = dict.fromkeys([i.qid for i in items], Verdict.CORRECT)
    b = dict(a)
    b["q2"] = Verdict.INCORRECT
    b["q0_abs"] = Verdict.INCORRECT
    return a, b


def _split_panel() -> JudgePanel:
    a, b = _split_scripts()
    return _panel({"judge_1": a, "judge_2": b})


def test_run_panel_returns_one_verdict_per_item_and_judge() -> None:
    items = _items(4)

    run = run_panel(_split_panel(), items)

    assert run.n_items == len(items)
    assert run.n_judges == 2
    assert len(run.verdicts.verdicts) == len(items) * 2
    assert run.verdicts.items() == sorted(i.key for i in items)
    for judge_id in ("judge_1", "judge_2"):
        assert set(run.verdicts.for_judge(judge_id)) == {i.key for i in items}


def test_run_panel_returns_the_disagreeing_items_with_their_own_verdicts() -> None:
    run = run_panel(_split_panel(), _items(4))

    assert {(d.qid, d.judge_a, d.judge_b) for d in run.disagreements} == {
        ("q2", "judge_1", "judge_2"),
        ("q0_abs", "judge_1", "judge_2"),
    }
    by_qid = {d.qid: d for d in run.disagreements}
    assert by_qid["q2"].verdict_a is Verdict.CORRECT
    assert by_qid["q2"].verdict_b is Verdict.INCORRECT
    assert by_qid["q2"].is_abstention is False
    assert by_qid["q2"].answer == "synthetic answer 2"
    assert by_qid["q0_abs"].is_abstention is True
    assert by_qid["q0_abs"].answer == "I don't know."


def test_the_disagreements_agree_with_the_verdict_set() -> None:
    """One comparison, not two: kappa's matrix and the disagreement rows must not
    be able to come from different implementations."""
    run = run_panel(_split_panel(), _items(4))

    assert sorted(run.verdicts.disagreements("judge_1", "judge_2")) == sorted(
        (d.condition_key, d.qid) for d in run.disagreements
    )


def test_a_panel_run_carries_no_accuracy_scalar() -> None:
    """A function that returns 0.62 can be read without opening anything, and 0.62
    cannot say WHICH items the panel is wrong about (R8.3).

    The check is on the KEYS, not on the word "accuracy" anywhere in the payload:
    `retention_note` is required to say out loud that no aggregate is returned, so a
    substring test would fail on the very sentence that documents the invariant.
    """
    payload = run_panel(_split_panel(), _items(4)).as_dict()

    assert "accuracy" not in payload
    assert "score" not in payload
    scalars = {
        k: v
        for k, v in payload.items()
        if isinstance(v, float | int) and not isinstance(v, bool)
    }
    assert set(scalars) <= {"n_items", "n_judges", "n_verdicts", "pass_index",
                            "prompt_tokens", "completion_tokens",
                            "wall_seconds", "resident_peak_gb"}
    assert len(payload["verdicts"]) == 10
    assert all(
        {"condition_key", "qid", "judge_id", "verdict", "pass_index"} <= set(row)
        for row in payload["verdicts"]
    )
    assert {d["qid"] for d in payload["disagreements"]} == {"q2", "q0_abs"}


def test_n_abstention_counts_items_not_disagreements() -> None:
    """R8.4 reports abstention separately from recall@K, so the number has to be a
    count of abstention ITEMS. Counting disagreements here would put a statistic
    about panel stability under a statistic about coverage's name."""
    run = run_panel(_split_panel(), _items(4))

    assert run.n_abstention == 1
    assert run.n_items == 5


def test_judge_item_returns_one_verdict_per_judge() -> None:
    item = _items(1)[0]

    verdicts = judge_item(_split_panel(), item)

    assert sorted(v.judge_id for v in verdicts) == ["judge_1", "judge_2"]
    assert {v.qid for v in verdicts} == {item.qid}


def test_judge_item_runs_the_same_discipline_as_the_sweep() -> None:
    """One code path, not a second one that could drift. `judge_item` pays a
    load/unload per judge, which is why the 75-hour sweep uses `run_panel`."""
    residency = _Residency()
    item = _items(1)[0]
    panel = _panel(
        {"judge_1": {item.qid: Verdict.CORRECT}, "judge_2": {item.qid: Verdict.INCORRECT}},
        residency,
    )

    judge_item(panel, item)

    assert residency.max_live == 1
    assert residency.live == []


def test_unparseable_judge_output_is_unscored_not_incorrect() -> None:
    """Scoring a hedge INCORRECT charges the reader for the judge's behaviour, and
    reading hedges as verdicts manufactures agreement (the failure `as_verdict`
    exists to refuse).

    `_items(1)` is TWO items, not one: it appends the `_abs` item. So the refusal
    count is items x judges, and asserting `2` here would have silently pinned the
    test to a panel that silently skipped abstention items.
    """
    panel = _panel({}, decide=_repeat("probably correct"))
    items = _items(1)

    run = run_panel(panel, items, on_backend_unavailable="exclude")

    assert run.verdicts.verdicts == []
    assert len(run.unscored) == len(items) * panel.n_judges
    assert all("not a verdict" in u.reason for u in run.unscored)


def test_a_truncated_judge_output_is_unscored() -> None:
    """R5.5: `length` is an explicit finish reason, and it is missing data."""
    panel = _panel({}, decide=_repeat("corr", finish_reason="length"))

    run = run_panel(panel, _items(1), on_backend_unavailable="exclude")

    assert run.verdicts.verdicts == []
    assert all(u.reason == "finish_reason=length" for u in run.unscored)


def test_prose_is_refused_not_read_as_a_verdict() -> None:
    """A judge told to "reply with exactly one word" sometimes replies with a
    sentence. Reading that sentence's verdict out of it is the change that
    manufactures agreement, so it is refused and COUNTED.

    This is the deliberate reading of `verdicts.as_verdict`, which states that
    "the answer looks correct" is not converted, and of `_first_line_verdict`,
    which narrows the reduction to the first line precisely so that meaning does
    not have to be reached for inside a sentence. Nothing is lost: the refusal
    lands in `run.unscored` as an explicit row, so a judge that does this
    everywhere is caught by the pilot instead of quietly reshaping the panel.
    """
    panel = _panel({}, decide=_repeat("The answer is correct."))
    items = _items(1)

    run = run_panel(panel, items, on_backend_unavailable="exclude")

    assert run.verdicts.verdicts == []
    assert len(run.unscored) == len(items) * panel.n_judges
    assert all("not a verdict" in u.reason for u in run.unscored)


def test_the_reduction_is_the_first_non_empty_line_not_the_best_one() -> None:
    """Leading prose does not hand the run a second chance.

    If the reduction scanned for whichever line looked most like a verdict, a
    judge could emit a refusal and a verdict and be scored on the second. Taking
    the first non-empty line means the refusal is what counts.
    """
    panel = _panel({}, decide=_repeat("I'm not sure.\n\ncorrect"))

    run = run_panel(panel, _items(1), on_backend_unavailable="exclude")

    assert run.verdicts.verdicts == []
    assert run.unscored


def test_an_empty_judge_output_is_unscored() -> None:
    panel = _panel({}, decide=_repeat(""))

    run = run_panel(panel, _items(1), on_backend_unavailable="exclude")

    assert run.verdicts.verdicts == []
    assert all("no output" in u.reason for u in run.unscored)


# ═══ R8.2: the 0.118 bar is a refusal, not a footnote ═══════════════════════


def test_kappa_below_the_bar_raises_and_the_message_states_the_bar() -> None:
    """A panel that agrees no better than chance-corrected cannot support a headline."""
    panel, items = _pilot_with(_alternating(100), _all(100, Verdict.INCORRECT))

    with pytest.raises(PanelKappaBelowBarError) as excinfo:
        panel_health(panel, items)

    message = str(excinfo.value)
    assert "0.118" in message
    assert "2511.02246" in message
    assert "NOT usable" in message
    assert "PILOT" in message, "the refusal must say WHEN to run the check"


def test_the_measured_kappa_really_is_below_the_bar() -> None:
    """The fixture is checked against `cohen_kappa`, not against a typed literal."""
    panel, items = _pilot_with(_alternating(100), _all(100, Verdict.INCORRECT))

    run = run_panel(panel, items)
    measured = cohen_kappa(
        run.verdicts.for_judge("judge_1"),
        run.verdicts.for_judge("judge_2"),
        label_a="judge_1",
        label_b="judge_2",
    ).kappa

    assert measured is not None
    assert measured < CAUTIONARY_KAPPA
    with pytest.raises(PanelKappaBelowBarError):
        panel_health(panel, items)


def test_the_bar_error_carries_the_agreement_block() -> None:
    """R8.7: the audit is written whether or not the panel passed, so the caller
    still needs the numbers after the refusal."""
    panel, items = _pilot_with(_alternating(100), _all(100, Verdict.INCORRECT))

    with pytest.raises(PanelKappaBelowBarError) as excinfo:
        panel_health(panel, items)

    assert excinfo.value.bar == CAUTIONARY_KAPPA == 0.118
    assert excinfo.value.agreement["panel_usable"] is False
    assert excinfo.value.agreement["verdicts"][0]["cautionary_bar"] == CAUTIONARY_KAPPA
    assert excinfo.value.agreement["base_rate_correct"]["judge_2"] == 0.0


def test_the_health_kappa_is_the_one_kappa_module_computes() -> None:
    """One implementation of the statistic, not two that agree until a verdict is
    missing from one of them."""
    panel, items = _pilot_with(
        _alternating(100), [Verdict.INCORRECT if i % 2 == 0 else Verdict.CORRECT for i in range(100)]
    )

    health = panel_health(panel, items, on_unusable="report")
    by_judge = run_panel(panel, items).verdicts_by_judge()
    direct = cohen_kappa(
        by_judge["judge_1"], by_judge["judge_2"], label_a="judge_1", label_b="judge_2"
    )

    assert health.agreement["pairs"][0]["agreement_matrix"] == direct.cells()
    assert health.agreement["pairs"][0]["kappa"] == direct.kappa
    assert health.agreement["verdicts"][0]["cautionary_bar"] == CAUTIONARY_KAPPA


def test_a_panel_that_clears_the_bar_is_accepted() -> None:
    a = _alternating(100)
    b = list(a)
    for i in range(5):
        b[i] = Verdict.INCORRECT
    panel, items = _pilot_with(a, b)

    health = panel_health(panel, items)

    assert health.usable is True
    assert health.kappa_min is not None
    assert health.kappa_min >= CAUTIONARY_KAPPA
    assert health.n_items == 100
    assert health.blocked == []
    assert health.headline()["panel_usable"] is True
    assert health.headline()["n_judges"] == 2


def test_an_undefined_kappa_is_reported_as_undefined_not_as_zero() -> None:
    """Unanimous, one-sided verdicts give p_e == 1: the statistic does not exist,
    and reporting 0.0 would say 'the panel disagrees' when it agrees perfectly."""
    panel, items = _pilot_with(_all(20, Verdict.INCORRECT), _all(20, Verdict.INCORRECT))

    with pytest.raises(PanelKappaBelowBarError) as excinfo:
        panel_health(panel, items)

    assert "undefined" in str(excinfo.value)
    assert excinfo.value.agreement["kappa_min"] is None
    assert "UNDEFINED" in excinfo.value.agreement["verdicts"][0]["verdict"]


def test_health_can_report_instead_of_raising() -> None:
    """`on_unusable="report"` exists for the R8.7 standalone artifact, which must be
    written even when the finding is 'nothing'."""
    panel, items = _pilot_with(_alternating(100), _all(100, Verdict.INCORRECT))

    health = panel_health(panel, items, on_unusable="report")

    assert health.usable is False
    assert any("0.118" in note for note in health.notes)
    assert health.as_dict()["bar_source"].startswith("2511.02246")


def test_an_unknown_on_unusable_is_refused() -> None:
    panel, items = _pilot_with(_all(4, Verdict.CORRECT), _all(4, Verdict.CORRECT))

    with pytest.raises(ValueError, match="raise\\|report"):
        panel_health(panel, items, on_unusable="ignore")  # type: ignore[arg-type]


def test_health_is_runnable_on_a_pilot_so_a_noisy_judge_is_found_at_hour_two() -> None:
    """A pilot is a handful of items. The cost of running health must not scale
    with the 75-hour sweep, and the refusal must name the moment to run it."""
    panel, items = _pilot_with(_all(12, Verdict.CORRECT), _all(12, Verdict.CORRECT))

    health = panel_health(panel, items, on_unusable="report")
    rendered = format_health(health)

    assert health.n_items == 12
    assert [r.judge_id for r in health.readiness] == ["judge_1", "judge_2"]
    assert "0.118" in rendered
    assert "judge_1" in rendered and "judge_2" in rendered


def test_health_reports_its_judges_ready_without_loading_weights() -> None:
    """`LocalBackend.health()` is the project's weight-free probe; paying 30 seconds
    and 4 GB to learn the answer is 'no' is what that interface avoids."""
    panel, items = _pilot_with(_all(4, Verdict.CORRECT), _all(4, Verdict.CORRECT))

    health = panel_health(panel, items, on_unusable="report")

    for row in health.readiness:
        assert row.ready is True
        assert row.fits_alone is True
        assert row.family in {"mistral", "granite"}
        assert row.version != ""


# ═══ a judge that cannot be reached is REPORTED, never scored zero ═══════════


def test_an_unavailable_judge_raises_with_its_own_reason() -> None:
    items = _items(2)
    panel = _panel(
        {"judge_1": {i.qid: Verdict.CORRECT for i in items}, "judge_2": {}},
        unloadable={"judge_2": "transformers is not installed on this box"},
    )

    with pytest.raises(JudgeBackendUnavailableError) as excinfo:
        run_panel(panel, items)

    assert "transformers is not installed" in str(excinfo.value)
    assert excinfo.value.reasons == {"judge_2": "transformers is not installed on this box"}


def test_the_partial_run_survives_the_exception() -> None:
    """Whatever was judged before the backend died is not thrown away with it."""
    items = _items(2)
    panel = _panel(
        {"judge_1": {i.qid: Verdict.CORRECT for i in items}, "judge_2": {}},
        unloadable={"judge_2": "CUDA init failed"},
    )

    with pytest.raises(JudgeBackendUnavailableError) as excinfo:
        run_panel(panel, items)

    run = excinfo.value.run
    assert run is not None
    assert len(run.verdicts.for_judge("judge_1")) == len(items)
    assert run.verdicts.for_judge("judge_2") == {}


def test_an_unavailable_judge_is_named_not_scored_incorrect() -> None:
    """A judge scored INCORRECT for being unreachable drags kappa down for a reason
    that has nothing to do with the panel — and that is a figure which gets 'fixed'
    by changing the rubric."""
    items = _items(2)
    panel = _panel(
        {"judge_1": {i.qid: Verdict.CORRECT for i in items}, "judge_2": {}},
        unloadable={"judge_2": "out of memory"},
    )

    run = run_panel(panel, items, on_backend_unavailable="exclude")

    assert run.verdicts.for_judge("judge_2") == {}
    assert run.unscored == [], "an unreachable judge is a gap, not a refusal"
    assert [u.judge_id for u in run.unavailable] == ["judge_2"]
    assert "out of memory" in run.unavailable[0].reason
    assert run.unavailable[0].n_items_affected == len(items)


def test_a_backend_dying_mid_run_reports_every_item_it_never_judged() -> None:
    """Reporting `1` would understate the gap by whatever is left in the queue, and
    an understated gap is what lets a run claim it judged the full set.

    The engine dies on its third call, so each judge clears two items of the FIVE
    `_items(4)` returns (four answerable plus the appended `_abs`) and must report
    the remaining three. Asserting the literal `2` would have pinned the arithmetic
    to a fixture that silently dropped the abstention item from the denominator.
    """
    items = _items(4)
    residency = _Residency()

    def load(spec: JudgeSpec) -> FakeJudgeBackend:
        calls = {"n": 0}

        def decide(request: GenerationRequest) -> GenerationResult:
            calls["n"] += 1
            if calls["n"] > 2:
                raise BackendUnavailableError("engine died at hour 3")
            return _ok("correct")

        return FakeJudgeBackend(spec, residency, decide)

    panel = judge_panel_from_config(
        copy.deepcopy(PRICING),
        templates=SYNTHETIC_TEMPLATES,
        abstention_rule=ABSTENTION_RULE,
        reader_families=READER_FAMILIES,
        load=load,
    )

    run = run_panel(panel, items, on_backend_unavailable="exclude")

    assert len(run.unavailable) == 2, "both judges hit the same dead engine"
    assert all(u.n_items_affected == len(items) - 2 for u in run.unavailable), (
        f"items 3..{len(items)} were never judged; reporting 1 hides the rest of the gap"
    )
    assert len(run.verdicts.verdicts) == 4


def test_panel_health_raises_when_a_judge_cannot_be_reached() -> None:
    """Kappa over one surviving judge is not weak, it is wrong: it measures one rater."""
    panel, items = _pilot_with(_all(40, Verdict.CORRECT), _all(40, Verdict.CORRECT))
    panel = _panel(
        {
            "judge_1": {i.qid: Verdict.CORRECT for i in items},
            "judge_2": {i.qid: Verdict.CORRECT for i in items},
        },
        unloadable={"judge_2": "checkpoint not in the local cache"},
    )

    with pytest.raises(JudgeBackendUnavailableError) as excinfo:
        panel_health(panel, items)

    assert "checkpoint not in the local cache" in str(excinfo.value)
    assert "judge_2" in excinfo.value.reasons
    assert excinfo.value.readiness, "the per-judge readiness rows travel with the refusal"


def test_a_pilot_with_one_dead_judge_reports_zero_pairs_rather_than_a_kappa() -> None:
    panel, items = _pilot_with(_all(40, Verdict.CORRECT), _all(40, Verdict.CORRECT))
    panel = _panel(
        {
            "judge_1": {i.qid: Verdict.CORRECT for i in items},
            "judge_2": {i.qid: Verdict.CORRECT for i in items},
        },
        unloadable={"judge_2": "dead"},
    )

    health = panel_health(panel, items, on_unusable="report")

    assert health.agreement["n_pairs"] == 0
    assert health.agreement["kappa_min"] is None
    assert health.usable is False
    assert any("NOT a statement about the panel" in n for n in health.notes)


def test_readiness_reports_a_judge_whose_engine_says_no() -> None:
    """A probe that reports rather than crashes: being told 'no' is cheap, a
    traceback from `transformers` at hour 0 of the sweep is not."""
    panel = _panel({}, unloadable={"judge_2": "CUDA driver version is insufficient"})

    rows = {r.judge_id: r for r in panel_readiness(panel)}

    assert rows["judge_1"].ready is True
    assert rows["judge_2"].ready is False
    assert "CUDA driver version is insufficient" in rows["judge_2"].reason
    assert rows["judge_2"].family == "granite"
    assert rows["judge_2"].weights_gb > 0


# ═══ R8.6: the second pass must not be a cache hit ══════════════════════════


def test_the_two_passes_do_not_share_a_seed() -> None:
    """`canonical_body()` includes `seed`, so a shared seed is a byte-identical
    request — and the content-addressed store (R6.1) would serve pass 1's verdict,
    giving a flip count of 0 by CONSTRUCTION rather than by determinism."""
    log: list[GenerationRequest] = []
    panel = _panel(
        {
            "judge_1": {f"p{i}": Verdict.CORRECT for i in range(4)},
            "judge_2": {f"p{i}": Verdict.CORRECT for i in range(4)},
        },
        log=log,
    )
    items = _pilot_n(4)

    run_panel(panel, items, pass_index=1)
    first = [r.canonical_body() for r in log]
    log.clear()
    run_panel(panel, items, pass_index=2)
    second = [r.canonical_body() for r in log]

    assert all(b["seed"] == JUDGE_SEED for b in first)
    assert all(b["seed"] == JUDGE_SEED + 1 for b in second)
    assert first != second, "the two passes must not be one cached request"
    assert all(b["temperature"] == 0.0 for b in second), (
        "R8.6 measures the judge's instability, so a sampling judge would be "
        "measuring entropy instead"
    )


def test_both_passes_are_retained_separately() -> None:
    """One deterministic judge run twice must be able to produce two records for one
    item, and a record that cannot say which pass it came from cannot be compared."""
    panel, items = _pilot_with(_all(4, Verdict.CORRECT), _all(4, Verdict.CORRECT))

    combined = VerdictSet()
    for index in (1, 2):
        combined.verdicts.extend(run_panel(panel, items, pass_index=index).verdicts.verdicts)

    assert len(combined.by_pass(1).verdicts) == 8
    assert len(combined.by_pass(2).verdicts) == 8
    assert combined.items() == sorted(i.key for i in items)
    assert {v.judge_id for v in combined.by_pass(2).verdicts} == {"judge_1", "judge_2"}


def test_the_replication_protocol_can_read_the_panels_output() -> None:
    """`audit.replicate_across_panel` consumes `verdicts_by_judge` directly — the
    panel produces the shape the two-pass protocol is written against."""
    from msat.judges.audit import replicate_across_panel

    panel, items = _pilot_with(_alternating(40), _alternating(40))
    first = run_panel(panel, items, pass_index=1).verdicts_by_judge()
    second = run_panel(panel, items, pass_index=2).verdicts_by_judge()

    report = replicate_across_panel(first, second)

    assert report["worst_flip_rate"] == 0.0
    assert [j["n_flips"] for j in report["judges"]] == [0, 0]
    assert "worst judge" in report["noise_floor_rule"]


# ═══ shape of the declaration and of the artifact ════════════════════════════


def _null_paths(node: Any, path: str) -> list[str]:
    if isinstance(node, Mapping):
        return [bad for k, v in node.items() for bad in _null_paths(v, f"{path}.{k}")]
    if isinstance(node, (list, tuple)):
        return [bad for i, v in enumerate(node) for bad in _null_paths(v, f"{path}[{i}]")]
    return [path] if node is None else []


def test_nothing_in_the_serialised_panel_is_null_r1_1() -> None:
    """A null in a readiness row reads as a value the reader failed to fill in, which
    is this project's whole complaint about numbers that mean something else."""
    panel = _split_panel()

    assert _null_paths(panel.as_dict(), "panel") == []
    assert _null_paths(run_panel(panel, _items(4)).as_dict(), "run") == []
    assert _null_paths(panel_readiness(panel)[0].as_dict(), "readiness") == []


def test_the_declared_constants_are_declared() -> None:
    """R1.1: no unnamed numbers. The card, the prompt cap and the seed are named so a
    config change reads as a diff rather than a mystery."""
    assert panel_mod.CARD_GB_DEFAULT == 8.0
    assert PRICING["judge"]["input_tokens_approx"] == panel_mod.JUDGE_INPUT_TOKENS_APPROX
    assert panel_mod.MIN_JUDGES == 2
    assert panel_mod.JUDGE_TEMPERATURE == 0.0
    assert panel_mod.JUDGE_TOP_P == 1.0
    assert JUDGE_SEED == 20261012
    assert panel_mod.ABSTENTION_RULE_KEY == "abstention"
    assert ABSTENTION_YAML[panel_mod.ABSTENTION_RULE_KEY]["suffix"] == "_abs"
    assert panel_mod.UNRECOGNISED_LINEAGE == "unrecognised"


def test_the_judges_never_import_transformers_directly() -> None:
    """They go through `LocalBackend`, so a backend switch is a config change rather
    than a rewrite of the 75-hour loop."""
    source = Path(panel_mod.__file__ or "").read_text(encoding="utf-8")

    assert "import transformers" not in source
    assert "from transformers" not in source
    assert "select_backend" in source
    assert "LocalBackend" in source


def test_the_public_surface_is_re_exported_from_the_package() -> None:
    """One import for a caller, matching how `arms/` and `analysis/` expose theirs."""
    import msat.judges as pkg

    for name in (
        "JudgePanel",
        "JudgeSpec",
        "JudgeItem",
        "run_panel",
        "judge_item",
        "panel_health",
        "PanelKappaBelowBarError",
        "JudgeSequentialError",
        "JudgeIndependenceError",
    ):
        assert name in pkg.__all__
        assert getattr(pkg, name) is not None
