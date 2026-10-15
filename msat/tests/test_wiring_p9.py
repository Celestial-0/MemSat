"""P9 wiring — the seven modules that were built, tested, and unreachable.

Each of these tests proves a WIRING, not a function's return value. The modules
under test (`providers/local.py`, `gate.py`, `planning.py`, `metrics/latency.py`,
`arms/negative.py`) have their own test files and always passed. What was missing
was a caller: no CLI command could reach them, so a requirement could be declared,
tested, and still not enforced by any run.

The requirements this file is about:

    R3.2   a cell that exceeds a reader's SERVABLE ceiling is REJECTED, with a
           named reason, and never silently truncated
    R4.3   the Arm E negative control actually RUNS, and a control that does not
           collapse says so loudly
    P5     the four-branch GO / NO-GO verdict comes from `gate.py`, once
    R4.4   `verify_crossed_grid` is asked whether arms A and G are crossed
    R8.5   latency is reduced to medians, IQR and quantiles, TTFT apart from total

Everything here runs on SYNTHETIC FIXTURES. No real run exists, no checkpoint is
loaded, and no number in this file is a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from msat.cli import (
    _capacity_preflight,
    _check_capacity,
    _crossing,
    _evaluate_gate,
    _gate_exit,
    _load_gold_sessions,
    _panel,
    _plan_from,
    _ttft_samples,
    app,
    capacity_report,
    latency_artifact,
    run_negative_control,
)
from msat.config import load_bundle
from msat.gate import Branch
from msat.providers.base import GenerationResult
from msat.providers.local import LocalReader
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse, cache_key

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MANIFEST = Path(__file__).resolve().parent.parent / "results" / "manifest.json"
RUNNER = CliRunner()

GOLD_ANSWER = "the vault code is 4711"


@pytest.fixture(scope="module")
def bundle() -> Any:
    return load_bundle(CONFIGS)


@pytest.fixture(scope="module")
def panel(bundle: Any) -> list[LocalReader]:
    return _panel(bundle)


@pytest.fixture
def complete_manifest(tmp_path: Path) -> Path:
    """A copy of the SHIPPED manifest — 108 accepted cells, none rejected.

    Copied rather than referenced so a test that writes next to it cannot dirty the
    repository's own artifact.
    """
    if not MANIFEST.exists():
        pytest.skip("run `msat build-plan --out results/manifest.json` first")
    dest = tmp_path / "manifest.json"
    dest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _run(tmp_path: Path, *args: str) -> Any:
    return RUNNER.invoke(app, ["run", *args, "--config", str(CONFIGS)])


# ══════════════════════════════════════════════ R3.2 · the servable ceiling ════
# Every family in readers.yaml declares 131k-262k of context and can serve
# 2.7k-36k on an 8 GB card. Nothing compared a cell against the second number, so
# a deep cell on a small reader was dispatched and OOM'd at hour 60.


def test_a_prompt_past_the_ceiling_is_rejected_with_a_named_reason(panel) -> None:
    """Rejected, and the rejection says WHICH ceiling and WHY it is not a truncate."""
    gemma = next(r for r in panel if r.id == "gemma")
    beyond = gemma.ceiling_tokens() + 1

    verdict = _check_capacity(gemma, beyond, label="S/gemma/K=32/B=512")

    assert verdict.accepted is False
    assert verdict.prompt_tokens == beyond, "the request is unchanged; nothing was trimmed"
    assert verdict.ceiling_tokens == gemma.ceiling_tokens()
    assert verdict.binding_constraint == "vram"
    assert gemma.ceiling_tokens() < gemma.declared_context_window, (
        "the ceiling that binds is the VRAM one; a window-only check finds nothing "
        "wrong and then OOMs"
    )
    # The reason is a sentence a reader can act on, not a boolean.
    assert str(gemma.ceiling_tokens()) in verdict.reason.replace(",", "")
    assert "Truncating would leave the" in verdict.reason
    assert "R3.2" in verdict.reason


def test_a_prompt_inside_the_ceiling_is_accepted(panel) -> None:
    gemma = next(r for r in panel if r.id == "gemma")
    verdict = _check_capacity(gemma, gemma.ceiling_tokens(), label="S/gemma/K=1/B=128")
    assert verdict.accepted is True
    assert verdict.ceiling_tokens == gemma.ceiling_tokens()
    assert verdict.as_dict()["overflow_by"] == 0
    assert verdict.reason


def test_the_ceiling_is_computed_not_typed(panel) -> None:
    """The check must go through `LocalReader`, which delegates to
    `hardware/vram.kv_ceiling_tokens`. A constant here would survive a card swap
    and then be wrong, which is the failure this whole module was written against.
    """
    gemma = next(r for r in panel if r.id == "gemma")
    verdict = _check_capacity(gemma, 10_000)
    assert verdict.vram_ceiling_tokens == gemma.ceiling_tokens()
    assert verdict.vram_ceiling_tokens == gemma.ceiling_at_headroom(gemma.headroom_gb)
    # D11: the allowance is a guess, not a constant, and it moves the answer by 7.8x.
    sensitivity = gemma.ceiling_sensitivity()
    assert sensitivity["tight_0.8gb"] > sensitivity["default"]
    assert gemma.ceiling_at_headroom(0.8) != gemma.ceiling_at_headroom(3.0)


def test_a_rejection_carries_prompt_ceiling_and_what_bound_it() -> None:
    """Three numbers, every rejection. One of them is missing most often."""
    reader = _panel(load_bundle(CONFIGS))
    gemma = next(r for r in reader if r.id == "gemma")
    verdicts = [
        _check_capacity(gemma, 1_000, label="fits"),
        _check_capacity(gemma, 999_999, label="overflows"),
    ]
    report = capacity_report(verdicts)
    assert report["n_checked"] == 2
    assert report["n_accepted"] == 1
    assert report["n_rejected"] == 1
    assert report["on_overflow"] == "reject_cell"
    assert report["never_truncate_silently"] is True

    rejected = report["rejected"][0]
    assert rejected["prompt_tokens"] == 999_999 > rejected["ceiling_tokens"]
    assert rejected["overflow_by"] == 999_999 - rejected["ceiling_tokens"]
    assert rejected["binding_constraint"] in ("vram", "declared_window", "both")
    assert rejected["vram_ceiling_tokens"] < rejected["declared_context_window"]


def test_the_preflight_rejects_the_deep_cells_nothing_used_to_check(
    bundle, panel, tmp_path
) -> None:
    """The shipped grid, against the shipped panel.

    qwen's ceiling at the default 3 GB headroom is 2,682 tokens; the grid's deepest
    cell is 16,884. Before this wiring nothing compared them, so twelve cells were
    declared servable that cannot be served.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    plan = _plan_from(manifest, bundle, {})
    report = _capacity_preflight(bundle, plan, panel)

    assert report["n_checked"] == len(plan.cells)
    assert report["n_rejected"] > 0, (
        "nothing was rejected, so this check is not the one that matters"
    )
    for entry in report["rejected"]:
        assert entry["prompt_tokens"] > entry["ceiling_tokens"]
        assert entry["reason"]
        assert entry["reader"] == "qwen", (
            "only qwen's ceiling is below the grid's deepest prompts on this card"
        )
    assert report["never_truncate_silently"] is True
    assert report["on_overflow"] == "reject_cell"
    assert "D11" in report["headroom_gb_is_open_decision"]
    assert set(report["panel_coverage"]["served_per_reader"]) == {"gemma", "qwen", "phi"}
    # The cheap K-against-ceiling proxy CANNOT see this: every K in the grid is
    # below 2,682, so it reports uniform coverage while the exact per-cell check
    # above rejects a third of qwen's cells. That gap is why both are reported, and
    # `coverage_over` says so in its own `note`.
    assert report["panel_coverage"]["uniform_coverage"] is True
    assert "cheap bound" in report["panel_coverage"]["readers"][0]["note"]
    assert report["panel_coverage"]["common_support"] == 108
    assert report["panel_coverage"]["n_cells"] == 108


def test_msat_run_writes_the_capacity_report(bundle, tmp_path) -> None:
    """The check is reachable from the command, not only from a function."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    r = _run(tmp_path, "pilot", "--dry-run", "--results", str(results))
    assert r.exit_code == 0, r.output

    report = json.loads((results / "pilot" / "capacity.json").read_text(encoding="utf-8"))
    assert report["requirement"] == "PRD §R3.2"
    assert report["n_rejected"] > 0
    assert "REJECTED" in r.output
    assert "R3.2" in r.output
    # Every reader's ceiling is printed WITH its binding constraint, so nobody has
    # to guess which of the two numbers they are looking at.
    assert "bound by vram" in r.output


# ══════════════════════════════════════════════ P5 · the four-branch gate ═════
# `cli.py` raised EXIT_NO_GO from inline `if` statements in three places. The
# four-branch verdict in `gate.py` — and the refusal to GO on a check that never
# ran — was not consulted by any of them.


def test_the_gate_reports_all_four_branches(tmp_path) -> None:
    verdict = _evaluate_gate(
        tmp_path,
        context="unit",
        run_id="r",
        config_hash="h",
        notes=["a note"],
    )
    assert {str(b.branch) for b in verdict.branches} == {str(b) for b in Branch}
    payload = json.loads((tmp_path / "go_no_go.json").read_text(encoding="utf-8"))
    assert len(payload["branches"]) == 4
    assert payload["context"] == "unit"
    assert "a note" in payload["notes"]


_EMPTY_GATE: dict[str, Any] = {
    "accuracy_by_cell": {},
    "recall_by_k": {},
    "stratified": None,
    "degenerate_cells": [],
    "n_interaction_cells": 0,
}


@pytest.mark.parametrize(
    ("branch", "inputs"),
    [
        (Branch.ACCURACY, {"accuracy_by_cell": {"a": 0.20, "b": 0.80}}),
        (Branch.RECALL, {"recall_by_k": {1: 0.10, "8": 0.70}}),
        (
            Branch.STRATIFIED,
            {
                "stratified": {
                    "multi_session_slope": 0.2,
                    "single_session_slope": 0.0,
                    "structure_detected": True,
                }
            },
        ),
        (Branch.INTERACTION, {"n_interaction_cells": 108}),
    ],
    ids=lambda v: str(v).split(")")[0],
)
def test_each_branch_alone_reaches_go_through_the_cli_helper(tmp_path, branch, inputs) -> None:
    """All four are reachable, and any one of them is enough — Architecture §P5."""
    verdict = _evaluate_gate(
        tmp_path,
        context=branch.name,
        run_id="r",
        config_hash="h",
        **{**_EMPTY_GATE, **inputs},
    )
    assert verdict.proceed is True, f"{branch} did not reach GO through the CLI helper"
    assert verdict.passed_branches == [str(branch)]


def test_a_check_that_never_ran_does_not_go(tmp_path) -> None:
    """`n_interaction_cells == 0` on an empty degenerate list is the worst failure
    `gate.py` could have, and it is the one the CLI must not introduce."""
    verdict = _evaluate_gate(
        tmp_path,
        context="empty",
        run_id="r",
        config_hash="h",
        **_EMPTY_GATE,
    )
    assert verdict.proceed is False
    assert verdict.interaction_estimable is False
    assert "NOT EVALUATED" in verdict.branches[0].detail


def test_the_cli_helper_defaults_to_no_data_not_to_a_pass(tmp_path) -> None:
    """A caller with only a design fact to report still gets the honest verdict."""
    verdict = _evaluate_gate(tmp_path, context="bare", run_id="r", config_hash="h")
    assert verdict.proceed is False
    assert verdict.outcome == "BOUNDED_NEGATIVE"
    assert verdict.n_cells == 0


def test_gate_exit_is_the_only_place_exit_no_go_is_raised_from_a_verdict(tmp_path) -> None:
    verdict = _evaluate_gate(tmp_path, context="exit", run_id="r", config_hash="h", **_EMPTY_GATE)
    from msat.constants import EXIT_NO_GO

    with pytest.raises(Exception) as excinfo:  # typer.Exit
        _gate_exit(verdict, message="because")
    assert getattr(excinfo.value, "exit_code", None) == EXIT_NO_GO


def test_msat_run_writes_the_preflight_verdict(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    r = _run(tmp_path, "pilot", "--dry-run", "--results", str(results))
    assert r.exit_code == 0, r.output

    payload = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "GO"
    assert payload["n_interaction_cells"] == 108
    assert payload["passed_branches"] == ["(iv) K x B interaction estimable"]
    assert len(payload["branches"]) == 4


def test_an_uncrossed_grid_is_a_no_go(tmp_path) -> None:
    """R4.4, enforced. The synthetic fixture rejects K=16/32 on S, so the plan is
    NOT a full crossing — and a grid with holes cannot carry the interaction."""
    results = tmp_path / "results"
    results.mkdir()
    build = RUNNER.invoke(
        app,
        ["build-plan", "--corpus", "synthetic", "--out", str(results / "manifest.json")],
    )
    assert build.exit_code == 3, build.output

    r = _run(tmp_path, "pilot", "--dry-run", "--results", str(results))
    assert r.exit_code == 7, r.output
    assert "NO-GO" in r.output
    payload = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "BOUNDED_NEGATIVE"
    assert payload["interaction_estimable"] is False
    assert payload["degenerate_cells"], "the holes must be NAMED, not just counted"
    assert any(d["reason"] == "missing cell" for d in payload["degenerate_cells"])


def test_a_ladder_violation_is_decided_by_the_gate(tmp_path) -> None:
    """Entering the full rung on a fresh ladder. The verdict is a pre-run refusal,
    and it says so — a BOUNDED_NEGATIVE that implies measured results would be a
    lie about what happened."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    r = _run(tmp_path, "pilot", "--rung", "full", "--results", str(results))
    assert r.exit_code == 7, r.output
    payload = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "BOUNDED_NEGATIVE"
    assert payload["n_interaction_cells"] == 0
    assert any("Ladder order violation" in n for n in payload["notes"])
    assert any("pre-run refusal" in n for n in payload["notes"])


def test_judge_audit_is_gated_by_the_same_implementation(tmp_path) -> None:
    """The third former inline `EXIT_NO_GO`. One judge is not a panel (R8.2)."""
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(
        json.dumps(
            [
                {
                    "judge_id": "judge-a",
                    "condition_key": "A|1|512.0|budget|S|gemma",
                    "qid": "q1",
                    "verdict": "correct",
                    "pass_index": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    r = RUNNER.invoke(
        app,
        ["judge-audit", "pilot-01", "--verdicts", str(verdicts), "--out", str(tmp_path / "a.json")],
    )
    assert r.exit_code == 7, r.output
    payload = json.loads((tmp_path / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "BOUNDED_NEGATIVE"
    assert len(payload["branches"]) == 4


# ══════════════════════════════════════════════ R4.4 · the crossed grid ═══════


def test_build_plan_writes_the_executable_plan_and_the_crossing_report(tmp_path) -> None:
    r = RUNNER.invoke(
        app,
        ["build-plan", "--corpus", "synthetic", "--out", str(tmp_path / "m.json")],
    )
    assert r.exit_code == 3, r.output  # rejected cells — unchanged behaviour
    plan = json.loads((tmp_path / "m.plan.json").read_text(encoding="utf-8"))
    assert plan["n_cells"] > 0
    assert plan["n_units"] > 0
    crossing = json.loads((tmp_path / "crossed_grid.json").read_text(encoding="utf-8"))
    assert crossing["requirement"].startswith("PRD §R4.4")
    assert crossing["reader_balanced"] is True
    assert crossing["crossed"] is False, "the synthetic fixture genuinely has holes"
    assert "NOT CROSSED" in r.output


def test_a_fully_crossed_plan_reports_no_problems(bundle, complete_manifest) -> None:
    plan = _plan_from(complete_manifest, bundle, {})
    report, n_interaction_cells, degenerate = _crossing(plan, bundle)
    assert report["crossed"] is True
    assert report["problems"] == []
    assert degenerate == []
    assert n_interaction_cells == len(plan.cells)


# ══════════════════════════════════════════════ R4.3 · Arm E actually runs ═══
# The negative control is the gate on the whole experiment: it draws memories from
# OTHER questions, so a correct reader scores near zero. A control that does not
# collapse means the pipeline is not respecting the condition it claims to
# manipulate — and every other arm is measured through the same path.

POOL_QIDS = ("q_target", "q_donor1", "q_donor2")


def _pool_payload(
    n_per_donor: int = 6,
    n_per_target: int = 6,
    *,
    n_tokens: int = 40,
    filler: int = 0,
) -> dict[str, Any]:
    """A three-question pool. Only `q_target` declares gold, and its gold lives in
    a session the donors never carry — which is what makes the control a control.

    `filler` pads every memory with that many extra words, so a test can build a
    control prompt that genuinely overflows a ceiling rather than asserting it.
    """

    def memories(qid: str, n: int) -> list[dict[str, Any]]:
        padding = " padding" * filler
        return [
            {
                "memory_id": f"{qid}_s{i}#0",
                "session_id": f"{qid}_s{i}",
                "text": f"unrelated memory {i} for {qid}, mentioning lamppost {i}{padding}",
                "n_tokens": n_tokens,
            }
            for i in range(n)
        ]

    target = memories("q_target", n_per_target)
    if target:
        target[0]["session_id"] = "q_target_gold0"
        target[0]["memory_id"] = "q_target_gold0#0"
        target[0]["text"] = f"the vault code is 4711{' padding' * filler}"
    return {
        "q_target": {
            "question": "what did we agree about the vault code?",
            "memories": target,
        },
        "q_donor1": {"question": "donor one", "memories": memories("q_donor1", n_per_donor)},
        "q_donor2": {"question": "donor two", "memories": memories("q_donor2", n_per_donor)},
    }


def _arm_e_inputs(tmp_path: Path) -> dict[str, str]:
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(_pool_payload()), encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps({"q_target": ["q_target_gold0"], "q_donor1": [], "q_donor2": []}),
        encoding="utf-8",
    )
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"q_target": GOLD_ANSWER}), encoding="utf-8")
    return {"pool": str(pool), "questions": str(questions), "gold": str(gold)}


def _make_fake(answer: str) -> Any:
    """A `LocalReader.generate` that returns a canned completion.

    Everything above this — draw, render, capacity, request construction, scoring,
    evaluate — is the production path. Only the token generation is replaced, which
    is the same substitution `runner/executor.Generator` exists for: the loop has
    to be checkable on a machine that has no GPU, or the check happens on the
    execution box and nowhere else.

    `answer` IS the experiment. A reader that abstains collapses the control; a
    reader that answers correctly OFF OTHER QUESTIONS' MEMORIES does not, and that
    is the failure the arm exists to catch.
    """

    def generate(self: LocalReader, request: Any) -> GenerationResult:
        return GenerationResult(
            text=answer,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            ttft_ms=0.5,
            queue_wait_ms=0.0,
            backend="test-double",
            model_version=self.version,
        )

    return generate


def _install_reader(monkeypatch: Any, *, answer: str) -> None:
    monkeypatch.setattr(LocalReader, "generate", _make_fake(answer))


def _control(
    bundle: Any,
    tmp_path: Path,
    *,
    answer: str,
    k: int = 1,
    b: float = 128.0,
    readers: tuple[str, ...] = ("gemma",),
    pool_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`run_negative_control` end to end, through its real inputs."""
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps(pool_payload if pool_payload is not None else _pool_payload()),
        encoding="utf-8",
    )
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    by_id = {r.id: r for r in _panel(bundle)}
    fake = _make_fake(answer)
    return run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[ConditionKey("E", k, b, "per_memory_tokens", "S", r) for r in readers],
        readers=[by_id[r] for r in readers],
        pool=pool,
        question_text=question_text,
        gold_by_question={"q_target": ["q_target_gold0"]},
        gold_answers={"q_target": GOLD_ANSWER},
        generate=lambda reader, request: fake(reader, request),
    )


def test_a_collapsing_arm_e_is_reported_as_a_pass(bundle, tmp_path) -> None:
    """The reader abstains: accuracy 0, recall 0, control collapsed."""
    report = _control(bundle, tmp_path, answer="I have no idea, sorry.")

    assert report["verdict"] == "PASS"
    assert report["proceed"] is True
    assert report["accuracy"] == pytest.approx(0.0, abs=1e-9)
    assert report["recall"] == pytest.approx(0.0, abs=1e-9)
    assert report["gate"] == "arm_e_negative_control"
    assert "collapse" in report["reason"]


def test_the_control_is_count_matched_and_drawn_from_other_questions(bundle, tmp_path) -> None:
    """Only the CONTENT differs between Arm A and Arm E. Anything else and a
    collapse could be attributed to the template rather than to the absence of
    evidence."""
    report = _control(bundle, tmp_path, answer="I have no idea, sorry.", k=4)

    assert report["control"]["count_matched"] is True
    assert report["n_generated"] == 1
    for draw in report["control"]["draws"]:
        assert draw["n_drawn"] == 4, "count-matched to the Arm A condition at K=4"
        assert draw["gold_sessions_drawn"] == [], (
            "a control that can see the answer is not a control"
        )
        assert not any(s.startswith("q_target") for s in draw["session_ids"])
        assert all(s.startswith(("q_donor1", "q_donor2")) for s in draw["session_ids"])


def test_a_non_collapsing_arm_e_fails_loudly(bundle, tmp_path) -> None:
    """The reader answers correctly OFF OTHER QUESTIONS' MEMORIES.

    This is the failure the whole arm exists to catch: the pipeline is scoring as if
    it had the evidence, so every other number in the project is suspect.
    """
    report = _control(bundle, tmp_path, answer=GOLD_ANSWER)

    assert report["verdict"] == "FAIL"
    assert report["proceed"] is False
    assert report["gating"] is True
    assert report["accuracy"] == pytest.approx(1.0)
    assert "did NOT collapse" in report["reason"]
    assert "Stop" in report["reason"]
    assert "STOP" in report["next"]


def test_the_control_runs_on_every_declared_reader_by_default(bundle, tmp_path) -> None:
    """Arm E is the gate on the WHOLE experiment, and a pipeline that flatters itself
    on the primary reader may not flatter itself on the others."""
    report = _control(
        bundle,
        tmp_path,
        answer="I have no idea, sorry.",
        readers=("gemma", "qwen", "phi"),
    )
    assert report["n_generated"] == 3
    assert {d["label"].split("/")[2] for d in report["control"]["draws"]} == {
        "gemma",
        "qwen",
        "phi",
    }


def test_gold_is_excluded_by_session_id_even_from_a_donor(bundle, tmp_path) -> None:
    """The harder check: recall on a count-matched random set should be ~0, and a
    plausible-looking recall is the signature of a retrieval path that is not
    respecting the condition at all.

    The target's gold session is planted in a DONOR's memory list. `draw_control_memories`
    excludes gold by session id regardless of which question carries it.
    """
    payload = _pool_payload()
    payload["q_donor1"]["memories"][0]["session_id"] = "q_target_gold0"
    payload["q_donor1"]["memories"][0]["memory_id"] = "q_target_gold0#0"

    report = _control(bundle, tmp_path, answer="I have no idea, sorry.", pool_payload=payload)
    assert report["recall"] == pytest.approx(0.0, abs=1e-9), (
        "gold is excluded by SESSION id even when a donor carries it"
    )
    assert report["verdict"] == "PASS"


def test_msat_run_executes_arm_e_and_reports_go_no_go(tmp_path, monkeypatch) -> None:
    """The whole path from the command: draw, render, generate, score, evaluate."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)
    _install_reader(monkeypatch, answer="I have no idea, sorry.")

    r = _run(
        tmp_path,
        "pilot",
        "--results",
        str(results),
        "--questions",
        inputs["questions"],
        "--gold",
        inputs["gold"],
        "--arm-e-pool",
        inputs["pool"],
        "--reader",
        "gemma",
    )
    assert r.exit_code == 0, r.output

    report = json.loads((results / "pilot" / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["proceed"] is True
    assert report["arm"] == "E"
    assert report["n_generated"] == 1
    assert report["capacity"]["on_overflow"] == "reject_cell"
    assert "ARM E COLLAPSED" in r.output

    gate = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert gate["outcome"] == "GO"
    assert any("Arm E negative control: PASS" in n for n in gate["notes"])


def test_a_non_collapsing_arm_e_stops_the_run_with_a_loud_verdict(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)
    _install_reader(monkeypatch, answer=GOLD_ANSWER)

    r = _run(
        tmp_path,
        "pilot",
        "--results",
        str(results),
        "--questions",
        inputs["questions"],
        "--gold",
        inputs["gold"],
        "--arm-e-pool",
        inputs["pool"],
        "--reader",
        "gemma",
    )
    out = r.output + str(getattr(r, "stderr", "") or "")
    assert r.exit_code == 7, out
    assert "ARM E DID NOT COLLAPSE" in out
    assert "flatters every other arm" in out

    report = json.loads((results / "pilot" / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    assert report["proceed"] is False

    gate = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert gate["outcome"] == "BOUNDED_NEGATIVE"
    assert any("did NOT collapse" in n for n in gate["notes"])


def test_an_unscorable_control_gates_and_says_why(tmp_path, monkeypatch) -> None:
    """INCONCLUSIVE gates exactly like FAIL: proceeding on 'we don't know' is how a
    broken pipeline gets twelve days of GPU time.

    Reached with thresholds that ARE numeric and a control that generated fine, so
    the arm is INCONCLUSIVE for a different reason — and the report has to name it.
    `evaluate()` returns one INCONCLUSIVE reason for three causes, and telling
    someone their thresholds are null when the real problem is a gold file missing
    the question sends them to edit the wrong file.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)
    # A gold file that does not mention the question the control drew.
    (tmp_path / "gold-empty.json").write_text(json.dumps({"some_other_qid": "x"}), encoding="utf-8")
    _install_reader(monkeypatch, answer="I have no idea, sorry.")

    r = _run(
        tmp_path,
        "pilot",
        "--results",
        str(results),
        "--questions",
        inputs["questions"],
        "--gold",
        str(tmp_path / "gold-empty.json"),
        "--arm-e-pool",
        inputs["pool"],
        "--reader",
        "gemma",
    )
    out = r.output + str(getattr(r, "stderr", "") or "")
    assert r.exit_code == 7, out

    report = json.loads((results / "pilot" / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["gating"] is True
    assert report["proceed"] is False
    assert report["n_generated"] == 1, "the control DID generate; it just could not be scored"
    assert report["n_scored"] == 0
    assert report["diagnostics"]["thresholds_measurable"] is True
    assert report["diagnostics"]["accuracy_is_none_because"]
    assert report["diagnostics"]["n_unattributed"] == 1
    assert "ARM E DID NOT COLLAPSE" in out


@pytest.mark.parametrize(
    ("omitted", "flag"),
    [("--questions", "--questions"), ("--gold", "--gold"), ("--arm-e-pool", "--arm-e-pool")],
)
def test_arm_e_says_exactly_which_input_it_is_missing(tmp_path, monkeypatch, omitted, flag) -> None:
    """No defaults. A negative control built from invented memories, or scored
    against an invented answer, would pass for the wrong reason — which is the one
    thing the arm exists to prevent.

    Every missing input is named at once, and the backend is named with them,
    because both are true and both block: fixing one and re-running only to be
    told about the next costs a round-trip per round-trip.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)
    _install_reader(monkeypatch, answer="I have no idea, sorry.")

    supplied = {
        "--questions": inputs["questions"],
        "--gold": inputs["gold"],
        "--arm-e-pool": inputs["pool"],
    }
    argv = ["pilot", "--results", str(results), "--reader", "gemma"]
    for other, value in supplied.items():
        if other != omitted:
            argv += [other, value]

    r = _run(tmp_path, *argv)
    out = r.output + str(getattr(r, "stderr", "") or "")
    assert r.exit_code == 1, out
    assert flag in out, out
    assert "needs a backend" in out, out
    assert "is missing input" in out


def test_a_missing_backend_is_reported_as_a_missing_backend_not_a_verdict(bundle, tmp_path) -> None:
    """The one seam this harness cannot cross without an engine.

    An optional dependency that is not installed must never be readable as a
    scientific result. `evaluate()` returns the same INCONCLUSIVE verdict for
    "thresholds are null", "nothing was scored" and "nothing ran", so the report
    and the CLI both have to name the real cause — otherwise someone reads
    "pass thresholds are null", goes and edits `arm_e.j2`, and installs nothing.
    """
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool
    from msat.providers.base import BackendUnavailableError

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_payload()), encoding="utf-8")
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    gemma = next(r for r in _panel(bundle) if r.id == "gemma")

    def dead_backend(reader: Any, request: Any) -> Any:
        raise BackendUnavailableError(
            "Using `bitsandbytes` 4-bit quantization requires bitsandbytes"
        )

    report = run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[ConditionKey("E", 1, 128.0, "per_memory_tokens", "S", "gemma")],
        readers=[gemma],
        pool=pool,
        question_text=question_text,
        gold_by_question={"q_target": ["q_target_gold0"]},
        gold_answers={"q_target": GOLD_ANSWER},
        generate=dead_backend,
    )
    assert report["backend"]["available"] is False
    assert report["backend"]["error"] == "BackendUnavailableError"
    assert "bitsandbytes" in report["backend"]["detail"]
    assert report["backend"]["n_generated_before_failure"] == 0
    assert report["n_generated"] == 0
    assert report["proceed"] is False, "a control that never ran cannot pass the gate"
    # The refusal that got us HERE was the capacity check passing, so the arm was
    # fully built and only the engine was missing.
    assert report["capacity"]["n_accepted"] == 1


def test_msat_run_refuses_the_seam_without_claiming_a_verdict(tmp_path, monkeypatch) -> None:
    """From the command: exit 1, no INCONCLUSIVE banner, and a named remedy."""
    from msat.providers.base import BackendUnavailableError

    def dead_generate(self: LocalReader, request: Any) -> Any:
        raise BackendUnavailableError(
            "Using `bitsandbytes` 4-bit quantization requires bitsandbytes"
        )

    monkeypatch.setattr(LocalReader, "generate", dead_generate)
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)

    r = _run(
        tmp_path,
        "pilot",
        "--results",
        str(results),
        "--questions",
        inputs["questions"],
        "--gold",
        inputs["gold"],
        "--arm-e-pool",
        inputs["pool"],
        "--reader",
        "gemma",
    )
    out = r.output + str(getattr(r, "stderr", "") or "")
    assert r.exit_code == 1, out
    assert "no loadable" in out
    assert "bitsandbytes" in out
    assert "Nothing was simulated" in out
    assert "arm E       INCONCLUSIVE" not in out, (
        "a broken engine must not be printed as a scientific verdict"
    )
    artifact = json.loads((results / "pilot" / "arm_e_control.json").read_text(encoding="utf-8"))
    assert artifact["backend"]["available"] is False
    assert artifact["capacity"]["n_accepted"] == 1, "the prompt was built and accepted"


def test_a_control_that_refuses_everything_is_inconclusive_not_a_pass(bundle, tmp_path) -> None:
    """Nothing generated is not a collapse. `evaluate` returns the same INCONCLUSIVE
    verdict for a refusal as for a missing measurement, so the refusal has to be
    named — an arm that quietly refused everything would look identical to one that
    simply had no thresholds."""
    report = _control(bundle, tmp_path, answer="", k=4, pool_payload=_pool_payload(0))
    assert report["n_generated"] == 0
    assert report["n_refused"] == 1
    refusal = report["refusals"][0]
    assert refusal["kind"] == "underfilled_control"
    assert "count-matched" in refusal["reason"]
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["proceed"] is False
    assert report["diagnostics"]["refusal_kinds"] == ["underfilled_control"]


def test_the_control_prompt_is_capacity_checked_before_generation(bundle, tmp_path) -> None:
    """The request that reaches the backend carries the DECODING config, not a
    default, and the prompt is split rather than flattened into one string."""
    seen: list[Any] = []
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_payload()), encoding="utf-8")
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    gemma = next(r for r in _panel(bundle) if r.id == "gemma")
    fake = _make_fake("I have no idea, sorry.")

    report = run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[ConditionKey("E", 1, 128.0, "per_memory_tokens", "S", "gemma")],
        readers=[gemma],
        pool=pool,
        question_text=question_text,
        gold_by_question={"q_target": ["q_target_gold0"]},
        gold_answers={"q_target": GOLD_ANSWER},
        generate=lambda reader, request: (seen.append(request), fake(reader, request))[1],
    )
    assert report["capacity"]["n_checked"] == 1
    assert report["n_generated"] == 1
    assert len(seen) == 1
    assert seen[0].model == gemma.model
    assert seen[0].max_tokens == bundle.decoding.max_tokens
    assert seen[0].temperature == bundle.decoding.temperature
    assert seen[0].top_p == bundle.decoding.top_p
    assert seen[0].system, "the prompt is split, not flattened into one string"


def test_an_overflowing_control_prompt_is_refused_not_truncated(bundle, tmp_path) -> None:
    """R3.2 inside the arm, not only around the grid.

    K=32 at B=2048 is a cell this project actually declared until 2026-10-14: 32
    memories of 2,048 tokens is 65,536 prompt tokens, which is past gemma's 36,326
    ceiling on 8 GB. The control is dispatched through the same path as every other
    arm, so it is refused for the same reason — and refused, never trimmed.
    """
    report = _control(
        bundle,
        tmp_path,
        answer="I have no idea, sorry.",
        k=32,
        b=2048.0,
        pool_payload=_pool_payload(40, 40, n_tokens=6_000, filler=4_000),
    )
    assert report["n_generated"] == 0
    assert report["capacity"]["n_checked"] == 1
    assert report["capacity"]["n_rejected"] == 1
    rejected = report["capacity"]["rejected"][0]
    assert rejected["prompt_tokens"] > rejected["ceiling_tokens"]
    assert rejected["overflow_by"] > 0
    assert rejected["binding_constraint"] == "vram"
    assert report["capacity"]["never_truncate_silently"] is True
    assert report["n_refused"] == 1
    assert report["refusals"][0]["kind"] == "context_overflow"
    assert report["proceed"] is False, "a control that never ran cannot pass"


def test_the_shipped_control_thresholds_are_numeric() -> None:
    """`null` would make Arm E INCONCLUSIVE and gate every run forever. The values
    themselves are D5; what is pinned here is that they are numbers."""
    from msat.arms.render import strip_leading_jinja_comment

    spec = yaml.safe_load(
        strip_leading_jinja_comment((CONFIGS / "prompts" / "arm_e.j2").read_text(encoding="utf-8"))
    )
    control = spec["control"]
    assert isinstance(control["pass_threshold_accuracy_below"], float)
    assert isinstance(control["pass_threshold_recall_below"], float)


# ══════════════════════════════════════════════ R8.5 · latency, reduced ══════
# `store/responses.py` has carried latency_ms / ttft_ms / queue_wait_ms since P4
# and nothing read them.


def _latency_store(tmp_path: Path, n: int = 120) -> ResponseStore:
    """A heavy-tailed store: a tight TTFT and a generator whose tail is 50x it."""
    store = ResponseStore(tmp_path / "responses")
    cond = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")
    for i in range(n):
        req = {
            "model": "google/gemma-4-E4B-it",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "temperature": 0.0,
        }
        tail = 20_000.0 if i % 20 == 0 else 900.0
        store.put(
            StoredResponse(
                key=cache_key(cond, {**req, "messages": req["messages"] + [{"c": str(i)}]}),
                condition=cond.as_dict(),
                request=req,
                text="paris",
                finish_reason="stop",
                model_version="v1",
                prompt_tokens=100,
                completion_tokens=4,
                latency_ms=tail,
                ttft_ms=180.0 + i % 7,
                queue_wait_ms=5.0,
                qid=f"q{i}",
            )
        )
    return store


def test_the_latency_artifact_separates_ttft_from_total(tmp_path) -> None:
    """They measure different things. One is the reader; the other is the
    generator's own tail, and quoting either as the other misreads the run."""
    artifact = latency_artifact(_latency_store(tmp_path))

    total = artifact["total_latency"]["overall"]
    ttft = artifact["ttft"]
    assert ttft["n"] == total["n"] == 120
    assert ttft["quantiles_ms"]["p50"] == pytest.approx(183.0, abs=3.0)
    assert total["quantiles_ms"]["p50"] == pytest.approx(900.0, abs=60.0)
    assert ttft["quantiles_ms"]["p50"] < total["quantiles_ms"]["p50"] / 2
    assert total["skewed"] is True
    assert "not a representative latency" in total["caveat"]


def test_the_latency_artifact_reports_median_iqr_and_every_quantile(tmp_path) -> None:
    """R8.5 verbatim: medians, IQR, and quantiles."""
    artifact = latency_artifact(_latency_store(tmp_path))
    for block in (artifact["total_latency"]["overall"], artifact["ttft"]):
        for q in ("p25", "p50", "p75", "p90", "p95", "p99"):
            assert q in block["quantiles_ms"], q
        assert block["iqr_ms"] == pytest.approx(
            block["quantiles_ms"]["p75"] - block["quantiles_ms"]["p25"]
        )
        assert block["n"] > 0


def test_there_is_no_bootstrap_ci_anywhere_in_the_written_artifact(tmp_path) -> None:
    """The explicit requirement. A CI on a heavy-tailed mean reports uncertainty
    about a statistic nobody should quote.

    Checked over the artifact's KEYS, recursively: what the requirement forbids is a
    latency summary that CONTAINS such a statistic, and a key is how a statistic
    enters a JSON artifact. The prose that says the CI is deliberately absent is
    checked separately so it cannot be deleted to make this pass.
    """

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            out = {str(k).lower() for k in node}
            for value in node.values():
                out |= keys(value)
            return out
        if isinstance(node, list):
            out: set[str] = set()
            for value in node:
                out |= keys(value)
            return out
        return set()

    artifact = latency_artifact(_latency_store(tmp_path, n=40))
    assert "no_bootstrap_ci" in artifact
    # Excluded deliberately: that key is the DECLARATION of their absence, and it
    # is checked on its own line below.
    present = keys(artifact) - {"no_bootstrap_ci"}
    for forbidden in ("ci", "confidence", "bootstrap", "stderr", "se_", "interval"):
        assert not any(forbidden in k for k in present), (
            f"R8.5 forbids {forbidden!r} in a latency summary"
        )
    assert "Deliberately absent" in artifact["no_bootstrap_ci"]
    assert artifact["ttft"]["p99_reliable"] is False, (
        "a p99 over 40 calls is one or two observations"
    )


def test_ttft_samples_reject_an_impossible_decomposition(tmp_path) -> None:
    """TTFT is measured FROM the end of the queue. A record saying otherwise is a
    client bug, and it is counted rather than allowed to abort the report."""
    from msat.metrics.latency import LatencySample

    good = LatencySample(latency_ms=900, queue_wait_ms=100, ttft_ms=300)
    assert len(_ttft_samples([good])) == 1

    store = ResponseStore(tmp_path / "responses")
    cond = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")
    req = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    store.put(
        StoredResponse(
            key=cache_key(cond, req),
            condition=cond.as_dict(),
            request=req,
            text="x",
            finish_reason="stop",
            model_version="v1",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=500.0,
            ttft_ms=10.0,
            queue_wait_ms=400.0,
            qid="q0",
        )
    )
    artifact = latency_artifact(store)
    assert artifact["n_impossible_records"] == 1
    assert artifact["n_latency_samples"] == 0


def test_queue_wait_is_reported_as_ours_not_the_readers(tmp_path) -> None:
    store = ResponseStore(tmp_path / "responses")
    cond = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="qwen")
    for i in range(10):
        req = {"model": "m", "messages": [{"role": "user", "content": f"q{i}"}]}
        store.put(
            StoredResponse(
                key=cache_key(cond, {**req, "messages": req["messages"] + [{"c": str(i)}]}),
                condition=cond.as_dict(),
                request=req,
                text="x",
                finish_reason="stop",
                model_version="v1",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1_000.0,
                ttft_ms=950.0,
                queue_wait_ms=900.0,
                qid=f"q{i}",
            )
        )
    artifact = latency_artifact(store)
    queue = artifact["total_latency"]["queue_wait"]
    assert queue["recorded"] is True
    assert queue["recording_concurrency"] is False
    assert "measures this harness" in queue["note"]
    per_reader = artifact["total_latency"]["by_reader"]["qwen"]
    assert per_reader["queue_excluded_mean_ms"] == pytest.approx(100.0, abs=1.0)


def test_msat_score_writes_the_latency_artifact(tmp_path) -> None:
    """Reachable from the command, which is the whole point."""
    d = tmp_path / "results" / "pilot-01"
    d.mkdir(parents=True)
    _latency_store(d, n=120)
    gold = d / "gold.json"
    gold.write_text(json.dumps({f"q{i}": "paris" for i in range(120)}), encoding="utf-8")

    r = RUNNER.invoke(
        app,
        [
            "score",
            "pilot-01",
            "--results",
            str(tmp_path / "results"),
            "--gold",
            str(gold),
        ],
    )
    assert r.exit_code == 0, r.output
    artifact = json.loads((d / "latency.json").read_text(encoding="utf-8"))
    assert artifact["requirement"] == "PRD §R8.5"
    assert artifact["total_latency"]["overall"]["n"] == 120
    assert artifact["ttft"]["n"] == 120
    assert "latency (R8.5)" in r.output
    assert "total" in r.output and "ttft" in r.output


def test_gold_sessions_load_is_typed(tmp_path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"q1": ["s1", "s2"]}), encoding="utf-8")
    assert _load_gold_sessions(path) == {"q1": ["s1", "s2"]}
