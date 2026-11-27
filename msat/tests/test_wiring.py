from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from msat.cli import (
    app,
    capacity_preflight,
    capacity_report,
    check_capacity,
    crossing,
    evaluate_gate,
    gate_exit,
    latency_artifact,
    load_gold_sessions,
    panel,
    plan_from,
    run_negative_control,
)
from msat.config import load_bundle
from msat.gate import Branch
from msat.providers.base import GenerationResult
from msat.providers.local import LocalReader
from msat.store import Condition, ResponseStore, StoredResponse, cache_key

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MANIFEST = Path(__file__).resolve().parent.parent / "results" / "manifest.json"
RUNNER = CliRunner()

GOLD_ANSWER = "the vault code is 4711"


@pytest.fixture(scope="module")
def bundle() -> Any:
    return load_bundle(CONFIGS)


@pytest.fixture(scope="module")
def reader_panel(bundle: Any) -> list[LocalReader]:
    return panel(bundle)


@pytest.fixture
def complete_manifest(tmp_path: Path) -> Path:
    if not MANIFEST.exists():
        pytest.skip("run `msat build-manifest --out results/manifest.json` first")
    dest = tmp_path / "manifest.json"
    dest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _run(tmp_path: Path, *args: str) -> Any:
    manifest = tmp_path / "results" / "manifest.json"
    plan_args = ["--plan", str(manifest)] if manifest.is_file() and "--plan" not in args else []
    return RUNNER.invoke(app, ["run", *args, *plan_args, "--config", str(CONFIGS)])


def test_a_prompt_past_the_ceiling_is_rejected_with_a_named_reason(reader_panel) -> None:
    gemma = next(r for r in reader_panel if r.id == "gemma")
    beyond = gemma.vram_capacity_tokens() + 1

    verdict = check_capacity(gemma, beyond, label="S/gemma/K=32/B=512")

    assert verdict.supported is False
    assert verdict.prompt_tokens == beyond, "the request is unchanged; nothing was trimmed"
    assert verdict.reader_capacity_tokens == gemma.reader_capacity_tokens
    assert verdict.binding_constraint == "vram"
    assert gemma.reader_capacity_tokens < gemma.declared_context_window, (
        "the ceiling that binds is the VRAM one; a window-only check finds nothing "
        "wrong and then OOMs"
    )
    assert str(gemma.reader_capacity_tokens) in verdict.reason.replace(",", "")
    assert "Truncating would leave the" in verdict.reason
    assert "R3.2" in verdict.reason


def test_a_prompt_inside_the_ceiling_is_accepted(reader_panel) -> None:
    gemma = next(r for r in reader_panel if r.id == "gemma")
    verdict = check_capacity(gemma, gemma.reader_capacity_tokens, label="S/gemma/K=1/B=128")
    assert verdict.supported is True
    assert verdict.reader_capacity_tokens == gemma.reader_capacity_tokens
    assert verdict.as_dict()["overflow_by"] == 0
    assert verdict.reason


def test_the_ceiling_is_computed_not_typed(reader_panel) -> None:
    gemma = next(r for r in reader_panel if r.id == "gemma")
    verdict = check_capacity(gemma, 10_000)
    assert verdict.vram_capacity_tokens == gemma.vram_capacity_tokens()
    assert verdict.vram_capacity_tokens == gemma.vram_capacity_at_headroom(gemma.headroom_gb)
    sensitivity = gemma.capacity_sensitivity()
    assert sensitivity["tight_0.8gb"] > sensitivity["default"]
    assert gemma.vram_capacity_at_headroom(0.8) != gemma.vram_capacity_at_headroom(3.0)


def test_a_rejection_carries_prompt_ceiling_and_what_bound_it() -> None:
    reader = panel(load_bundle(CONFIGS))
    gemma = next(r for r in reader if r.id == "gemma")
    verdicts = [
        check_capacity(gemma, 1_000, label="fits"),
        check_capacity(gemma, 999_999, label="overflows"),
    ]
    report = capacity_report(verdicts)
    assert report["n_checked"] == 2
    assert report["n_supported"] == 1
    assert report["n_unsupported"] == 1
    assert report["on_overflow"] == "reject_cell"
    assert report["never_truncate_silently"] is True

    rejected = report["unsupported"][0]
    assert rejected["prompt_tokens"] == 999_999 > rejected["reader_capacity_tokens"]
    assert rejected["overflow_by"] == 999_999 - rejected["reader_capacity_tokens"]
    assert rejected["binding_constraint"] in ("vram", "declared_window", "both")
    assert rejected["vram_capacity_tokens"] < rejected["declared_context_window"]


def test_the_preflight_rejects_the_deep_cells_nothing_used_to_check(
    bundle, reader_panel, tmp_path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    plan = plan_from(manifest, bundle, {})
    report = capacity_preflight(bundle, plan, reader_panel)

    assert report["n_checked"] == len(plan.conditions)
    assert report["n_unsupported"] > 0, (
        "nothing was rejected, so this check is not the one that matters"
    )
    for entry in report["unsupported"]:
        assert entry["prompt_tokens"] > entry["reader_capacity_tokens"]
        assert entry["reason"]
        assert entry["reader"] == "qwen", (
            "only qwen's ceiling is below the grid's deepest prompts on this card"
        )
    assert report["never_truncate_silently"] is True
    assert report["on_overflow"] == "reject_cell"
    assert "D11" in report["headroom_gb_is_open_decision"]
    assert set(report["panel_coverage"]["supported_per_reader"]) == {"gemma", "qwen", "phi"}
    assert report["panel_coverage"]["uniform_coverage"] is False
    assert "modeled prompt tokens" in report["panel_coverage"]["readers"][0]["note"]
    assert report["panel_coverage"]["common_support"] < 36
    assert ["S", 32, 512.0, "per_memory_tokens"] not in report["panel_coverage"][
        "common_support_cells"
    ]
    assert report["panel_coverage"]["n_cells"] == 108


def test_msat_run_writes_the_capacity_report(bundle, tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    r = _run(tmp_path, "pilot", "--dry-run", "--results", str(results))
    assert r.exit_code == 0, r.output

    report = json.loads((results / "pilot" / "capacity.json").read_text(encoding="utf-8"))
    assert report["requirement"] == "PRD §R3.2"
    assert report.get("n_unsupported", report.get("n_rejected", 0)) > 0
    assert "REJECTED" in r.output or "UNSUPPORTED" in r.output
    assert "R3.2" in r.output
    assert "bound by vram" in (results / "pilot" / "capacity.json").read_text(encoding="utf-8")


def test_the_gate_reports_all_four_branches(tmp_path) -> None:
    verdict = evaluate_gate(
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
    verdict = evaluate_gate(
        tmp_path,
        context=branch.name,
        run_id="r",
        config_hash="h",
        **{**_EMPTY_GATE, **inputs},
    )
    assert verdict.proceed is True, f"{branch} did not reach GO through the CLI helper"
    assert verdict.passed_branches == [str(branch)]


def test_a_check_that_never_ran_does_not_go(tmp_path) -> None:
    verdict = evaluate_gate(
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
    verdict = evaluate_gate(tmp_path, context="bare", run_id="r", config_hash="h")
    assert verdict.proceed is False
    assert verdict.outcome == "BOUNDED_NEGATIVE"
    assert verdict.n_conditions == 0


def test_gate_exit_is_the_only_place_exit_no_go_is_raised_from_a_verdict(tmp_path) -> None:
    verdict = evaluate_gate(tmp_path, context="exit", run_id="r", config_hash="h", **_EMPTY_GATE)
    from msat.constants import EXIT_NO_GO

    with pytest.raises(Exception) as excinfo:
        gate_exit(verdict, message="because")
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
    results = tmp_path / "results"
    results.mkdir()
    build = RUNNER.invoke(
        app,
        ["build-manifest", "--corpus", "synthetic", "--out", str(results / "manifest.json")],
    )
    assert build.exit_code == 3, build.output

    r = _run(tmp_path, "pilot", "--dry-run", "--results", str(results))
    assert r.exit_code == 7, r.output
    assert "NO-GO" in r.output
    payload = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "BOUNDED_NEGATIVE"
    assert payload["interaction_estimable"] is False
    degen = payload.get("degenerate_conditions", payload.get("degenerate_cells"))
    assert degen, "the holes must be NAMED, not just counted"
    assert any(d["reason"] == "missing cell" for d in degen)


def test_a_ladder_violation_is_decided_by_the_gate(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    r = _run(tmp_path, "pilot", "--rung", "crossed", "--results", str(results))
    assert r.exit_code == 7, r.output
    payload = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "BOUNDED_NEGATIVE"
    assert payload.get("n_interaction_conditions", payload.get("n_interaction_cells")) == 0
    assert any("Ladder order violation" in n for n in payload["notes"])
    assert any("pre-run refusal" in n for n in payload["notes"])


def test_judge_audit_is_gated_by_the_same_implementation(tmp_path) -> None:
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


def test_build_plan_writes_the_executable_plan_and_the_crossing_report(tmp_path) -> None:
    r = RUNNER.invoke(
        app,
        ["build-manifest", "--corpus", "synthetic", "--out", str(tmp_path / "m.json")],
    )
    assert r.exit_code == 3, r.output
    plan = json.loads((tmp_path / "m.plan.json").read_text(encoding="utf-8"))
    assert plan.get("n_conditions", plan.get("n_cells", 0)) > 0
    assert plan.get("n_units", plan.get("n_observations", 0)) > 0
    crossing_report = json.loads((tmp_path / "crossed_grid.json").read_text(encoding="utf-8"))
    assert crossing_report["requirement"].startswith("PRD §R4.4")
    assert crossing_report["reader_balanced"] is True
    assert crossing_report["crossed"] is False, "the synthetic fixture genuinely has holes"
    assert "NOT CROSSED" in r.output


def test_a_fully_crossed_plan_reports_no_problems(bundle, complete_manifest) -> None:
    plan = plan_from(complete_manifest, bundle, {})
    report, n_interaction_cells, degenerate = crossing(plan, bundle)
    assert report["crossed"] is True
    assert report["problems"] == []
    assert degenerate == []
    assert n_interaction_cells == len(plan.conditions)


POOL_QIDS = ("q_target", "q_donor1", "q_donor2")


def _pool_payload(
    n_per_donor: int = 6,
    n_per_target: int = 6,
    *,
    n_tokens: int = 40,
    filler: int = 0,
) -> dict[str, Any]:

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
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps(pool_payload if pool_payload is not None else _pool_payload()),
        encoding="utf-8",
    )
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    by_id = {r.id: r for r in panel(bundle)}
    fake = _make_fake(answer)
    return run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[Condition("E", k, b, "per_memory_tokens", "S", r) for r in readers],
        readers=[by_id[r] for r in readers],
        pool=pool,
        question_text=question_text,
        gold_by_question={"q_target": ["q_target_gold0"]},
        gold_answers={"q_target": GOLD_ANSWER},
        generate=lambda reader, request: fake(reader, request),
    )


def test_a_collapsing_arm_e_is_reported_as_a_pass(bundle, tmp_path) -> None:
    report = _control(bundle, tmp_path, answer="I have no idea, sorry.")

    assert report["verdict"] == "PASS"
    assert report["proceed"] is True
    assert report["accuracy"] == pytest.approx(0.0, abs=1e-9)
    assert report["recall"] == pytest.approx(0.0, abs=1e-9)
    assert report["gate"] == "arm_e_negative_control"
    assert "collapse" in report["reason"]


def test_the_control_is_count_matched_and_drawn_from_other_questions(bundle, tmp_path) -> None:
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
    report = _control(bundle, tmp_path, answer=GOLD_ANSWER)

    assert report["verdict"] == "FAIL"
    assert report["proceed"] is False
    assert report["gating"] is True
    assert report["accuracy"] == pytest.approx(1.0)
    assert "did NOT collapse" in report["reason"]
    assert "Stop" in report["reason"]
    assert "STOP" in report["next"]


def test_the_control_runs_on_every_declared_reader_by_default(bundle, tmp_path) -> None:
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
    payload = _pool_payload()
    payload["q_donor1"]["memories"][0]["session_id"] = "q_target_gold0"
    payload["q_donor1"]["memories"][0]["memory_id"] = "q_target_gold0#0"

    report = _control(bundle, tmp_path, answer="I have no idea, sorry.", pool_payload=payload)
    assert report["recall"] == pytest.approx(0.0, abs=1e-9), (
        "gold is excluded by SESSION id even when a donor carries it"
    )
    assert report["verdict"] == "PASS"


def test_msat_run_executes_arm_e_and_reports_go_no_go(tmp_path, monkeypatch) -> None:
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
    assert "accuracy 1.000" in out

    report = json.loads((results / "pilot" / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    assert report["proceed"] is False

    gate = json.loads((results / "pilot" / "go_no_go.json").read_text(encoding="utf-8"))
    assert gate["outcome"] == "BOUNDED_NEGATIVE"
    assert any("did NOT collapse" in n for n in gate["notes"])


def test_an_unscorable_control_gates_and_says_why(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    inputs = _arm_e_inputs(tmp_path)
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
    assert report["n_generated"] == 0, "unscorable control inputs must refuse generation"
    assert report["n_scored"] == 0
    assert report["diagnostics"]["thresholds_measurable"] is True
    assert report["diagnostics"]["accuracy_is_none_because"]
    assert report["diagnostics"]["n_unattributed"] == 1
    assert "ARM E NOT EVALUATED" in out


@pytest.mark.parametrize(
    ("omitted", "flag"),
    [("--questions", "--questions"), ("--gold", "--gold"), ("--arm-e-pool", "--arm-e-pool")],
)
def test_arm_e_says_exactly_which_input_it_is_missing(tmp_path, monkeypatch, omitted, flag) -> None:
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
    assert "is missing input" in out


def test_a_missing_backend_is_reported_as_a_missing_backend_not_a_verdict(bundle, tmp_path) -> None:
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool
    from msat.providers.base import BackendUnavailableError

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_payload()), encoding="utf-8")
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    gemma = next(r for r in panel(bundle) if r.id == "gemma")

    def dead_backend(reader: Any, request: Any) -> Any:
        raise BackendUnavailableError(
            "Using `bitsandbytes` 4-bit quantization requires bitsandbytes"
        )

    report = run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[Condition("E", 1, 128.0, "per_memory_tokens", "S", "gemma")],
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
    assert report["capacity"]["n_supported"] == 1


def test_msat_run_refuses_the_seam_without_claiming_a_verdict(tmp_path, monkeypatch) -> None:
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
    assert artifact["capacity"]["n_supported"] == 1, "the prompt was built and accepted"


def test_a_control_that_refuses_everything_is_inconclusive_not_a_pass(bundle, tmp_path) -> None:
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
    seen: list[Any] = []
    from msat.arms.registry import build_registry
    from msat.cli import load_control_pool

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_payload()), encoding="utf-8")
    registry, _ = build_registry(bundle)
    pool, question_text = load_control_pool(pool_path)
    gemma = next(r for r in panel(bundle) if r.id == "gemma")
    fake = _make_fake("I have no idea, sorry.")

    report = run_negative_control(
        bundle=bundle,
        arm=registry.get("E"),
        cells=[Condition("E", 1, 128.0, "per_memory_tokens", "S", "gemma")],
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
    assert report["capacity"]["n_unsupported"] == 1
    rejected = report["capacity"]["unsupported"][0]
    assert rejected["prompt_tokens"] > rejected["reader_capacity_tokens"]
    assert rejected["overflow_by"] > 0
    assert rejected["binding_constraint"] == "vram"
    assert report["capacity"]["never_truncate_silently"] is True
    assert report["n_refused"] == 1
    assert report["refusals"][0]["kind"] == "context_overflow"
    assert report["proceed"] is False, "a control that never ran cannot pass"


def test_the_shipped_control_thresholds_are_numeric() -> None:
    from msat.arms.render import strip_leading_jinja_comment

    spec = yaml.safe_load(
        strip_leading_jinja_comment((CONFIGS / "prompts" / "arm_e.j2").read_text(encoding="utf-8"))
    )
    control = spec["control"]
    assert isinstance(control["pass_threshold_accuracy_below"], float)
    assert isinstance(control["pass_threshold_recall_below"], float)


def _latency_store(tmp_path: Path, n: int = 120) -> ResponseStore:
    store = ResponseStore(tmp_path / "responses")
    cond = Condition(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")
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
    artifact = latency_artifact(_latency_store(tmp_path))
    for block in (artifact["total_latency"]["overall"], artifact["ttft"]):
        for q in ("p25", "p50", "p75", "p90", "p95", "p99"):
            assert q in block["quantiles_ms"], q
        assert block["iqr_ms"] == pytest.approx(
            block["quantiles_ms"]["p75"] - block["quantiles_ms"]["p25"]
        )
        assert block["n"] > 0


def test_there_is_no_bootstrap_ci_anywhere_in_the_written_artifact(tmp_path) -> None:

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
    store = ResponseStore(tmp_path / "responses")
    cond = Condition(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")
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
    cond = Condition(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="qwen")
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
    assert load_gold_sessions(path) == {"q1": ["s1", "s2"]}
