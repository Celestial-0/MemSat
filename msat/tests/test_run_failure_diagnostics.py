"""Regression tests for run failure diagnostics and Arm E error handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from msat.arms.negative import ControlConfig, Verdict, evaluate, go_no_go
from msat.arms.registry import build_registry
from msat.datasets.schema import CorpusStats, Question, Session, Turn
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)
from msat.providers.local import LocalReader, ReaderVramSpec
from msat.runner.input import RunInputs
from msat.runner.lifecycle import (
    GateFailedError,
    RunLifecycle,
    read_ladder,
)


class DummyTestBackend:
    """In-memory test backend simulating local generation."""

    def __init__(
        self,
        answer: str = "There is no answer to this question in the provided history.",
        available: bool = True,
    ) -> None:
        self.name = "dummy-test-backend"
        self.answer = answer
        self.available = available
        self.calls: list[GenerationRequest] = []

    def health(self) -> dict[str, Any]:
        return {"ready": self.available, "name": self.name}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.available:
            raise BackendUnavailableError("Simulated backend offline")
        self.calls.append(request)
        text = self.answer
        if request.request_id == "E":
            text = "There is no answer to this question in the provided history."
        elif "Paris is France's capital" in request.prompt:
            text = "Paris"
        elif "official currency is the Yen" in request.prompt:
            text = "Yen"
        return GenerationResult(
            text=text,
            finish_reason="stop",
            prompt_tokens=15,
            completion_tokens=4,
            latency_ms=2.5,
            ttft_ms=1.0,
            queue_wait_ms=0.1,
            backend=self.name,
            model_version="test-v1",
        )

    def context_window(self, model: str) -> int:
        return 4096

    def close(self) -> None:
        pass


def make_test_reader(
    reader_id: str = "test-reader",
    declared_context_window: int = 4096,
    backend: Any = None,
) -> LocalReader:
    reader = LocalReader(
        id=reader_id,
        model=f"local/{reader_id}",
        version="v1.0",
        released="2026-01-01",
        role="primary",
        reader_strength=1.0,
        declared_context_window=declared_context_window,
        vram=ReaderVramSpec(params_b=2.0, layers=18, kv_heads=4, head_dim=64),
        backend_name="transformers",
        card_gb=24.0,
        headroom_gb=1.0,
    )
    if backend is not None:
        object.__setattr__(reader, "_backend", backend)
    return reader


def make_test_bundle(prompts: dict[str, str] | None = None) -> Any:
    default_arm_e_template = """system: |
  You are answering a question about a conversation history.
user: |
  {{ memories }}

  Question: {{ question }}
control:
  match: count
  source: other_questions
  seed: 20261012
  exclude_gold: true
  exclude_overlap_with_target_question: true
  pass_threshold_accuracy_below: 0.25
  pass_threshold_recall_below: 0.10
"""
    default_arm_a_template = """system: |
  System prompt.
user: |
  {{ memories }}

  Question: {{ question }}
"""
    all_prompts = {
        "prompts/arm_e.j2": default_arm_e_template,
        "prompts/arm_a.j2": default_arm_a_template,
        "prompts/arm_d.j2": default_arm_a_template,
        "prompts/arm_f.j2": default_arm_a_template,
        "prompts/arm_g.j2": default_arm_a_template,
        "prompts/arm_b.j2": default_arm_a_template,
        "prompts/arm_c.j2": default_arm_a_template,
    }
    if prompts:
        all_prompts.update(prompts)

    class DummyB:
        value = 256.0
        mode = "per_memory_tokens"

    class DummyDesign:
        K_grid = (1, 4)
        B_grid = (DummyB(),)
        b_values = (256.0,)
        scales = ("S",)
        readers = ("test-reader",)
        runs_per_cell = 1

    arms_list = [
        {
            "id": "A",
            "isolates": "retrieval",
            "prompt_template": "prompts/arm_a.j2",
            "enabled": True,
        },
        {
            "id": "E",
            "isolates": "negative_control",
            "prompt_template": "prompts/arm_e.j2",
            "enabled": True,
        },
        {
            "id": "D",
            "isolates": "full_context",
            "prompt_template": "prompts/arm_d.j2",
            "enabled": True,
        },
        {"id": "F", "isolates": "oracle", "prompt_template": "prompts/arm_f.j2", "enabled": True},
        {"id": "G", "isolates": "recency", "prompt_template": "prompts/arm_g.j2", "enabled": True},
        {"id": "B", "isolates": "padding", "prompt_template": "prompts/arm_b.j2", "enabled": True},
        {"id": "C", "isolates": "position", "prompt_template": "prompts/arm_c.j2", "enabled": True},
    ]

    class DummyBundle:
        config_hash = "test_config_hash_1234"
        seed = 20261012
        design = DummyDesign()
        pricing: ClassVar[dict[str, Any]] = {
            "token_estimates": {
                "chars_per_token": 4.0,
                "S": {
                    "sessions_mean": 10.0,
                    "tokens_per_session": 256.0,
                    "corpus_tokens": 2560.0,
                },
            }
        }
        prompts = all_prompts
        arms = type("Arms", (), {"arms": arms_list, "crossed_grid": ["A", "G"]})()
        raw: ClassVar[dict[str, Any]] = {
            "arms": {
                "arms": arms_list,
                "crossed_grid": ["A", "G"],
            }
        }
        decoding = None

    return DummyBundle()


def make_synthetic_questions() -> list[Question]:
    return [
        Question(
            qid="q1",
            scale="S",
            question="What is the capital of France?",
            question_date="2026-01-01",
            gold_answers=("Paris",),
            gold_session_ids=("sess_q1_gold",),
            ability="knowledge-retrieval",
            answerable=True,
            corpus=CorpusStats(n_sessions=2, n_turns=2, n_chars=200, n_gold_sessions=1),
            sessions=(
                Session(
                    session_id="sess_q1_gold",
                    date="2026-01-01",
                    turns=(
                        Turn(
                            role="user",
                            content="What is the capital of France? Paris is France's capital city.",
                            has_answer=True,
                        ),
                    ),
                ),
                Session(
                    session_id="sess_q1_distractor",
                    date="2026-01-02",
                    turns=(
                        Turn(
                            role="user",
                            content="The weather in France is temperate.",
                            has_answer=False,
                        ),
                    ),
                ),
            ),
        ),
        Question(
            qid="q2",
            scale="S",
            question="What is the currency of Japan?",
            question_date="2026-01-01",
            gold_answers=("Yen",),
            gold_session_ids=("sess_q2_gold",),
            ability="knowledge-retrieval",
            answerable=True,
            corpus=CorpusStats(n_sessions=2, n_turns=2, n_chars=200, n_gold_sessions=1),
            sessions=(
                Session(
                    session_id="sess_q2_gold",
                    date="2026-01-01",
                    turns=(
                        Turn(
                            role="user",
                            content="Japan's official currency is the Yen.",
                            has_answer=True,
                        ),
                    ),
                ),
                Session(
                    session_id="sess_q2_distractor",
                    date="2026-01-02",
                    turns=(
                        Turn(
                            role="user", content="Tokyo is a bustling metropolis.", has_answer=False
                        ),
                    ),
                ),
            ),
        ),
    ]


# =============================================================================
# 1. Negative Control Unit Evaluation Tests
# =============================================================================


def test_evaluate_null_thresholds_reports_null_thresholds_reason() -> None:
    """When thresholds are genuinely null, reason cites missing thresholds."""
    cfg = ControlConfig()  # pass_threshold_accuracy_below=None, pass_threshold_recall_below=None
    assert not cfg.thresholds_measurable

    outcome = evaluate(None, None, cfg, n_questions=0)
    assert outcome.verdict is Verdict.INCONCLUSIVE
    assert outcome.gating is True
    assert "pass thresholds are null" in outcome.reason
    assert outcome.thresholds == {"accuracy": None, "recall": None}

    gate_dict = go_no_go(outcome)
    assert gate_dict["proceed"] is False
    assert "numeric threshold in place" in gate_dict["next"]


def test_evaluate_configured_thresholds_with_missing_outcomes_reports_missing_outcomes() -> None:
    """When thresholds are configured but outcomes are None, reason must NOT say thresholds are null."""
    cfg = ControlConfig(pass_threshold_accuracy_below=0.15, pass_threshold_recall_below=0.01)
    assert cfg.thresholds_measurable is True

    outcome = evaluate(None, None, cfg, n_questions=0)
    assert outcome.verdict is Verdict.INCONCLUSIVE
    assert outcome.gating is True
    assert "pass thresholds are null" not in outcome.reason
    assert "measured control outcomes are missing" in outcome.reason
    assert outcome.thresholds == {"accuracy": 0.15, "recall": 0.01}

    gate_dict = go_no_go(outcome)
    assert gate_dict["proceed"] is False
    assert "numeric threshold in place" not in gate_dict["next"]
    assert "STOP" in gate_dict["next"]


def test_evaluate_partial_missing_outcomes_reports_exact_missing_state() -> None:
    """When one outcome is measured but another is missing, reason details the missing outcome."""
    cfg = ControlConfig(pass_threshold_accuracy_below=0.15, pass_threshold_recall_below=0.01)
    outcome = evaluate(0.05, None, cfg, n_questions=1)
    assert outcome.verdict is Verdict.INCONCLUSIVE
    assert "accuracy=0.05" in outcome.reason
    assert "recall=None" in outcome.reason
    assert outcome.thresholds == {"accuracy": 0.15, "recall": 0.01}


# =============================================================================
# 2. Lifecycle Interface Regression Tests with Fake Provider / Backend Errors
# =============================================================================


class FailingBackend:
    """Simulates a backend that fails during generation with a specified exception."""

    def __init__(self, exc: Exception, ready: bool = True) -> None:
        self.name = "failing-backend"
        self.exc = exc
        self.ready = ready
        self.calls: list[GenerationRequest] = []

    def health(self) -> dict[str, Any]:
        return {"ready": self.ready, "name": self.name}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        raise self.exc

    def context_window(self, model: str) -> int:
        return 4096

    def close(self) -> None:
        pass


class LeakingBackend(DummyTestBackend):
    """Simulates a backend that leaks answers so negative control fails to collapse."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return GenerationResult(
            text="Paris is France's capital city.",
            finish_reason="stop",
            prompt_tokens=15,
            completion_tokens=4,
            latency_ms=2.5,
            ttft_ms=1.0,
            queue_wait_ms=0.1,
            backend=self.name,
            model_version="test-v1",
        )


class TruncatedBackend(DummyTestBackend):
    """Simulates a backend that returns truncated/unusable responses."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return GenerationResult(
            text="Incomplete answer due to token limit...",
            finish_reason="length",
            prompt_tokens=15,
            completion_tokens=4,
            latency_ms=2.5,
            ttft_ms=1.0,
            queue_wait_ms=0.1,
            backend=self.name,
            model_version="test-v1",
        )


def test_lifecycle_backend_unavailable_error_during_generation(tmp_path: Path) -> None:
    """When model loading or provider raises BackendUnavailableError during generate(),
    lifecycle records backend.available=False with error details and raises BackendUnavailableError.
    Ladder must never advance.
    """
    run_dir = tmp_path / "run_backend_unavailable"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = FailingBackend(
        BackendUnavailableError("CUDA device out of memory: model load failed")
    )
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    with pytest.raises(BackendUnavailableError, match="CUDA device out of memory"):
        lifecycle.run_rung("negative")

    # Durable report verification
    control_path = run_dir / "arm_e_control.json"
    assert control_path.exists()
    report = json.loads(control_path.read_text(encoding="utf-8"))
    assert report["backend"]["available"] is False
    assert report["backend"]["error"] == "BackendUnavailableError"
    assert "CUDA device out of memory" in report["backend"]["detail"]
    assert report["verdict"] is None, "A broken backend must not claim a scientific verdict"
    assert report["proceed"] is False
    assert report["thresholds"]["accuracy"] == 0.25
    assert report["thresholds"]["recall"] == 0.10

    # Ladder must not advance
    ladder_state = read_ladder(run_dir / "ladder.json")
    assert "negative" not in ladder_state.passed


def test_lifecycle_real_fast_mode_value_error_reproduction(tmp_path: Path) -> None:
    """Exact reproduction of the real fast-mode failure (ValueError: Some modules dispatched on CPU/disk).
    Lifecycle must:
    1. Distinguish backend execution failure from a measured collapse failure.
    2. Mark backend.available=False in arm_e_control.json (not falsely True).
    3. Include actual causal detail in arm_e_control.json and gate notes.
    4. Not cite 'pass thresholds are null' when thresholds exist.
    5. Not cite 'Arm E did NOT collapse' when generator failed to produce responses.
    6. Never advance ladder on missing outcomes.
    """
    run_dir = tmp_path / "run_real_fast_failure"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    error_msg = (
        "ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough "
        "GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the "
        "disk while keeping these modules in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` "
        "and pass a custom `device_map` to `from_pretrained`."
    )
    backend = FailingBackend(ValueError(error_msg))
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    with pytest.raises(GateFailedError) as exc_info:
        lifecycle.run_rung("negative")

    assert "generator failed to produce responses" in str(exc_info.value)
    assert "Some modules are dispatched on the CPU" in str(exc_info.value)

    # 1. arm_e_control.json verification
    control_path = run_dir / "arm_e_control.json"
    assert control_path.exists()
    report = json.loads(control_path.read_text(encoding="utf-8"))

    # Backend availability must NOT be falsely true
    assert report["backend"]["available"] is False
    assert report["backend"]["error"] == "ValueError"
    assert "Some modules are dispatched on the CPU" in report["backend"]["detail"]
    assert report["backend"]["n_failed"] > 0

    # Thresholds must NOT be reported as null
    assert report["thresholds"]["accuracy"] == 0.25
    assert report["thresholds"]["recall"] == 0.10
    assert "pass thresholds are null" not in report["reason"]
    assert "generator failed to produce responses" in report["reason"]

    # Diagnostics section
    assert report["diagnostics"]["thresholds_measurable"] is True
    assert "Some modules are dispatched on the CPU" in report["diagnostics"]["causal_detail"]
    assert report["diagnostics"]["n_failed"] > 0

    # 2. go_no_go.json gate notes verification
    gate_path = run_dir / "go_no_go.json"
    assert gate_path.exists()
    gate_data = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_notes = " ".join(gate_data.get("notes", []))

    # Gate note must diagnose execution failure, NOT a collapse failure
    assert "Arm E execution failure: generator failed to produce responses" in gate_notes
    assert "Some modules are dispatched on the CPU" in gate_notes
    assert "Arm E did NOT collapse" not in gate_notes

    # 3. Ladder must NOT advance
    ladder_state = read_ladder(run_dir / "ladder.json")
    assert "negative" not in ladder_state.passed


# =============================================================================
# 3. Distinction Tests: Execution Failure vs Collapse Failure vs Pass
# =============================================================================


def test_distinguish_measured_collapse_failure_from_execution_failure(tmp_path: Path) -> None:
    """When generation succeeds but model answers exceed pass thresholds, it is a
    MEASURED COLLAPSE FAILURE:
    - backend.available is True
    - coverage_complete is True
    - verdict is FAIL
    - gate note states 'Arm E did NOT collapse: FAIL'
    """
    run_dir = tmp_path / "run_measured_collapse_failure"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)

    # Backend answers with gold answer, meaning accuracy will be high and will NOT collapse
    backend = LeakingBackend(answer="Paris")
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    with pytest.raises(GateFailedError) as exc_info:
        lifecycle.run_rung("negative")

    assert "Arm E negative control failed to collapse" in str(exc_info.value)
    assert "control did NOT collapse" in str(exc_info.value)

    # arm_e_control.json must report measured scientific failure
    report = json.loads((run_dir / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["backend"]["available"] is True
    assert report["coverage_complete"] is True
    assert report["verdict"] == "FAIL"
    assert report["accuracy"] is not None and report["accuracy"] > 0.25

    # Gate note must cite measured collapse failure
    gate_data = json.loads((run_dir / "go_no_go.json").read_text(encoding="utf-8"))
    notes = " ".join(gate_data["notes"])
    assert "Arm E did NOT collapse: FAIL" in notes
    assert "execution failure" not in notes

    # Ladder must not advance
    assert "negative" not in read_ladder(run_dir / "ladder.json").passed


def test_distinguish_incomplete_measured_output(tmp_path: Path) -> None:
    """When generation produces unusable finish reasons (e.g. 'length' truncation),
    it is INCOMPLETE MEASURED OUTPUT, not an Arm E collapse failure:
    - coverage_complete is False
    - gate note states 'Arm E incomplete output'
    - does NOT say 'Arm E did NOT collapse'
    """
    run_dir = tmp_path / "run_incomplete_output"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)

    # Backend answers with truncated finish reason
    backend = TruncatedBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    with pytest.raises(GateFailedError) as exc_info:
        lifecycle.run_rung("negative")

    assert "Arm E output incomplete" in str(exc_info.value)

    # arm_e_control.json
    report = json.loads((run_dir / "arm_e_control.json").read_text(encoding="utf-8"))
    assert report["coverage_complete"] is False
    assert len(report["unusable_finish"]) > 0

    # go_no_go.json
    gate_data = json.loads((run_dir / "go_no_go.json").read_text(encoding="utf-8"))
    notes = " ".join(gate_data["notes"])
    assert "Arm E incomplete output" in notes
    assert "Arm E did NOT collapse" not in notes

    # Ladder must not advance
    assert "negative" not in read_ladder(run_dir / "ladder.json").passed


def test_successful_arm_e_collapse_advances_ladder(tmp_path: Path) -> None:
    """When generation succeeds and accuracy/recall are below thresholds:
    - verdict is PASS
    - gate note states 'Arm E negative control: PASS'
    - ladder advances to include 'negative'
    """
    run_dir = tmp_path / "run_successful_collapse"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend(answer="I have no idea.")
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    report = lifecycle.run_rung("negative")
    assert report.passed is True

    ctrl_report = json.loads((run_dir / "arm_e_control.json").read_text(encoding="utf-8"))
    assert ctrl_report["backend"]["available"] is True
    assert ctrl_report["coverage_complete"] is True
    assert ctrl_report["verdict"] == "PASS"

    gate_data = json.loads((run_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert any("Arm E negative control: PASS" in n for n in gate_data["notes"])

    # Ladder must advance
    assert "negative" in read_ladder(run_dir / "ladder.json").passed


# =============================================================================
# 4. Baselines & Full Rungs Diagnostics
# =============================================================================


def test_baselines_rung_generator_failure_diagnostics(tmp_path: Path) -> None:
    """When generator failure happens during baselines rung, gate note and exception
    include actual causal detail, and ladder does not advance.
    """
    run_dir = tmp_path / "run_baselines_fail"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)

    # First run negative rung successfully
    backend = DummyTestBackend(answer="I don't know")
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    inputs = RunInputs.from_synthetic(make_synthetic_questions())

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    rep_neg = lifecycle.run_rung("negative")
    assert rep_neg.passed

    # Now make backend fail for baselines
    object.__setattr__(
        reader,
        "_backend",
        FailingBackend(RuntimeError("Provider connection reset during baseline generation")),
    )

    with pytest.raises(GateFailedError) as exc_info:
        lifecycle.run_rung("baselines")

    assert "Provider connection reset" in str(exc_info.value)

    gate_data = json.loads((run_dir / "go_no_go.json").read_text(encoding="utf-8"))
    notes = " ".join(gate_data["notes"])
    assert "Baselines rung execution failure" in notes
    assert "Provider connection reset" in notes

    # Ladder must not advance to baselines
    state = read_ladder(run_dir / "ladder.json")
    assert "negative" in state.passed
    assert "baselines" not in state.passed
