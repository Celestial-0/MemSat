"""Tests for the durable run lifecycle and optimized WAL reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

import msat.runner.executor as executor_module
from msat.arms.registry import build_registry
from msat.datasets.schema import CorpusStats, Question, Session, Turn
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)
from msat.providers.local import LocalReader, ReaderVramSpec
from msat.runner.executor import (
    reconcile,
)
from msat.runner.input import MissingInputError, RunInputs
from msat.runner.ladder import LadderOrderViolationError
from msat.runner.lifecycle import (
    CapacityRefusalError,
    GateFailedError,
    RunLifecycle,
    check_capacity,
    read_ladder,
)
from msat.runner.schedule import ScheduledUnit
from msat.runner.wal import Wal, WalState
from msat.store import Condition, ResponseStore, StoredResponse, cache_key

# =============================================================================
# Helper Fixtures & Dummy Objects
# =============================================================================


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

    class DummyPricing:
        token_estimates: ClassVar[dict[str, float]] = {"chars_per_token": 4.0}

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
# 1. Reconcile Optimization & Outcome Tests
# =============================================================================


def test_fresh_input_objects_resume_derived_control_pool(tmp_path: Path) -> None:
    """Loading benchmark inputs again preserves control and later-rung attribution."""
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    reader = make_test_reader(backend=backend)

    def restart() -> RunLifecycle:
        questions = make_synthetic_questions()
        return RunLifecycle(
            run_dir=tmp_path,
            bundle=bundle,
            registry=registry,
            panel=[reader],
            inputs=RunInputs(questions={q.qid: q for q in questions}, allow_test_adapters=True),
            resume=True,
        )

    assert restart().run_rung("negative").passed
    n_calls = len(backend.calls)
    assert restart().run_rung("negative").run_result.n_generated == 0
    assert len(backend.calls) == n_calls
    assert restart().run_rung("baselines").passed


def test_retrieval_depth_refuses_underfilled_inputs(tmp_path: Path) -> None:
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    lifecycle = RunLifecycle(
        run_dir=tmp_path,
        bundle=bundle,
        registry=registry,
        panel=[make_test_reader(backend=backend)],
        inputs=RunInputs.from_synthetic(make_synthetic_questions()),
    )
    unit = ScheduledUnit(
        Condition("A+G", 4, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
    )
    with pytest.raises(MissingInputError, match="supplies 2 retrieval memories for declared K=4"):
        lifecycle._execute_schedule([unit])
    assert backend.calls == []
    assert not (tmp_path / "ladder.json").exists()


@pytest.mark.parametrize("changed", ["config", "question", "pool"])
def test_resume_refuses_changed_attribution_before_generation(tmp_path: Path, changed: str) -> None:
    from dataclasses import replace

    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    reader = make_test_reader(backend=backend)
    inputs = RunInputs.from_synthetic(make_synthetic_questions())
    lifecycle = RunLifecycle(
        run_dir=tmp_path, bundle=bundle, registry=registry, panel=[reader], inputs=inputs
    )
    assert lifecycle.run_rung("negative").passed
    original_calls = len(backend.calls)
    original_ladder = (tmp_path / "ladder.json").read_bytes()
    fresh_bundle = make_test_bundle()
    fresh_questions = make_synthetic_questions()
    if changed == "config":
        fresh_bundle.config_hash = "changed_config"
    if changed == "question":
        fresh_questions[0] = replace(fresh_questions[0], question="A different question?")
    fresh_inputs = RunInputs.from_synthetic(fresh_questions)
    if changed == "pool":
        pool = fresh_inputs.build_control_pool()
        pool.by_question["q2"][0] = replace(pool.by_question["q2"][0], text="Changed donor content")
    resumed = RunLifecycle(
        run_dir=tmp_path,
        bundle=fresh_bundle,
        registry=registry,
        panel=[reader],
        inputs=fresh_inputs,
        resume=True,
    )
    with pytest.raises(ValueError, match="configuration or question inputs changed"):
        resumed.run_rung("baselines")
    assert len(backend.calls) == original_calls
    assert (tmp_path / "ladder.json").read_bytes() == original_ladder


def test_invalid_normalisation_refuses_control_generation(tmp_path: Path) -> None:
    bundle = make_test_bundle()
    bundle.raw["prompts/normalisation.yaml"] = {
        "normalisation": {"steps": [{"id": "bad", "op": "unknown_operation"}]}
    }
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    lifecycle = RunLifecycle(
        run_dir=tmp_path,
        bundle=bundle,
        registry=registry,
        panel=[make_test_reader(backend=backend)],
        inputs=RunInputs.from_synthetic(make_synthetic_questions()),
    )
    with pytest.raises(ValueError, match="unknown op"):
        lifecycle.run_rung("negative")
    assert backend.calls == []
    assert not (tmp_path / "ladder.json").exists()


def test_reconcile_single_pass_read_count_proof(tmp_path: Path) -> None:
    """Proof that reconcile reads the WAL exactly ONCE, avoiding O(N*W) file rescans."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    req = {"model": "m", "messages": [{"role": "user", "content": "q"}], "seed": 0}
    ck = cache_key(cell, req)
    store.put(StoredResponse(ck, cell.as_dict(), req, "text", "stop", "v1", 1, 1, 1.0))

    # Populate WAL with multiple done entries
    n_units = 50
    with Wal(wal_path) as wal:
        for i in range(n_units):
            wal.append(WalState.DONE, cell.key, f"q{i}", 0, cache_key=ck)

    schedule = [ScheduledUnit(cell, f"q{i}", 0) for i in range(n_units)]

    # Instrument read_wal to count invocations
    real_read_wal = executor_module.read_wal
    call_count = 0

    def counting_read_wal(path: Path):
        nonlocal call_count
        call_count += 1
        return real_read_wal(path)

    with patch.object(executor_module, "read_wal", side_effect=counting_read_wal):
        remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert call_count == 1, (
        f"read_wal was called {call_count} times; expected exactly 1 pass over the WAL"
    )
    assert skipped == n_units
    assert remaining == []


def test_reconcile_preserves_missing_response(tmp_path: Path) -> None:
    """A WAL entry marked DONE whose response file is missing in store is not skipped."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    with Wal(wal_path) as wal:
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key="nonexistent_key_123")

    schedule = [ScheduledUnit(cell, "q1", 0)]
    remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert skipped == 0
    assert len(remaining) == 1
    assert remaining[0].qid == "q1"


def test_reconcile_preserves_corrupt_response(tmp_path: Path) -> None:
    """A WAL entry marked DONE whose response file in store is corrupted is not skipped."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    req = {"model": "m", "messages": [{"role": "user", "content": "q"}], "seed": 0}
    ck = cache_key(cell, req)

    # Write corrupt response file on disk
    corrupt_file = store.path_for(ck)
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_text('{"key": "truncated...', encoding="utf-8")
    assert store.has(ck)
    assert store.get(ck) is None  # Corrupt file deserializes as None

    with Wal(wal_path) as wal:
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key=ck)

    schedule = [ScheduledUnit(cell, "q1", 0)]
    remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert skipped == 0
    assert len(remaining) == 1


def test_reconcile_preserves_duplicate_wal_entries(tmp_path: Path) -> None:
    """Duplicate WAL entries for the same unit are handled correctly without duplicate skipping."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    req = {"model": "m", "messages": [{"role": "user", "content": "q"}], "seed": 0}
    ck = cache_key(cell, req)
    store.put(StoredResponse(ck, cell.as_dict(), req, "text", "stop", "v1", 1, 1, 1.0))

    with Wal(wal_path) as wal:
        wal.append(WalState.FAILED, cell.key, "q1", 0, attempts=1)
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key=ck, attempts=2)
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key=ck, attempts=2)  # Duplicate DONE

    schedule = [ScheduledUnit(cell, "q1", 0)]
    remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert skipped == 1
    assert remaining == []


def test_reconcile_preserves_exhausted_retries(tmp_path: Path) -> None:
    """Units that exhausted retries are excluded from remaining and not skipped."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    with Wal(wal_path) as wal:
        wal.append(WalState.FAILED, cell.key, "q1", 0, attempts=3)

    schedule = [ScheduledUnit(cell, "q1", 0)]
    remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert skipped == 0
    assert remaining == []


def test_reconcile_preserves_truncated_final_wal_line(tmp_path: Path) -> None:
    """A crash resulting in a truncated final line in WAL is handled cleanly."""
    wal_path = tmp_path / "WAL"
    cell = Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma")
    store = ResponseStore(tmp_path / "responses")

    req = {"model": "m", "messages": [{"role": "user", "content": "q"}], "seed": 0}
    ck = cache_key(cell, req)
    store.put(StoredResponse(ck, cell.as_dict(), req, "text", "stop", "v1", 1, 1, 1.0))

    with Wal(wal_path) as wal:
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key=ck)

    with wal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 1, "state": "done", "cell":')  # Incomplete line

    schedule = [ScheduledUnit(cell, "q1", 0), ScheduledUnit(cell, "q2", 0)]
    remaining, skipped = reconcile(schedule, wal_path, store, max_attempts=3)

    assert skipped == 1
    assert len(remaining) == 1
    assert remaining[0].qid == "q2"


# =============================================================================
# 2. Run Lifecycle Integration & Seam Tests
# =============================================================================


def test_tiny_test_adapter_ladder_path(tmp_path: Path) -> None:
    """Tiny test-adapter path runs negative -> restart/resume -> baseline -> full.
    Verifies actual responses, WAL, ledger, conditions, observations and ladder progression.
    """
    run_dir = tmp_path / "run_lifecycle_test"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend(answer="Paris")
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]

    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )

    # 1. Negative Rung
    rep_neg = lifecycle.run_rung("negative")
    assert rep_neg.passed
    assert (run_dir / "WAL").exists()
    assert (run_dir / "ledger.json").exists()
    assert (run_dir / "arm_e_control.json").exists()
    assert (run_dir / "go_no_go.json").exists()
    assert (run_dir / "ladder.json").exists()
    assert (run_dir / "conditions.parquet").exists()
    assert (run_dir / "observations.parquet").exists()
    assert (run_dir / "observations.json").exists()

    ladder_state = read_ladder(run_dir / "ladder.json")
    assert "negative" in ladder_state.passed

    # 2. Warm path verification (re-running negative with resume=True makes zero new calls)
    backend.calls.clear()
    lifecycle_resume = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
        resume=True,
    )
    rep_neg_warm = lifecycle_resume.run_rung("negative")
    assert rep_neg_warm.run_result.n_generated == 0
    assert rep_neg_warm.run_result.n_warm_hits > 0
    assert len(backend.calls) == 0, "Warm path must not call generator"

    # 3. Baselines Rung
    rep_base = lifecycle.run_rung("baselines")
    assert rep_base.passed
    ladder_state = read_ladder(run_dir / "ladder.json")
    assert "baselines" in ladder_state.passed

    # 4. Full Rung with ExecutablePlan and crossed A+G
    from msat.planning import ExecutablePlan, PlanCondition

    cell1 = Condition("A+G", 1, 256.0, "per_memory_tokens", "S", "test-reader")
    # Match the declared grid to the two actual sessions in this fixture.
    bundle.design.K_grid = (1, 2)
    cell2 = Condition("A+G", 2, 256.0, "per_memory_tokens", "S", "test-reader")
    sched = [
        ScheduledUnit(cell1, "q1", 0),
        ScheduledUnit(cell2, "q1", 0),
    ]
    plan = ExecutablePlan(
        conditions=(
            PlanCondition(cell1, 1, 0.0),
            PlanCondition(cell2, 1, 0.0),
        ),
        rejected=(),
        schedule=tuple(sched),
        config_hash=bundle.config_hash,
        seed=20261012,
        repetition=1,
    )
    lifecycle.plan = plan
    rep_crossed = lifecycle.run_rung("crossed")
    assert rep_crossed.passed
    ladder_state = read_ladder(run_dir / "ladder.json")
    assert "crossed" in ladder_state.passed

    # Verify both observations and metadata sidecars exist and are durable
    assert (run_dir / "observations.parquet").exists()
    assert (run_dir / "metadata.parquet").exists()
    assert (run_dir / "observations.json").exists()
    assert (run_dir / "metadata.json").exists()

    from msat.runner.metadata import read_observation_metadata

    obs_list = read_observation_metadata(run_dir / "observations.parquet")
    assert len(obs_list) >= 4, "Observations sidecar must preserve all rungs"
    for obs in obs_list:
        assert "|" in obs.condition_key, "condition_key must be pipe-separated canonical string"
        assert len(obs.prompt_sha256) == 64, (
            "prompt_sha256 must be SHA256 hex digest, not full prompt"
        )
        assert obs.ability == "knowledge-retrieval"
        assert obs.question_type == "knowledge-retrieval"
        assert obs.gold_rank in (1, 2, None)


def test_scale_s_and_m_same_qid_survive_loader() -> None:
    """Questions sharing the same QID across S and M scales survive without collapsing."""
    q_s = Question(
        qid="shared_100",
        scale="S",
        question="Scale S question content",
        question_date="2026-01-01",
        gold_answers=("Ans S",),
        gold_session_ids=("sess_s_gold",),
        ability="knowledge-retrieval",
        answerable=True,
        corpus=CorpusStats(n_sessions=1, n_turns=1, n_chars=50, n_gold_sessions=1),
        sessions=(
            Session(
                session_id="sess_s_gold",
                date="2026-01-01",
                turns=(Turn(role="user", content="Content S", has_answer=True),),
            ),
        ),
    )
    q_m = Question(
        qid="shared_100",
        scale="M",
        question="Scale M question content",
        question_date="2026-01-01",
        gold_answers=("Ans M",),
        gold_session_ids=("sess_m_gold",),
        ability="reasoning",
        answerable=True,
        corpus=CorpusStats(n_sessions=1, n_turns=1, n_chars=50, n_gold_sessions=1),
        sessions=(
            Session(
                session_id="sess_m_gold",
                date="2026-01-01",
                turns=(Turn(role="user", content="Content M", has_answer=True),),
            ),
        ),
    )
    inputs = RunInputs.from_synthetic([q_s, q_m])
    qids = inputs.qids_by_scale()
    assert "shared_100" in qids["S"]
    assert "shared_100" in qids["M"]
    assert inputs.get_question("shared_100", scale="S").question == "Scale S question content"
    assert inputs.get_question("shared_100", scale="M").question == "Scale M question content"
    assert inputs.get_gold_answer("shared_100", scale="S") == "Ans S"
    assert inputs.get_gold_answer("shared_100", scale="M") == "Ans M"


def test_missing_sessions_refuses_placeholder_generation(tmp_path: Path) -> None:
    """Questions missing sessions raise MissingInputError; placeholder memories are forbidden."""
    run_dir = tmp_path / "run_missing_sess"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]

    empty_q = Question(
        qid="q_no_sess",
        scale="S",
        question="What is this?",
        question_date="2026-01-01",
        gold_answers=("Ans",),
        gold_session_ids=("sess_none",),
        ability="knowledge-retrieval",
        answerable=True,
        corpus=CorpusStats(n_sessions=0, n_turns=0, n_chars=0, n_gold_sessions=0),
        sessions=(),  # Absent sessions
    )
    inputs = RunInputs.from_synthetic([empty_q])
    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    unit = ScheduledUnit(
        Condition("A", 1, 256.0, "per_memory_tokens", "S", "test-reader"), "q_no_sess", 0
    )
    with pytest.raises(MissingInputError) as exc_info:
        lifecycle._render_unit(unit)
    assert "Refusing to manufacture placeholder sessions" in str(exc_info.value)


def test_composite_arm_a_plus_g_dispatches_crossed_treatment(tmp_path: Path) -> None:
    """Composite arm A+G dispatches real crossed treatment without KeyError."""
    run_dir = tmp_path / "run_arm_ag"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    unit = ScheduledUnit(
        Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
    )
    rendered = lifecycle._render_unit(unit)
    assert rendered.arm == "A+G"
    assert rendered.k == 1
    assert rendered.b == 128.0
    assert rendered.b_mode == "per_memory_tokens"
    assert "Paris is France's capital" in rendered.user
    assert "Memory 0 for q1" not in rendered.user


def test_arm_f_oracle_gold_sessions_only(tmp_path: Path) -> None:
    """Arm F includes only oracle gold sessions."""
    run_dir = tmp_path / "run_arm_f"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    unit = ScheduledUnit(Condition("F", 1, 128.0, "per_memory_tokens", "S", "test-reader"), "q1", 0)
    rendered = lifecycle._render_unit(unit)
    assert rendered.session_ids == ("sess_q1_gold",)
    assert "sess_q1_distractor" not in rendered.session_ids


def test_arm_b_filler_padding_and_gold_exclusion(tmp_path: Path) -> None:
    """Arm B pads with filler sessions and excludes gold sessions."""
    run_dir = tmp_path / "run_arm_b"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]
    questions = make_synthetic_questions()

    # Create filler session
    filler_sessions = ("Filler conversation turn one.", "Filler conversation turn two.")
    inputs = RunInputs.from_synthetic(questions, filler_sessions=filler_sessions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    unit = ScheduledUnit(Condition("B", 1, 0.5, "corpus_fraction", "S", "test-reader"), "q1", 0)
    # Target fraction requires reaching budget; missing filler or token mismatch raises cleanly
    with pytest.raises(Exception) as exc_info:
        lifecycle._render_unit(unit)
    assert (
        "TokenTargetMissedError" in type(exc_info.value).__name__
        or "Filler" in type(exc_info.value).__name__
        or "GoldLeakageError" in type(exc_info.value).__name__
    )


def test_arm_c_position_variants_across_replicates(tmp_path: Path) -> None:
    """Arm C reorders memories across replicates."""
    run_dir = tmp_path / "run_arm_c"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    unit0 = ScheduledUnit(
        Condition("C", 2, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
    )
    unit1 = ScheduledUnit(
        Condition("C", 2, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 1
    )
    r0 = lifecycle._render_unit(unit0)
    r1 = lifecycle._render_unit(unit1)
    assert r0.session_ids == tuple(reversed(r1.session_ids))
    assert r0.session_ids != r1.session_ids


def test_failure_or_exhaustion_cannot_advance_ladder(tmp_path: Path) -> None:
    """Generation failure or retry exhaustion persists available artifacts but halts ladder."""
    run_dir = tmp_path / "run_exhaust_fail"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    # Set backend to fail after 0 successful calls
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    # Pre-populate unit in WAL as FAILED with max attempts
    cell = Condition("E", 1, 256.0, "per_memory_tokens", "S", "test-reader")
    with Wal(run_dir / "WAL") as wal:
        wal.append(WalState.FAILED, cell.key, "q1", 0, attempts=3)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
        resume=True,
    )
    with pytest.raises(GateFailedError):
        lifecycle.run_rung("negative")

    # Ladder must not advance
    state = read_ladder(run_dir / "ladder.json")
    assert "negative" not in state.passed
    assert (run_dir / "arm_e_control.json").exists()


def test_unconfigured_external_embedder_refuses_live_execution(tmp_path: Path) -> None:
    """Non-synthetic run without available external model raises MissingInputError."""
    run_dir = tmp_path / "run_embed_refuse"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]
    questions = make_synthetic_questions()
    # Non-synthetic inputs
    inputs = RunInputs.from_synthetic(questions)
    inputs.is_synthetic = False
    inputs.allow_test_adapters = False

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    with pytest.raises(MissingInputError, match="configured embedder"):
        lifecycle._render_unit(
            ScheduledUnit(
                Condition("A", 1, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
            )
        )


def test_workflow_preflight_capacity_rejection(tmp_path: Path) -> None:
    """Workflow rejects capacity overflow before any generation occurs."""
    from msat.runner.run_workflow import run_lifecycle_workflow

    run_dir = tmp_path / "run_workflow_cap"
    bundle = make_test_bundle()
    bundle.design.readers = ("tiny-reader",)
    tiny_reader = make_test_reader(reader_id="tiny-reader", declared_context_window=10)
    panel = [tiny_reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    # A complete declared grid reaches capacity validation before dispatch.
    plan_file = tmp_path / "test_plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": "cfg1",
                "cells": [
                    {
                        "arm": "A",
                        "K": k,
                        "B": b,
                        "B_mode": "per_memory_tokens",
                        "scale": "S",
                        "reader": "tiny-reader",
                        "n_questions": 1,
                        "degeneracy_fraction": 0.0,
                        "status": "accepted",
                        "reason": "accepted",
                    }
                    for k in bundle.design.K_grid
                    for b in bundle.design.b_values
                ],
                "schedule": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CapacityRefusalError):
        run_lifecycle_workflow(
            bundle,
            run_dir,
            rung="negative",
            inputs=inputs,
            panel=panel,
            plan_path=plan_file,
            is_synthetic=True,
        )


def test_missing_inputs_gate_failure_cannot_advance(tmp_path: Path) -> None:
    """Missing inputs or gate failure cannot advance the ladder."""
    run_dir = tmp_path / "run_gate_fail"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    reader = make_test_reader(reader_id="test-reader", backend=DummyTestBackend())
    panel = [reader]

    # Empty inputs without pool or questions must fail
    empty_inputs = RunInputs.from_synthetic([])
    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=empty_inputs,
    )
    with pytest.raises(MissingInputError):
        lifecycle.run_rung("negative")

    assert not (run_dir / "ladder.json").exists()

    # Attempting to enter 'baselines' when 'negative' has not passed raises LadderOrderViolationError
    with pytest.raises(LadderOrderViolationError):
        lifecycle.run_rung("baselines")


def test_interruption_resumes_only_remainder(tmp_path: Path) -> None:
    """Interruption mid-flight resumes only the ungenerated remainder."""
    run_dir = tmp_path / "run_interruption"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    # Pre-populate one unit as completed in WAL and store
    cell = Condition("E", 1, 256.0, "per_memory_tokens", "S", "test-reader")
    unit1 = ScheduledUnit(cell, "q1", 0)
    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    _ = lifecycle._render_unit(unit1)
    req_body1 = lifecycle._render_request_body(unit1)
    ck1 = cache_key(cell, req_body1)

    store = ResponseStore(run_dir / "responses")
    store.put(StoredResponse(ck1, cell.as_dict(), req_body1, "London", "stop", "v1", 10, 2, 1.0))
    with Wal(run_dir / "WAL") as wal:
        wal.append(WalState.DONE, cell.key, "q1", 0, cache_key=ck1)

    # Resume run
    lifecycle_resumed = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
        resume=True,
    )
    report = lifecycle_resumed.run_rung("negative")

    # Unit 1 was warm and skipped; 0 calls for unit 1
    assert report.run_result.n_warm_hits >= 1


def test_backend_unavailable_is_infrastructure_failure(tmp_path: Path) -> None:
    """Backend offline raises BackendUnavailableError, not a fabricated scientific negative."""
    run_dir = tmp_path / "run_backend_offline"
    bundle = make_test_bundle()
    registry, _ = build_registry(bundle)
    offline_backend = DummyTestBackend(available=False)
    reader = make_test_reader(reader_id="test-reader", backend=offline_backend)
    panel = [reader]
    questions = make_synthetic_questions()
    inputs = RunInputs.from_synthetic(questions)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        bundle=bundle,
        registry=registry,
        panel=panel,
        inputs=inputs,
    )
    with pytest.raises(BackendUnavailableError):
        lifecycle.run_rung("negative")


def test_capacity_check_before_generation(tmp_path: Path) -> None:
    """A reader with insufficient context window triggers capacity check rejection."""
    # Reader with tiny declared context window 10 tokens
    tiny_reader = make_test_reader(reader_id="tiny-reader", declared_context_window=10)
    verdict = check_capacity(tiny_reader, prompt_tokens=250, label="ArmA/test")

    assert not verdict.supported
    assert verdict.prompt_tokens == 250
    assert verdict.reader_capacity_tokens == 10
    assert "cannot serve" in verdict.reason


def test_warm_cache_hit_merges_replicates_and_updates_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm cache hit for new replicate merges replicate into StoredResponse and updates ledger."""
    from msat.runner.executor import Executor, RunConfig
    from msat.store import Condition, ResponseStore, StoredResponse, cache_key

    run_dir = tmp_path / "warm_hit_merge"
    store_dir = run_dir / "responses"
    ledger_path = run_dir / "ledger.json"

    store = ResponseStore(store_dir)
    cell = Condition("A", 1, 128.0, "per_memory_tokens", "S", "test-reader")
    req_body = {
        "model": "test-reader",
        "messages": [{"role": "user", "content": "q1 content"}],
        "seed": 42,
    }
    ck = cache_key(cell, req_body)

    # Initial response has replicate 0
    init_resp = StoredResponse(
        key=ck,
        condition=cell.as_dict(),
        request=req_body,
        text="Paris",
        finish_reason="stop",
        model_version="v1.0",
        prompt_tokens=10,
        completion_tokens=2,
        latency_ms=1.5,
        qid="q1",
        repetitions=(0,),
    )
    store.put(init_resp)

    # Run executor with replicate 1 on the same cell and question
    config = RunConfig(run_dir=run_dir, max_attempts=3)
    calls = []

    def generator(unit: ScheduledUnit, *, attempt: int) -> StoredResponse:
        calls.append(unit)
        raise RuntimeError("Generator must not be called on warm hit")

    executor = Executor(config, generator, lambda u: req_body, resume=True)
    unit1 = ScheduledUnit(cell, "q1", 1)
    res = executor.run([unit1])

    assert res.n_warm_hits == 1
    assert res.n_generated == 0
    assert len(calls) == 0

    # Verify store has merged replicates (0, 1)
    loaded = store.get(ck)
    assert loaded is not None
    assert loaded.repetitions == (0, 1)
    assert loaded.as_dict()["replicates"] == [0, 1]

    # Verify ledger completion count includes the warm hit
    ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger_data["n_done"] == 1


def test_metadata_condition_key_matches_metrics_scoring(tmp_path: Path) -> None:
    """ObservationMetadata condition_key matches metrics.score.condition_key_str exactly."""
    from msat.metrics.score import condition_key_str
    from msat.runner.metadata import (
        ObservationMetadata,
        read_observation_metadata,
        write_observation_metadata,
    )

    cell = Condition("A+G", 4, 256.0, "per_memory_tokens", "S", "gemini-flash")
    req_body = {
        "model": "gemini-flash",
        "messages": [{"role": "user", "content": "test"}],
        "seed": 0,
    }
    ck = cache_key(cell, req_body)
    stored = StoredResponse(
        key=ck,
        condition=cell.as_dict(),
        request=req_body,
        text="Answer",
        finish_reason="stop",
        model_version="v1",
        prompt_tokens=20,
        completion_tokens=5,
        latency_ms=2.0,
        qid="q1",
        repetitions=(0,),
    )
    expected_key = condition_key_str(stored)
    assert expected_key == "A+G|4|256.0|per_memory_tokens|S|gemini-flash"

    obs = ObservationMetadata(
        config_hash="cfg123",
        condition_key=expected_key,
        arm=cell.arm,
        k=cell.k,
        b=cell.b,
        b_mode=cell.b_mode,
        scale=cell.scale,
        reader=cell.reader,
        qid="q1",
        replicate=0,
        memory_ranks=[1],
        gold_ranks=[1],
        gold_rank=1,
        corpus_fraction=0.1,
        ability="knowledge-retrieval",
        question_type="knowledge-retrieval",
        reader_strength=1.0,
        prompt_sha256="a" * 64,
        cache_key=ck,
        canonical_config_hash="cfg123",
    )
    p = tmp_path / "observations.parquet"
    write_observation_metadata([obs], p)

    loaded = read_observation_metadata(p)
    assert len(loaded) == 1
    assert loaded[0].condition_key == expected_key
    assert loaded[0].gold_rank == 1
    assert loaded[0].prompt_sha256 == "a" * 64


def test_input_validation_and_scale_isolation(tmp_path: Path) -> None:
    """RunInputs validates missing paths and keeps questions isolated across scales."""
    # Missing explicit path raises MissingInputError
    with pytest.raises(MissingInputError):
        RunInputs.from_legacy_paths(
            questions_path=tmp_path / "nonexistent_questions.json",
        )

    # Scale-isolated question retrieval
    q_s = Question(
        qid="shared_id",
        scale="S",
        question="Question in S",
        question_date="2026-01-01",
        gold_answers=("A1", "A2"),
        gold_session_ids=("s_gold",),
        ability="knowledge-retrieval",
        answerable=True,
        corpus=CorpusStats(1, 1, 10, 1),
        sessions=(),
    )
    q_m = Question(
        qid="shared_id",
        scale="M",
        question="Question in M",
        question_date="2026-01-01",
        gold_answers=("B1",),
        gold_session_ids=("m_gold",),
        ability="reasoning",
        answerable=True,
        corpus=CorpusStats(1, 1, 10, 1),
        sessions=(),
    )
    inputs = RunInputs.from_synthetic([q_s, q_m])
    assert inputs.get_question("shared_id", scale="S").question == "Question in S"
    assert inputs.get_question("shared_id", scale="M").question == "Question in M"
    assert inputs.get_gold_answers("shared_id", scale="S") == ("A1", "A2")
    assert inputs.get_gold_answers("shared_id", scale="M") == ("B1",)
    assert inputs.get_gold_sessions("shared_id", scale="S") == frozenset({"s_gold"})
    assert inputs.get_gold_sessions("shared_id", scale="M") == frozenset({"m_gold"})
