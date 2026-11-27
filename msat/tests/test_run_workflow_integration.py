"""Integration tests for MemSat run lifecycle, scoring, and analysis workflows.

Exercises the full experimental progression:
negative control -> restart/warm resume -> baselines -> full crossed grid -> score -> analyze.

Verifies:
- Zero additional generations on warm resume.
- Conditions and metadata retain prior rungs.
- Incompatible inputs and configurations refuse before dispatch.
- Primary scoring and analysis reproducibility offline without model downloads.
- Coherent session counts >= K on complete minimal declared grid.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from msat.analysis.analysis_workflow import run_analysis_workflow
from msat.analysis.mixedfx import NonBinaryOutcomeError
from msat.arms.registry import build_registry
from msat.cli import app
from msat.config import load_bundle
from msat.datasets.schema import CorpusStats, Question, Session, Turn
from msat.experiment import resolve_effective_experiment
from msat.memory.embed import HashEmbedder
from msat.metrics.score import read_scores
from msat.metrics.scoring_workflow import run_scoring_workflow
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)
from msat.providers.local import LocalReader, ReaderVramSpec
from msat.runner.input import MissingInputError, RunInputs
from msat.runner.lifecycle import (
    GateFailedError,
    RunLifecycle,
    check_capacity,
    read_ladder,
)
from msat.runner.metadata import read_observation_metadata
from msat.runner.run_workflow import run_lifecycle_workflow
from msat.runner.schedule import ScheduledUnit
from msat.store import Condition

# =============================================================================
# Test Doubles & Fixture Helpers
# =============================================================================


class DummyTestBackend:
    """In-memory test backend simulating local generation without external model downloads."""

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
            # For Arm E negative control, collapse by returning no relevant answer
            text = "There is no answer to this question in the provided history."
        elif "Paris is France's capital" in request.prompt:
            text = "Paris"
        elif "official currency is the Yen" in request.prompt:
            text = "Yen"
        elif "Berlin is Germany's capital" in request.prompt:
            text = "Berlin"
        elif "Rome is Italy's capital" in request.prompt:
            text = "Rome"
        elif "Madrid is Spain's capital" in request.prompt and "256" in request.prompt:
            text = "Madrid"
        elif "Ottawa is Canada's capital" in request.prompt and "256" in request.prompt:
            text = "Ottawa"
        else:
            text = "There is no answer."
        return GenerationResult(
            text=text,
            finish_reason="stop",
            prompt_tokens=20,
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


def make_minimal_bundle(
    k_grid: tuple[int, ...] = (1, 2),
    b_values: tuple[float, ...] = (128.0, 256.0),
    reader_id: str = "test-reader",
) -> Any:
    """Minimal bundle declaring a crossed (K, B) grid."""
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
    default_treatment_template = """system: |
  System prompt.
user: |
  {{ memories }}

  Question: {{ question }}
"""
    all_prompts = {
        "prompts/arm_e.j2": default_arm_e_template,
        "prompts/arm_a.j2": default_treatment_template,
        "prompts/arm_d.j2": default_treatment_template,
        "prompts/arm_f.j2": default_treatment_template,
        "prompts/arm_g.j2": default_treatment_template,
        "prompts/arm_b.j2": default_treatment_template,
        "prompts/arm_c.j2": default_treatment_template,
    }

    class DummyB:
        def __init__(self, val: float) -> None:
            self.value = val
            self.mode = "per_memory_tokens"

    b_grid_tuple = tuple(DummyB(b) for b in b_values)

    dummy_design = type(
        "DummyDesign",
        (),
        {
            "K_grid": k_grid,
            "B_grid": b_grid_tuple,
            "b_values": b_values,
            "scales": ("S",),
            "readers": (reader_id,),
            "runs_per_cell": 1,
            "seed": 20261012,
        },
    )

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
        config_hash = "integration_config_hash_42"
        canonical_config_hash = "integration_config_hash_42"
        seed = 20261012
        design = dummy_design()
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
            "readers.yaml": {
                "families": [
                    {
                        "id": reader_id,
                        "model": f"local/{reader_id}",
                        "version": "v1.0",
                        "released": "2026-01-01",
                        "role": "primary",
                        "reader_strength": 1.0,
                        "context_window": 4096,
                    }
                ]
            },
            "arms": {
                "arms": arms_list,
                "crossed_grid": ["A", "G"],
            },
        }
        mode = "fast"
        decoding = None

    return DummyBundle()


def make_coherent_questions(n_questions: int = 6) -> list[Question]:
    """Build synthetic benchmark questions with session counts >= max(K) and varied lengths."""
    data_specs = [
        ("q0", "What is the capital of France?", "Paris is France's capital city.", "Paris"),
        ("q1", "What is the currency of Japan?", "Japan's official currency is the Yen.", "Yen"),
        ("q2", "What is the capital of Germany?", "Berlin is Germany's capital city.", "Berlin"),
        ("q3", "What is the capital of Italy?", "Rome is Italy's capital city.", "Rome"),
        ("q4", "What is the capital of Spain?", "Madrid is Spain's capital city.", "Madrid"),
        ("q5", "What is the capital of Canada?", "Ottawa is Canada's capital city.", "Ottawa"),
    ]

    questions = []
    for idx in range(min(n_questions, len(data_specs))):
        qid, q_text, gold_content, gold_ans = data_specs[idx]
        n_sessions = 5 + idx  # 5, 6, 7, 8, 9, 10 sessions -> all >= max(K) = 2

        sessions = [
            Session(
                session_id=f"sess_{qid}_gold",
                date="2026-01-01",
                turns=(Turn(role="user", content=gold_content, has_answer=True),),
            )
        ]
        for i in range(1, n_sessions):
            sessions.append(
                Session(
                    session_id=f"sess_{qid}_distractor_{i}",
                    date=f"2026-01-{i + 1:02d}",
                    turns=(
                        Turn(
                            role="user",
                            content=f"General conversation turn {i} for question {qid}.",
                            has_answer=False,
                        ),
                    ),
                )
            )

        questions.append(
            Question(
                qid=qid,
                scale="S",
                question=q_text,
                question_date="2026-01-01",
                gold_answers=(gold_ans,),
                gold_session_ids=(f"sess_{qid}_gold",),
                ability="knowledge-retrieval",
                answerable=True,
                corpus=CorpusStats(
                    n_sessions=len(sessions),
                    n_turns=len(sessions),
                    n_chars=400,
                    n_gold_sessions=1,
                ),
                sessions=tuple(sessions),
            )
        )
    return questions


def make_manifest_file(
    path: Path,
    bundle: Any,
    qids: Sequence[str],
    k_grid: tuple[int, ...] = (1, 2),
    b_values: tuple[float, ...] = (128.0, 256.0),
    reader_id: str = "test-reader",
    scale: str = "S",
) -> Path:
    """Generate and write a valid minimal crossed-grid manifest JSON."""
    cells = []
    for k in k_grid:
        for b in b_values:
            cells.append(
                {
                    "arm": "A+G",
                    "K": k,
                    "B": float(b),
                    "B_mode": "per_memory_tokens",
                    "scale": scale,
                    "reader": reader_id,
                    "status": "supported",
                    "n_questions": len(qids),
                    "degeneracy_fraction": 0.0,
                    "reason": "ok",
                }
            )

    manifest_data = {
        "config_hash": bundle.config_hash,
        "canonical_config_hash": bundle.canonical_config_hash,
        "effective_config_hash": bundle.config_hash,
        "mode": getattr(bundle, "mode", "fast"),
        "grid": {
            "K_grid": list(k_grid),
            "B_grid": [{"value": float(b), "mode": "per_memory_tokens"} for b in b_values],
            "readers": [reader_id],
            "scales": [scale],
            "seed": 20261012,
            "runs_per_cell": 1,
        },
        "metadata": {
            "seed": 20261012,
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8")
    return path


# =============================================================================
# 1. End-to-End Workflow Integration Fixture
# =============================================================================


def test_full_workflow_progression_negative_to_analyze(tmp_path: Path) -> None:
    """Exercise complete progression: negative -> warm resume -> baselines -> full -> score -> analyze.

    Asserts:
    - Zero additional generations on warm resume.
    - Conditions and metadata retain prior rungs.
    - Scoring and analysis reproduce offline without model downloads.
    - All session counts >= max(K).
    """
    run_dir = tmp_path / "full_pipeline_run"
    bundle = make_minimal_bundle()
    backend = DummyTestBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    panel = [reader]

    questions = make_coherent_questions(n_questions=6)
    inputs = RunInputs.from_synthetic(questions)
    manifest_path = make_manifest_file(
        tmp_path / "manifest.json",
        bundle,
        qids=[q.qid for q in questions],
    )

    # -------------------------------------------------------------------------
    # Step 1: Negative Control Rung
    # -------------------------------------------------------------------------
    rep_neg = run_lifecycle_workflow(
        bundle,
        run_dir,
        rung="negative",
        inputs=inputs,
        panel=panel,
        plan_path=manifest_path,
        seed=20261012,
        embedder=HashEmbedder(),
    )
    assert rep_neg.passed
    assert (run_dir / "WAL").exists()
    assert (run_dir / "ledger.json").exists()
    assert (run_dir / "arm_e_control.json").exists()
    assert (run_dir / "go_no_go.json").exists()
    assert (run_dir / "ladder.json").exists()
    assert (run_dir / "conditions.parquet").exists()
    assert (run_dir / "observations.parquet").exists()

    ctrl_report = json.loads((run_dir / "arm_e_control.json").read_text(encoding="utf-8"))
    assert ctrl_report["verdict"] == "PASS", "Arm E negative control must collapse and pass"

    ladder_1 = read_ladder(run_dir / "ladder.json")
    assert "negative" in ladder_1.passed
    assert "baselines" not in ladder_1.passed

    calls_after_negative = len(backend.calls)
    assert calls_after_negative > 0

    # -------------------------------------------------------------------------
    # Step 2: Restart / Warm Resume (Zero additional generations)
    # -------------------------------------------------------------------------
    rep_warm = run_lifecycle_workflow(
        bundle,
        run_dir,
        rung="negative",
        inputs=inputs,
        panel=panel,
        plan_path=manifest_path,
        resume=True,
        seed=20261012,
        embedder=HashEmbedder(),
    )
    assert rep_warm.run_result.n_generated == 0, "Warm resume must generate 0 units"
    assert rep_warm.run_result.n_warm_hits > 0, "Warm resume must register warm cache hits"
    assert len(backend.calls) == calls_after_negative, "Zero additional model calls on warm resume"

    # -------------------------------------------------------------------------
    # Step 3: Baselines Rung
    # -------------------------------------------------------------------------
    rep_base = run_lifecycle_workflow(
        bundle,
        run_dir,
        rung="baselines",
        inputs=inputs,
        panel=panel,
        plan_path=manifest_path,
        seed=20261012,
        embedder=HashEmbedder(),
    )
    assert rep_base.passed
    ladder_2 = read_ladder(run_dir / "ladder.json")
    assert "negative" in ladder_2.passed
    assert "baselines" in ladder_2.passed
    assert "crossed" not in ladder_2.passed

    # Verify metadata and conditions retained negative + baselines
    obs_after_base = read_observation_metadata(run_dir / "observations.parquet")
    arms_in_obs = {obs.arm for obs in obs_after_base}
    assert "E" in arms_in_obs, "Negative control observations must be preserved"
    assert {"A", "D", "F"}.issubset(arms_in_obs), "Baseline observations must be added"

    # -------------------------------------------------------------------------
    # Step 4: Full Crossed Grid Rung
    # -------------------------------------------------------------------------
    rep_full = run_lifecycle_workflow(
        bundle,
        run_dir,
        rung="crossed",
        inputs=inputs,
        panel=panel,
        plan_path=manifest_path,
        seed=20261012,
        embedder=HashEmbedder(),
    )
    assert rep_full.passed
    ladder_3 = read_ladder(run_dir / "ladder.json")
    assert {"negative", "baselines", "crossed"}.issubset(ladder_3.passed)

    # Verify crossed grid artifact
    assert (run_dir / "crossed_grid.json").exists()
    crossed_info = json.loads((run_dir / "crossed_grid.json").read_text(encoding="utf-8"))
    assert crossed_info["crossed"], "Declared grid must be crossed without degeneracy"

    # Verify observations retention across all rungs
    obs_after_full = read_observation_metadata(run_dir / "observations.parquet")
    all_arms = {obs.arm for obs in obs_after_full}
    assert "E" in all_arms
    assert "A+G" in all_arms
    assert len(obs_after_full) >= 4 + len(bundle.design.K_grid) * len(bundle.design.b_values)

    for obs in obs_after_full:
        assert len(obs.prompt_sha256) == 64
        assert "|" in obs.condition_key
        assert obs.corpus_fraction > 0.0
        assert obs.ability == "knowledge-retrieval"

    # -------------------------------------------------------------------------
    # Step 5: Scoring (Offline, zero model calls)
    # -------------------------------------------------------------------------
    gold_map = {q.qid: q.gold_answers[0] for q in questions if q.gold_answers}
    config_dir = tmp_path / "configs"
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "normalisation.yaml").write_text(
        yaml.safe_dump({"normalisation": {"steps": [{"id": "strip", "op": "strip"}]}}),
        encoding="utf-8",
    )

    calls_before_score = len(backend.calls)
    score_res = run_scoring_workflow(
        run_dir,
        gold=gold_map,
        config_dir=config_dir,
        mode="fast",
        metric=["em", "f1"],
    )
    assert score_res.report.n_rows == 2 * len(obs_after_full)
    assert len(backend.calls) == calls_before_score, "Scoring must make 0 generation calls"
    assert (run_dir / "scores.parquet").exists()

    scores_table = read_scores(run_dir / "scores.parquet")
    score_metrics = set(scores_table["metric"].to_pylist())
    assert score_metrics == {"em", "f1"}

    # -------------------------------------------------------------------------
    # Step 6: Analysis (Offline & Reproducible)
    # -------------------------------------------------------------------------
    analysis_res_1 = run_analysis_workflow(
        run_dir,
        config_root=config_dir,
        selected_outcome="em",
        strict=False,
    )
    assert analysis_res_1.analysis_json_path.exists()
    assert analysis_res_1.results_md_path.exists()
    assert analysis_res_1.n_observations > 0

    results_md_1 = (run_dir / "results.md").read_text(encoding="utf-8")
    assert "MemSat — Analysis Results" in results_md_1

    # Analysis reproducibility check: second run yields exact identical markdown
    analysis_res_2 = run_analysis_workflow(
        run_dir,
        config_root=config_dir,
        selected_outcome="em",
        strict=False,
    )
    results_md_2 = (run_dir / "results.md").read_text(encoding="utf-8")
    assert results_md_1 == results_md_2, "Analysis results.md must be byte-for-byte reproducible"
    assert analysis_res_1.n_observations == analysis_res_2.n_observations


# =============================================================================
# 2. Production CLI Commands Integration
# =============================================================================


def test_cli_public_commands_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise production CLI commands (run, score, analyze) with injected test reader."""
    results_dir = tmp_path / "results"
    run_id = "test_cli_run"
    run_dir = results_dir / run_id

    # Set up config bundle directory
    cfg_dir = tmp_path / "configs"
    prompts_dir = cfg_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path("configs"), cfg_dir, dirs_exist_ok=True)

    # Modify design.yaml in cfg_dir to minimal grid K=[1, 2], B=[128, 256]
    des_file = cfg_dir / "design.yaml"
    raw_design = yaml.safe_load(des_file.read_text(encoding="utf-8"))
    raw_design["K_grid"] = [1, 2]
    raw_design["B_grid"] = raw_design["B_grid"][:2]
    raw_design["reference"]["K"] = 2
    raw_design["reference"]["B"] = 128.0
    des_file.write_text(yaml.safe_dump(raw_design), encoding="utf-8")

    backend = DummyTestBackend()
    test_reader = make_test_reader(reader_id="gemma", backend=backend)

    # Inject test reader into CLI panel and HashEmbedder into SentenceTransformerEmbedder
    monkeypatch.setattr("msat.cli._panel", lambda bundle: [test_reader])
    monkeypatch.setattr(
        "msat.memory.embed.SentenceTransformerEmbedder", lambda *a, **k: HashEmbedder()
    )

    # Build coherent questions and input files
    questions = make_coherent_questions(n_questions=6)
    q_file = tmp_path / "questions.json"
    gold_file = tmp_path / "gold.json"
    pool_file = tmp_path / "pool.json"
    q_file.write_text(
        json.dumps(
            [
                {
                    "question_id": q.qid,
                    "question": q.question,
                    "question_date": q.question_date,
                    "question_type": "single-session-user",
                    "answer": list(q.gold_answers),
                    "answer_session_ids": list(q.gold_session_ids),
                    "haystack_session_ids": [session.session_id for session in q.sessions],
                    "haystack_dates": [session.date for session in q.sessions],
                    "haystack_sessions": [
                        [
                            {
                                "role": turn.role,
                                "content": turn.content,
                                "has_answer": turn.has_answer,
                            }
                            for turn in session.turns
                        ]
                        for session in q.sessions
                    ],
                }
                for q in questions
            ]
        ),
        encoding="utf-8",
    )

    # Write gold answers
    gold_data = {q.qid: q.gold_answers[0] for q in questions if q.gold_answers}
    gold_file.write_text(json.dumps(gold_data), encoding="utf-8")

    # Write control pool
    pool_data = {
        q.qid: {
            "question": q.question,
            "memories": [
                {
                    "memory_id": f"{q.qid}_m0",
                    "session_id": f"{q.qid}_s0",
                    "text": "Control pooled text",
                    "n_tokens": 10,
                }
            ],
        }
        for q in questions
    }
    pool_file.write_text(json.dumps(pool_data), encoding="utf-8")

    # Generate manifest_fast.json matching cfg_dir's fast bundle
    canonical = load_bundle(cfg_dir)
    fast_bundle = resolve_effective_experiment(canonical, "fast").bundle
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = make_manifest_file(
        results_dir / "manifest_fast.json",
        fast_bundle,
        qids=[q.qid for q in questions],
        k_grid=fast_bundle.design.K_grid,
        b_values=fast_bundle.design.b_values,
        reader_id="gemma",
    )

    runner = CliRunner()
    common = ["--profile", "fast", "--config", str(cfg_dir), "--results", str(results_dir)]
    run_inputs = [
        "--questions",
        str(q_file),
        "--gold",
        str(gold_file),
        "--arm-e-pool",
        str(pool_file),
        "--plan",
        str(manifest_path),
    ]

    def run_stage(rung: str, *, resume: bool = False):
        extra = ["--resume"] if resume else []
        result = runner.invoke(app, ["run", run_id, "--rung", rung, *extra, *common, *run_inputs])
        assert result.exit_code == 0, result.output
        assert rung in read_ladder(run_dir / "ladder.json").passed
        return result

    run_stage("negative")
    calls_before_resume = len(backend.calls)
    warm = run_stage("negative", resume=True)
    assert "generated   0" in warm.output
    assert len(backend.calls) == calls_before_resume
    run_stage("baselines", resume=True)
    run_stage("crossed", resume=True)
    calls_after_crossed = len(backend.calls)
    warm_crossed = run_stage("crossed", resume=True)
    assert "generated   0" in warm_crossed.output
    assert len(backend.calls) == calls_after_crossed

    observed_arms = {row.arm for row in read_observation_metadata(run_dir / "observations.parquet")}
    assert {"E", "D", "F", "A+G"} <= observed_arms
    score_result = runner.invoke(
        app, ["score", run_id, *common, "--metric", "em", "--gold", str(gold_file)]
    )
    assert score_result.exit_code == 0, score_result.output
    assert (run_dir / "scores.parquet").exists()

    analysis_args = ["analyze", run_id, *common, "--outcome", "em"]
    analysis = runner.invoke(app, analysis_args)
    assert analysis.exit_code == 0, analysis.output
    assert (run_dir / "analysis.json").exists()
    first_json = (run_dir / "analysis.json").read_bytes()
    first_md = (run_dir / "results.md").read_bytes()
    repeated = runner.invoke(app, analysis_args)
    assert repeated.exit_code == 0, repeated.output
    assert (run_dir / "analysis.json").read_bytes() == first_json
    assert (run_dir / "results.md").read_bytes() == first_md
    assert len(backend.calls) == calls_after_crossed, "Stored scoring and analysis cannot generate."


# =============================================================================
# 3. Refusal and Incompatibility Guard Tests
# =============================================================================


def test_incompatible_manifest_uncrossed_refuses_before_generation(tmp_path: Path) -> None:
    """A manifest missing declared grid cells is refused before generation without dirtying store."""
    run_dir = tmp_path / "run_uncrossed"
    bundle = make_minimal_bundle(k_grid=(1, 2), b_values=(128.0, 256.0))
    backend = DummyTestBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    questions = make_coherent_questions(n_questions=4)
    inputs = RunInputs.from_synthetic(questions)

    # Manifest intentionally missing cell (K=2, B=256.0)
    uncrossed_manifest = make_manifest_file(
        tmp_path / "uncrossed_manifest.json",
        bundle,
        qids=[q.qid for q in questions],
        k_grid=(1,),  # Missing K=2
        b_values=(128.0,),
    )

    with pytest.raises(GateFailedError, match="The declared grid is not a crossed grid"):
        run_lifecycle_workflow(
            bundle,
            run_dir,
            rung="negative",
            inputs=inputs,
            panel=[reader],
            plan_path=uncrossed_manifest,
            embedder=HashEmbedder(),
        )

    assert len(backend.calls) == 0, "No generation allowed when crossed grid check fails"
    assert (run_dir / "crossed_grid.json").exists()
    grid_info = json.loads((run_dir / "crossed_grid.json").read_text(encoding="utf-8"))
    assert not grid_info["crossed"]


def test_capacity_overflow_refuses_before_dispatch(tmp_path: Path) -> None:
    """Reader context window smaller than required prompt tokens refuses before dispatch."""
    tiny_reader = make_test_reader(reader_id="tiny-reader", declared_context_window=15)
    verdict = check_capacity(tiny_reader, prompt_tokens=500, label="ArmA/test")

    assert not verdict.supported
    assert verdict.prompt_tokens == 500
    assert verdict.reader_capacity_tokens == 15
    assert "cannot serve" in verdict.reason


def test_missing_sessions_refuses_placeholder_memories(tmp_path: Path) -> None:
    """Questions with empty sessions raise MissingInputError; placeholder memories are refused."""
    run_dir = tmp_path / "run_missing_sessions"
    bundle = make_minimal_bundle()
    registry, _ = build_registry(bundle)
    backend = DummyTestBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)

    empty_q = Question(
        qid="q_empty",
        scale="S",
        question="What is missing?",
        question_date="2026-01-01",
        gold_answers=("Nothing",),
        gold_session_ids=("sess_empty",),
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
        panel=[reader],
        inputs=inputs,
        embedder=HashEmbedder(),
    )
    unit = ScheduledUnit(
        Condition("A", 1, 128.0, "per_memory_tokens", "S", "test-reader"),
        "q_empty",
        0,
    )
    with pytest.raises(MissingInputError, match="Refusing to manufacture placeholder sessions"):
        lifecycle._render_unit(unit)


def test_ladder_order_violation_refuses_advance(tmp_path: Path) -> None:
    """Attempting to run baselines or full before negative passes is strictly refused."""
    run_dir = tmp_path / "run_ladder_violation"
    bundle = make_minimal_bundle()
    backend = DummyTestBackend()
    reader = make_test_reader(reader_id="test-reader", backend=backend)
    questions = make_coherent_questions(n_questions=4)
    inputs = RunInputs.from_synthetic(questions)

    with pytest.raises(GateFailedError, match="cannot enter rung"):
        run_lifecycle_workflow(
            bundle,
            run_dir,
            rung="baselines",  # Skips negative!
            inputs=inputs,
            panel=[reader],
            embedder=HashEmbedder(),
        )
    assert len(backend.calls) == 0


def test_analysis_workflow_refuses_non_binary_f1(tmp_path: Path) -> None:
    """Analysis workflow rejects continuous F1 outcome without explicit binarization."""
    run_dir = tmp_path / "run_f1_outcome"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write dummy scores.parquet containing F1
    import pyarrow as pa
    import pyarrow.parquet as pq

    scores_data = [
        {
            "schema_version": 1,
            "cache_key": "ck1",
            "condition_key": "A|1|128.0|per_memory_tokens|S|test-reader",
            "qid": "q1",
            "arm": "A",
            "K": 1,
            "B": 128.0,
            "B_mode": "per_memory_tokens",
            "scale": "S",
            "reader": "test-reader",
            "repetition": 0,
            "cached": False,
            "metric": "f1",
            "value": 0.85,
            "is_abstention": False,
            "answerable": True,
            "finish_reason": "stop",
            "latency_ms": 10.0,
            "ttft_ms": 5.0,
            "queue_wait_ms": 0.1,
            "config_hash": "cfg_test",
        }
    ]
    pq.write_table(pa.Table.from_pylist(scores_data), run_dir / "scores.parquet")

    with pytest.raises(NonBinaryOutcomeError, match="F1 is a continuous token overlap score"):
        run_analysis_workflow(run_dir, selected_outcome="f1", strict=False)
