"""An explicitly enabled pilot baseline remains distinguishable from full context."""

from dataclasses import replace
from pathlib import Path

import pytest

from msat.arms.registry import build_registry
from msat.datasets.schema import CorpusStats, Session, Turn
from msat.runner.input import RunInputs
from msat.runner.lifecycle import CapacityRefusalError, RunLifecycle
from msat.runner.metadata import read_observation_metadata
from msat.runner.schedule import ScheduledUnit
from msat.store import Condition
from tests.test_lifecycle import (
    DummyTestBackend,
    make_synthetic_questions,
    make_test_bundle,
    make_test_reader,
)


def lifecycle(tmp_path: Path, *, mode: str = "fast", ceiling: int = 2000) -> RunLifecycle:
    bundle = make_test_bundle()
    bundle.mode = mode
    bundle.arms.arms = [
        *bundle.arms.arms,
        {
            "id": "D-max",
            "isolates": "maximum feasible context",
            "enabled": True,
            "prompt_template": "prompts/arm_d.j2",
        },
    ]
    questions = []
    for question in make_synthetic_questions():
        extra_sessions = tuple(
            Session(
                session_id=f"{question.qid}_extra_{i}",
                date="2026-01-01",
                turns=(Turn(role="user", content="irrelevant " * 100),),
            )
            for i in range(9)
        )
        sessions = (*extra_sessions, question.sessions[0])
        questions.append(
            replace(
                question,
                sessions=sessions,
                corpus=CorpusStats(
                    n_sessions=len(sessions),
                    n_turns=len(sessions),
                    n_chars=sum(s.n_chars for s in sessions),
                    n_gold_sessions=1,
                ),
            )
        )
    registry, _ = build_registry(bundle)
    return RunLifecycle(
        run_dir=tmp_path,
        bundle=bundle,
        registry=registry,
        panel=[make_test_reader(declared_context_window=ceiling, backend=DummyTestBackend())],
        inputs=RunInputs.from_synthetic(questions),
    )


def test_explicit_pilot_max_baseline_uses_complete_sessions_and_correct_evidence(
    tmp_path: Path,
) -> None:
    run = lifecycle(tmp_path)
    rendered = run._render_unit(
        ScheduledUnit(
            Condition("D-max", 1, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
        )
    )
    assert rendered.arm == "D-max"
    assert 0 < rendered.corpus_fraction < 1
    assert rendered.evidence_survival == 1.0
    assert run.run_rung("negative").passed
    report = run.run_rung("baselines")
    assert report.passed
    assert report.details["arms"] == ["A", "D-max", "F"]
    observations = read_observation_metadata(tmp_path / "observations.parquet")
    assert "D" not in {observation.arm for observation in observations}
    assert "D-max" in {observation.arm for observation in observations}
    assert any("not full-context D" in note for note in report.gate_verdict.notes)


def test_full_profile_keeps_the_full_context_refusal(tmp_path: Path) -> None:
    run = lifecycle(tmp_path, mode="full")
    assert run.run_rung("negative").passed
    with pytest.raises(CapacityRefusalError, match="arm D would need"):
        run.run_rung("baselines")
    assert "baselines" not in run.get_ladder_state().passed


def test_max_context_refuses_when_no_complete_session_fits(tmp_path: Path) -> None:
    run = lifecycle(tmp_path, ceiling=20)
    with pytest.raises(CapacityRefusalError, match="cannot fit any complete session"):
        run._render_unit(
            ScheduledUnit(
                Condition("D-max", 1, 256.0, "per_memory_tokens", "S", "test-reader"), "q1", 0
            )
        )
