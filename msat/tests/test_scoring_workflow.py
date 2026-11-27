"""Tests for stored scoring workflow (Recommendation 04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from msat.metrics.normalisation import Normaliser
from msat.metrics.score import condition_key_str, score_store
from msat.metrics.scoring_workflow import (
    AmbiguousOfficialJoinError,
    InvalidScoringConfigurationError,
    load_gold,
    load_normaliser,
    run_scoring_workflow,
    validate_mode_and_metrics,
    validate_official_join,
)
from msat.store import Condition, ResponseStore, StoredResponse, cache_key

COND_A = Condition(arm="A", k=4, b=512.0, b_mode="budget", scale="S", reader="gemma")
COND_B = Condition(arm="A", k=8, b=512.0, b_mode="budget", scale="S", reader="gemma")


def _put_response(
    store: ResponseStore,
    qid: str,
    text: str,
    cond: Condition = COND_A,
    reps: int = 1,
    latency_ms: float = 25.0,
    ttft_ms: float | None = 10.0,
    queue_wait_ms: float | None = 2.0,
) -> StoredResponse:
    req = {
        "model": "google/gemma-4-E4B-it",
        "messages": [{"role": "user", "content": f"Question {qid}"}],
        "temperature": 0.0,
    }
    rec = StoredResponse(
        key=cache_key(cond, req),
        condition=cond.as_dict(),
        request=req,
        text=text,
        finish_reason="stop",
        model_version="v1",
        prompt_tokens=120,
        completion_tokens=6,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        queue_wait_ms=queue_wait_ms,
        qid=qid,
        repetitions=tuple(range(reps)),
    )
    store.put(rec)
    return rec


def test_load_normaliser_custom_dir_and_path(tmp_path: Path) -> None:
    # 1. Spec dict directly
    spec = {
        "normalisation": {
            "steps": [{"id": "lower", "op": "lowercase"}],
            "apply_to": ["prediction", "gold"],
        }
    }
    norm = load_normaliser(spec=spec)
    assert norm.normalise("HELLO") == "hello"

    # 2. Custom path
    custom_path = tmp_path / "custom_norm.yaml"
    custom_path.write_text(yaml.dump(spec), encoding="utf-8")
    norm_from_path = load_normaliser(config_path=custom_path)
    assert norm_from_path.normalise("WORLD") == "world"

    # 3. Custom dir with prompts/normalisation.yaml
    custom_dir = tmp_path / "my_configs"
    prompts_dir = custom_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "normalisation.yaml").write_text(yaml.dump(spec), encoding="utf-8")
    norm_from_dir = load_normaliser(config_dir=custom_dir)
    assert norm_from_dir.normalise("TEST") == "test"

    # 4. Missing file fails with FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Normalisation configuration not found"):
        load_normaliser(config_dir=tmp_path / "nonexistent_dir")


def test_validate_mode_and_metrics() -> None:
    # Valid modes and defaults
    mode, metrics = validate_mode_and_metrics(mode="fast", metric="all")
    assert mode == "fast"
    assert metrics == ("em", "f1")
    assert validate_mode_and_metrics(mode="full", metric="all") == ("full", ("em", "f1"))

    # With official present
    mode, metrics = validate_mode_and_metrics(mode="cached", metric="all", has_official=True)
    assert mode == "cached"
    assert metrics == ("em", "f1", "official")

    # Explicit single metric
    _, m_em = validate_mode_and_metrics(metric="em")
    assert m_em == ("em",)
    _, m_f1 = validate_mode_and_metrics(metric="f1")
    assert m_f1 == ("f1",)
    assert validate_mode_and_metrics(metric="judge")[1] == ("judge",)

    # Explicit list of metrics
    _, m_seq = validate_mode_and_metrics(metric=["EM", "f1"])
    assert m_seq == ("em", "f1")

    # Invalid mode
    with pytest.raises(InvalidScoringConfigurationError, match="unknown mode 'bogus'"):
        validate_mode_and_metrics(mode="bogus")

    # Unknown metric
    with pytest.raises(InvalidScoringConfigurationError, match="unknown metric"):
        validate_mode_and_metrics(metric="bleu")


def test_cached_judge_metric_preserves_judges_and_repetitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = ResponseStore(run_dir / "responses")
    record = _put_response(store, "q1", "Paris", reps=2)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "normalisation.yaml").write_text(
        yaml.safe_dump({"normalisation": {"steps": [{"id": "lower", "op": "lowercase"}]}}),
        encoding="utf-8",
    )
    verdicts = run_dir / "verdicts.json"
    entries = [
        {
            "condition_key": condition_key_str(record),
            "qid": "q1",
            "judge_id": judge,
            "verdict": verdict,
        }
        for judge, verdict in (("j1", "correct"), ("j2", "incorrect"))
    ]
    verdicts.write_text(json.dumps(entries), encoding="utf-8")
    result = run_scoring_workflow(
        run_dir, gold={"q1": "Paris"}, config_dir=config_dir, metric="judge"
    )
    assert {(row.metric, row.repetition, row.value) for row in result.rows} == {
        ("judge:j1", 0, 1.0),
        ("judge:j1", 1, 1.0),
        ("judge:j2", 0, 0.0),
        ("judge:j2", 1, 0.0),
    }
    assert result.report.metrics_written == ("judge:j1", "judge:j2")
    entries.append({**entries[0], "verdict": "incorrect"})
    verdicts.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(InvalidScoringConfigurationError, match="Conflicting cached verdicts"):
        run_scoring_workflow(run_dir, gold={"q1": "Paris"}, config_dir=config_dir, metric="judge")
    verdicts.unlink()
    with pytest.raises(InvalidScoringConfigurationError, match="requires cached verdicts"):
        run_scoring_workflow(run_dir, gold={"q1": "Paris"}, config_dir=config_dir, metric="judge")


def test_load_gold_oracle_list_and_qid_map(tmp_path: Path) -> None:
    # 1. Plain qid map
    gold_map = {"q1": "Paris", "q2": ["London", "City of London"]}
    loaded = load_gold(gold_map)
    assert loaded["q1"] == "Paris"
    assert loaded["q2"] == ("London", "City of London")

    # 2. JSON file with qid map
    map_file = tmp_path / "gold_map.json"
    map_file.write_text(json.dumps(gold_map), encoding="utf-8")
    loaded_file = load_gold(map_file)
    assert loaded_file["q1"] == "Paris"
    assert loaded_file["q2"] == ("London", "City of London")

    # 3. Oracle list
    oracle = [
        {"question_id": "q10", "answer": "Berlin"},
        {"qid": "q20", "gold_answers": ["Tokyo", "Edo"]},
        {"question_id": "q30", "gold": "Rome"},
    ]
    loaded_oracle = load_gold(oracle)
    assert loaded_oracle["q10"] == "Berlin"
    assert loaded_oracle["q20"] == ("Tokyo", "Edo")
    assert loaded_oracle["q30"] == "Rome"

    # 4. JSON file with oracle list
    oracle_file = tmp_path / "oracle.json"
    oracle_file.write_text(json.dumps(oracle), encoding="utf-8")
    loaded_oracle_file = load_gold(oracle_file)
    assert loaded_oracle_file["q10"] == "Berlin"
    assert loaded_oracle_file["q20"] == ("Tokyo", "Edo")

    # 5. Empty fails
    with pytest.raises(ValueError, match="Gold answer collection is empty"):
        load_gold({})

    # 6. Nonexistent file fails
    with pytest.raises(FileNotFoundError):
        load_gold(tmp_path / "missing.json")


def test_score_store_multi_gold_answers(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "responses")
    _put_response(store, "q1", "Paris")
    _put_response(store, "q2", "City of London")
    _put_response(store, "q3", "Berlin")

    spec = {
        "normalisation": {
            "steps": [{"id": "lower", "op": "lowercase"}],
            "apply_to": ["prediction", "gold"],
        }
    }
    norm = Normaliser.from_config(spec)

    gold = {
        "q1": "Paris",
        "q2": ("London", "City of London"),  # exact match on second option
        "q3": "Tokyo",
    }

    rows, _report = score_store(store, gold, norm)
    by_key = {(r.qid, r.metric): r.value for r in rows}

    assert by_key[("q1", "em")] == 1.0
    assert by_key[("q2", "em")] == 1.0  # matched second variant
    assert by_key[("q3", "em")] == 0.0


def test_official_cached_result_condition_identity(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "responses")
    # Response under COND_A
    _put_response(store, "q1", "answer A", cond=COND_A)

    # 1. Unambiguous join for single condition
    official_single = {"q1": 1.0}
    validated = validate_official_join(official_single, store)
    assert validated == official_single

    # 2. Add second condition with same qid
    _put_response(store, "q1", "answer B", cond=COND_B)

    # Bare qid without condition key must raise AmbiguousOfficialJoinError
    with pytest.raises(AmbiguousOfficialJoinError, match="Refusing ambiguous broadcast"):
        validate_official_join({"q1": 1.0}, store)

    # Condition-aware key succeeds
    cond_a_str = "|".join(str(x) for x in COND_A.key)
    cond_b_str = "|".join(str(x) for x in COND_B.key)
    official_aware = {
        (cond_a_str, "q1"): 1.0,
        (cond_b_str, "q1"): 0.0,
    }
    validated_aware = validate_official_join(official_aware, store)
    assert validated_aware is not None

    spec = {
        "normalisation": {
            "steps": [{"id": "lower", "op": "lowercase"}],
            "apply_to": ["prediction", "gold"],
        }
    }
    norm = Normaliser.from_config(spec)
    rows, _report = score_store(store, {"q1": "gold"}, norm, official=official_aware)

    official_rows = [r for r in rows if r.metric == "official"]
    assert len(official_rows) == 2
    row_a = next(r for r in official_rows if r.condition_key == cond_a_str)
    row_b = next(r for r in official_rows if r.condition_key == cond_b_str)
    assert row_a.value == 1.0
    assert row_b.value == 0.0


def test_run_scoring_workflow_full_pipeline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    responses_dir = run_dir / "responses"
    store = ResponseStore(responses_dir)

    r1 = _put_response(store, "q1", "Paris", COND_A)
    r2 = _put_response(store, "q2", "the city is London", COND_A)
    _r3 = _put_response(store, "q3_abs", "I don't know", COND_A)  # abstention

    # Gold
    gold_path = run_dir / "gold.json"
    gold_path.write_text(json.dumps({"q1": "Paris", "q2": "London"}), encoding="utf-8")

    # Official
    official_path = run_dir / "official.json"
    official_path.write_text(json.dumps({"q1": 1.0, "q2": 0.5}), encoding="utf-8")

    # Verdicts with 2 judges
    cond_str = "|".join(str(x) for x in COND_A.key)
    verdicts = [
        {
            "pass_index": 1,
            "condition_key": cond_str,
            "qid": "q1",
            "judge_id": "judge1",
            "verdict": "correct",
            "cache_key": r1.key,
        },
        {
            "pass_index": 1,
            "condition_key": cond_str,
            "qid": "q1",
            "judge_id": "judge2",
            "verdict": "correct",
            "cache_key": r1.key,
        },
        {
            "pass_index": 1,
            "condition_key": cond_str,
            "qid": "q2",
            "judge_id": "judge1",
            "verdict": "correct",
            "cache_key": r2.key,
        },
        {
            "pass_index": 1,
            "condition_key": cond_str,
            "qid": "q2",
            "judge_id": "judge2",
            "verdict": "incorrect",
            "cache_key": r2.key,
        },
    ]
    (run_dir / "verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")

    result = run_scoring_workflow(
        run_dir,
        gold=gold_path,
        mode="cached",
        metric="all",
        official=official_path,
    )

    assert result.scores_path.is_file()
    assert result.latency_path.is_file()
    assert result.disagreements_path is not None
    assert result.disagreements_path.is_file()
    assert result.disagreements_count == 1  # q2 has disagreement between judge1 and judge2
    assert result.verdicts_missing is False

    # Check abstention row value is None
    abs_row = next(r for r in result.rows if r.qid == "q3_abs" and r.metric == "em")
    assert abs_row.value is None
    assert abs_row.is_abstention is True

    # Deterministic offline rescoring check
    scores_before = result.scores_path.read_bytes()
    latency_before = result.latency_path.read_text(encoding="utf-8")

    result2 = run_scoring_workflow(
        run_dir,
        gold=gold_path,
        mode="cached",
        metric="all",
        official=official_path,
    )
    assert result2.scores_path.read_bytes() == scores_before
    assert result2.latency_path.read_text(encoding="utf-8") == latency_before


def test_run_scoring_workflow_missing_verdicts_visibly_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_no_verdicts"
    store = ResponseStore(run_dir / "responses")
    _put_response(store, "q1", "Paris", COND_A)

    result = run_scoring_workflow(
        run_dir,
        gold={"q1": "Paris"},
        mode="fast",
        metric="em",
    )

    assert result.verdicts_missing is True
    assert result.disagreements_path is None
    assert result.disagreements_count == 0
    assert any("no verdicts.json" in n for n in result.notes)


def test_exact_metric_subset_honored(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_subset"
    store = ResponseStore(run_dir / "responses")
    _put_response(store, "q1", "Paris", COND_A)

    # 1. metric="em" -> only EM written
    res_em = run_scoring_workflow(
        run_dir,
        gold={"q1": "Paris"},
        mode="fast",
        metric="em",
    )
    metrics_em = {r.metric for r in res_em.rows}
    assert metrics_em == {"em"}
    assert res_em.report.metrics_written == ("em",)

    # 2. metric="f1" -> only F1 written
    res_f1 = run_scoring_workflow(
        run_dir,
        gold={"q1": "Paris"},
        mode="fast",
        metric="f1",
    )
    metrics_f1 = {r.metric for r in res_f1.rows}
    assert metrics_f1 == {"f1"}
    assert res_f1.report.metrics_written == ("f1",)


def test_cached_repetitions_remain_distinct_observations(tmp_path: Path) -> None:
    run_dir = tmp_path / "repeated"
    store = ResponseStore(run_dir / "responses")
    _put_response(store, "q1", "Paris", reps=3)
    result = run_scoring_workflow(run_dir, gold={"q1": "Paris"}, mode="fast", metric="em")
    assert {row.repetition for row in result.rows} == {0, 1, 2}
    assert result.report.n_attributable == 3


def test_fast_default_does_not_include_cached_official_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "fast-official"
    store = ResponseStore(run_dir / "responses")
    _put_response(store, "q1", "Paris")
    (run_dir / "official.json").write_text(json.dumps({"q1": 1.0}), encoding="utf-8")
    result = run_scoring_workflow(run_dir, gold={"q1": "Paris"}, mode="fast")
    assert {row.metric for row in result.rows} == {"em", "f1"}


def test_missing_explicit_official_scores_fail_before_artifact_write(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-official"
    _put_response(ResponseStore(run_dir / "responses"), "q1", "Paris")
    with pytest.raises(FileNotFoundError, match="Official score file"):
        run_scoring_workflow(run_dir, gold={"q1": "Paris"}, official=tmp_path / "absent.json")
    assert not (run_dir / "scores.parquet").exists()


def test_missing_gold_answers_remain_missing_and_duplicates_fail() -> None:
    assert load_gold({"q1": None, "q2": "Paris"}) == {"q2": "Paris"}
    with pytest.raises(ValueError, match="Duplicate gold question"):
        load_gold([{"qid": "q1", "answer": "Paris"}, {"qid": "q1", "answer": "London"}])
