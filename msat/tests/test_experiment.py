"""Tests for Recommendation 01: Effective Experiment Design & Planning.

Verifies:
- Hash distinctness (fast mode gets distinct reproducible hash, canonical hash preserved)
- Legacy manifest & plan compatibility
- Dynamic primary reader selection (no hardcoded gemma)
- Full tier agreement (design, readers, raw, panel)
- Canonical reader validation remains >= 2 while pilot selects single primary
- Invalid mode failure
- Manifest validation before execution
- QID/seed/repeat metadata preservation & validation
- Crossed grid verification rejects duplicate, extra, and unexpected cells
- Condition consolidation with store
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import msat.manifest
import msat.store
from msat.config import ConfigBundle, load_bundle
from msat.experiment import (
    ExecutionProfile,
    IncompatibleManifestError,
    InvalidProfileError,
    resolve_effective_experiment,
    validate_manifest_for_execution,
)
from msat.manifest import Condition, build_manifest, write_manifest
from msat.planning import (
    ExecutablePlan,
    PlanCondition,
    plan_from_manifest,
    verify_crossed_grid,
)
from msat.providers.local import readers_from_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_effective_fast_resolution_is_idempotent(bundle: ConfigBundle) -> None:
    first = resolve_effective_experiment(bundle, "fast")
    second = resolve_effective_experiment(first.bundle, "fast")
    assert second.summary() == first.summary()


def test_runtime_manifest_rejects_seed_question_and_grid_changes(bundle: ConfigBundle) -> None:
    from msat.manifest import stats_from_corpus

    experiment = resolve_effective_experiment(bundle, "fast")
    manifest = build_manifest(
        experiment,
        {"S": [stats_from_corpus("q1", 100, 400_000, chars_per_token=4.0)]},
        chars_per_token=4.0,
        raw_counts={"S": [(100, 400_000)]},
        metadata={"qids_by_scale": {"S": ["q1"]}, "seed": bundle.design.seed},
    ).as_dict()
    validate_manifest_for_execution(manifest, experiment, qids_by_scale={"S": ["q1"]})
    with pytest.raises(IncompatibleManifestError, match="question IDs"):
        validate_manifest_for_execution(manifest, experiment, qids_by_scale={"S": ["q2"]})
    with pytest.raises(IncompatibleManifestError, match="seed"):
        validate_manifest_for_execution(
            manifest, experiment, seed=bundle.design.seed + 1, qids_by_scale={"S": ["q1"]}
        )
    incomplete = {**manifest, "cells": manifest["cells"][1:]}
    with pytest.raises(IncompatibleManifestError, match="missing"):
        validate_manifest_for_execution(incomplete, experiment, qids_by_scale={"S": ["q1"]})
    forged = {**manifest, "config_hash": experiment.canonical_config_hash}
    with pytest.raises(IncompatibleManifestError, match="config_hash"):
        validate_manifest_for_execution(forged, experiment, qids_by_scale={"S": ["q1"]})


@pytest.fixture
def bundle() -> ConfigBundle:
    return load_bundle(CONFIGS)


def test_condition_is_consolidated_from_store() -> None:
    """Consolidate Condition in manifest by importing store's class."""
    assert msat.manifest.Condition is msat.store.Condition
    assert Condition is msat.store.Condition


def test_hash_distinctness_in_fast_mode(bundle: ConfigBundle) -> None:
    """Fast mode produces a distinct reproducible hash while preserving canonical identity."""
    exp_fast = resolve_effective_experiment(bundle, mode="fast")
    assert exp_fast.mode == "fast"
    assert exp_fast.is_fast is True

    # Hashes are non-empty hex strings
    assert len(exp_fast.canonical_config_hash) == 64
    assert len(exp_fast.effective_config_hash) == 64

    # Distinct attribution in fast mode
    assert exp_fast.effective_config_hash != exp_fast.canonical_config_hash
    assert exp_fast.canonical_config_hash == bundle.config_hash

    # Fully deterministic & reproducible
    exp_fast_2 = resolve_effective_experiment(bundle, mode="fast")
    assert exp_fast_2.effective_config_hash == exp_fast.effective_config_hash

    # Full mode has matching canonical and effective hashes
    exp_full = resolve_effective_experiment(bundle, mode="full")
    assert exp_full.effective_config_hash == exp_full.canonical_config_hash
    assert exp_full.effective_config_hash == bundle.config_hash


def test_primary_reader_dynamically_selected_not_hardcoded(bundle: ConfigBundle) -> None:
    """Fast mode dynamically chooses the declared primary reader, not hardcoded 'gemma'."""
    raw = dict(bundle.raw)
    readers_raw = dict(raw["readers.yaml"])

    # Simulate renaming primary reader to 'custom_model'
    custom_families = [
        {**readers_raw["families"][0], "id": "custom_model", "role": "primary"},
        {**readers_raw["families"][1], "id": "secondary_1", "role": "secondary"},
    ]
    readers_raw["families"] = custom_families
    raw["readers.yaml"] = readers_raw

    # Build bundle with renamed primary
    custom_bundle = bundle.model_copy(
        update={
            "raw": raw,
            "readers": bundle.readers.model_copy(
                update={
                    "families": [
                        bundle.readers.families[0].model_copy(
                            update={"id": "custom_model", "role": "primary"}
                        ),
                        bundle.readers.families[1].model_copy(
                            update={"id": "secondary_1", "role": "secondary"}
                        ),
                    ]
                }
            ),
        }
    )

    exp = resolve_effective_experiment(custom_bundle, mode="fast")
    assert exp.primary_reader == "custom_model"
    assert exp.primary_reader != "gemma"
    assert list(exp.bundle.design.readers) == ["custom_model"]
    assert [r.id for r in exp.panel] == ["custom_model"]


def test_effective_design_raw_and_panel_all_agree(bundle: ConfigBundle) -> None:
    """In fast mode, filtered effective design, raw configs, and selected panel agree."""
    exp = resolve_effective_experiment(bundle, mode="fast")
    primary = exp.primary_reader

    # 1. Effective design
    assert list(exp.bundle.design.scales) == ["S"]
    assert list(exp.bundle.design.readers) == [primary]
    assert exp.bundle.design.runs_per_cell == 1

    # 2. Effective readers config
    assert [f.id for f in exp.bundle.readers.families] == [primary]

    # 3. Effective raw config dicts
    assert exp.bundle.raw["design.yaml"]["scales"] == ["S"]
    assert exp.bundle.raw["design.yaml"]["readers"] == [primary]
    assert exp.bundle.raw["design.yaml"]["runs_per_cell"] == 1
    assert [f["id"] for f in exp.bundle.raw["readers.yaml"]["families"]] == [primary]

    # 4. Reader panel
    assert len(exp.panel) == 1
    assert exp.panel[0].id == primary


def test_canonical_reader_validation_remains_ge_2(bundle: ConfigBundle) -> None:
    """Canonical reader validation requires >= 2 while pilot selects one explicitly."""
    single_raw = {
        "families": [bundle.raw["readers.yaml"]["families"][0]],
    }
    # Canonical validation rejects single family
    with pytest.raises(ValueError, match="at least 2"):
        readers_from_config(single_raw, allow_single=False)

    # Pilot path explicitly allows single family
    panel = readers_from_config(single_raw, allow_single=True)
    assert len(panel) == 1


def test_invalid_mode_values_fail(bundle: ConfigBundle) -> None:
    """Invalid execution modes fail loudly."""
    with pytest.raises(InvalidProfileError):
        resolve_effective_experiment(bundle, mode="invalid")
    with pytest.raises(InvalidProfileError):
        resolve_effective_experiment(bundle, mode="ultra_fast")
    with pytest.raises(InvalidProfileError):
        ExecutionProfile.from_str("medium")


def test_manifest_validation_before_execution(bundle: ConfigBundle, tmp_path: Path) -> None:
    """Manifest must match experiment mode, hash, and question sets before execution."""
    exp_fast = resolve_effective_experiment(bundle, mode="fast")
    exp_full = resolve_effective_experiment(bundle, mode="full")

    # Build manifest for fast mode
    man_fast = build_manifest(exp_fast, {}, chars_per_token=4.0)
    p_fast = write_manifest(man_fast, tmp_path / "manifest_fast.json")

    # Valid fast execution
    res = validate_manifest_for_execution(
        p_fast,
        exp_fast,
        qids_by_scale={"S": ["q1", "q2"]},
        strict_questions=False,
    )
    assert res["valid"] is True
    assert res["mode"] == "fast"

    # Mismatched mode: fast manifest with full experiment
    with pytest.raises(IncompatibleManifestError, match="mode"):
        validate_manifest_for_execution(p_fast, exp_full, strict_questions=False)

    # Missing questions in strict mode
    with pytest.raises(IncompatibleManifestError, match="question"):
        validate_manifest_for_execution(
            p_fast,
            exp_fast,
            qids_by_scale={"S": []},
            strict_questions=True,
        )


def test_legacy_manifest_and_plan_compatibility(tmp_path: Path) -> None:
    """Legacy manifest without new metadata fields loads seamlessly and defaults cleanly."""
    legacy_manifest_data = {
        "schema_version": 1,
        "config_hash": "legacy_deadbeef" * 4,
        "grid": {
            "K": [1, 2],
            "B": [{"value": 128, "mode": "per_memory_tokens"}],
            "scales": ["S"],
            "readers": ["gemma"],
            "runs_per_cell": 1,
        },
        "cells": [
            {
                "arm": "A+G",
                "K": 1,
                "B": 128.0,
                "B_mode": "per_memory_tokens",
                "scale": "S",
                "reader": "gemma",
                "n_questions": 1,
                "n_degenerate": 0,
                "degeneracy_fraction": 0.0,
                "status": "accepted",
                "reason": "ok",
                "example_reason": "",
            }
        ],
        "corpus": {},
        "guards": {},
        "notes": [],
    }

    manifest_file = tmp_path / "legacy_manifest.json"
    manifest_file.write_text(json.dumps(legacy_manifest_data), encoding="utf-8")

    # plan_from_manifest handles legacy format
    plan = plan_from_manifest(
        manifest_file,
        {"S": ["q_001"]},
        seed=12345,
        runs_per_cell=1,
    )

    assert plan.mode == "full"
    assert plan.canonical_config_hash == legacy_manifest_data["config_hash"]
    assert plan.effective_config_hash == legacy_manifest_data["config_hash"]
    assert len(plan.conditions) == 1
    assert plan.conditions[0].condition.k == 1


def test_strict_planning_validation(tmp_path: Path) -> None:
    """Strict runtime planning rejects missing questions, but non-strict preserves dry-runs."""
    manifest_data = {
        "schema_version": 1,
        "config_hash": "abc" * 21 + "a",
        "grid": {"K": [1], "B": [{"value": 128, "mode": "per_memory_tokens"}]},
        "cells": [
            {
                "arm": "A+G",
                "K": 1,
                "B": 128.0,
                "B_mode": "per_memory_tokens",
                "scale": "S",
                "reader": "gemma",
                "n_questions": 1,
                "n_degenerate": 0,
                "degeneracy_fraction": 0.0,
                "status": "accepted",
                "reason": "ok",
            }
        ],
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Non-strict dry run with no questions succeeds
    plan_dry = plan_from_manifest(manifest_file, {"S": []}, seed=42, runs_per_cell=1, strict=False)
    assert len(plan_dry.schedule) == 0

    # Strict runtime planning raises ValueError
    with pytest.raises(ValueError, match="Strict runtime planning requires questions"):
        plan_from_manifest(manifest_file, {"S": []}, seed=42, runs_per_cell=1, strict=True)


def test_crossed_verification_rejects_extra_duplicate_and_unexpected_cells() -> None:
    """verify_crossed_grid rejects unexpected scales/readers, duplicate cells, and extra cells."""
    valid_c1 = PlanCondition(Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma"), 1, 0.0)
    valid_c2 = PlanCondition(Condition("A+G", 2, 128.0, "per_memory_tokens", "S", "gemma"), 1, 0.0)

    # 1. Base valid plan
    plan_valid = ExecutablePlan(
        conditions=(valid_c1, valid_c2),
        rejected=(),
        schedule=(),
        config_hash="test",
        seed=42,
        repetition=1,
    )
    res_valid = verify_crossed_grid(
        plan_valid, k_grid=[1, 2], b_grid=[128.0], scales=["S"], readers=["gemma"]
    )
    assert res_valid["crossed"] is True
    assert len(res_valid["problems"]) == 0

    # 2. Duplicate cell
    plan_dup = ExecutablePlan(
        conditions=(valid_c1, valid_c2, valid_c1),
        rejected=(),
        schedule=(),
        config_hash="test",
        seed=42,
        repetition=1,
    )
    res_dup = verify_crossed_grid(
        plan_dup, k_grid=[1, 2], b_grid=[128.0], scales=["S"], readers=["gemma"]
    )
    assert res_dup["crossed"] is False
    assert any("duplicate" in p.lower() for p in res_dup["problems"])

    # 3. Unexpected scale
    cell_bad_scale = PlanCondition(
        Condition("A+G", 1, 128.0, "per_memory_tokens", "XL", "gemma"), 1, 0.0
    )
    plan_bad_scale = ExecutablePlan(
        conditions=(valid_c1, valid_c2, cell_bad_scale),
        rejected=(),
        schedule=(),
        config_hash="test",
        seed=42,
        repetition=1,
    )
    res_bad_scale = verify_crossed_grid(
        plan_bad_scale, k_grid=[1, 2], b_grid=[128.0], scales=["S"], readers=["gemma"]
    )
    assert res_bad_scale["crossed"] is False
    assert any("unexpected scale" in p.lower() for p in res_bad_scale["problems"])

    # 4. Unexpected reader
    cell_bad_reader = PlanCondition(
        Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "llama"), 1, 0.0
    )
    plan_bad_reader = ExecutablePlan(
        conditions=(valid_c1, valid_c2, cell_bad_reader),
        rejected=(),
        schedule=(),
        config_hash="test",
        seed=42,
        repetition=1,
    )
    res_bad_reader = verify_crossed_grid(
        plan_bad_reader, k_grid=[1, 2], b_grid=[128.0], scales=["S"], readers=["gemma"]
    )
    assert res_bad_reader["crossed"] is False
    assert any("unexpected reader" in p.lower() for p in res_bad_reader["problems"])

    # 5. Extra (K, B) cell
    cell_extra_kb = PlanCondition(
        Condition("A+G", 99, 128.0, "per_memory_tokens", "S", "gemma"), 1, 0.0
    )
    plan_extra_kb = ExecutablePlan(
        conditions=(valid_c1, valid_c2, cell_extra_kb),
        rejected=(),
        schedule=(),
        config_hash="test",
        seed=42,
        repetition=1,
    )
    res_extra_kb = verify_crossed_grid(
        plan_extra_kb, k_grid=[1, 2], b_grid=[128.0], scales=["S"], readers=["gemma"]
    )
    assert res_extra_kb["crossed"] is False
    assert any("extra" in p.lower() for p in res_extra_kb["problems"])


def test_canonical_vocabulary_observation_condition_repetition(tmp_path: Path) -> None:
    from msat.manifest import Condition
    from msat.metrics.normalisation import Normaliser
    from msat.metrics.score import Observation, read_observations, score_store, write_observations
    from msat.planning import ExecutablePlan, PlanCondition
    from msat.runner.metadata import ObservationMetadata
    from msat.runner.schedule import ScheduledUnit, build_schedule
    from msat.store import ResponseStore, StoredResponse, cache_key

    # 1. Condition identity
    cond = Condition("A", 2, 128.0, "per_memory_tokens", "S", "gemma")
    assert cond.key == ("A", 2, 128.0, "per_memory_tokens", "S", "gemma")

    # 2. ScheduledUnit repetition
    unit = ScheduledUnit(cond, "q1", repetition=1)
    assert unit.repetition == 1

    # 3. Schedule building with repetition
    sched = build_schedule(conditions=[cond], qids=["q1"], repetition=2, seed=42)
    assert len(sched) == 2
    assert all(u.repetition in (0, 1) for u in sched)

    # 4. ResponseStore with StoredResponse repetitions
    store = ResponseStore(tmp_path / "store")
    req = {"model": "gemma", "messages": [{"role": "user", "content": "hi"}]}
    k = cache_key(cond, req)
    record = StoredResponse(
        key=k,
        condition=cond.as_dict(),
        request=req,
        text="answer",
        finish_reason="stop",
        model_version="v1",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=10.0,
        qid="q1",
        repetitions=(0, 1),
    )
    store.put(record)
    loaded = store.get(k)
    assert loaded is not None
    assert loaded.repetitions == (0, 1)
    assert loaded.as_dict()["replicates"] == [0, 1]

    # 5. Observation and scoring seam
    norm = Normaliser(steps=())
    observations, _report = score_store(
        store,
        gold={"q1": "answer"},
        norm=norm,
        metrics=("em",),
        expand_repetitions=True,
    )
    assert len(observations) == 2
    assert isinstance(observations[0], Observation)
    assert observations[0].repetition in (0, 1)

    # 6. Parquet serialization roundtrip as Observations
    scores_path = tmp_path / "scores.parquet"
    write_observations(observations, scores_path)
    loaded_obs = read_observations(scores_path)
    assert len(loaded_obs) == 2
    assert isinstance(loaded_obs[0], Observation)
    assert loaded_obs[0].qid == "q1"
    assert loaded_obs[0].repetition == observations[0].repetition

    # 7. PlanCondition and ExecutablePlan
    pcond = PlanCondition(cond, 1, 0.0)
    plan = ExecutablePlan(
        conditions=(pcond,),
        rejected=(),
        schedule=sched,
        config_hash="test",
        seed=42,
        repetition=2,
    )
    assert len(plan.conditions) == 1
    assert plan.repetition == 2
    assert plan.as_dict()["repetition"] == 2
    assert plan.as_dict()["runs_per_cell"] == 2
    assert plan.as_dict()["n_conditions"] == 1

    # 8. ObservationMetadata
    meta = ObservationMetadata(
        config_hash="h1",
        condition_key="A|2|128.0|per_memory_tokens|S|gemma",
        arm="A",
        k=2,
        b=128.0,
        b_mode="per_memory_tokens",
        scale="S",
        reader="gemma",
        qid="q1",
        repetition=1,
        memory_ranks=[0],
        gold_ranks=[0],
        corpus_fraction=0.1,
        ability="single-session",
        reader_strength=1.0,
        prompt_sha256="sha",
        cache_key="key",
    )
    assert meta.repetition == 1
    assert meta.replicate == 1
    d = meta.as_dict()
    assert d["repetition"] == 1
    assert d["replicate"] == 1
    restored = ObservationMetadata.from_dict(d)
    assert restored.repetition == 1
