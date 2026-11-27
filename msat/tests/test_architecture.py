from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from msat.cli import app
from msat.manifest import stats_from_corpus
from msat.metrics.score import read_scores
from msat.store import Condition, ResponseStore, StoredResponse, cache_key


@pytest.fixture
def stored_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "results" / "selected-metrics"
    condition = Condition("A", 1, 128.0, "per_memory_tokens", "S", "gemma")
    request = {
        "model": "fixture-reader",
        "messages": [{"role": "user", "content": "What city?"}],
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 1,
    }
    store = ResponseStore(run_dir / "responses")
    store.put(
        StoredResponse(
            key=cache_key(condition, request),
            condition=condition.as_dict(),
            request=request,
            text="paris",
            finish_reason="stop",
            model_version="fixture-v1",
            prompt_tokens=8,
            completion_tokens=1,
            latency_ms=1.0,
            qid="fixture-city",
            repetitions=(1,),
        )
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({"fixture-city": "Paris"}), encoding="utf-8")
    return run_dir, gold_path


def score_args(run_dir: Path, gold_path: Path) -> list[str]:
    return ["score", run_dir.name, "--results", str(run_dir.parent), "--gold", str(gold_path)]


@pytest.mark.parametrize("metric", ["em", "f1"])
def test_cli_honors_explicit_metric(stored_run: tuple[Path, Path], metric: str) -> None:
    run_dir, gold_path = stored_run
    result = CliRunner().invoke(app, [*score_args(run_dir, gold_path), "--metric", metric])
    assert result.exit_code == 0, result.output
    rows = read_scores(run_dir / "scores.parquet").to_pylist()
    assert {row["metric"] for row in rows} == {metric}


def test_cli_honors_configured_normalization(stored_run: tuple[Path, Path], tmp_path: Path) -> None:
    run_dir, gold_path = stored_run
    custom_config = tmp_path / "custom-config"
    shutil.copytree(Path("configs"), custom_config)
    normalization = custom_config / "prompts" / "normalisation.yaml"
    normalization.write_text(
        yaml.safe_dump({"normalisation": {"steps": [{"id": "strip", "op": "strip"}]}}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [*score_args(run_dir, gold_path), "--metric", "em", "--config", str(custom_config)],
    )
    assert result.exit_code == 0, result.output
    rows = read_scores(run_dir / "scores.parquet").to_pylist()
    assert len(rows) == 1
    assert rows[0]["value"] == 0.0


def test_cli_rejects_unknown_scoring_profile(stored_run: tuple[Path, Path]) -> None:
    run_dir, gold_path = stored_run
    result = CliRunner().invoke(app, [*score_args(run_dir, gold_path), "--profile", "typo"])
    assert result.exit_code == 2, result.output
    assert not (run_dir / "scores.parquet").exists()


def test_cli_rejects_mode_flag(stored_run: tuple[Path, Path]) -> None:
    run_dir, gold_path = stored_run
    result = CliRunner().invoke(app, [*score_args(run_dir, gold_path), "--mode", "fast"])
    assert result.exit_code == 2, result.output
    assert "no such option" in result.output.lower()


def test_cli_fast_scoring_keeps_deterministic_metric_pair(
    stored_run: tuple[Path, Path], no_network: None
) -> None:
    run_dir, gold_path = stored_run
    result = CliRunner().invoke(app, [*score_args(run_dir, gold_path), "--profile", "fast"])
    assert result.exit_code == 0, result.output
    assert set(read_scores(run_dir / "scores.parquet")["metric"].to_pylist()) == {"em", "f1"}


@pytest.mark.parametrize("profile", ["fast", "full"])
def test_default_plan_destination_agrees_with_run_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    source_configs = Path("configs").resolve()
    shutil.copytree(source_configs, tmp_path / "configs")
    corpus = {
        scale: [stats_from_corpus("fixture", 100, 400_000, chars_per_token=4.0)]
        for scale in ("S", "M")
    }
    counts = {scale: [(100, 400_000)] for scale in corpus}
    monkeypatch.setattr("msat.selftest.synthetic_corpus", lambda _: (corpus, counts))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["build-manifest", "--profile", profile, "--corpus", "synthetic"]
    )
    assert result.exit_code == 0, result.output
    filename = "manifest_fast.json" if profile == "fast" else "manifest.json"
    assert (tmp_path / "work" / filename).exists()
    assert not (tmp_path / "configs" / filename).exists()


def test_fast_run_does_not_fall_back_to_full_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(Path("configs").resolve(), tmp_path / "configs")
    work = tmp_path / "work"
    work.mkdir()
    (work / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "pilot", "--profile", "fast", "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "manifest_fast.json" in result.output


def test_fast_effective_bundle_and_runtime_panel_agree() -> None:
    from msat.cli import panel, resolve_profile_bundle
    from msat.config import load_bundle

    canonical = load_bundle(Path("configs"))
    fast, profile = resolve_profile_bundle(canonical, "fast")
    assert profile == "fast"
    assert fast.mode == "fast"
    assert fast.config_hash != canonical.config_hash
    assert fast.canonical_config_hash == canonical.config_hash
    assert [r.id for r in panel(fast)] == fast.design.readers
    assert len(panel(fast)) == 1


def test_cli_rung_crossed_accepted_and_full_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(Path("configs").resolve(), tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    r_full = CliRunner().invoke(app, ["run", "r001", "--rung", "full", "--dry-run"])
    assert r_full.exit_code != 0
    assert "unknown rung 'full'" in r_full.output

    r_crossed = CliRunner().invoke(app, ["run", "r001", "--rung", "crossed", "--dry-run"])
    assert "unknown rung" not in r_crossed.output
    assert "entering rung 'crossed'" in r_crossed.output


def test_reachability_invariant_every_module_has_production_importer() -> None:
    """Reachability invariant (DEC-005): Every module under src/msat must have at least one

    production importer within src/msat, or be explicitly listed in EXEMPTIONS with a documented reason.
    """
    import ast
    from collections import defaultdict

    root = Path(__file__).resolve().parent.parent / "src" / "msat"
    assert root.is_dir(), f"Expected msat package root at {root}"

    modules: dict[str, Path] = {}
    for p in root.rglob("*.py"):
        rel = p.relative_to(root.parent)
        if p.name == "__init__.py":
            continue
        mod_name = ".".join(rel.with_suffix("").parts)
        modules[mod_name] = p

    importers: dict[str, set[str]] = defaultdict(set)

    for file_path in root.rglob("*.py"):
        rel = file_path.relative_to(root.parent)
        if file_path.name == "__init__.py":
            mod_name = ".".join(rel.parent.parts)
            package = mod_name
        else:
            mod_name = ".".join(rel.with_suffix("").parts)
            package = mod_name.rpartition(".")[0]

        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    for m in modules:
                        if (imported == m or imported.startswith(m + ".")) and mod_name != m:
                            importers[m].add(mod_name)
            elif isinstance(node, ast.ImportFrom):
                level = node.level
                mod_part = node.module or ""
                if level == 0:
                    full_mod = mod_part
                else:
                    parts = package.split(".")
                    if level <= len(parts):
                        base = ".".join(parts[: len(parts) - level + 1])
                        full_mod = f"{base}.{mod_part}" if mod_part else base
                    else:
                        continue
                for alias in node.names:
                    candidate = f"{full_mod}.{alias.name}" if full_mod else alias.name
                    if candidate in modules and mod_name != candidate:
                        importers[candidate].add(mod_name)
                for m in modules:
                    if full_mod == m and mod_name != m:
                        importers[m].add(mod_name)

    exemptions: dict[str, str] = {
        "msat.cli": "Top-level console script entry point registered in pyproject.toml [project.scripts].",
        "msat.selftest": "Standalone diagnostic self-test suite invoked via CLI callback and CI preflight.",
    }

    unreachable = {
        m: "Zero production importers found"
        for m in sorted(modules)
        if not importers[m] and m not in exemptions
    }
    assert not unreachable, (
        f"Found unreachable production modules without exemptions: {unreachable}"
    )
