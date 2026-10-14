"""`conditions.parquet` and the `msat run` dry-run path.

The manifest is the R4.2 deliverable and the only thing `msat analyze` will read.
Three properties are load-bearing and each is pinned here:

  * `session_ids` travels beside `memory_ids` — gold is a session, a memory is a
    chunk of one, and dropping the column forces recall to parse ids
  * `null` evidence_survival survives the round trip as `null`, not `1.0`
  * two cells that render identical bytes are reported as degenerate
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from msat.arms.conditions import (
    CONDITIONS_SCHEMA_VERSION,
    arrow_schema,
    cells_from,
    read_conditions,
    row_for,
    verify_manifest,
    write_conditions,
)
from msat.arms.render import RenderedPrompt
from msat.cli import app
from msat.store.responses import ConditionKey

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
RUNNER = CliRunner()


def _prompt(
    arm="A",
    k=4,
    b=512,
    *,
    scale="S",
    reader="gemma",
    sessions=None,
    qid="q1",
    survival=1.0,
    variant="natural",
) -> RenderedPrompt:
    sessions = sessions or [f"s{i}" for i in range(k)]
    return RenderedPrompt(
        arm=arm,
        condition=ConditionKey(arm=arm, k=k, b=b, b_mode="budget", scale=scale, reader=reader),
        system="sys",
        user=f"user {qid} {arm} {k} {b} {sessions}",
        memory_ids=tuple(f"{s}#0" for s in sessions),
        session_ids=tuple(sessions),
        ranks=tuple(range(1, len(sessions) + 1)),
        k=len(sessions),
        b=b,
        b_mode="budget",
        corpus_fraction=0.1,
        position_variant=variant,
        evidence_survival=survival,
        qid=qid,
    )


# ── the schema ─────────────────────────────────────────────────────────────────


def test_the_schema_is_declared_not_inferred() -> None:
    """An inferred schema takes the type of whatever is in row zero, so a K that is
    an int in the first chunk and a float in a later one produces two incompatible
    files with the same name."""
    names = [f.name for f in arrow_schema()]
    for required in ("qid", "session_ids", "evidence_survival", "ranks", "prompt_sha256"):
        assert required in names, required


def test_the_session_column_is_present_and_populated() -> None:
    """The column `recall@K` is computed over."""
    row = row_for(_prompt())
    assert row["session_ids"] == ["s0", "s1", "s2", "s3"]
    assert row["memory_ids"] == [f"s{i}#0" for i in range(4)]


def test_one_row_per_condition_question_and_variant_not_one_per_condition() -> None:
    """Collapsing the question away makes recall impossible without re-running."""
    rows = [row_for(_prompt(qid=f"q{i}", k=k)) for i in range(3) for k in (2, 4)]
    assert len({(r["qid"], r["K"]) for r in rows}) == 6


# ── the round trip ─────────────────────────────────────────────────────────────


def test_a_written_manifest_reads_back_intact(tmp_path: Path) -> None:
    p = tmp_path / "conditions.parquet"
    prompts = [_prompt(qid=f"q{i}", k=4) for i in range(5)]
    write_conditions(prompts, p)
    rows = read_conditions(p).to_pylist()
    assert len(rows) == 5
    assert rows[0]["session_ids"] == ["s0", "s1", "s2", "s3"]
    assert rows[0]["ranks"] == [1, 2, 3, 4]


def test_null_survival_stays_null_and_does_not_become_one(tmp_path: Path) -> None:
    """A missing survival and a survival of 1.0 are different facts."""
    p = tmp_path / "c.parquet"
    prompts = [RenderedPrompt(**_as_kwargs(_prompt(qid="q1"), evidence_survival=None))]
    write_conditions(prompts, p)
    row = read_conditions(p).to_pylist()[0]
    assert row["evidence_survival"] is None, "null was coerced to 1.0"
    assert row["evidence_survival"] != 1.0


def _as_kwargs(p: RenderedPrompt, **over) -> dict:
    d = {
        "arm": p.arm,
        "condition": p.condition,
        "system": p.system,
        "user": p.user,
        "memory_ids": p.memory_ids,
        "session_ids": p.session_ids,
        "ranks": p.ranks,
        "k": p.k,
        "b": p.b,
        "b_mode": p.b_mode,
        "corpus_fraction": p.corpus_fraction,
        "position_variant": p.position_variant,
        "evidence_survival": p.evidence_survival,
        "qid": p.qid,
    }
    d.update(over)
    return d


def test_a_cut_arm_records_its_survival(tmp_path: Path) -> None:
    p = tmp_path / "c.parquet"
    write_conditions([_prompt(arm="D-max", survival=0.4)], p)
    assert read_conditions(p).to_pylist()[0]["evidence_survival"] == pytest.approx(0.4)


def test_the_schema_version_is_recorded(tmp_path: Path) -> None:
    p = tmp_path / "c.parquet"
    rep = write_conditions([_prompt()], p)
    assert rep["schema_version"] == CONDITIONS_SCHEMA_VERSION
    assert read_conditions(p).to_pylist()[0]["schema_version"] == CONDITIONS_SCHEMA_VERSION


# ── identical prompts ──────────────────────────────────────────────────────────


def test_two_cells_rendering_identical_bytes_are_reported(tmp_path: Path) -> None:
    """The failure the absolute-token budget produced: 78 of 126 cells rendered the
    same prompt, so they measured nothing. The ROW count would not reveal it."""
    p = tmp_path / "c.parquet"
    # Two different conditions whose prompts are byte-identical: same text, same
    # memories, different K in the condition key. This is exactly what the
    # absolute-token budget produced - the key said one thing, the bytes another.
    a = _prompt(arm="A", k=2, b=128, sessions=["s0", "s1"])
    b = RenderedPrompt(
        **_as_kwargs(_prompt(arm="A", k=4, b=128, sessions=["s0", "s1"]), user=a.user)
    )
    rep = write_conditions([a, b], p)
    assert rep["indistinguishable_cells"], "two conditions, one prompt digest"
    assert rep["n_cells"] == 2
    assert rep["n_distinct_prompts"] == 1
    assert "indistinguishable" in rep["note"]


def test_the_check_is_across_cells_not_within_one(tmp_path: Path) -> None:
    """Prompts WITHIN a cell are supposed to vary across questions. Treating that as
    degeneracy would flag every healthy manifest - which is the same class of error
    as the bug it was written to catch."""
    p = tmp_path / "c.parquet"
    prompts = [_prompt(qid=f"q{i}", sessions=["a", "b", f"c{i}"]) for i in range(3)]
    rep = write_conditions(prompts, p)
    assert rep["n_cells"] == 1
    assert rep["n_distinct_prompts"] == 3
    assert rep["indistinguishable_cells"] == []


def test_a_single_row_cell_is_not_called_degenerate(tmp_path: Path) -> None:
    """One row per cell is normal for a one-question pilot."""
    assert write_conditions([_prompt()], tmp_path / "c.parquet")["indistinguishable_cells"] == []


def test_the_digest_changes_with_the_prompt_text() -> None:
    a = row_for(_prompt(qid="q1"))["prompt_sha256"]
    b = row_for(_prompt(qid="q2"))["prompt_sha256"]
    assert a != b


# ── R4.4 read from the artifact ───────────────────────────────────────────────


def _crossed(tmp_path: Path, k_grid, b_grid) -> Path:
    p = tmp_path / "c.parquet"
    write_conditions(
        [
            _prompt(arm=arm, k=k, b=b, scale=scale, reader=reader)
            for arm in ("A", "G")
            for scale in ("S", "M")
            for reader in ("gemma", "qwen")
            for k in k_grid
            for b in b_grid
        ],
        p,
    )
    return p


def test_a_complete_crossing_verifies_from_the_artifact(tmp_path: Path) -> None:
    p = _crossed(tmp_path, [1, 2, 4], [128, 512])
    rep = verify_manifest(p, expected_k=[1, 2, 4], expected_b=[128, 512])
    assert rep["crossed"] is True
    assert rep["reader_balanced"] is True


def test_a_missing_cell_is_visible_in_the_artifact(tmp_path: Path) -> None:
    """Read from the FILE, so a hand-edited manifest cannot pass."""
    p = tmp_path / "c.parquet"
    write_conditions([_prompt(arm="A", k=k, b=b) for k in (1, 2) for b in (128, 512)], p)
    rep = verify_manifest(p, expected_k=[1, 2, 4], expected_b=[128, 512])
    assert rep["crossed"] is False
    assert rep["problems"]


def test_cells_from_returns_the_distinct_conditions(tmp_path: Path) -> None:
    p = _crossed(tmp_path, [1, 2], [128])
    assert len(cells_from(p)) == 16  # 2 arms x 2 scales x 2 readers x 2 K


def test_the_report_carries_the_path_and_a_size(tmp_path: Path) -> None:
    dest = tmp_path / "conditions.parquet"
    rep = write_conditions([_prompt()], dest)
    assert Path(rep["path"]).name == "conditions.parquet"
    assert rep["n_bytes"] == dest.stat().st_size


# ═══════════════════════════════════════════════════════ msat run ══════════════


def test_run_dry_run_validates_and_generates_nothing(tmp_path: Path) -> None:
    r = RUNNER.invoke(
        app,
        [
            "run",
            "pilot-01",
            "--dry-run",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "DRY RUN" in r.output
    assert "A, B, C, D, E, F, G" in r.output
    assert not (tmp_path / "pilot-01" / "responses").exists()


def test_run_dry_run_names_the_prompt_and_config_arms(tmp_path: Path) -> None:
    """A reader of `arms.yaml` will otherwise go looking for `arm_e.j2` as a prompt."""
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert "config-only ['B', 'E', 'F']" in r.output
    assert "manipulated in the memory set" in r.output


def test_run_refuses_the_full_grid_from_a_fresh_ladder(tmp_path: Path) -> None:
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--rung",
            "full",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert r.exit_code == 7, r.output
    assert "cannot be climbed from the top" in r.output


def test_run_enters_the_first_unpassed_rung_by_default(tmp_path: Path) -> None:
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert "entering rung 'negative'" in r.output


def test_an_unknown_rung_is_refused_rather_than_defaulting_to_full(tmp_path: Path) -> None:
    """Defaulting a typo to `full` would skip the ladder, which is the one thing the
    ladder exists to prevent."""
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--rung",
            "ful",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert r.exit_code == 1
    assert "unknown rung" in r.output


def test_run_passes_the_arm_e_gate_once_thresholds_are_set(tmp_path: Path) -> None:
    """The shipped config carries numbers, so `run` clears all four checks.

    This test asserted `EXIT_NO_GO` with "thresholds are NULL" until D5 was answered
    on 2026-10-13. Pinning the blocked state would have failed the moment someone
    did the work - so the direction of the pin is reversed, and the blocked path is
    kept where it belongs: `test_run_would_block_on_null_thresholds_below`.
    """
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--results",
            str(tmp_path),
            "--plan",
            str(CONFIGS.parent / "results" / "manifest.json"),
        ],
    )
    assert "thresholds are NULL" not in r.output
    assert "needs a backend" in r.output, "all four checks cleared; only the backend is missing"


def test_run_would_block_on_null_thresholds_below(tmp_path: Path) -> None:
    """`null` means NOT EVALUABLE, so `proceed` is False. The behaviour is real and
    is tested against a config that has none - not against the shipped one, which
    now has numbers."""
    from msat.arms.negative import ControlConfig

    assert ControlConfig().thresholds_measurable is False
    assert not ControlConfig(pass_threshold_accuracy_below=0.15).thresholds_measurable
    assert not ControlConfig(pass_threshold_recall_below=0.01).thresholds_measurable
    assert ControlConfig(
        pass_threshold_accuracy_below=0.15, pass_threshold_recall_below=0.01
    ).thresholds_measurable


def test_run_needs_a_manifest(tmp_path: Path) -> None:
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--results",
            str(tmp_path),
            "--plan",
            str(tmp_path / "absent.json"),
        ],
    )
    assert r.exit_code == 1
    assert "run `msat build-plan` first" in r.output


def test_run_requires_a_run_id() -> None:
    assert RUNNER.invoke(app, ["run", "--dry-run"]).exit_code == 1


def test_run_checks_the_registry_before_the_plan(tmp_path: Path) -> None:
    """Order matters: the cheap checks refuse a run before the expensive one. A bad
    registry with no manifest still reports the REGISTRY problem."""
    r = RUNNER.invoke(
        app,
        [
            "run",
            "p",
            "--dry-run",
            "--results",
            str(tmp_path),
            "--plan",
            str(tmp_path / "absent.json"),
        ],
    )
    assert "arms        A, B, C, D, E, F, G" in r.output, "registry ran first"
    assert "run `msat build-plan` first" in r.output


def test_the_manifest_fixture_used_by_these_tests_exists() -> None:
    assert (CONFIGS.parent / "results" / "manifest.json").exists(), (
        "run `msat build-plan --out results/manifest.json`"
    )
