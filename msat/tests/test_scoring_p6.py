"""`msat score`, the blinded human sample, and the standalone judge audit.

Three properties that are easy to lose and expensive to rediscover:

  * abstention items score NULL, never 0.0 (R8.4)
  * the labeller's CSV contains no judge verdict (R8.2, anchoring)
  * the audit is produced with NO run and a ZERO interaction (R8.7)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from msat.cli import app
from msat.judges.human_sample import (
    DEFAULT_MIN_PER_CELL,
    HUMAN_COLUMNS,
    allocate,
    draw_sample,
    judge_vs_human,
    read_human_csv,
    write_human_csv,
    write_judge_side,
)
from msat.judges.report import build_audit, format_audit
from msat.metrics.normalisation import Normaliser
from msat.metrics.score import (
    build_disagreements,
    read_scores,
    score_store,
    write_disagreements,
    write_scores,
)
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse, cache_key

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
RUNNER = CliRunner()
COND = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")


@pytest.fixture(scope="module")
def norm() -> Normaliser:
    spec = yaml.safe_load((CONFIGS / "prompts" / "normalisation.yaml").read_text(encoding="utf-8"))
    return Normaliser.from_config(spec)


def _store(tmp_path: Path, items: list[tuple[str, str, int]]) -> ResponseStore:
    """items: (qid, answer_text, replicate)."""
    store = ResponseStore(tmp_path / "responses")
    for qid, text, rep in items:
        req = {
            "model": "google/gemma-4-E4B-it",
            "messages": [{"role": "user", "content": qid}],
            "temperature": 0.0,
        }
        store.put(
            StoredResponse(
                key=cache_key(COND, req),
                condition=COND.as_dict(),
                request=req,
                text=text,
                finish_reason="stop",
                model_version="v1",
                prompt_tokens=100,
                completion_tokens=4,
                latency_ms=10.0,
                qid=qid,
                replicates=tuple(range(rep)),
            )
        )
    return store


# ═══════════════════════════════════════════════════════ msat score ════════════


def test_em_and_f1_are_written_side_by_side(tmp_path: Path, norm) -> None:
    """R8.1: never one metric alone. A reader shown only EM calls the system worse
    than it is; only F1 and they call it better. Neither is supportable from one."""
    store = _store(tmp_path, [("q1", "paris", 1)])
    rows, rep = score_store(store, {"q1": "Paris"}, norm)
    assert {r.metric for r in rows} == {"em", "f1"}
    assert rep.metrics_written == ("em", "f1")


def test_em_and_f1_disagree_on_a_paraphrase(tmp_path: Path, norm) -> None:
    """The reason both are needed, shown rather than asserted."""
    store = _store(tmp_path, [("q1", "the city is Paris", 1)])
    rows, _ = score_store(store, {"q1": "Paris"}, norm)
    by = {r.metric: r.value for r in rows}
    assert by["em"] == 0.0, "EM is unforgiving of paraphrase"
    assert by["f1"] > 0.0, "F1 credits the overlapping token"


def test_a_missing_official_evaluator_is_reported_not_tolerated(tmp_path: Path, norm) -> None:
    """A two-metric table is indistinguishable from a three-metric one in the
    artifact unless the report says so."""
    store = _store(tmp_path, [("q1", "paris", 1)])
    _, rep = score_store(store, {"q1": "Paris"}, norm)
    assert any("only EM and token-F1" in n for n in rep.notes)


def test_an_official_score_joins_the_same_rows(tmp_path: Path, norm) -> None:
    store = _store(tmp_path, [("q1", "paris", 1)])
    rows, rep = score_store(store, {"q1": "Paris"}, norm, official={"q1": 1.0})
    assert "official" in {r.metric for r in rows}
    assert rep.metrics_written == ("em", "f1", "official")


def test_abstention_items_score_null_never_zero(tmp_path: Path, norm) -> None:
    """R8.4. 0.0 would say a perfect retriever retrieved nothing on every
    abstention item; NULL says the question does not apply."""
    store = _store(tmp_path, [("q1", "Paris", 1), ("q9_abs", "I don't know", 1)])
    rows, rep = score_store(store, {"q1": "Paris"}, norm)
    abst = [r for r in rows if r.is_abstention]
    assert {r.qid for r in abst} == {"q9_abs"}
    assert {r.value for r in abst} == {None}
    assert rep.n_abstention == 1
    # The answerable item is unaffected and still scored.
    assert {r.value for r in rows if not r.is_abstention} == {1.0}


def test_the_abstention_suffix_is_the_rule(tmp_path: Path, norm) -> None:
    store = _store(tmp_path, [("q9_abs", "no idea", 1)])
    rows, _ = score_store(store, {}, norm)
    assert all(r.value is None for r in rows)


def test_a_missing_gold_is_retained_as_a_null_row(tmp_path: Path, norm) -> None:
    """Dropping it would shrink the denominator of every mean without saying so."""
    store = _store(tmp_path, [("q1", "paris", 1)])
    rows, rep = score_store(store, {}, norm)
    assert len(rows) == 2
    assert rep.n_missing_gold == 1


def test_a_means_is_none_with_nothing_scored(tmp_path: Path, norm) -> None:
    store = _store(tmp_path, [("q9_abs", "no", 1)])
    _, rep = score_store(store, {}, norm)
    assert rep.means["em"]["mean"] is None


def test_the_mean_is_reported_with_its_n(tmp_path: Path, norm) -> None:
    """0.42 over 12 items and over 12,000 are different claims."""
    store = _store(tmp_path, [("q1", "paris", 1), ("q2", "lyon", 1)])
    _, rep = score_store(store, {"q1": "Paris", "q2": "Paris"}, norm)
    assert rep.means["em"]["mean"] == pytest.approx(0.5)
    assert rep.means["em"]["n_scored"] == 2


def test_cached_replicates_are_flagged(tmp_path: Path, norm) -> None:
    """A run reporting 3 replicates of a cell may have PAID for one. Reporting the
    apparent replicate count as the independent sample count overstates by 3x."""
    one = _store(tmp_path / "a", [("q1", "paris", 1)])
    many = _store(tmp_path / "b", [("q1", "paris", 3)])
    r1, _ = score_store(one, {"q1": "Paris"}, norm)
    r3, _ = score_store(many, {"q1": "Paris"}, norm)
    assert r1[0].cached is False and r1[0].replicate == 0
    assert r3[0].cached is True and r3[0].replicate == 2


def test_an_unattributable_response_is_counted_not_scored(tmp_path: Path, norm) -> None:
    """A response with no qid cannot be joined to a question. Inventing one puts
    numbers in a table with no way to trace them back."""
    store = _store(tmp_path, [("q1", "paris", 1)])
    # The response DOES carry a qid, so it is attributable and scores (to NULL,
    # because no gold was supplied). The unattributable path is covered separately.
    rows, rep = score_store(store, {}, norm)
    assert rep.n_unattributable == 0
    assert rep.n_attributable == 1
    assert {r.value for r in rows} == {None}
    assert store.get(_keys(store)[0]) is not None


def _keys(store: ResponseStore) -> list[str]:
    return [r.key for r in store.iter_all()]


def test_a_store_of_orphans_scores_nothing_and_says_so(tmp_path: Path, norm) -> None:
    store = ResponseStore(tmp_path / "responses")
    req = {"model": "m", "messages": [], "temperature": 0.0}
    store.put(
        StoredResponse(
            key=cache_key(COND, req),
            condition=COND.as_dict(),
            request=req,
            text="x",
            finish_reason="stop",
            model_version="v1",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
        )
    )
    rows, rep = score_store(store, {"q": "g"}, norm)
    assert rows == []
    assert rep.n_unattributable == 1


def test_scores_round_trip_through_parquet(tmp_path: Path, norm) -> None:
    store = _store(tmp_path, [("q1", "paris", 1), ("q9_abs", "no", 1)])
    rows, _ = score_store(store, {"q1": "Paris"}, norm)
    dest = tmp_path / "scores.parquet"
    write_scores(rows, dest)
    back = read_scores(dest).to_pylist()
    assert len(back) == 4
    assert {r["qid"] for r in back} == {"q1", "q9_abs"}
    assert any(r["value"] is None for r in back), "null survived the write"


def test_one_row_per_item_per_metric_never_an_aggregate(tmp_path: Path, norm) -> None:
    """R8.3. A stored aggregate cannot be re-derived under a different metric."""
    store = _store(tmp_path, [(f"q{i}", "paris", 1) for i in range(3)])
    rows, rep = score_store(store, {f"q{i}": "Paris" for i in range(3)}, norm)
    assert rep.n_rows == 6
    assert len(rows) == 6


def test_disagreements_are_written_per_item_with_content(tmp_path: Path, norm) -> None:
    from msat.judges.verdicts import Verdict

    key = ("A|4|512.0|budget|S|gemma", "q1")
    rows = build_disagreements(
        {"j1": {key: Verdict.CORRECT}, "j2": {key: Verdict.INCORRECT}},
        {key: "Paris"},
        {"q1": "Paris"},
        norm,
    )
    assert len(rows) == 1
    assert rows[0]["answer"] == "Paris"
    assert rows[0]["gold"] == "Paris"
    assert rows[0]["em"] == pytest.approx(1.0), "the metric travels with the disagreement"

    dest = tmp_path / "disagreements.parquet"
    write_disagreements(rows, dest)
    import pyarrow.parquet as pq

    assert pq.read_table(dest).num_rows == 1


def test_agreeing_judges_produce_no_disagreement_row(tmp_path: Path, norm) -> None:
    from msat.judges.verdicts import Verdict

    key = ("A|4|512.0|budget|S|gemma", "q1")
    rows = build_disagreements(
        {"j1": {key: Verdict.CORRECT}, "j2": {key: Verdict.CORRECT}},
        {key: "Paris"},
        {"q1": "Paris"},
        norm,
    )
    assert rows == []


# ═══════════════════════════════════════════════════ the CLI ══════════════════


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results" / "pilot-01"
    d.mkdir(parents=True)
    _store(d, [("q1", "paris", 1), ("q9_abs", "no idea", 1)])
    gold = d / "gold.json"
    gold.write_text(json.dumps({"q1": "Paris"}), encoding="utf-8")
    return d


def test_score_writes_the_parquet_and_reports_both_metrics(tmp_path: Path) -> None:
    d = _run_dir(tmp_path)
    r = RUNNER.invoke(
        app,
        [
            "score",
            "pilot-01",
            "--results",
            str(tmp_path / "results"),
            "--gold",
            str(d / "gold.json"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (d / "scores.parquet").exists()
    assert "em" in r.output and "f1" in r.output
    assert "scored NULL, not 0.0" in r.output


def test_score_requires_gold(tmp_path: Path) -> None:
    _run_dir(tmp_path)
    r = RUNNER.invoke(app, ["score", "pilot-01", "--results", str(tmp_path / "results")])
    assert r.exit_code == 1
    assert "--gold is required" in r.output


def test_score_says_when_disagreements_cannot_be_written(tmp_path: Path) -> None:
    """A missing verdicts file is not evidence the judges agreed."""
    d = _run_dir(tmp_path)
    r = RUNNER.invoke(
        app,
        [
            "score",
            "pilot-01",
            "--results",
            str(tmp_path / "results"),
            "--gold",
            str(d / "gold.json"),
        ],
    )
    assert "not evidence that the judges agreed" in r.output
    assert not (d / "disagreements.parquet").exists()


def test_score_needs_a_real_run(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    r = RUNNER.invoke(app, ["score", "nope", "--results", str(tmp_path / "results")])
    assert r.exit_code == 1
    assert "no run directory" in r.output


# ═══════════════════════════════════════════════ the human sample ══════════════


def _candidates(n_per_cell: dict[str, int]) -> list[dict]:
    out = []
    for cell, n in n_per_cell.items():
        for i in range(n):
            out.append(
                {
                    "condition_key": cell,
                    "qid": f"{cell}#q{i}",
                    "arm": "A",
                    "K": 4,
                    "B": 512,
                    "scale": "S",
                    "reader": "gemma",
                    "question": f"question {i}?",
                    "answer": "Paris",
                    "gold": "Paris",
                }
            )
    return out


def test_allocation_is_proportional_with_a_floor() -> None:
    assert allocate(1000) == 100
    assert allocate(100) == 10
    # The floor lifts a small cell so it is present rather than vanishing - but never
    # beyond the cell's own size, because a cell draws ALL its rows or none.
    assert allocate(3) == 3
    assert allocate(8) == DEFAULT_MIN_PER_CELL
    assert allocate(0) == 0


def test_a_small_cell_is_not_allowed_to_vanish_from_the_sample() -> None:
    """Without the floor, the small cells - where the K x B crossover lives - would
    get a couple of rows and judge-human agreement would be measured on the easy end."""
    plan = draw_sample(_candidates({"big": 1000, "small": 2}), min_per_cell=5)
    assert plan.per_cell["small"] == 2, "a 2-row cell draws both rows or neither"


def test_the_sample_is_reproducible_from_its_seed() -> None:
    cands = _candidates({"c1": 50, "c2": 30})
    a = draw_sample(cands, seed=7)
    b = draw_sample(cands, seed=7)
    assert [i.sample_id for i in a.items] == [i.sample_id for i in b.items]


def test_a_different_seed_draws_different_items() -> None:
    cands = _candidates({"c1": 100})
    a = {i.sample_id for i in draw_sample(cands, seed=1).items}
    b = {i.sample_id for i in draw_sample(cands, seed=2).items}
    assert a != b


def test_adding_a_cell_does_not_reshuffle_an_existing_one() -> None:
    """Otherwise a half-finished labelling round has to restart every time a condition
    is added."""
    before = draw_sample(_candidates({"c1": 100}), seed=5)
    after = draw_sample(_candidates({"c1": 100, "c2": 100}), seed=5)
    assert [i.sample_id for i in before.items] == [
        i.sample_id for i in after.items if i.condition_key == "c1"
    ]


# ── blinding ───────────────────────────────────────────────────────────────────


def test_the_labeller_file_contains_no_judge_verdict(tmp_path: Path) -> None:
    """A labeller shown a judge's answer agrees with it more often, and the
    resulting 'judges agree with humans' figure measures the anchor."""
    plan = draw_sample(_candidates({"c1": 20}))
    dest = write_human_csv(plan, tmp_path / "human_sample.csv")
    with dest.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(HUMAN_COLUMNS)
        for row in reader:
            assert not any("judge" in c.lower() for c in row)
            assert row["human_verdict"] == "", "the human's column starts BLANK"


def test_the_judges_answers_are_written_to_a_separate_file(tmp_path: Path) -> None:
    """The join happens AFTER labelling, never before."""
    plan = draw_sample(_candidates({"c1": 20}))
    verdict_rows = {
        (i.condition_key, i.qid): {
            "judge_a": "j1",
            "verdict_a": "correct",
            "judge_b": "j2",
            "verdict_b": "incorrect",
            "em": 1.0,
            "f1": 1.0,
        }
        for i in plan.items
    }
    side = write_judge_side(plan, verdict_rows, tmp_path / "judge_side.csv")
    write_human_csv(plan, tmp_path / "human_sample.csv")
    with side.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert "verdict_a" in reader.fieldnames
        assert "human_verdict" not in reader.fieldnames
    # ... and the labeller's own file still contains none of it.
    assert "judge" not in (tmp_path / "human_sample.csv").read_text(encoding="utf-8")


def dest_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_judge_vers_human_separates_panel_only_from_human_only() -> None:
    """Two counts that hide a third. A panel right far more often than humans is not
    more reliable, it is differently wrong, and `panel_only` is where that shows."""
    labelled = {
        "s1": {"human_verdict": "correct"},
        "s2": {"human_verdict": "incorrect"},
        "s3": {"human_verdict": "incorrect"},
        "s4": {"human_verdict": "correct"},
    }
    judge = {
        "s1": {"judge_a": "j1", "judge_b": "j2", "verdict_a": "correct", "verdict_b": "correct"},
        "s2": {"judge_a": "j1", "judge_b": "j2", "verdict_a": "correct", "verdict_b": "correct"},
        "s3": {
            "judge_a": "j1",
            "judge_b": "j2",
            "verdict_a": "incorrect",
            "verdict_b": "incorrect",
        },
        "s4": {
            "judge_a": "j1",
            "judge_b": "j2",
            "verdict_a": "incorrect",
            "verdict_b": "incorrect",
        },
    }
    out = judge_vs_human(labelled, judge)
    s = out["pairs"]["j1 vs j2"]
    assert s["n"] == 4
    assert s["both_correct"] == 1
    assert s["panel_only"] == 1
    assert s["human_only"] == 1
    assert s["both_wrong"] == 1


def test_unlabelled_items_are_excluded_and_named() -> None:
    labelled = {"s1": {"human_verdict": "correct"}, "s2": {"human_verdict": ""}}
    judge = {
        "s1": {"judge_a": "j1", "judge_b": "j2", "verdict_a": "correct", "verdict_b": "correct"},
        "s2": {"judge_a": "j1", "judge_b": "j2", "verdict_a": "correct", "verdict_b": "correct"},
    }
    out = judge_vs_human(labelled, judge)
    assert out["n_unlabelled"] == 1
    assert out["pairs"]["j1 vs j2"]["n"] == 1
    assert "no human verdict yet" in out["note"]


def test_reading_back_a_filled_labeller_file(tmp_path: Path) -> None:
    plan = draw_sample(_candidates({"c1": 10}))
    dest = write_human_csv(plan, tmp_path / "h.csv")
    text = dest.read_text(encoding="utf-8").replace(",,\n", ",correct,\n", 1)
    dest.write_text(text, encoding="utf-8")
    back = read_human_csv(dest)
    assert any(r["human_verdict"] for r in back.values())


# ══════════════════════════════════════════════════ the standalone audit ═══════


def _records(n_items: int = 20, judges=("j1", "j2"), passes=(1,)) -> list[dict]:
    out = []
    for j in judges:
        for p in passes:
            for i in range(n_items):
                out.append(
                    {
                        "condition_key": "A|4|512|budget|S|gemma",
                        "qid": f"q{i}",
                        "judge_id": j,
                        "pass_index": p,
                        "verdict": "correct" if (i + p) % 5 else "incorrect",
                    }
                )
    return out


def test_the_audit_is_produced_with_no_run_at_all() -> None:
    """R8.7's whole point. A zero interaction still produces an audit, because that
    is exactly the case where an unreliable panel is the reason."""
    a = build_audit(_records())
    assert a.n_judges == 2
    assert a.n_items == 20
    assert a.panel["n_pairs"] == 1


def test_a_single_judge_produces_no_kappa_and_says_why() -> None:
    a = build_audit(_records(judges=("j1",)))
    assert a.panel["panel_usable"] is False
    assert any("requires >= 2 judges" in w for w in a.warnings)


def test_a_single_pass_reports_the_noise_floor_as_unmeasured() -> None:
    """Without it, no result can be distinguished from the judge's own instability."""
    a = build_audit(_records(passes=(1,)))
    assert a.replication["status"] == "NOT RUN"
    assert a.noise_floor is None
    assert any("R8.6 was not run" in w for w in a.warnings)


def test_two_passes_report_a_measured_floor() -> None:
    a = build_audit(_records(passes=(1, 2)))
    assert a.replication["status"] == "RUN"
    assert a.noise_floor is not None
    assert a.noise_floor > 0, "passes 1 and 2 disagree by construction"
    assert "WORST judge" in a.replication["noise_floor_rule"]


def test_a_missing_human_sample_is_reported_not_tolerated() -> None:
    """A panel can agree with itself while being consistently wrong."""
    a = build_audit(_records())
    assert a.human["status"] == "NOT RUN"
    assert any("human-labelled 10% is missing" in w for w in a.warnings)


def test_an_unusable_panel_warns_in_words() -> None:
    a = build_audit(
        [
            {
                "condition_key": "c",
                "qid": f"q{i}",
                "judge_id": "j1",
                "pass_index": 1,
                "verdict": "correct" if i < 8 else "incorrect",
            }
            for i in range(10)
        ]
        + [
            {
                "condition_key": "c",
                "qid": f"q{i}",
                "judge_id": "j2",
                "pass_index": 1,
                "verdict": "correct" if i % 2 == 0 else "incorrect",
            }
            for i in range(10)
        ]
    )
    assert a.panel_usable is False
    assert any("NOT usable" in w for w in a.warnings)


def test_the_audit_is_json_serialisable_and_says_it_is_standalone(tmp_path: Path) -> None:
    dest = build_audit(_records(passes=(1, 2))).write(tmp_path / "judge_audit.json")
    raw = json.loads(dest.read_text(encoding="utf-8"))
    assert raw["standalone"].startswith("Built from verdict records only")
    assert raw["headline"]["cautionary_bar"] == 0.118
    assert raw["reference_bound"]["n_flips"] == 8


def test_the_audit_surfaces_the_aggregate_is_not_the_headline() -> None:
    a = build_audit(_records(passes=(1, 2)))
    assert "n_flips counts ROWS" in a.replication["aggregate_is_not_the_headline"]


def test_the_cli_summary_names_the_three_numbers() -> None:
    text = format_audit(build_audit(_records(passes=(1, 2))))
    assert "panel usable" in text
    assert "kappa min" in text
    assert "judge-noise floor" in text
    assert "2609.38021" in text


def test_judge_audit_requires_verdicts(tmp_path: Path) -> None:
    r = RUNNER.invoke(app, ["judge-audit", "pilot-01", "--out", str(tmp_path / "a.json")])
    assert r.exit_code == 1
    assert "--verdicts is required" in r.output


def test_judge_audit_writes_the_file_and_exits_no_go_on_a_bad_panel(tmp_path: Path) -> None:
    v = tmp_path / "v.json"
    v.write_text(json.dumps(_records(judges=("j1",))), encoding="utf-8")
    out = tmp_path / "judge_audit.json"
    r = RUNNER.invoke(app, ["judge-audit", "pilot-01", "--verdicts", str(v), "--out", str(out)])
    assert r.exit_code == 7, r.output
    assert out.exists(), "the artifact is written even when the panel is unusable"
    assert json.loads(out.read_text(encoding="utf-8"))["headline"]["panel_usable"] is False


def test_judge_audit_succeeds_with_a_usable_panel(tmp_path: Path) -> None:
    v = tmp_path / "v.json"
    v.write_text(json.dumps(_records()), encoding="utf-8")
    out = tmp_path / "judge_audit.json"
    r = RUNNER.invoke(app, ["judge-audit", "pilot-01", "--verdicts", str(v), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text(encoding="utf-8"))["headline"]["panel_usable"] is True
