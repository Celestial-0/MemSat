from __future__ import annotations

import pytest

from msat.judges.audit import (
    REFERENCE_FLIPS,
    REFERENCE_N,
    REFERENCE_PASS1_CORRECT,
    REFERENCE_PASS2_CORRECT,
    flip_verdict,
    flipped_items,
    format_flip_table,
    judged_items_for_pass,
    reference_bound,
    replicate_across_panel,
    two_pass_replication,
)
from msat.judges.verdicts import ItemKey, Verdict

CELL = "A|1|512"


def _v(text: str) -> dict[ItemKey, Verdict]:
    bad = [c for c in text if c not in {"c", "i"}]
    if bad:
        raise ValueError(f"verdict fixtures use 'c'/'i' only; got {bad}")
    return {
        (CELL, f"q{i}"): (Verdict.CORRECT if c == "c" else Verdict.INCORRECT)
        for i, c in enumerate(text)
    }


def _reference_shape() -> tuple[dict, dict]:
    items = ["i"] * REFERENCE_N
    for i in range(REFERENCE_PASS1_CORRECT):
        items[i] = "c"
    pass1 = _v("".join(items))

    items2 = list(items)
    for i in range(6):
        items2[i] = "i"
    items2[490] = "c"
    items2[491] = "c"
    pass2 = _v("".join(items2))
    return pass1, pass2


def test_the_published_numbers_are_the_constants() -> None:
    assert (REFERENCE_N, REFERENCE_PASS1_CORRECT, REFERENCE_PASS2_CORRECT, REFERENCE_FLIPS) == (
        500,
        479,
        475,
        8,
    )


def test_the_reference_bound_records_both_numbers_and_says_why() -> None:
    ref = reference_bound()
    assert ref["net_delta"] == -4
    assert ref["n_flips"] == 8
    assert "8 rows" in ref["note"] and "-4" in ref["note"]
    assert "n_flips is the headline" in ref["note"]


def test_reconstructing_the_reference_gives_eight_flips_not_four() -> None:
    p1, p2 = _reference_shape()
    rep = two_pass_replication(p1, p2, judge_id="judge-1")
    assert rep.correct_pass1 == 479
    assert rep.correct_pass2 == 475
    assert rep.net_delta == -4
    assert rep.n_flips == 8, "flips cancel in the aggregate; count the rows"
    assert rep.correct_to_wrong == 6
    assert rep.wrong_to_correct == 2


def test_the_flip_rate_is_what_gets_compared_not_the_delta() -> None:
    p1, p2 = _reference_shape()
    rep = two_pass_replication(p1, p2)
    assert rep.flip_rate == pytest.approx(8 / 500)


def test_two_identical_passes_report_zero_flips() -> None:
    v = _v("cici")
    rep = two_pass_replication(v, dict(v))
    assert rep.identical is True
    assert rep.n_flips == 0
    assert rep.directional_balance is None


def test_identical_passes_never_read_as_a_strong_verdict() -> None:
    v = _v("cici")
    out = flip_verdict(two_pass_replication(v, dict(v)))
    assert out["verdict"].startswith("IDENTICAL")


def test_two_different_judges_are_not_a_replication() -> None:
    rep = two_pass_replication(_v("cici"), _v("iiii"), pass1_judge="judge-1", pass2_judge="judge-2")
    assert rep.judge_ids_differ is True
    assert flip_verdict(rep)["verdict"].startswith("INVALID")


def test_balanced_flips_and_drifted_flips_are_different_problems() -> None:
    balanced = two_pass_replication(_v("cici"), _v("icci"))
    assert balanced.n_flips == 2
    assert balanced.net_delta == 0
    assert balanced.directional_balance == pytest.approx(0.0)

    drifted = two_pass_replication(_v("cccc"), _v("iiic"))
    assert drifted.n_flips == 3
    assert drifted.net_delta == -3
    assert drifted.directional_balance == pytest.approx(1.0)


def test_a_single_flip_is_always_fully_directional() -> None:
    rep = two_pass_replication(_v("cc"), _v("ic"))
    assert rep.directional_balance == pytest.approx(1.0)


def test_a_flip_rate_below_the_reference_is_reported_as_such() -> None:
    p1 = _v("c" * 100)
    p2 = _v("i" + "c" * 99)
    out = flip_verdict(two_pass_replication(p1, p2))
    assert "at or below" in out["verdict"]
    assert out["flip_rate_ratio_vs_reference"] < 1.0
    assert out["reference"]["n_flips"] == 8


def test_a_flip_rate_above_the_reference_says_results_inside_it_are_not_results() -> None:
    p1 = _v("c" * 20)
    p2 = _v("i" * 20)
    out = flip_verdict(two_pass_replication(p1, p2))
    assert "ABOVE" in out["verdict"]
    assert "inside the judge's own noise" in out["verdict"]
    assert out["flip_rate_ratio_vs_reference"] > 1.0


def test_an_empty_comparison_says_so_rather_than_reporting_a_clean_run() -> None:
    out = flip_verdict(two_pass_replication({}, {}))
    assert "No items" in out["verdict"]


def test_items_in_only_one_pass_are_excluded_not_counted_as_flips() -> None:
    p1 = _v("cici")
    p2 = dict(p1)
    p2[(CELL, "q9")] = Verdict.INCORRECT
    rep = two_pass_replication(p1, p2)
    assert rep.n_items == 4
    assert rep.n_excluded == 1
    assert rep.n_flips == 0


def test_the_flipped_items_are_retained_with_their_direction() -> None:
    rep = two_pass_replication(_v("cici"), _v("icci"))
    rows = flipped_items(rep, {(CELL, "q0"): {"gold": "Paris"}})
    assert [r["qid"] for r in rows] == ["q0", "q1"]
    assert rows[0]["direction"] == "correct_to_wrong"
    assert rows[0]["gold"] == "Paris", "the gold travels with the flip"
    assert rows[1]["direction"] == "wrong_to_correct"


def test_the_reference_reconstruction_retains_all_eight_rows() -> None:
    p1, p2 = _reference_shape()
    rep = two_pass_replication(p1, p2)
    rows = flipped_items(rep, {})
    assert len(rows) == 8
    assert sum(1 for r in rows if r["direction"] == "correct_to_wrong") == 6
    assert sum(1 for r in rows if r["direction"] == "wrong_to_correct") == 2


def test_the_noise_floor_is_the_worst_judge_not_the_mean() -> None:
    good = {
        "judge-1": _v("c" * 20),
    }
    out = replicate_across_panel(
        {"judge-1": _v("c" * 20), "judge-2": _v("c" * 20)},
        {"judge-1": _v("c" * 20), "judge-2": _v("i" * 10 + "c" * 10)},
    )
    assert out["worst_flip_rate"] == pytest.approx(0.5)
    assert out["best_flip_rate"] == pytest.approx(0.0)
    assert "WORST judge" in out["noise_floor_rule"]
    assert good


def test_the_table_shows_the_reference_row() -> None:
    out = replicate_across_panel({"j1": _v("c" * 10)}, {"j1": _v("i" + "c" * 9)})
    text = format_flip_table(out)
    assert "2609.38021" in text
    assert "j1" in text
    assert "flips" in text


def test_extracting_one_passes_verdicts_from_a_flat_record_list() -> None:
    records = [
        {
            "condition_key": CELL,
            "qid": "q0",
            "judge_id": "j1",
            "verdict": "correct",
            "pass_index": 1,
        },
        {
            "condition_key": CELL,
            "qid": "q0",
            "judge_id": "j1",
            "verdict": "incorrect",
            "pass_index": 2,
        },
        {
            "condition_key": CELL,
            "qid": "q1",
            "judge_id": "j2",
            "verdict": "correct",
            "pass_index": 1,
        },
    ]
    p1 = judged_items_for_pass(records, "j1", 1)
    p2 = judged_items_for_pass(records, "j1", 2)
    assert p1 == {(CELL, "q0"): Verdict.CORRECT}
    assert p2 == {(CELL, "q0"): Verdict.INCORRECT}
    assert judged_items_for_pass(records, "j2", 2) == {}
