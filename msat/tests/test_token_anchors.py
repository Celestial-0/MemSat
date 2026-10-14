"""The shipped token anchors are wrong, and that must stay visible.

`configs/pricing/latest.yaml` carries `prompt_tokens_at_K1/K64` written as
order-of-magnitude placeholders before P1 measured anything. P1 then measured the
corpus, which makes the anchors checkable. They do not survive the check:

    S: 489,297 chars/question / 47.73 sessions = 10,250 chars per session
    M: 4,838,071 chars/question / 475.31 sessions = 10,179 chars per session

S and M sessions are the same size to within 0.7%. The config nonetheless claims
S/K=64 is 12,000 tokens while M/K=64 is 120,000 - asserting a 10x difference in
per-session size that is not present in the data.

These numbers decide which cells fit in VRAM, so the discrepancy is not academic.

THE REGRESSION BELOW IS DELIBERATE. It asserts the CURRENT, WRONG values are
flagged. When someone fixes `configs/pricing/latest.yaml` this test fails, and
failing is the correct outcome: the fix needs a real tokenizer measurement on the
execution box, not a constant that makes a test green. Fixing the config means
replacing these anchors with measured values AND updating this test to pin the
new ones - so the correction is a reviewed act rather than a silent edit.
"""

from __future__ import annotations

import pytest

from msat.design import validate_token_anchors

# Measured in P1 (results/dataset_report.json). These are FACTS, not assumptions.
MEASURED_CHARS = {"S": 489_297.0, "M": 4_838_071.0}
MEASURED_SESSIONS = {"S": 47.734, "M": 475.31}

SHIPPED_ANCHORS = {
    "prompt_overhead_tokens": 500,
    "token_estimates": {
        "S": {"prompt_tokens_at_K1": 400, "prompt_tokens_at_K64": 12_000},
        "M": {"prompt_tokens_at_K1": 2_000, "prompt_tokens_at_K64": 120_000},
    },
}


def test_the_two_scales_have_the_same_session_size() -> None:
    """The premise of everything below. If this stops holding, re-derive everything."""
    per_session = {s: MEASURED_CHARS[s] / MEASURED_SESSIONS[s] for s in ("S", "M")}
    assert per_session["S"] / per_session["M"] == pytest.approx(1.0, rel=0.02)


def test_the_shipped_anchors_are_flagged_against_the_measured_corpus() -> None:
    report = validate_token_anchors(SHIPPED_ANCHORS, MEASURED_CHARS, MEASURED_SESSIONS)
    assert not report["consistent"], (
        "the shipped S anchors should be flagged. If this now passes, someone "
        "changed configs/pricing/latest.yaml - re-measure with a real tokenizer "
        "and update this test deliberately."
    )
    assert report["n_off"] == 2

    off = {(f["scale"], f["anchor"]) for f in report["findings"] if not f["ok"]}
    assert off == {
        ("S", "prompt_tokens_at_K1"),
        ("S", "prompt_tokens_at_K64"),
    }


def test_the_s_anchor_understates_by_an_order_of_magnitude() -> None:
    report = validate_token_anchors(SHIPPED_ANCHORS, MEASURED_CHARS, MEASURED_SESSIONS)
    by_key = {(f["scale"], f["anchor"]): f for f in report["findings"]}
    k64 = by_key[("S", "prompt_tokens_at_K64")]
    # declared 12,000 vs ~122,000 expected: the config is ~10x low.
    assert k64["ratio"] < 0.15


def test_the_m_anchors_are_within_tolerance() -> None:
    report = validate_token_anchors(SHIPPED_ANCHORS, MEASURED_CHARS, MEASURED_SESSIONS)
    by_key = {(f["scale"], f["anchor"]): f for f in report["findings"]}
    for anchor in ("prompt_tokens_at_K1", "prompt_tokens_at_K64"):
        assert by_key[("M", anchor)]["ok"], f"M {anchor} should be within 2x"


def test_the_verdict_does_not_depend_on_the_chars_per_token_assumption() -> None:
    """4.0 is an assumption. A 10x discrepancy survives any plausible ratio."""
    verdicts = set()
    for cpt in (3.0, 4.0, 5.0, 6.0):
        report = validate_token_anchors(
            SHIPPED_ANCHORS, MEASURED_CHARS, MEASURED_SESSIONS, chars_per_token=cpt
        )
        verdicts.add(tuple(sorted({f["scale"] for f in report["findings"] if not f["ok"]})))
    assert verdicts == {("S",)}, "only S is wrong, at every plausible ratio"
