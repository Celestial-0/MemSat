"""Arms D and F, and the shape of the thing that cannot be run.

The Arm D tests exist because the failure they guard against is silent. A
truncating full-context arm produces plausible prompts, plausible answers, and a
column labelled `full_context` that is none of those things. The refusal is the
deliverable here; the rendering is incidental.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.arms.baselines import (
    FULL_CONTEXT_B,
    FullContextArm,
    FullContextTruncationRefusedError,
    OracleCeiling,
    format_memories,
    full_context_feasible,
    generation_ceiling,
)
from msat.arms.render import Memory
from msat.store.responses import ConditionKey

TEMPLATE = (
    "system: You are precise.\nuser: |\n  Memories:\n  {{ memories }}\n  Question: {{ question }}"
)
COND = ConditionKey(
    arm="D",
    k=48,
    b=FULL_CONTEXT_B,
    b_mode="full_context",
    scale="S",
    reader="gemma",
)


def _mem(i: int, n_tokens: int = 100) -> Memory:
    return Memory(
        memory_id=f"m{i}#0",
        session_id=f"m{i}",
        source_question_id="q0",
        text=f"session {i} happened",
        n_tokens=n_tokens,
    )


# ── the measured hardware finding ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reader", "ceiling"),
    [("gemma", 66_160), ("phi", 44_318), ("qwen", 20_889)],
)
def test_no_reader_can_hold_a_full_s_conversation(reader: str, ceiling: int) -> None:
    """S full context is 122,324 tokens. Every reader is short by >1.8x.

    Pinned because it is an empirical result about the panel on an 8 GB card, not
    a property of the code. If a future reader is VRAM-frugal enough to fit, this
    test FAILS and that is the day Arm D becomes runnable.
    """
    v = full_context_feasible(122_324, context_window=262_144, vram_ceiling=ceiling)
    assert not v.ok
    assert v.shortfall > 0
    assert v.multiple_needed > 1.8, f"{reader} is only {v.multiple_needed:.2f}x short"


def test_feasibility_checks_both_ceilings() -> None:
    """Checking the declared window alone is the error this project already made.

    Every candidate declares 131k-262k of context; none can deliver it on 8 GB.
    """
    declared = 262_144
    assert full_context_feasible(122_324, context_window=declared, vram_ceiling=66_160).ok is False
    # Same corpus, a card that could actually hold it.
    assert full_context_feasible(122_324, context_window=declared, vram_ceiling=200_000).ok is True


def test_the_window_is_the_binding_ceiling_when_it_is_smaller() -> None:
    v = full_context_feasible(200_000, context_window=131_072, vram_ceiling=262_144)
    assert v.ceiling == 131_072
    assert not v.ok


# ── the refusal, which is the real deliverable ─────────────────────────────────


def test_arm_d_refuses_rather_than_truncating() -> None:
    arm = FullContextArm(template=TEMPLATE)
    with pytest.raises(FullContextTruncationRefusedError) as e:
        arm.render(
            "what happened?",
            [_mem(i) for i in range(48)],
            COND,
            corpus_tokens=122_324,
            ceiling=66_160,
        )
    msg = str(e.value)
    assert "122,324" in msg and "66,160" in msg
    # The refusal must name the alternative, not just refuse.
    assert "x this size" in msg


def test_arm_d_renders_when_it_actually_fits() -> None:
    arm = FullContextArm(template=TEMPLATE)
    out = arm.render(
        "what happened?",
        [_mem(i) for i in range(4)],
        COND,
        corpus_tokens=1_000,
        ceiling=66_160,
    )
    assert out.k == 4
    assert out.b == FULL_CONTEXT_B
    assert out.b_mode == "full_context"
    assert out.corpus_fraction == 1.0
    assert out.memory_ids == ("m0#0", "m1#0", "m2#0", "m3#0")
    assert out.session_ids == ("m0", "m1", "m2", "m3")
    assert "[4] session 3 happened" in out.user


def test_arm_d_never_truncates_a_memory_even_a_gigantic_one() -> None:
    """The per-memory budget is the trap. Arm D passes `None`, and a memory of
    10,000 tokens comes through whole."""
    arm = FullContextArm(template=TEMPLATE)
    out = arm.render("q", [_mem(0, n_tokens=10_000)], COND, corpus_tokens=10_000, ceiling=66_160)
    assert "session 0 happened" in out.user
    assert format_memories([_mem(0, n_tokens=10_000)], None) == "[1] session 0 happened"


def test_arm_d_is_a_distinct_budget_mode() -> None:
    """It must not collide with a numeric budget in a condition key or a ledger."""
    assert FULL_CONTEXT_B != 128 and FULL_CONTEXT_B != 2048
    assert (
        COND.key
        != ConditionKey(arm="A", k=48, b=2048, b_mode="budget", scale="S", reader="gemma").key
    )


# ── arm F, the oracle ceiling ──────────────────────────────────────────────────


def test_oracle_loads_and_covers(tmp_path: Path) -> None:
    p = tmp_path / "oracle.json"
    p.write_text(
        json.dumps([{"question_id": "s1", "answer": "a"}, {"qid": "s2", "answer": ["b"]}]),
        encoding="utf-8",
    )
    o = OracleCeiling.load(p)
    assert o.gold_for("s1") == "a"
    assert o.gold_for("s2") == "b"
    cov = o.coverage(["s1", "s2", "s3"])
    assert cov == {"n_questions": 3, "n_covered": 2, "coverage": pytest.approx(2 / 3)}


def test_oracle_reports_partial_coverage_rather_than_implying_full() -> None:
    """A ceiling computed over 4% of the questions bounds nothing, and a reader
    of the number has no way to tell."""
    o = OracleCeiling(answers={"s1": "a"})
    assert o.coverage(["s1", "s2", "s3", "s4", "s5"])["coverage"] == pytest.approx(0.2)


# ── which side is binding ──────────────────────────────────────────────────────


def test_a_large_gap_means_retrieval_is_binding() -> None:
    r = generation_ceiling(0.55, 0.80, 1.0)
    assert r["verdict"] == "retrieval is binding"
    assert r["gap"] == pytest.approx(0.25)


def test_a_small_gap_means_the_write_path_is_binding() -> None:
    r = generation_ceiling(0.55, 0.58, 1.0)
    assert r["verdict"] == "the write path is binding"


def test_an_unattributable_gap_says_so() -> None:
    """The middle band exists because 'inconclusive' is a real answer and forcing
    a verdict there would pick a side to work on for a week."""
    assert generation_ceiling(0.55, 0.62, 1.0)["verdict"].startswith("inconclusive")


def test_a_ceiling_from_a_subset_is_labelled_as_one() -> None:
    assert generation_ceiling(0.55, 0.80, 0.04)["caveat"] == "estimated on the covered subset only"
    assert generation_ceiling(0.55, 0.80, 1.0)["caveat"] == "measured on the full run"
