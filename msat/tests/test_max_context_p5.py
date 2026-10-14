"""Arm D-max: the arm that is honest about having cut the conversation.

The tests concentrate on the three ways this arm could lie:

1. by being labelled `full_context` when it is not
2. by reporting a low score that was caused by the CUT rather than by distraction
3. by trusting a MEAN session length that the real sessions do not honour

Each has a test that fails if it regresses. The second is the one that matters
most: it produces a confident, publishable, meaningless number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from msat.arms.baselines import FULL_CONTEXT_B
from msat.arms.max_context import (
    DEFAULT_RESERVE_TOKENS,
    MAX_CONTEXT_B_MODE,
    MaxContextArm,
    MaxContextFit,
    Recency,
    evidence_survival,
    fit_to_ceiling,
    fit_to_memories,
    max_context_plan,
    max_context_verdict,
    select_sessions,
)
from msat.arms.render import Memory
from msat.store.responses import ConditionKey

TEMPLATE = "system: You are precise.\nuser: |\n  {{ memories }}\n  Q: {{ question }}"
COND = ConditionKey(arm="D-max", k=8, b=17838, b_mode=MAX_CONTEXT_B_MODE, scale="S", reader="qwen")

# The real S scale: 47.73 sessions/q, 489,297 chars/q, 4 chars/token.
S_CORPUS_TOKENS = 122_324
S_SESSIONS = 48.0
S_MEAN_SESSION_TOKENS = S_CORPUS_TOKENS / S_SESSIONS  # ~2548


def _mem(i: int, *, gold: bool = False, n_tokens: int = 2548) -> Memory:
    session = "s_q7" if gold else f"decoy{i}"
    return Memory(
        memory_id=f"{session}#0" if gold else f"decoy{i}#0",
        session_id=session,
        source_question_id="s_q7" if gold else f"decoy{i}",
        text=f"session {i}",
        n_tokens=n_tokens,
    )


# ── the label ──────────────────────────────────────────────────────────────────


def test_the_budget_mode_is_not_full_context() -> None:
    assert MAX_CONTEXT_B_MODE != "full_context"


def test_the_condition_key_never_collides_with_arm_d() -> None:
    arm_d = ConditionKey(
        arm="D", k=48, b=FULL_CONTEXT_B, b_mode="full_context", scale="S", reader="qwen"
    )
    assert COND.key != arm_d.key


def test_the_arm_carries_a_real_finite_budget() -> None:
    """An infinite B here would be a lie: this arm is budgeted by construction."""
    fit = fit_to_ceiling(
        reader="qwen",
        ceiling=20_889,
        mean_session_tokens=S_MEAN_SESSION_TOKENS,
        n_sessions_available=int(S_SESSIONS),
    )
    assert 0 < fit.budget < float("inf")
    assert fit.budget == fit.kept_tokens


# ── the fit ────────────────────────────────────────────────────────────────────


def test_the_fit_reserves_room_to_decode() -> None:
    """A prompt that exactly fills the ceiling OOMs one token into the answer."""
    fit = fit_to_ceiling(
        reader="gemma",
        ceiling=66_160,
        mean_session_tokens=S_MEAN_SESSION_TOKENS,
        n_sessions_available=48,
        reserve_tokens=512,
    )
    assert fit.n_sessions_kept * S_MEAN_SESSION_TOKENS <= 66_160 - 512


def test_the_fit_never_exceeds_the_conversation() -> None:
    fit = fit_to_ceiling(
        reader="qwen",
        ceiling=10_000_000,
        mean_session_tokens=S_MEAN_SESSION_TOKENS,
        n_sessions_available=5,
    )
    assert fit.n_sessions_kept == 5
    assert fit.corpus_fraction == 1.0


def test_a_zero_reserve_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_to_ceiling(
            reader="q",
            ceiling=100,
            mean_session_tokens=10,
            n_sessions_available=10,
            reserve_tokens=-1,
        )


@pytest.mark.parametrize(
    ("reader", "ceiling", "expected_sessions"),
    [("gemma", 66_160, 25), ("phi", 44_318, 17), ("qwen", 20_889, 7)],
)
def test_the_measured_fits_match_the_real_panel(reader, ceiling, expected_sessions) -> None:
    """Pinned against the real `readers.yaml` ceilings and the measured S corpus.

    If a checkpoint or a ceiling changes, this fails and the answer changes with
    it - which is the point of computing it rather than asserting a round number.
    """
    fit = fit_to_ceiling(
        reader=reader,
        ceiling=ceiling,
        mean_session_tokens=S_MEAN_SESSION_TOKENS,
        n_sessions_available=int(S_SESSIONS),
        reserve_tokens=DEFAULT_RESERVE_TOKENS,
    )
    assert fit.n_sessions_kept == expected_sessions
    assert 0.1 < fit.corpus_fraction < 0.6


# ── the overflow check: mean lengths are an estimate, not a promise ────────────


def test_sessions_heavier_than_the_mean_shrink_the_fit() -> None:
    """Late sessions longer than average overflow the estimate. Caught, not OOMed."""
    fit = fit_to_ceiling(
        reader="qwen",
        ceiling=20_889,
        mean_session_tokens=2_000,
        n_sessions_available=48,
    )
    mems = [_mem(i, n_tokens=2_000) for i in range(7)] + [
        _mem(7 + i, n_tokens=9_000) for i in range(4)
    ]
    assert fit.n_sessions_kept == 10

    shrunk = fit_to_memories(fit, mems)
    assert shrunk.n_sessions_kept < fit.n_sessions_kept
    assert shrunk.dropped_for_decode > 0
    assert shrunk.kept_tokens <= fit.ceiling - fit.reserve_tokens


def test_a_fit_that_already_fits_is_left_alone() -> None:
    fit = fit_to_ceiling(
        reader="qwen", ceiling=20_889, mean_session_tokens=2_000, n_sessions_available=48
    )
    mems = [_mem(i, n_tokens=2_000) for i in range(48)]
    assert fit_to_memories(fit, mems) == fit


# ── which sessions survive ─────────────────────────────────────────────────────


def test_most_recent_is_the_default_and_keeps_the_tail() -> None:
    kept, dropped = select_sessions([_mem(i) for i in range(10)], 3)
    assert [m.memory_id for m in kept] == ["decoy7#0", "decoy8#0", "decoy9#0"]
    assert dropped == 7


def test_the_oldest_policy_is_available_because_it_is_an_experimental_choice() -> None:
    """A silent default would hide that this was a decision."""
    kept, _ = select_sessions([_mem(i) for i in range(10)], 3, Recency.OLDEST)
    assert [m.memory_id for m in kept] == ["decoy0#0", "decoy1#0", "decoy2#0"]


def test_asking_for_more_than_exists_returns_everything() -> None:
    kept, dropped = select_sessions([_mem(0), _mem(1)], 99)
    assert len(kept) == 2
    assert dropped == 0


# ── evidence survival: the measurement that makes the arm readable ──────────────


def test_survival_is_the_fraction_of_gold_that_survived() -> None:
    kept = [_mem(0, gold=True), _mem(1, gold=True), _mem(2)]
    assert evidence_survival(kept, ["s_q7", "s_q8"]) == 0.5


def test_survival_is_one_when_no_evidence_exists() -> None:
    """An abstention item has nothing to lose.

    Reporting NaN would drop those items from every average and quietly shrink
    the denominator - which is how a broken cell turns into a flattering result.
    """
    assert evidence_survival([_mem(0)], []) == 1.0


def test_survival_is_zero_when_the_cut_deleted_all_of_it() -> None:
    assert evidence_survival([_mem(0)], ["s_q7"]) == 0.0


# ── the refusal: a confounded result is not a distraction result ───────────────


def test_a_low_score_with_the_answer_cut_out_is_refused() -> None:
    """This is the test that matters most.

    The arm would otherwise report "accuracy 0.22 -> strong distraction effect",
    which is a real-looking, publishable number produced entirely by having
    deleted the evidence.
    """
    v = max_context_verdict(0.22, survival=0.0)
    assert v["verdict"].startswith("CONFOUNDED")
    assert v["reportable_as_distraction"] is False


def test_a_low_score_with_the_evidence_present_is_distraction() -> None:
    v = max_context_verdict(0.22, survival=1.0)
    assert "genuine distraction" in v["verdict"]
    assert v["reportable_as_distraction"] is True


def test_a_high_score_is_clean_only_if_nothing_was_cut() -> None:
    assert "clean" in max_context_verdict(0.8, 1.0)["verdict"]
    partial = max_context_verdict(0.8, 0.6)
    assert "robust to the cut" in partial["verdict"]


def test_survival_of_three_of_three_is_not_confounded() -> None:
    """1 - 1/3 is 0.666, but 3/3 must not come back as 0.9999999999999998 and
    turn a clean distraction result into a refusal."""
    v = max_context_verdict(0.2, survival=3 / 3)
    assert v["reportable_as_distraction"] is True


def test_a_genuinely_partial_survival_is_still_confounded_when_the_score_is_low() -> None:
    v = max_context_verdict(0.2, survival=2 / 3)
    assert v["reportable_as_distraction"] is False


# ── rendering ──────────────────────────────────────────────────────────────────


def test_the_rendered_prompt_carries_survival_and_its_own_mode() -> None:
    arm = MaxContextArm(template=TEMPLATE)
    # The fit is built for 2000-token sessions and the fixture supplies exactly
    # those, so the overflow check has nothing to correct. (Hand the arm 2548-token
    # sessions against a 2000-token fit and it shrinks - see
    # `test_sessions_heavier_than_the_mean_shrink_the_fit`.)
    fit = fit_to_ceiling(
        reader="qwen", ceiling=20_889, mean_session_tokens=2_000, n_sessions_available=48
    )
    mems = [_mem(i, gold=False, n_tokens=2_000) for i in range(48)]
    mems[5] = _mem(5, gold=True, n_tokens=2_000)
    out = arm.render("q", mems, COND, fit=fit, gold_ids=["s_q5"])
    assert out.b_mode == MAX_CONTEXT_B_MODE
    assert out.position_variant == "full/most_recent"
    assert out.k == len(out.memory_ids) == fit.n_sessions_kept == 10
    # session 5 was cut (only the last 10 survive) -> the evidence is gone.
    assert out.evidence_survival == 0.0
    assert "[1] session 38" in out.user
    assert "[10] session 47" in out.user


def test_the_render_carries_survival_when_the_evidence_survives() -> None:
    arm = MaxContextArm(template=TEMPLATE)
    fit = fit_to_ceiling(
        reader="qwen", ceiling=20_889, mean_session_tokens=2_000, n_sessions_available=48
    )
    mems = [_mem(i, n_tokens=2_000) for i in range(48)]
    mems[45] = _mem(45, gold=True, n_tokens=2_000)  # `_mem(gold=True)` -> session "s_q7"
    out = arm.render("q", mems, COND, fit=fit, gold_ids=["s_q7"])
    assert "s_q7#0" in out.memory_ids
    assert "s_q7" in out.session_ids
    assert out.evidence_survival == 1.0


def test_an_arm_with_no_template_refuses() -> None:
    arm = MaxContextArm(template="")
    fit = fit_to_ceiling(reader="q", ceiling=1000, mean_session_tokens=100, n_sessions_available=5)
    with pytest.raises(ValueError, match="no template"):
        arm.render("q", [_mem(0)], COND, fit=fit)


# ── the plan, so the cap is visible before a run ───────────────────────────────


def test_the_plan_reports_arm_d_infeasible_and_the_cap() -> None:
    plan = max_context_plan(
        corpus_tokens=S_CORPUS_TOKENS,
        sessions_median=S_SESSIONS,
        limits={"gemma": 66_160, "qwen": 20_889},
    )
    assert plan["arm_d_infeasible"] is True
    assert plan["per_reader"]["gemma"]["full_context"] is False
    assert plan["per_reader"]["gemma"]["n_sessions_kept"] == 25
    assert "NOT full context" in plan["note"]


def test_the_plan_would_not_claim_infeasibility_it_cannot_see() -> None:
    """If a bigger reader ever lands, the plan says Arm D is runnable again."""
    plan = max_context_plan(
        corpus_tokens=S_CORPUS_TOKENS,
        sessions_median=S_SESSIONS,
        limits={"big": 200_000},
    )
    assert plan["arm_d_infeasible"] is False
    assert plan["per_reader"]["big"]["full_context"] is True


def test_the_fit_round_trips_through_its_dict() -> None:
    fit = fit_to_ceiling(
        reader="gemma",
        ceiling=66_160,
        mean_session_tokens=S_MEAN_SESSION_TOKENS,
        n_sessions_available=48,
    )
    d = fit.as_dict()
    assert d["reader"] == "gemma"
    assert 0.0 <= d["corpus_fraction"] <= 1.0
    assert d["kept_tokens"] > 0


def test_a_fit_is_immutable_so_a_shrink_cannot_corrupt_a_shared_one() -> None:
    fit: MaxContextFit = fit_to_ceiling(
        reader="q", ceiling=100_000, mean_session_tokens=1_000, n_sessions_available=10
    )
    assert fit.n_sessions_kept == 10
    fit_to_memories(fit, [_mem(0, n_tokens=99_000)])
    assert fit.n_sessions_kept == 10  # untouched


def test_no_config_file_defines_a_full_context_max_budget() -> None:
    """The cap is DERIVED from the ceilings. A hand-written one would drift."""
    root = Path(__file__).resolve().parent.parent / "configs"
    for path in root.rglob("*.yaml"):
        assert "full_context_max" not in path.read_text(encoding="utf-8"), (
            f"{path} names full_context_max; it must be computed from the VRAM "
            f"ceilings, not declared."
        )
