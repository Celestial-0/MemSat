"""The ladder cannot be climbed from the top.

These tests are almost entirely about ordering. That is deliberate: the ladder's
job is to make a specific mistake impossible, and the mistake is spending 292
hours before finding out something `ls` would have said.
"""

from __future__ import annotations

import pytest

from msat.runner.ladder import (
    LADDER,
    LadderOrderViolationError,
    LadderState,
    check_order,
)


def test_the_ladder_starts_with_the_cheapest_arm_that_must_fail_loudly() -> None:
    assert LADDER[0].name == "negative"
    assert LADDER[0].arms == ("E",)
    assert LADDER[0].n_questions == 1


def test_the_ladder_ends_with_everything() -> None:
    assert LADDER[-1].name == "full"
    assert LADDER[-1].n_questions == -1
    assert set(LADDER[-1].scales) == {"S", "M"}


def test_every_rung_depends_on_the_one_before_it() -> None:
    names = [r.name for r in LADDER]
    for i, rung in enumerate(LADDER):
        expected = None if i == 0 else names[i - 1]
        assert rung.requires_prior == expected, f"{rung.name} can be entered out of order"


# ── the refusal ────────────────────────────────────────────────────────────────


def test_a_fresh_state_can_enter_the_first_rung() -> None:
    ok, reason = LadderState().may_enter("negative")
    assert ok and "entry point" in reason


def test_the_full_grid_is_blocked_on_a_fresh_state() -> None:
    ok, reason = LadderState().may_enter("full")
    assert not ok
    assert "negative" in reason


def test_the_blocking_reason_names_where_you_actually_are() -> None:
    """ "you skipped rung 1" is actionable at 3am. "validation failed" is not."""
    _, reason = LadderState().may_enter("full")
    assert "blocked at negative" in reason


def test_check_order_raises_rather_than_returning() -> None:
    """Returning False gets ignored. Raising does not."""
    with pytest.raises(LadderOrderViolationError) as e:
        check_order(LadderState(), "full")
    assert "cannot be climbed from the top" in str(e.value)


def test_check_order_passes_once_the_prior_rung_has() -> None:
    s = LadderState()
    check_order(s, "negative")
    s.pass_rung("negative", "arm E confidence below threshold")
    check_order(s, "baselines")
    assert not s.may_enter("full")[0]
    s.pass_rung("baselines")
    check_order(s, "full")


def test_a_rung_cannot_be_passed_twice_to_become_a_key() -> None:
    """Passing a rung is not a vote; there is nothing to accumulate."""
    s = LadderState()
    s.pass_rung("negative")
    s.pass_rung("negative")
    assert s.as_dict()["passed"] == ["negative"]


def test_the_next_rung_is_the_first_unpassed_one_not_the_last() -> None:
    s = LadderState()
    assert s.first_unpassed().name == "negative"
    s.pass_rung("negative")
    assert s.first_unpassed().name == "baselines"
    s.pass_rung("baselines")
    s.pass_rung("full")
    assert s.first_unpassed() is None
    assert s.as_dict()["next"] is None


def test_notes_are_kept_because_a_bare_pass_flag_tells_you_nothing() -> None:
    s = LadderState()
    s.pass_rung("negative", "accuracy 0.19 < 0.40; reader may be ignoring context")
    assert "0.19" in s.as_dict()["notes"]["negative"]


def test_the_state_survives_a_round_trip() -> None:
    """The ladder has to outlive the process, or a restart resets the evidence."""
    s = LadderState()
    s.pass_rung("negative", "ok")
    restored = LadderState(passed=set(s.as_dict()["passed"]), notes=dict(s.as_dict()["notes"]))
    assert restored.may_enter("baselines")[0]
    assert restored.notes == s.notes
