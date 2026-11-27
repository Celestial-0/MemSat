from __future__ import annotations

import pytest

from msat.arms.position import (
    VARIANTS,
    PositionArm,
    PositionPlan,
    PositionVariant,
    build_position_plan,
    reorder,
    verify_same_set,
)
from msat.arms.render import Memory


def _memories(count: int, *, gold_at: tuple[int, ...] = (0,)) -> list[Memory]:
    gold = set(gold_at)
    return [
        Memory(
            memory_id=f"m{i}",
            session_id=(f"gold{i}" if i in gold else f"s{i}"),
            source_question_id="q1",
            text=f"body {i}",
            n_tokens=100,
        )
        for i in range(count)
    ]


def _gold(gold_at: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"gold{i}" for i in sorted(gold_at))


def _session_order(mems: list[Memory]) -> list[str]:
    return [m.session_id for m in mems]


def test_every_variant_carries_the_same_set() -> None:
    gold_at = (0, 3)
    mems = _memories(8, gold_at=gold_at)
    variants = {v: reorder(mems, v, gold_sessions=_gold(gold_at), shuffle_seed=7) for v in VARIANTS}
    report = verify_same_set(variants)
    assert report["identical"] is True
    assert report["n_variants"] == 4
    assert all(v == [] for v in report["missing"].values()), report["missing"]


def test_the_orders_actually_differ() -> None:
    gold_at = (0, 3)
    mems = _memories(8, gold_at=gold_at)
    orders = {
        v: tuple(_session_order(reorder(mems, v, gold_sessions=_gold(gold_at), shuffle_seed=7)))
        for v in VARIANTS
    }
    assert len(set(orders.values())) > 1, "the variants are indistinguishable"


def test_natural_is_the_identity() -> None:
    mems = _memories(6)
    assert _session_order(reorder(mems, PositionVariant.NATURAL)) == _session_order(mems)


def test_reversed_is_the_exact_mirror() -> None:
    mems = _memories(6)
    got = _session_order(reorder(mems, PositionVariant.REVERSED))
    assert got == list(reversed(_session_order(mems)))


def test_gold_first_hoists_only_gold() -> None:
    gold_at = (3, 5)
    mems = _memories(6, gold_at=gold_at)
    got = _session_order(reorder(mems, PositionVariant.GOLD_FIRST, gold_sessions=_gold(gold_at)))
    assert tuple(got[:2]) == _gold(gold_at)
    assert set(got) == set(_session_order(mems))


def test_gold_first_preserves_relative_order_within_each_group() -> None:
    gold_at = (1, 4)
    mems = _memories(6, gold_at=gold_at)
    natural = _session_order(mems)
    got = _session_order(reorder(mems, PositionVariant.GOLD_FIRST, gold_sessions=_gold(gold_at)))
    assert tuple(s for s in got if s.startswith("gold")) == _gold(gold_at)
    assert tuple(s for s in natural if s.startswith("gold")) == _gold(gold_at)
    tails = [s for s in got if not s.startswith("gold")]
    assert tails == [s for s in natural if not s.startswith("gold")]


def test_reorder_never_mutates_its_input() -> None:
    gold_at = (0,)
    mems = _memories(6, gold_at=gold_at)
    before = _session_order(mems)
    for v in VARIANTS:
        reorder(mems, v, gold_sessions=_gold(gold_at), shuffle_seed=3)
    assert _session_order(mems) == before


def test_shuffled_without_a_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a seed"):
        reorder(_memories(6), PositionVariant.SHUFFLED)


def test_gold_first_with_no_gold_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reorder(_memories(6), PositionVariant.GOLD_FIRST, gold_sessions=[])


def test_a_shuffled_variant_is_reproducible_from_its_seed() -> None:
    mems = _memories(10)
    a = _session_order(reorder(mems, PositionVariant.SHUFFLED, shuffle_seed=99))
    b = _session_order(reorder(mems, PositionVariant.SHUFFLED, shuffle_seed=99))
    c = _session_order(reorder(mems, PositionVariant.SHUFFLED, shuffle_seed=100))
    assert a == b, "same seed must give the same order, or the run is not replayable"
    assert a != c, "different seeds should give different orders"


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown position variant"):
        reorder(_memories(4), "sideways")  # type: ignore[arg-type]


def test_the_plan_is_balanced_across_variants() -> None:
    plan = build_position_plan(n_repetitions=3, seed=100)
    assert plan.balanced is True
    assert len(VARIANTS) == 4


def test_balance_compares_sets_not_sequences() -> None:
    staggered = PositionPlan(
        n_repetitions=2,
        seed=1,
        assignments={
            PositionVariant.NATURAL: (1, 2),
            PositionVariant.REVERSED: (2, 1),
        },
    )
    assert staggered.balanced is True


def test_a_genuinely_unbalanced_plan_is_reported_unbalanced() -> None:
    unbalanced = PositionPlan(
        n_repetitions=2,
        seed=1,
        assignments={
            PositionVariant.NATURAL: (1, 2),
            PositionVariant.REVERSED: (3, 4),
        },
    )
    assert unbalanced.balanced is False


def test_an_empty_plan_is_not_balanced() -> None:
    assert PositionPlan(n_repetitions=0, seed=1, assignments={}).balanced is False


def test_derived_seeds_differ_per_variant_but_cover_the_same_set() -> None:
    plan = build_position_plan(n_repetitions=4, seed=50)
    for v in VARIANTS:
        assert plan.seed_for(v, 0) == 50
        assert plan.seed_for(v, 3) == 53
    assert len({plan.seed_for(v, 0) for v in VARIANTS}) == 1


def test_seed_lookup_wraps_rather_than_overrunning() -> None:
    plan = build_position_plan(n_repetitions=2, seed=7)
    assert plan.seed_for(PositionVariant.SHUFFLED, 5) == 8


def test_an_unassigned_variant_is_a_named_error() -> None:
    plan = build_position_plan(n_repetitions=2, seed=7, variants=(PositionVariant.NATURAL,))
    with pytest.raises(KeyError, match="no seeds assigned"):
        plan.seed_for(PositionVariant.SHUFFLED, 0)


def test_the_plan_serialises_with_its_balance_flag() -> None:
    d = build_position_plan(n_repetitions=2, seed=9).as_dict()
    assert d["balanced"] is True
    assert d["seed"] == 9
    assert set(d["assignments"]) == {str(v) for v in VARIANTS}


_TEMPLATE = "system: s\nuser: |\n  {{ question }}\n  {{ memories }}"


def _condition(b: float = 128.0) -> object:
    from msat.store import Condition

    return Condition(arm="C", k=4, b=b, b_mode="budget", scale="S", reader="gemma")


def _rendered_order(prompt: object) -> list[Memory]:
    mems = [
        Memory(
            memory_id=mid,
            session_id=sid,
            source_question_id="q1",
            text="",
            n_tokens=0,
        )
        for mid, sid in zip(prompt.memory_ids, prompt.session_ids, strict=True)
    ]
    return mems


def test_the_arm_records_the_variant_it_used() -> None:
    gold_at = (0,)
    arm = PositionArm(template=_TEMPLATE, prompt_template_name="c")
    for v in VARIANTS:
        prompt = arm.render(
            "why?",
            _memories(4, gold_at=gold_at),
            _condition(),  # type: ignore[arg-type]
            variant=v,
            gold_sessions=_gold(gold_at),
            shuffle_seed=5,
        )
        assert prompt.position_variant == f"position/{v}"


def test_the_ranks_are_the_new_positions_not_the_old_ones() -> None:
    gold_at = (0,)
    arm = PositionArm(template=_TEMPLATE, prompt_template_name="c")
    prompt = arm.render(
        "why?",
        _memories(4, gold_at=gold_at),
        _condition(),  # type: ignore[arg-type]
        variant=PositionVariant.REVERSED,
        gold_sessions=_gold(gold_at),
    )
    assert prompt.ranks == (1, 2, 3, 4)
    assert prompt.session_ids[-1] == "gold0", "gold should now be LAST under REVERSED"


def test_the_session_ids_and_ranks_stay_aligned() -> None:
    gold_at = (0, 2)
    arm = PositionArm(template=_TEMPLATE, prompt_template_name="c")
    prompt = arm.render(
        "why?",
        _memories(5, gold_at=gold_at),
        _condition(),  # type: ignore[arg-type]
        variant=PositionVariant.GOLD_FIRST,
        gold_sessions=_gold(gold_at),
    )
    assert len(prompt.session_ids) == len(prompt.ranks) == len(prompt.memory_ids)
    assert sorted(prompt.ranks) == list(range(1, len(prompt.ranks) + 1))
    assert prompt.session_ids[:2] == _gold(gold_at)


def test_an_arm_with_no_template_refuses_rather_than_rendering_blank() -> None:
    with pytest.raises(ValueError, match="no template"):
        PositionArm().render("why?", _memories(3), _condition())  # type: ignore[arg-type]


def test_every_cell_renders_all_four_variants() -> None:
    from msat.arms.evidence import gold_session_ranks

    gold_at = (0, 3)
    mems = _memories(8, gold_at=gold_at)
    arm = PositionArm(template=_TEMPLATE, prompt_template_name="c")
    prompts = [
        arm.render(
            "why?",
            mems,
            _condition(),  # type: ignore[arg-type]
            variant=v,
            gold_sessions=_gold(gold_at),
            shuffle_seed=11,
        )
        for v in VARIANTS
    ]
    assert len(prompts) == len(VARIANTS) == 4
    assert len({p.position_variant for p in prompts}) == 4
    assert len({frozenset(p.memory_ids) for p in prompts}) == 1, "the set must be shared"

    seen: dict[str, tuple[int, ...]] = {}
    for p in prompts:
        ranks, missing = gold_session_ranks(_rendered_order(p), _gold(gold_at))
        assert missing == (), f"gold missing from {p.position_variant}"
        seen[p.position_variant] = ranks

    assert seen["position/natural"] == (1, 4)
    assert seen["position/reversed"] == (5, 8)
    assert seen["position/gold_first"] == (1, 2)
    assert len({seen[f"position/{v}"] for v in VARIANTS}) == 4, (
        f"four variants produced only {len(set(seen.values()))} distinct rank vectors"
    )
