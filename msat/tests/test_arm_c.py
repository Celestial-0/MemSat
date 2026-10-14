"""Arm C — identical set, varied order (PRD §P7).

THE CLAIM UNDER TEST
--------------------
Position is a property of the *ordering*, not of the retrieved set. Arm C holds the
set fixed and permutes it, so any accuracy difference is attributable to order.
"Lost in the middle" is only testable if the middle holds the same memories in every
condition.

THE EXIT GATE: ARM C MANIFESTS COMPLETE
---------------------------------------
Every (K, B, scale, reader) cell must render all four variants. A cell that silently
renders three is indistinguishable in the ledger from one that legitimately lacks a
variant, and the position effect is then computed over a fraction of the grid with
no marker.

THE TWO FAILURES THIS FILE IS MOST ABOUT
-----------------------------------------
1. `shuffled` without a seed is a single arbitrary ordering pretending to be a
   randomised condition.
2. `gold_first` with no gold sessions renders IDENTICALLY to `natural` - so on exactly
   the items where position was under test, the "gold first" arm is a duplicate of
   the control, and a position effect computed from it is a duplicate-condition
   artifact rather than a finding.

Both raise rather than falling back. A silent fallback produces a duplicate
condition, and a duplicate condition produces a number that looks like a result.
"""

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
    """Memories with DISTINCT gold SESSIONS at the given indices.

    Distinct sessions, not distinct chunks of one session: `gold_session_ranks`
    ranks over sessions in first-appearance order, so two chunks of a single gold
    session correctly collapse to ONE rank. A fixture built that way would assert
    about chunking, not position.
    """
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
    """The `gold_sessions` argument matching a `_memories` fixture.

    A tuple because it is compared against slices of `RenderedPrompt.session_ids`,
    which are tuples - comparing to a list would fail on type and read as a content
    mismatch."""
    return tuple(f"gold{i}" for i in sorted(gold_at))


def _session_order(mems: list[Memory]) -> list[str]:
    return [m.session_id for m in mems]


# ── the set is held; only the order changes ───────────────────────────────────


def test_every_variant_carries_the_same_set() -> None:
    """The defining property, and the one a refactor breaks quietly: sort one arm's
    memories in place, or drop one in a variant, and the arm still runs, still
    scores, and is now measuring two different retrieval outcomes."""
    gold_at = (0, 3)
    mems = _memories(8, gold_at=gold_at)
    variants = {v: reorder(mems, v, gold_sessions=_gold(gold_at), shuffle_seed=7) for v in VARIANTS}
    report = verify_same_set(variants)
    assert report["identical"] is True
    assert report["n_variants"] == 4
    assert all(v == [] for v in report["missing"].values()), report["missing"]


def test_the_orders_actually_differ() -> None:
    """`identical` compares SETS. Four variants that all rendered `natural` would
    pass it while measuring nothing."""
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
    """A stable hoist, not a re-sort. If it also reordered within groups, two
    variants would differ in more than position and the arm would be measuring
    ordering-of-ordering."""
    gold_at = (1, 4)
    mems = _memories(6, gold_at=gold_at)
    natural = _session_order(mems)
    got = _session_order(reorder(mems, PositionVariant.GOLD_FIRST, gold_sessions=_gold(gold_at)))
    assert tuple(s for s in got if s.startswith("gold")) == _gold(gold_at)
    assert tuple(s for s in natural if s.startswith("gold")) == _gold(gold_at)
    tails = [s for s in got if not s.startswith("gold")]
    assert tails == [s for s in natural if not s.startswith("gold")]


def test_reorder_never_mutates_its_input() -> None:
    """In-place sorting of the caller's list would corrupt every other arm that
    shares it, and the damage would appear as a position effect."""
    gold_at = (0,)
    mems = _memories(6, gold_at=gold_at)
    before = _session_order(mems)
    for v in VARIANTS:
        reorder(mems, v, gold_sessions=_gold(gold_at), shuffle_seed=3)
    assert _session_order(mems) == before


# ── the two refusals that stop duplicate conditions ───────────────────────────


def test_shuffled_without_a_seed_is_refused() -> None:
    """Otherwise it is a single arbitrary ordering presenting as a randomised
    condition - unreproducible and uninterpretable."""
    with pytest.raises(ValueError, match="needs a seed"):
        reorder(_memories(6), PositionVariant.SHUFFLED)


def test_gold_first_with_no_gold_is_refused() -> None:
    """It would render identically to `natural`, and the arm would report a position
    effect that is really a duplicate-condition artifact - on precisely the items
    where position was the thing under test."""
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


# ── the plan: every variant sees the same SEEDS ───────────────────────────────


def test_the_plan_is_balanced_across_variants() -> None:
    """An unbalanced arm is not a weaker arm, it is a different and unstated
    experiment, and it looks identical in the ledger."""
    plan = build_position_plan(n_replicates=3, seed=100)
    assert plan.balanced is True
    assert len(VARIANTS) == 4


def test_balance_compares_sets_not_sequences() -> None:
    """Comparing in order would flag a design that is balanced but staggered -
    which is correct and deliberate."""
    staggered = PositionPlan(
        n_replicates=2,
        seed=1,
        assignments={
            PositionVariant.NATURAL: (1, 2),
            PositionVariant.REVERSED: (2, 1),
        },
    )
    assert staggered.balanced is True


def test_a_genuinely_unbalanced_plan_is_reported_unbalanced() -> None:
    """The property must be able to read False, or it is not measuring anything."""
    unbalanced = PositionPlan(
        n_replicates=2,
        seed=1,
        assignments={
            PositionVariant.NATURAL: (1, 2),
            PositionVariant.REVERSED: (3, 4),
        },
    )
    assert unbalanced.balanced is False


def test_an_empty_plan_is_not_balanced() -> None:
    """Empty is not trivially balanced. `len(set()) == 1` would say it is."""
    assert PositionPlan(n_replicates=0, seed=1, assignments={}).balanced is False


def test_derived_seeds_differ_per_variant_but_cover_the_same_set() -> None:
    """The bug this function exists to prevent: deriving the seed per variant
    (`seed + hash(variant)`) gives every variant a different, non-overlapping set and
    silently unbalances the arm."""
    plan = build_position_plan(n_replicates=4, seed=50)
    for v in VARIANTS:
        assert plan.seed_for(v, 0) == 50
        assert plan.seed_for(v, 3) == 53
    assert len({plan.seed_for(v, 0) for v in VARIANTS}) == 1


def test_seed_lookup_wraps_rather_than_overrunning() -> None:
    """Replicate indices past the seed list cycle. Indexing directly would raise
    IndexError in the middle of a run, after the ladder has already committed cells."""
    plan = build_position_plan(n_replicates=2, seed=7)
    assert plan.seed_for(PositionVariant.SHUFFLED, 5) == 8


def test_an_unassigned_variant_is_a_named_error() -> None:
    plan = build_position_plan(n_replicates=2, seed=7, variants=(PositionVariant.NATURAL,))
    with pytest.raises(KeyError, match="no seeds assigned"):
        plan.seed_for(PositionVariant.SHUFFLED, 0)


def test_the_plan_serialises_with_its_balance_flag() -> None:
    """The flag travels with the plan into the manifest, so the balance check
    `msat run` performs is visible in the artifact afterwards."""
    d = build_position_plan(n_replicates=2, seed=9).as_dict()
    assert d["balanced"] is True
    assert d["seed"] == 9
    assert set(d["assignments"]) == {str(v) for v in VARIANTS}


# ── the arm renders, and its prompt records what it did ───────────────────────


#: The render layer requires explicit `system:` and `user:` blocks. A template
#: missing one renders as an empty prompt, which looks like a retrieval failure
#: rather than a broken template - so the format is exercised here rather than
#: assumed.
_TEMPLATE = "system: s\nuser: |\n  {{ question }}\n  {{ memories }}"


def _condition(b: float = 128.0) -> object:
    from msat.arms.render import ConditionKey

    return ConditionKey(arm="C", k=4, b=b, b_mode="budget", scale="S", reader="gemma")


def _rendered_order(prompt: object) -> list[Memory]:
    """Rebuild memories in RENDERED order from a prompt's parallel id tuples.

    Using the prompt's own `memory_ids`/`session_ids` - rather than the input list -
    is what makes the gold-rank assertion below a statement about the order the
    reader saw rather than the order retrieval produced."""
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
    """A rendered prompt that does not say which ordering produced it cannot be
    grouped by variant downstream, and every position result collapses into one
    undifferentiated average."""
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
    """Ranks must describe where a memory ended up. Carrying the pre-reorder ranks
    would make every gold-rank histogram a function of the original retrieval order
    while claiming to describe the rendered one."""
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
    """`RenderedPrompt.__post_init__` rejects misaligned parallel tuples, so this is
    guaranteed by construction - asserted anyway, because the guarantee is the
    thing a later refactor could quietly remove."""
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


# ── the exit gate: every cell renders every variant ───────────────────────────


def test_every_cell_renders_all_four_variants() -> None:
    """Arm C manifests complete. Checked by rendering each variant of one cell and
    confirming four distinct prompts that share a set - the property a partially
    rendered cell would break without any error being raised anywhere."""
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

    # And the evidence machinery can still find the gold in every variant: the
    # gold-rank histogram is only meaningful if ranks describe the RENDERED order.
    # Under `natural` the gold sits at 1 and 4; under `reversed` at 5 and 8. A rank
    # computation that ignored the reorder would report the same numbers for all
    # four, which is the bug this assertion exists to catch.
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
