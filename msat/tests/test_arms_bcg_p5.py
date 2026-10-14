"""Arms B, C and G, and the registry that loads them from `arms.yaml`.

The theme: each of these three can be made to *look* correct while measuring a
different thing than its name claims.

  B  padding that misses its token target, or pads with gold
  C  an ordering that quietly drops a memory, or a "random" arm with one seed
  G  a budget sweep written as a second renderer, which loses the cancellation

The registry tests cover the failure that makes all three possible at once: arms
are supposed to be DATA, so a hardcoded dispatch table would be invisible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.arms.budget import BudgetArm, assert_crossed, crossed_cells
from msat.arms.padding import (
    FILLER_FILENAME,
    FillerMissingError,
    FillerSource,
    GoldLeakageError,
    PaddingBudget,
    TokenTargetMissedError,
    assert_no_gold_leakage,
    leakage_features,
    pad_to_length,
    size_leakage_matrix,
)
from msat.arms.position import (
    PositionPlan,
    PositionVariant,
    build_position_plan,
    reorder,
    verify_same_set,
)
from msat.arms.registry import RegistryError, build_registry
from msat.arms.render import Memory
from msat.store.responses import ConditionKey

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
COND = ConditionKey(arm="A", k=4, b=512, b_mode="budget", scale="S", reader="gemma")


def _mem(session: str, *, tokens: int = 100, text: str = "") -> Memory:
    return Memory(
        memory_id=f"{session}#0",
        session_id=session,
        source_question_id="q",
        text=text or ("x" * (tokens * 4)),
        n_tokens=tokens,
    )


# ═══════════════════════════════════════════════════════════ ARM B ═════════════


def _filler(tmp_path: Path, n: int = 60, tokens: int = 100) -> FillerSource:
    """Enough padding for a 40% target on a 10,000-token corpus (4,000 tokens)."""
    p = tmp_path / FILLER_FILENAME
    p.write_text(json.dumps(["y" * (tokens * 4)] * n), encoding="utf-8")
    return FillerSource.load(p)


def test_the_only_legal_padding_source_is_the_filler_file(tmp_path: Path) -> None:
    """Blocker B3 is OPEN, so this is the behaviour that matters most in arm B."""
    with pytest.raises(FillerMissingError, match="B3"):
        FillerSource.load(tmp_path / FILLER_FILENAME)


def test_another_filename_is_refused_by_name(tmp_path: Path) -> None:
    """The set of places that could invent padding is a set of size one."""
    p = tmp_path / "whatever.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(FillerMissingError, match="nothing else"):
        FillerSource.load(p)


def test_an_empty_filler_is_refused_not_used(tmp_path: Path) -> None:
    p = tmp_path / FILLER_FILENAME
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(FillerMissingError, match="declares no sessions"):
        FillerSource.load(p)


def test_padding_reaches_its_token_target(tmp_path: Path) -> None:
    filler = _filler(tmp_path)
    budget = PaddingBudget(corpus_tokens=10_000, fraction=0.40)
    _out, report = pad_to_length([_mem("a"), _mem("b")], budget, filler)
    assert report["achieved_fraction"] == pytest.approx(0.40, abs=0.02)
    assert budget.lower <= report["achieved_tokens"] <= budget.upper
    assert report["n_retrieved"] == 2
    assert report["n_padded"] == report["n_padded"]


def test_padding_that_cannot_reach_the_target_raises(tmp_path: Path) -> None:
    """A 40% arm that lands at 10% is measuring a different B than its condition key."""
    filler = _filler(tmp_path, n=2, tokens=100)
    budget = PaddingBudget(corpus_tokens=100_000, fraction=0.40)
    with pytest.raises(TokenTargetMissedError, match="filler is too small"):
        pad_to_length([_mem("a")], budget, filler)


def test_padding_never_adds_gold(tmp_path: Path) -> None:
    """R9.10. Checked on SESSIONS, and asserted rather than warned."""
    filler = _filler(tmp_path)
    budget = PaddingBudget(corpus_tokens=10_000, fraction=0.40)
    out, _ = pad_to_length([_mem("a")], budget, filler, gold_sessions=[])
    assert all(m.source_question_id == "__filler__" or m.session_id == "a" for m in out)
    assert not any(m.session_id in {"gold1"} for m in out)


def test_a_gold_session_in_the_retrieved_set_is_an_error() -> None:
    """The label leaking through memory size is what makes the arm invalid."""
    with pytest.raises(GoldLeakageError, match="invalid, not merely noisy"):
        assert_no_gold_leakage([_mem("g1")], ["g1"], what="retrieved")


def test_a_gold_session_split_into_chunks_is_still_caught() -> None:
    """Matching on ids would let three colliding chunk ids through."""
    mems = [
        Memory("g1#0", "g1", "q", "a", 10),
        Memory("g1#1", "g1", "q", "b", 10),
        Memory("g1#2", "g1", "q", "c", 10),
    ]
    with pytest.raises(GoldLeakageError):
        assert_no_gold_leakage(mems, ["g1"], what="retrieved")


def test_no_gold_declared_is_not_an_error() -> None:
    assert_no_gold_leakage([_mem("a")], [], what="retrieved")


def test_the_budget_is_corpus_relative_so_it_means_the_same_on_both_scales() -> None:
    """S and M differ ~10x in session count. An absolute budget would mean one
    thing on S and something else on M, and R9.11 would measure the budget's drift."""
    small = PaddingBudget(corpus_tokens=122_324, fraction=0.40)
    large = PaddingBudget(corpus_tokens=1_209_518, fraction=0.40)
    # `int()` truncation makes the achieved fractions equal only to within a token's
    # worth, so this compares approximately rather than exactly.
    assert small.target_tokens / small.corpus_tokens == pytest.approx(
        large.target_tokens / large.corpus_tokens, abs=1e-5
    )
    assert large.target_tokens > small.target_tokens


def test_the_tolerance_is_two_percent() -> None:
    b = PaddingBudget(corpus_tokens=1000, fraction=0.40, tolerance=0.02)
    assert b.target_tokens == 400
    assert b.lower == 392
    assert b.upper == 408


# ── R9.10's size features ──────────────────────────────────────────────────────


def test_the_size_features_exist_for_the_classifier() -> None:
    """P9 fits `features -> condition label` and asserts the separation fails. The
    features have to exist for that assertion to be anything but vacuous."""
    f = leakage_features([_mem("a", tokens=10), _mem("b", tokens=30)])
    assert f == {
        "n_memories": 2.0,
        "total_tokens": 40.0,
        "mean_tokens": 20.0,
        "max_tokens": 30.0,
        "n_over_mean": 1.0,
    }


def test_size_features_of_nothing_are_zero_not_an_error() -> None:
    assert leakage_features([])["n_memories"] == 0.0


def test_arm_b_holding_length_constant_is_checked_against_arm_a() -> None:
    """The one thing checkable without a classifier: if B's total_tokens drifts
    from A's, length is not held constant and the arm measures nothing."""
    feats = {
        ("A", 1, 128, "budget", "S", "gemma"): leakage_features([_mem("a", tokens=100)]),
        ("B", 1, 128, "budget", "S", "gemma"): leakage_features([_mem("a", tokens=100)] * 4),
    }
    assert size_leakage_matrix(feats)["warnings"], "400 vs 100 tokens is not constant"


def test_matching_arm_b_length_produces_no_warning() -> None:
    feats = {
        ("A", 1, 128, "budget", "S", "gemma"): leakage_features([_mem("a", tokens=100)] * 2),
        ("B", 1, 128, "budget", "S", "gemma"): leakage_features([_mem("a", tokens=100)] * 2),
    }
    assert size_leakage_matrix(feats)["warnings"] == []


# ═══════════════════════════════════════════════════════════ ARM C ═════════════


def test_every_variant_holds_the_same_set() -> None:
    """The defining property of Arm C, and the one a refactor breaks quietly."""
    mems = [_mem("a"), _mem("b"), _mem("c"), _mem("gold")]
    variants = {
        v: reorder(mems, v, gold_sessions=["gold"], shuffle_seed=7) for v in PositionVariant
    }
    rep = verify_same_set(variants)
    assert rep["identical"] is True
    assert rep["missing"] == {}


def test_a_dropped_memory_is_reported_as_a_missing_set_difference() -> None:
    """Dropping one in one variant does not crash the arm; it silently converts it
    into a different retrieval condition."""
    mems = [_mem("a"), _mem("b"), _mem("c")]
    variants = {
        PositionVariant.NATURAL: reorder(mems, PositionVariant.NATURAL),
        PositionVariant.SHUFFLED: reorder(mems[:-1], PositionVariant.SHUFFLED, shuffle_seed=1),
    }
    rep = verify_same_set(variants)
    assert rep["identical"] is False
    assert any(rep["missing"].values())


def test_reversed_is_the_exact_mirror_of_natural() -> None:
    mems = [_mem("a"), _mem("b"), _mem("c")]
    assert [m.session_id for m in reorder(mems, PositionVariant.REVERSED)] == ["c", "b", "a"]


def test_shuffled_requires_a_seed() -> None:
    """A seedless "random" arm is one arbitrary ordering pretending to be randomised."""
    with pytest.raises(ValueError, match="needs a seed"):
        reorder([_mem("a")], PositionVariant.SHUFFLED)


def test_gold_first_without_gold_is_refused_not_silently_natural() -> None:
    """Falling back to `natural` would make this variant a DUPLICATE of natural on
    exactly the items where position was the thing under test."""
    with pytest.raises(ValueError, match="duplicate-condition artifact"):
        reorder([_mem("a"), _mem("b")], PositionVariant.GOLD_FIRST, gold_sessions=[])


def test_gold_first_hoists_the_evidence() -> None:
    mems = [_mem("x"), _mem("y"), _mem("gold")]
    out = reorder(mems, PositionVariant.GOLD_FIRST, gold_sessions=["gold"])
    assert next(m.session_id for m in out) == "gold"


def test_the_shuffle_is_reproducible_for_a_seed() -> None:
    mems = [_mem(s) for s in "abcdefgh"]
    a = reorder(mems, PositionVariant.SHUFFLED, shuffle_seed=11)
    b = reorder(mems, PositionVariant.SHUFFLED, shuffle_seed=11)
    assert [m.session_id for m in a] == [m.session_id for m in b]


def test_reorder_never_mutates_its_input() -> None:
    mems = [_mem("a"), _mem("b")]
    before = list(mems)
    reorder(mems, PositionVariant.REVERSED)
    assert mems == before


# ── balanced seeds ─────────────────────────────────────────────────────────────


def test_every_variant_sees_the_same_seed_set() -> None:
    """A single shuffled seed is a confounded condition wearing the name of a
    randomised one."""
    plan = build_position_plan(n_replicates=3, seed=100)
    assert plan.balanced is True
    assert set(plan.assignments[PositionVariant.SHUFFLED]) == set(
        plan.assignments[PositionVariant.NATURAL]
    )


def test_an_unbalanced_plan_is_detectable() -> None:
    """Compared as SETS, not sequences: staggered seeds are correct and deliberate,
    disjoint sets are the bug."""
    good = build_position_plan(n_replicates=2, seed=1)
    bad = PositionPlan(
        n_replicates=2,
        seed=1,
        assignments={
            PositionVariant.NATURAL: (1, 2),
            PositionVariant.SHUFFLED: (9, 10),  # disjoint -> unbalanced
        },
    )
    assert good.balanced and not bad.balanced


def test_seeds_wrap_when_replicates_exceed_the_assigned_range() -> None:
    plan = build_position_plan(n_replicates=2, seed=50)
    assert plan.seed_for(PositionVariant.SHUFFLED, 0) == 50
    assert plan.seed_for(PositionVariant.SHUFFLED, 5) == 51


def test_an_unassigned_variant_is_a_key_error() -> None:
    plan = PositionPlan(n_replicates=1, seed=1, assignments={})
    with pytest.raises(KeyError, match="no seeds assigned"):
        plan.seed_for(PositionVariant.SHUFFLED, 0)


# ═══════════════════════════════════════════════════════════ ARM G ═════════════


def test_a_and_g_share_one_renderer_so_formatting_changes_cancel() -> None:
    """Two near-identical renderers would make a formatting change look like a
    budget effect - the confound G exists to be able to rule out."""
    from msat.arms.render import SimpleArm

    assert issubclass(BudgetArm, SimpleArm)
    assert BudgetArm.render is not SimpleArm.render, "G overrides only to add fields"


def test_the_crossed_pair_in_code_matches_the_crossed_pair_in_config() -> None:
    import yaml

    raw = yaml.safe_load((CONFIGS / "arms.yaml").read_text(encoding="utf-8"))
    assert sorted(
        assert_crossed(registry_stub(), raw["crossed_grid"]["arms"])["declared_in_config"]
    ) == ["A", "G"]


def registry_stub():
    from msat.arms.render import ArmRegistry

    return ArmRegistry(arms={"A": None, "G": None, "B": None, "C": None})


def test_a_config_that_declares_a_different_crossed_pair_is_flagged() -> None:
    """`arms.yaml` is the single source of truth. If code hard-codes a different
    pair, one of them is stale and the interaction runs over the wrong cells."""
    assert assert_crossed(registry_stub(), ["A", "B"])["agree"] is False


def test_crossed_cells_is_the_full_crossing_and_nothing_derived() -> None:
    cells = crossed_cells([1, 2, 4], [128, 512])
    assert len(cells) == 6
    assert cells[0] == (1, 128.0)
    assert cells[-1] == (4, 512.0)


# ═══════════════════════════════════════════════════════ THE REGISTRY ══════════


class _Bundle:
    """The minimum a registry reads. Deliberately not the real ConfigBundle, so the
    tests pin the registry's own contract rather than the loader's."""

    def __init__(self, arms, crossed=("A", "G")) -> None:
        self.arms = type("A", (), {"arms": arms})()
        self.prompts = {
            f"arm_{a['id'].lower()}.j2": "system: s\nuser: |\n  {{ question }}" for a in arms
        }
        self.raw = {"crossed_grid": {"arms": list(crossed)}}


def _spec(i: str, enabled: bool = True, template: str | None = None) -> dict:
    return {
        "id": i,
        "isolates": "x",
        "enabled": enabled,
        "prompt_template": template if template is not None else f"arm_{i.lower()}.j2",
    }


def test_the_registry_builds_every_enabled_arm() -> None:
    reg, rep = build_registry(_Bundle([_spec(a) for a in "ABCDEFG"]))
    assert sorted(reg.arms) == ["A", "B", "C", "D", "E", "F", "G"]
    assert rep["n_built"] == 7
    assert rep["refused"] == {}


def test_a_disabled_arm_is_refused_with_a_reason() -> None:
    reg, rep = build_registry(_Bundle([_spec("A"), _spec("G"), _spec("F", enabled=False)]))
    assert "F" not in reg.arms
    assert "disabled" in rep["refused"]["F"]


def test_a_missing_template_is_an_error_not_an_empty_prompt() -> None:
    """An empty prompt renders to a plausible score for a condition never run."""
    b = _Bundle([_spec("A"), _spec("G"), _spec("D", template="absent.j2")])
    b.prompts.pop("absent.j2", None)
    with pytest.raises(RegistryError, match="never run"):
        build_registry(b)


def test_a_missing_crossed_half_refuses_to_build_a_plan() -> None:
    """Without both A and G there is no interaction term and no headline result."""
    b = _Bundle([_spec("A"), _spec("F")])
    with pytest.raises(RegistryError, match="no interaction term"):
        build_registry(b)


def test_an_arm_with_no_renderer_is_refused_not_skipped_silently() -> None:
    reg, rep = build_registry(_Bundle([_spec("A"), _spec("G"), _spec("H")]))
    assert "H" not in reg.arms
    assert "no renderer is registered" in rep["refused"]["H"]


def test_the_registry_reports_the_crossed_verdict() -> None:
    _, rep = build_registry(_Bundle([_spec("A"), _spec("G")]))
    assert rep["crossed"]["agree"] is True
    assert rep["crossed"]["missing_from_registry"] == []


def test_the_shipped_arms_yaml_builds() -> None:
    """The real config, end to end. Every arm the PRD declares must render."""
    from msat.config import load_bundle

    reg, rep = build_registry(load_bundle(CONFIGS))
    assert rep["n_built"] == 7, rep["refused"]
    assert set("ABCDEFG") <= set(reg.arms)
    assert reg.get("A").template.strip(), "arm A's template must not be empty"


def test_every_declared_arm_has_a_prompt_file() -> None:
    import yaml

    raw = yaml.safe_load((CONFIGS / "arms.yaml").read_text(encoding="utf-8"))
    for entry in raw["arms"]:
        path = CONFIGS / "prompts" / entry["prompt_template"]
        assert path.exists(), f"arm {entry['id']} names a template that does not exist"


# ── B, E and F: config-block arms whose prompt IS the treatment's ───────────────


def test_the_config_arms_are_a_declared_set_not_a_convention() -> None:
    from msat.arms.registry import DELEGATING_ARMS, PROMPT_ARMS

    assert set(DELEGATING_ARMS) == {"B", "E", "F"}
    assert set(DELEGATING_ARMS).isdisjoint(PROMPT_ARMS)


def test_e_is_structurally_identical_to_a() -> None:
    """The identity is a TYPE-level fact, not a test that could be forgotten.

    `arm_e.j2` is arm E's CONFIG block, not a prompt. A collapse of the negative
    control has to be attributable to the absence of evidence and nothing else
    (PRD §R4.3), so arm E renders through arm A's template BY CONSTRUCTION rather
    than by two templates happening to agree today.
    """
    from msat.arms.render import DelegatingArm

    reg, rep = build_registry(_Bundle([_spec("A"), _spec("G"), _spec("E")]))
    assert rep["delegates"] == {"E": "A"}
    assert isinstance(reg.get("E"), DelegatingArm)
    assert reg.get("E").template == reg.get("A").template

    mems = [_mem("a"), _mem("b")]
    a_out = reg.get("A").render(
        "q?", mems, ConditionKey(arm="A", k=2, b=512, b_mode="b", scale="S", reader="gemma")
    )
    e_out = reg.get("E").render(
        "q?", mems, ConditionKey(arm="E", k=2, b=512, b_mode="b", scale="S", reader="gemma")
    )
    assert a_out.system == e_out.system
    assert a_out.user == e_out.user
    assert e_out.arm == "E", "the identity differs; only the PROMPT is shared"


def test_a_disabled_parent_refuses_the_whole_plan_before_it_can_be_reported() -> None:
    """A disabled A leaves E with no template to inherit, and also removes half the
    crossed grid - so the refusal fires at the cross, which is the stronger check."""
    with pytest.raises(RegistryError, match="no interaction term"):
        build_registry(_Bundle([_spec("A", enabled=False), _spec("G"), _spec("E")]))


def test_the_registry_names_which_arms_are_prompt_arms_and_which_are_config() -> None:
    """So a reader of `arms.yaml` does not go looking for `arm_e.j2` as a prompt."""
    _, rep = build_registry(_Bundle([_spec(a) for a in "ABCDEFG"]))
    assert rep["prompt_arms"] == ["A", "C", "D", "G"]
    assert rep["config_arms"] == ["B", "E", "F"]
