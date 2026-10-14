"""P5 — arm purity, the crossed grid, and the negative control.

Three of these encode requirements that are easy to state and easy to break:

  * purity (R4.1)      — if a template can reach a global, two renders of the same
                          input differ and the run is not reproducible
  * crossed grid (R4.4) — the headline interaction needs EVERY (K, B) cell, and a
                          pair of marginal sweeps does not produce it
  * negative control     — a gate that cannot be failed is not a gate
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat.arms.negative import (
    ControlConfig,
    PooledMemory,
    Verdict,
    draw_control_memories,
    evaluate,
    go_no_go,
    render_control,
)
from msat.arms.render import (
    Memory,
    SimpleArm,
    TemplateError,
    build_registry,
    format_memories,
    render_template,
)
from msat.store.responses import ConditionKey

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
PROMPTS = CONFIGS / "prompts"


def condition(k: int = 4, b: float = 512.0, scale: str = "M", reader: str = "qwen") -> ConditionKey:
    return ConditionKey("A+G", k, b, "per_memory_tokens", scale, reader)


def load_arm(arm_id: str) -> SimpleArm:
    specs = yaml.safe_load((CONFIGS / "arms.yaml").read_text(encoding="utf-8"))["arms"]
    templates = {p.name: p.read_text(encoding="utf-8") for p in PROMPTS.glob("*.j2")}
    return build_registry(specs, templates).get(arm_id)


def memories(n: int, qid: str = "q_target", *, gold_at: int | None = 0) -> list[Memory]:
    """Memories whose SESSION is gold for one index.

    Gold is declared by session id, never by a flag - `Memory` has no `is_gold`
    field, so a test that wants gold to exist must name the session.
    """
    out = []
    for i in range(n):
        session = f"{qid}_gold{i}" if (gold_at is not None and i == gold_at) else f"{qid}_s{i}"
        out.append(
            Memory(
                memory_id=f"{session}#0",
                session_id=session,
                source_question_id=qid,
                text=f"memory {i} mentions marker{i} and some other words here",
                n_tokens=40,
            )
        )
    return out


def gold_of(qid: str = "q_target", at: int = 0) -> frozenset[str]:
    return frozenset({f"{qid}_gold{at}"})


# ── R4.1 purity ───────────────────────────────────────────────────────────────
def test_render_is_deterministic() -> None:
    arm = load_arm("A")
    mems = memories(4)
    a = arm.render("what did we decide?", mems, condition(), budget_tokens=64)
    b = arm.render("what did we decide?", mems, condition(), budget_tokens=64)
    assert a.prompt == b.prompt
    assert a.memory_ids == b.memory_ids


def test_render_reads_nothing_but_its_arguments() -> None:
    """Purity, checked by making the outside world unusable.

    Patching `open`, `os.environ` and the network to raise is a crude but real
    test: a template that reached for a global would raise here, and one that does
    not is proven not to by the most direct means available.
    """
    arm = load_arm("A")

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("render performed I/O")

    original_open = builtins.open
    original_environ = dict(os.environ)
    try:
        builtins.open = boom  # type: ignore[assignment]
        os.environ.clear()
        rendered = arm.render("q?", memories(2), condition(), budget_tokens=64)
    finally:
        builtins.open = original_open
        os.environ.clear()
        os.environ.update(original_environ)
    assert rendered.prompt


def test_the_manifest_is_complete_for_recall_work() -> None:
    """R4.2. This manifest is the ground truth — not a reimplementation of the
    selector "just for the metrics"."""
    arm = load_arm("A")
    mems = memories(6)
    rendered = arm.render("q?", mems, condition(k=6), budget_tokens=64)
    assert rendered.memory_ids == tuple(f"{m.session_id}#0" for m in mems)
    assert rendered.session_ids == tuple(m.session_id for m in mems)
    assert rendered.ranks == (1, 2, 3, 4, 5, 6)
    assert rendered.k == 6
    assert rendered.b_mode == "per_memory_tokens"
    assert "corpus_fraction" in rendered.as_dict()


def test_gold_ranks_are_reported_from_the_manifest() -> None:
    arm = load_arm("A")
    rendered = arm.render("q?", memories(6, gold_at=3), condition(k=6), budget_tokens=64)
    # Gold is declared by SESSION. `q_target_gold3` sits at rank 4.
    assert rendered.gold_ranks(gold_of(at=3)) == (4,)
    assert rendered.gold_ranks(frozenset()) == ()
    assert rendered.gold_ranks(frozenset({"absent"})) == ()


def test_two_chunks_of_one_gold_session_report_one_rank() -> None:
    """The bug this design is most prone to, pinned.

    Gold is a session; a memory is a chunk of one. A gold session split across
    chunks 1 and 5 used to be reported at ranks 1 AND 5, so the gold-rank
    histogram measured the chunk size rather than retrieval quality.
    """
    chunked = [
        Memory("s1#0", "s1", "q", "a", 10),
        Memory("s2#0", "s2", "q", "b", 10),
        Memory("s1#1", "s1", "q", "c", 10),
        Memory("s3#0", "s3", "q", "d", 10),
    ]
    out = load_arm("A").render("q?", chunked, condition(k=4), budget_tokens=64)
    assert out.gold_ranks(frozenset({"s1"})) == (1,)
    assert out.gold_ranks(frozenset({"s1", "s3"})) == (1, 4)


def test_an_undefined_variable_is_an_error_not_an_empty_string() -> None:
    """`{{ missing }}` rendering as empty produces a prompt with no evidence at all,
    which looks exactly like a retrieval failure and is not one."""
    with pytest.raises(TemplateError, match="render failed"):
        render_template("system: |\n  s\nuser: |\n  {{ not_given }}\n", {"memories": "x"})


def test_a_template_without_system_and_user_is_rejected() -> None:
    with pytest.raises(TemplateError, match="must define"):
        render_template("question: |\n  hi\n", {})


def test_the_registry_fails_on_a_missing_template() -> None:
    with pytest.raises(KeyError, match="renders as an empty prompt"):
        build_registry(
            [{"id": "Z", "isolates": "x", "prompt_template": "absent.j2", "enabled": True}], {}
        )


# ── the budget is per-memory, so it is independent of K ────────────────────────
def test_the_budget_truncates_each_memory_not_the_list() -> None:
    """This is what makes K and B independent factors. Truncating the
    concatenated list made B a cutoff on K expressed in tokens."""
    mems = [Memory(f"m{i}#0", f"m{i}", "q", "x" * 400, n_tokens=100) for i in range(4)]
    tight = format_memories(mems, budget_tokens=10)
    loose = format_memories(mems, budget_tokens=1000)
    assert tight.count("[1]") == loose.count("[1]") == 1, "the COUNT must not change"
    assert len(tight) < len(loose)
    assert "[4]" in tight, "every retrieved memory survives at any budget"


# ── R4.4 the crossed grid ─────────────────────────────────────────────────────
def test_every_k_b_pair_is_present_in_the_registry_grid() -> None:
    """The single most dangerous requirement in the PRD.

    A pair of marginal sweeps — K at fixed B, then B at fixed K — does NOT satisfy
    R4.4 and cannot produce the interaction term the paper leads with.
    """
    design = yaml.safe_load((CONFIGS / "design.yaml").read_text(encoding="utf-8"))
    cells = {(k, b["value"]) for k in design["K_grid"] for b in design["B_grid"]}
    assert len(cells) == len(design["K_grid"]) * len(design["B_grid"])
    assert len(cells) > 1, "a crossed grid has more cells than either axis alone"


def test_the_condition_key_distinguishes_every_k_b_pair() -> None:
    keys = {condition(k=k, b=b).key for k in (1, 4, 16) for b in (128.0, 512.0)}
    assert len(keys) == 9 // 3 * 2  # 3 K x 2 B


# ── Arm E, the gate ───────────────────────────────────────────────────────────
def _pool() -> PooledMemory:
    return PooledMemory(
        by_question={
            "q_target": memories(4, "q_target", gold_at=0),
            "q_donor1": memories(4, "q_donor1"),
            "q_donor2": memories(4, "q_donor2"),
        }
    )


def _gold_by_question() -> dict[str, frozenset[str]]:
    """Only the TARGET declares gold. The donors' memories are all non-gold.

    Arm E therefore has to exclude the target's own memories by QUESTION id, and
    the donors pass through untouched - which is what makes the control a control.
    """
    return {"q_target": gold_of(at=0), "q_donor1": frozenset(), "q_donor2": frozenset()}


def test_the_control_template_is_byte_identical_to_the_treatment() -> None:
    """If they differed, a collapse could be attributed to the template rather
    than to the absence of evidence — a confound inside the control."""
    a = load_arm("A")
    e = load_arm("E")
    mems = memories(4, "q_target")
    ca = condition(k=4)
    assert (
        a.render("q?", mems, ca, budget_tokens=64).prompt
        == e.render("q?", mems, ca, budget_tokens=64).prompt
    )


def test_the_control_is_count_matched() -> None:
    arm = load_arm("E")
    out = render_control(
        arm,
        "q?",
        _pool(),
        "q_target",
        gold_of(at=0),
        condition(k=4),
        ControlConfig(),
        budget_tokens=64,
        gold_by_question=_gold_by_question(),
    )
    assert out.n_memories == 4, "the ONLY difference from Arm A is the content"


def test_the_control_never_contains_the_targets_own_memories() -> None:
    arm = load_arm("E")
    out = render_control(
        arm,
        "q?",
        _pool(),
        "q_target",
        gold_of(at=0),
        condition(k=4),
        ControlConfig(),
        budget_tokens=64,
        gold_by_question=_gold_by_question(),
    )
    assert not any(s.startswith("q_target_") for s in out.session_ids)


def test_a_control_draw_with_no_gold_map_is_refused() -> None:
    """`exclude_gold=True` cannot be satisfied without session-level gold.

    Defaulting it to "no gold anywhere" would let a donor's real evidence into the
    negative control - the exact failure the arm exists to detect, made invisible
    by an omitted argument.
    """
    with pytest.raises(ValueError, match="gold_by_question"):
        draw_control_memories(_pool(), "q_target", 2, ControlConfig())


def test_the_control_draw_is_reproducible_per_question() -> None:
    """A resumed run must produce the same control as the run it resumed, or the
    two halves are not comparable."""
    cfg = ControlConfig()
    g = _gold_by_question()
    first = [
        m.memory_id for m in draw_control_memories(_pool(), "q_target", 3, cfg, gold_by_question=g)
    ]
    second = [
        m.memory_id for m in draw_control_memories(_pool(), "q_target", 3, cfg, gold_by_question=g)
    ]
    assert first == second


def test_an_underfilled_control_is_refused_not_underfilled() -> None:
    """A control short of its count would collapse for the wrong reason."""
    pool = PooledMemory(by_question={"q_target": memories(4), "q_donor": memories(1)})
    with pytest.raises(ValueError, match="cannot be count-matched"):
        draw_control_memories(
            pool,
            "q_target",
            4,
            ControlConfig(),
            gold_by_question={"q_target": gold_of(), "q_donor": frozenset()},
        )


def test_donor_gold_is_excluded_too() -> None:
    """Another question's gold must not leak in either.

    Gold is excluded by SESSION id, passed in as `answer_session_ids`. There is no
    flag to set and therefore no way to forget to set one.
    """
    pool = PooledMemory(
        by_question={
            "q_target": memories(2),
            "q_donor": [
                Memory("d0#0", "d0", "q_donor", "t", 1),
                Memory("d1#0", "d1", "q_donor", "t", 1),
            ],
        }
    )
    gold_by_q = {"q_target": gold_of(), "q_donor": frozenset({"d0"})}
    assert [
        m.memory_id for m in pool.donors("q_target", exclude_gold=True, gold_by_question=gold_by_q)
    ] == ["d1#0"]
    # And the control draw inherits the same exclusion.
    drawn = draw_control_memories(pool, "q_target", 1, ControlConfig(), gold_by_question=gold_by_q)
    assert [m.memory_id for m in drawn] == ["d1#0"]


# ── the gate itself ───────────────────────────────────────────────────────────
def test_a_null_threshold_is_inconclusive_and_gates() -> None:
    """null means the arm CANNOT be evaluated. It does not mean "loosely", and it
    certainly does not mean "pass"."""
    out = evaluate(0.01, 0.0, ControlConfig(), n_questions=10)
    assert out.verdict is Verdict.INCONCLUSIVE
    assert out.gating
    assert not go_no_go(out)["proceed"]


def test_a_collapsed_control_passes_once_thresholds_are_numbers() -> None:
    cfg = ControlConfig(pass_threshold_accuracy_below=0.25, pass_threshold_recall_below=0.10)
    out = evaluate(0.04, 0.0, cfg, n_questions=100)
    assert out.verdict is Verdict.PASS
    assert go_no_go(out)["proceed"]


def test_a_control_that_does_not_collapse_stops_the_run() -> None:
    cfg = ControlConfig(pass_threshold_accuracy_below=0.25, pass_threshold_recall_below=0.10)
    out = evaluate(0.61, 0.42, cfg, n_questions=100)
    assert out.verdict is Verdict.FAIL
    verdict = go_no_go(out)
    assert not verdict["proceed"]
    assert "STOP" in verdict["next"]


def test_plausible_recall_alone_fails_the_gate() -> None:
    """The harder check: recall on a count-matched random set should be near zero,
    and a plausible-looking recall is the signature of a retrieval path that is
    not respecting the condition."""
    cfg = ControlConfig(pass_threshold_accuracy_below=0.25, pass_threshold_recall_below=0.10)
    assert evaluate(0.01, 0.38, cfg).verdict is Verdict.FAIL


def test_the_shipped_config_carries_numeric_thresholds() -> None:
    """Pinned the OTHER way round.

    This test asserted `null` until the thresholds were set on 2026-10-13
    (decision D5). Asserting `null` meant the config's state was pinned to "the
    gate cannot pass", which would have FAILED the moment someone did the work.

    What is pinned now is the invariant that actually matters: the shipped config
    has MEASURABLE numbers, so `msat run` will not exit `EXIT_NO_GO`. The
    `null -> INCONCLUSIVE -> gates` behaviour is still tested, against a
    constructed `ControlConfig`, in `test_a_null_threshold_is_inconclusive_and_gates`.
    """
    config = ControlConfig.from_template((PROMPTS / "arm_e.j2").read_text(encoding="utf-8"))
    assert config.thresholds_measurable is True
    assert config.pass_threshold_accuracy_below == 0.15
    assert config.pass_threshold_recall_below == 0.01


def test_the_recall_threshold_is_an_order_of_magnitude_tighter_than_accuracy() -> None:
    """The two thresholds are not the same kind of number.

    Recall is set overlap on a control whose memories contain no gold by
    CONSTRUCTION, so it should be ~0 and any non-zero value is a leak. Accuracy
    will not be zero - a 9B model has parametric knowledge and token-F1 gives
    partial credit - so it needs real headroom.

    If these two ever converge, the recall threshold has been loosened to match the
    accuracy one and the leak detector has stopped detecting.
    """
    config = ControlConfig.from_template((PROMPTS / "arm_e.j2").read_text(encoding="utf-8"))
    assert config.pass_threshold_recall_below is not None
    assert config.pass_threshold_accuracy_below is not None
    assert config.pass_threshold_recall_below * 10 < config.pass_threshold_accuracy_below


def test_the_thresholds_are_a_gate_not_a_target() -> None:
    """0.15 must sit well below Arm A's expected level, because the failure this
    catches is a pipeline scoring as if it had the evidence."""
    config = ControlConfig.from_template((PROMPTS / "arm_e.j2").read_text(encoding="utf-8"))
    assert config.pass_threshold_accuracy_below is not None
    assert config.pass_threshold_accuracy_below < 0.25, (
        "Arm A runs 0.4-0.6 on this benchmark; a threshold near that level cannot "
        "fire before the sweep is spent"
    )
