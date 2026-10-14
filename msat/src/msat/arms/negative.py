"""Arm E — the NEGATIVE CONTROL (PRD §R4.3, §G5).

THE GATE ON THE WHOLE EXPERIMENT
--------------------------------
Arm E retrieves memories from OTHER questions, count-matched to the Arm A
condition so that only the CONTENT differs. Its accuracy should collapse to
chance. If it does not, the pipeline is not respecting the condition, and every
other number in the project is suspect.

It runs FIRST in every smoke test and the ladder is never reversed (E -> A/D/F ->
full grid). That ordering is the whole point: this is the cheapest possible moment
to discover the pipeline is measuring nothing, and it costs one small arm.

WHY COUNT-MATCHING MATTERS
--------------------------
Without it, a collapse could be explained by the control getting fewer memories
than the treatment. Count-matched means the ONLY difference between Arm A and Arm
E is whether the evidence is the right evidence. A difference in accuracy is then
attributable to evidence, and to nothing else.

WHY THE TEMPLATE MUST BE IDENTICAL
----------------------------------
`arm_e.j2` and `arm_a.j2` carry byte-identical `system:` and `user:` blocks. If they
differed, a collapse could be attributed to the template rather than to the absence
of evidence — and the experiment would have a confound in its control.
`test_the_control_template_is_byte_identical_to_the_treatment` pins it.

THE THRESHOLD IS A NUMBER
-------------------------
`pass_threshold_accuracy_below` is `null` in the config, and null means the arm
CANNOT BE EVALUATED — not that it should be evaluated loosely. A threshold of
"far below" cannot be failed, so it cannot be passed either. Until a pilot sets a
number, `evaluate()` returns INCONCLUSIVE and `go_no_go` refuses to proceed.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from msat.arms.render import Memory, RenderedPrompt, SimpleArm
from msat.store.responses import ConditionKey


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """The `control:` block of `arm_e.j2`."""

    match: str = "count"
    source: str = "other_questions"
    seed: int = 20261012
    exclude_gold: bool = True
    exclude_overlap_with_target_question: bool = True
    pass_threshold_accuracy_below: float | None = None
    pass_threshold_recall_below: float | None = None

    @classmethod
    def from_template(cls, source: str) -> ControlConfig:
        import yaml

        from msat.arms.render import strip_leading_jinja_comment

        parsed = yaml.safe_load(strip_leading_jinja_comment(source)) or {}
        block = parsed.get("control") or {}
        return cls(
            match=str(block.get("match", "count")),
            source=str(block.get("source", "other_questions")),
            seed=int(block.get("seed", 20261012)),
            exclude_gold=bool(block.get("exclude_gold", True)),
            exclude_overlap_with_target_question=bool(
                block.get("exclude_overlap_with_target_question", True)
            ),
            pass_threshold_accuracy_below=block.get("pass_threshold_accuracy_below"),
            pass_threshold_recall_below=block.get("pass_threshold_recall_below"),
        )

    @property
    def thresholds_measurable(self) -> bool:
        return (
            self.pass_threshold_accuracy_below is not None
            and self.pass_threshold_recall_below is not None
        )


@dataclass
class PooledMemory:
    """A cross-question pool. The only thing Arm E reads from outside its args."""

    by_question: Mapping[str, Sequence[Memory]]

    def donors(
        self,
        target_question_id: str,
        *,
        exclude_gold: bool,
        gold_by_question: Mapping[str, frozenset[str]] | None = None,
    ) -> list[Memory]:
        """Memories from questions OTHER than the target.

        Gold is excluded even from the DONORS, not just from the target's own draw.
        Pulling another question's gold in would make the control partially correct
        for reasons that have nothing to do with the manipulation.

        `gold_by_question` maps qid -> `answer_session_ids`, and gold is DERIVED by
        session id. A donor whose question is absent from the map has no declared
        gold, so nothing of its own is excluded - which is stated rather than
        assumed, because silently treating "unknown gold" as "no gold" for the
        negative control would let real evidence into the condition the whole
        pipeline exists to check.
        """
        gold_by_question = gold_by_question or {}
        out: list[Memory] = []
        for qid, memories in self.by_question.items():
            if qid == target_question_id:
                continue
            gold = gold_by_question.get(qid, frozenset())
            for m in memories:
                if exclude_gold and m.is_gold_for(gold):
                    continue
                out.append(m)
        return out


@dataclass
class ControlOutcome:
    verdict: Verdict
    accuracy: float | None = None
    recall: float | None = None
    n_questions: int = 0
    reason: str = ""
    thresholds: dict[str, float | None] = field(default_factory=dict)

    @property
    def gating(self) -> bool:
        """INCONCLUSIVE gates exactly like FAIL. Proceeding on "we don't know" is
        how a broken pipeline gets twelve days of GPU time."""
        return self.verdict is not Verdict.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "accuracy": self.accuracy,
            "recall": self.recall,
            "n_questions": self.n_questions,
            "reason": self.reason,
            "thresholds": self.thresholds,
            "gating": self.gating,
        }


def draw_control_memories(
    pool: PooledMemory,
    target_question_id: str,
    k: int,
    config: ControlConfig,
    *,
    seed: int | None = None,
    gold_by_question: Mapping[str, frozenset[str]] | None = None,
) -> list[Memory]:
    """Count-matched random memories from OTHER questions.

    Seeded on `(seed, target_question_id)` so the draw is reproducible PER QUESTION,
    not just per run — a run resumed later must produce the same control as the run
    it resumed, or the two halves are not comparable.

    `gold_by_question` is REQUIRED when `config.exclude_gold` is true, because
    gold is derived from session ids and cannot be defaulted. A control that
    silently kept gold because the caller forgot a mapping is the one failure this
    whole arm exists to catch.
    """
    if config.exclude_gold and gold_by_question is None:
        raise ValueError(
            "arm E sets exclude_gold=True, so `gold_by_question` is required: gold is "
            "derived from `answer_session_ids` by session id, and there is no flag to "
            "default. Omitting it would draw another question's gold into the "
            "negative control."
        )
    donors = pool.donors(
        target_question_id,
        exclude_gold=config.exclude_gold,
        gold_by_question=gold_by_question,
    )
    if len(donors) < k:
        raise ValueError(
            f"control for {target_question_id!r} needs {k} donor memories but only "
            f"{len(donors)} are available across other questions. Arm E cannot be "
            f"count-matched at this K, so it cannot be run - and an under-filled "
            f"control would collapse for the wrong reason."
        )
    rng = random.Random(f"{seed if seed is not None else config.seed}:{target_question_id}")
    return rng.sample(donors, k)


def render_control(
    arm: SimpleArm,
    question: str,
    pool: PooledMemory,
    target_question_id: str,
    gold_sessions: frozenset[str],
    condition: ConditionKey,
    config: ControlConfig,
    *,
    budget_tokens: int | None = None,
    qid: str = "",
    gold_by_question: Mapping[str, frozenset[str]] | None = None,
) -> RenderedPrompt:
    """Render one Arm E prompt. Count-matched to the Arm A condition.

    `gold_sessions` is the TARGET's `answer_session_ids` - sessions, not chunk ids.
    The post-draw check re-verifies on sessions too, so a donor that slipped past
    `PooledMemory.donors` is still caught here by a different code path.
    """
    memories = draw_control_memories(
        pool, target_question_id, condition.k, config, gold_by_question=gold_by_question
    )
    leaked = [m.session_id for m in memories if m.session_id in gold_sessions]
    if leaked and config.exclude_gold:
        raise AssertionError(
            f"control for {target_question_id!r} pulled GOLD evidence {leaked}. A "
            f"control that can see the answer is not a control."
        )
    return arm.render(
        question,
        memories,
        condition,
        budget_tokens=budget_tokens,
        position_variant="control",
        qid=qid,
    )


def evaluate(
    outcome_accuracy: float | None,
    outcome_recall: float | None,
    config: ControlConfig,
    *,
    n_questions: int = 0,
) -> ControlOutcome:
    """Compare measured control performance against the NUMERIC thresholds.

    Both accuracy AND recall must be below. Recall is the harder check: a plausible
    -looking recall from a count-matched random set is the signature of a retrieval
    path that is not respecting the condition at all.
    """
    thresholds = {
        "accuracy": config.pass_threshold_accuracy_below,
        "recall": config.pass_threshold_recall_below,
    }
    if not config.thresholds_measurable or outcome_accuracy is None or outcome_recall is None:
        return ControlOutcome(
            verdict=Verdict.INCONCLUSIVE,
            accuracy=outcome_accuracy,
            recall=outcome_recall,
            n_questions=n_questions,
            reason=(
                "pass thresholds are null, so Arm E cannot be EVALUATED. null does "
                "not mean 'loosely' - it means the gate cannot be passed, and the "
                "gate gates."
            ),
            thresholds=thresholds,
        )

    acc_ok = outcome_accuracy < float(config.pass_threshold_accuracy_below)  # type: ignore[arg-type]
    rec_ok = outcome_recall < float(config.pass_threshold_recall_below)  # type: ignore[arg-type]
    ok = acc_ok and rec_ok
    return ControlOutcome(
        verdict=Verdict.PASS if ok else Verdict.FAIL,
        accuracy=outcome_accuracy,
        recall=outcome_recall,
        n_questions=n_questions,
        reason=(
            f"control collapsed as required: accuracy {outcome_accuracy:.3f} < "
            f"{config.pass_threshold_accuracy_below} and recall {outcome_recall:.3f} < "
            f"{config.pass_threshold_recall_below}"
            if ok
            else (
                f"control did NOT collapse: accuracy {outcome_accuracy:.3f} "
                f"(threshold {config.pass_threshold_accuracy_below}), recall "
                f"{outcome_recall:.3f} (threshold {config.pass_threshold_recall_below}). "
                f"Stop - the pipeline is not respecting the condition."
            )
        ),
        thresholds=thresholds,
    )


def go_no_go(outcome: ControlOutcome) -> dict[str, Any]:
    """The verdict that gates the run. Written to `go_no_go.json`."""
    return {
        "gate": "arm_e_negative_control",
        "requirement": "PRD §R4.3 / §G5",
        **outcome.as_dict(),
        "proceed": not outcome.gating,
        "next": (
            "Arms A / D / F, then the full grid."
            if not outcome.gating
            else "STOP. Do not launch the grid until the negative control collapses "
            "with a numeric threshold in place."
        ),
    }
