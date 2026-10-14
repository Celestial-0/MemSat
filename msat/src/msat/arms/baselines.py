"""Arms D and F — the two baselines the budgeted arms are measured against.

ARM D — TRUE FULL CONTEXT (PRD §R4.4)
------------------------------------
Every session in the conversation, unbudgeted. This is the reference point: it
answers "does holding a budget cost you anything?", and on M it cannot be run at
all (M averages ~1.2M tokens/question).

ARM D IS INFEASIBLE ON THIS HARDWARE, AND THAT IS A RESULT
----------------------------------------------------------
Measured against the panel in `readers.yaml` on an RTX 4060 (8 GB):

    S full corpus     122,324 tokens   (48 sessions median)
    gemma  ceiling     66,160          short by 56,164
    phi    ceiling     44,318          short by 78,006
    qwen   ceiling     20,889          short by 101,435

**No reader can hold a full S conversation.** Not a close call — every one is short
by more than a factor of 1.8x, and the shortfall grows for M by a further 8x.

So Arm D is REJECTED by `full_context_feasible()` rather than run. It is not
truncated, and it must never be: an arm labelled "full context" that silently
holds 54% of the conversation is worse than no arm at all, because the label
would be true and the condition false. That is the same failure as the S-scale
token anchors being 10x off, and the same lesson applies — a measurement that
cannot be taken must be reported as untaken.

`full_context_feasible()` returns the shortfall as a NUMBER so the requirement can
be quoted: the cheapest reader needs roughly 2x this card, or a corpus subset.

ARM F — THE ORACLE CEILING (PRD §R4.4)
---------------------------------------
The official `_oracle` file, free. It bounds the WRITE path: if Arm F sits far
above Arm A, retrieval is binding and a better retriever pays; if it sits close
to Arm A, the write path is binding and a better reader pays. Distinguishing those
two is the difference between a week of retrieval work and a week of reader work,
and it costs one file that already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.arms.render import Memory, RenderedPrompt, format_memories
from msat.store.responses import ConditionKey

FULL_CONTEXT_B = float("inf")
"""Sentinel for Arm D. Never passed to a budget function - `render_full_context`
refuses to truncate, so the value exists only to make "no budget" explicit in a
condition key rather than implied by a missing field."""


@dataclass(frozen=True, slots=True)
class FullContextVerdict:
    ok: bool
    corpus_tokens: int
    ceiling: int
    shortfall: int
    detail: str = ""

    @property
    def multiple_needed(self) -> float:
        """How much bigger the card would have to be. Quoting a multiple is more
        actionable than quoting a token count."""
        return self.corpus_tokens / self.ceiling if self.ceiling else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "corpus_tokens": self.corpus_tokens,
            "ceiling": self.ceiling,
            "shortfall": self.shortfall,
            "multiple_needed": round(self.multiple_needed, 2),
            "detail": self.detail,
        }


def full_context_feasible(
    corpus_tokens: int,
    *,
    context_window: int,
    vram_ceiling: int,
) -> FullContextVerdict:
    """Can this reader hold the WHOLE conversation? Both ceilings, both checked.

    Checking only the declared window is the mistake this project has already made
    once: every candidate declares 131k-262k of context and none of them can
    deliver it on 8 GB.
    """
    ceiling = min(context_window, vram_ceiling)
    shortfall = corpus_tokens - ceiling
    return FullContextVerdict(
        ok=shortfall <= 0,
        corpus_tokens=corpus_tokens,
        ceiling=ceiling,
        shortfall=max(0, shortfall),
        detail=(
            "full context fits"
            if shortfall <= 0
            else f"needs {corpus_tokens:,} tokens, ceiling is {ceiling:,}; "
            f"a card {corpus_tokens / ceiling:.1f}x this size would do it"
        ),
    )


class FullContextTruncationRefusedError(RuntimeError):
    """Someone tried to run Arm D on a conversation that does not fit.

    Refused rather than truncated. An arm labelled "full context" holding 54% of
    the conversation is a measurement whose label is true and whose condition is
    false — worse than no arm, because it looks correct.
    """


@dataclass
class FullContextArm:
    """Arm D. Every session, in order, unbudgeted."""

    id: str = "D"
    isolates: str = "full context"
    template: str = ""
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        *,
        corpus_tokens: int | None = None,
        ceiling: int | None = None,
        qid: str = "",
        **kwargs: Any,
    ) -> RenderedPrompt:
        from msat.arms.render import render_template

        if ceiling is not None and corpus_tokens is not None:
            verdict = full_context_feasible(
                corpus_tokens, context_window=2**31, vram_ceiling=ceiling
            )
            if not verdict.ok:
                raise FullContextTruncationRefusedError(
                    f"arm D would need {verdict.corpus_tokens:,} tokens but the "
                    f"ceiling is {verdict.ceiling:,} ({verdict.detail}). Refusing "
                    f"rather than truncating: a 'full context' arm holding a fraction "
                    f"of the conversation has a true label and a false condition."
                )

        if not self.template:
            raise ValueError("arm D has no template")
        system, user = render_template(
            self.template,
            {
                # `budget_tokens=None` is the whole point: no per-memory cap, so
                # nothing here can shrink a conversation behind the label's back.
                "memories": format_memories(memories, None),
                "question": question,
                "K": len(memories),
                "B": "unbudgeted",
                "arm": self.id,
            },
        )
        return RenderedPrompt(
            arm=self.id,
            condition=condition,
            system=system,
            user=user,
            memory_ids=tuple(m.memory_id for m in memories),
            session_ids=tuple(m.session_id for m in memories),
            ranks=tuple(range(1, len(memories) + 1)),
            k=len(memories),
            b=FULL_CONTEXT_B,
            b_mode="full_context",
            corpus_fraction=1.0,
            position_variant="full",
            qid=qid,
        )


@dataclass
class OracleCeiling:
    """Arm F - the official `_oracle` file.

    Free, and it bounds the write path. Load it, then compare Arm A's measured
    recall against it: a large gap means retrieval is binding, a small gap means
    the write path is.
    """

    answers: dict[str, str]

    @classmethod
    def load(cls, path: Any) -> OracleCeiling:
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else raw.get("questions", [])
        answers: dict[str, str] = {}
        for r in records:
            qid = str(r.get("question_id") or r.get("qid") or "")
            gold = r.get("answer") or r.get("answer_session_ids") or ""
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            if qid:
                answers[qid] = str(gold)
        return cls(answers=answers)

    def gold_for(self, qid: str) -> str | None:
        return self.answers.get(qid)

    def coverage(self, qids: Sequence[str]) -> dict[str, float | int]:
        """How much of the run the oracle actually covers.

        Reported because an oracle covering 4% of the questions bounds nothing, and
        a ceiling computed over the covered subset is not the ceiling for the run.
        """
        covered = sum(1 for q in qids if q in self.answers)
        return {
            "n_questions": len(qids),
            "n_covered": covered,
            "coverage": covered / len(qids) if qids else 0.0,
        }


def generation_ceiling(
    arm_a_recall: float,
    oracle_recall: float,
    coverage: float,
) -> dict[str, Any]:
    """Which side is binding - the retriever or the write path.

    `coverage` below 1.0 means the ceiling is estimated on a subset, and it is
    passed in so the verdict can say so rather than implying the number covers the
    whole run.
    """
    gap = oracle_recall - arm_a_recall
    verdict = (
        "retrieval is binding"
        if gap > 0.15
        else "the write path is binding"
        if gap <= 0.05
        else "inconclusive - the gap is too small to attribute"
    )
    return {
        "arm_a_recall": round(arm_a_recall, 4),
        "oracle_recall": round(oracle_recall, 4),
        "gap": round(gap, 4),
        "coverage": round(coverage, 4),
        "verdict": verdict,
        "caveat": (
            "estimated on the covered subset only" if coverage < 1.0 else "measured on the full run"
        ),
    }
