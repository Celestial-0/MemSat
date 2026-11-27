"""Arms D and F, the baselines the budgeted arms are measured against."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from msat.arms.render import Memory, RenderedPrompt, format_memories
from msat.store import Condition

FULL_CONTEXT_B = float("inf")


@dataclass(frozen=True, slots=True)
class FullContextVerdict:
    ok: bool
    corpus_tokens: int
    ceiling: int
    shortfall: int
    detail: str = ""

    @property
    def multiple_needed(self) -> float:
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
    """Raised rather than truncating a conversation that does not fit."""


@dataclass
class FullContextArm:
    id: str = "D"
    isolates: str = "full context"
    template: str = ""
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
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
