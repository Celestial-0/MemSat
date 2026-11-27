"""Arm D-max, the largest full context each reader can hold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from msat.arms.baselines import full_context_feasible
from msat.arms.render import Memory, RenderedPrompt, format_memories, render_template
from msat.store import Condition

MAX_CONTEXT_B_MODE = "full_context_max"

DEFAULT_RESERVE_TOKENS = 512


class Recency(StrEnum):
    WHICH_END_TO_KEEP = "which_end_to_keep"

    MOST_RECENT = "most_recent"
    OLDEST = "oldest"


@dataclass(frozen=True, slots=True)
class MaxContextFit:
    reader: str
    reader_capacity_tokens: int
    mean_session_tokens: float
    n_sessions_available: int
    n_sessions_kept: int
    kept_tokens: int
    reserve_tokens: int
    dropped_for_decode: int = 0

    @property
    def ceiling(self) -> int:
        return self.reader_capacity_tokens

    @property
    def corpus_fraction(self) -> float:
        if not self.n_sessions_available:
            return 0.0
        return self.n_sessions_kept / self.n_sessions_available

    @property
    def budget(self) -> int:
        return self.kept_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "reader_capacity_tokens": self.reader_capacity_tokens,
            "ceiling": self.reader_capacity_tokens,
            "mean_session_tokens": round(self.mean_session_tokens, 1),
            "n_sessions_available": self.n_sessions_available,
            "n_sessions_kept": self.n_sessions_kept,
            "kept_tokens": self.kept_tokens,
            "reserve_tokens": self.reserve_tokens,
            "dropped_for_decode": self.dropped_for_decode,
            "corpus_fraction": round(self.corpus_fraction, 4),
        }


def fit_to_reader_capacity(
    *,
    reader: str,
    reader_capacity_tokens: int = 0,
    ceiling: int | None = None,
    mean_session_tokens: float,
    n_sessions_available: int,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> MaxContextFit:
    cap = reader_capacity_tokens if ceiling is None else ceiling
    if mean_session_tokens <= 0:
        raise ValueError("mean_session_tokens must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens cannot be negative")
    usable = max(0, cap - reserve_tokens)
    n_keep = int(usable // mean_session_tokens)
    n_keep = max(0, min(n_keep, n_sessions_available))
    return MaxContextFit(
        reader=reader,
        reader_capacity_tokens=cap,
        mean_session_tokens=mean_session_tokens,
        n_sessions_available=n_sessions_available,
        n_sessions_kept=n_keep,
        kept_tokens=int(n_keep * mean_session_tokens),
        reserve_tokens=reserve_tokens,
    )


fit_to_ceiling = fit_to_reader_capacity


def select_sessions(
    memories: Sequence[Memory], n_keep: int, recency: Recency = Recency.MOST_RECENT
) -> tuple[list[Memory], int]:
    if n_keep < 0:
        raise ValueError("n_keep cannot be negative")
    if n_keep >= len(memories):
        return list(memories), 0
    ordered = list(memories)
    kept = ordered[-n_keep:] if n_keep and recency is Recency.MOST_RECENT else ordered[:n_keep]
    return kept, len(memories) - len(kept)


def evidence_survival(kept: Sequence[Memory], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0
    have = {m.session_id for m in kept}
    return len(gold & have) / len(gold)


def max_context_verdict(
    accuracy: float,
    survival: float,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    complete = survival >= 1.0 - tolerance
    if accuracy < 0.5 and not complete:
        verdict = "CONFOUNDED - the cut removed evidence; not a distraction result"
    elif accuracy < 0.5:
        verdict = "genuine distraction - evidence was present and the reader missed it"
    elif complete:
        verdict = "clean - full evidence and a high score"
    else:
        verdict = (
            "partial evidence survived and the score is still high; the reader is robust to the cut"
        )
    return {
        "accuracy": round(accuracy, 4),
        "evidence_survival": round(survival, 4),
        "verdict": verdict,
        "reportable_as_distraction": complete,
    }


def fit_to_memories(
    fit: MaxContextFit,
    memories: Sequence[Memory],
    *,
    ceil: float = 1.0,
) -> MaxContextFit:
    usable = max(0, fit.ceiling - fit.reserve_tokens)
    budget = usable * ceil
    n = fit.n_sessions_kept
    while n > 0 and sum(m.n_tokens for m in memories[-n:]) > budget:
        n -= 1
    return replace(
        fit,
        n_sessions_kept=n,
        kept_tokens=sum(m.n_tokens for m in memories[-n:]) if n else 0,
        dropped_for_decode=fit.dropped_for_decode + (fit.n_sessions_kept - n),
    )


@dataclass
class MaxContextArm:
    id: str = "D-max"
    isolates: str = "max full context (VRAM-capped)"
    template: str = ""
    prompt_template_name: str = ""
    recency: Recency = Recency.MOST_RECENT

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
        *,
        fit: MaxContextFit,
        gold_ids: Sequence[str] = (),
        qid: str = "",
        **kwargs: Any,
    ) -> RenderedPrompt:
        if not self.template:
            raise ValueError("arm D-max has no template")
        fit = fit_to_memories(fit, memories)
        kept, _ = select_sessions(memories, fit.n_sessions_kept, self.recency)
        system, user = render_template(
            self.template,
            {
                "memories": format_memories(kept, None),
                "question": question,
                "K": len(kept),
                "B": fit.budget,
                "arm": self.id,
            },
        )
        return RenderedPrompt(
            arm=self.id,
            condition=condition,
            system=system,
            user=user,
            memory_ids=tuple(m.memory_id for m in kept),
            session_ids=tuple(m.session_id for m in kept),
            ranks=tuple(range(1, len(kept) + 1)),
            k=len(kept),
            b=fit.budget,
            b_mode=MAX_CONTEXT_B_MODE,
            corpus_fraction=fit.corpus_fraction,
            position_variant=f"full/{self.recency}",
            evidence_survival=evidence_survival(kept, gold_ids),
            qid=qid,
        )


def max_context_plan(
    *,
    corpus_tokens: int,
    sessions_median: float,
    reader_capacity_tokens: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> dict[str, Any]:
    capacities = reader_capacity_tokens if limits is None else limits
    if capacities is None:
        raise ValueError("reader_capacity_tokens or limits required")
    mean_session = corpus_tokens / sessions_median if sessions_median else 0.0
    fits = {
        r: fit_to_reader_capacity(
            reader=r,
            reader_capacity_tokens=cap,
            mean_session_tokens=mean_session,
            n_sessions_available=int(sessions_median),
            reserve_tokens=reserve_tokens,
        )
        for r, cap in capacities.items()
    }
    verdicts = {
        r: full_context_feasible(corpus_tokens, context_window=2**31, vram_ceiling=c)
        for r, c in capacities.items()
    }
    return {
        "corpus_tokens": corpus_tokens,
        "sessions_median": sessions_median,
        "mean_session_tokens": round(mean_session, 1),
        "arm_d_infeasible": all(not v.ok for v in verdicts.values()),
        "per_reader": {
            r: {
                **f.as_dict(),
                "full_context": verdicts[r].ok,
                "shortfall": verdicts[r].shortfall,
            }
            for r, f in fits.items()
        },
        "note": (
            "`full_context_max` keeps the most recent sessions up to each reader's "
            "reader capacity. It is NOT full context and must never be reported as such: "
            "each reader holds a different number of sessions, so the three are three "
            "points on a curve, not replicates."
        ),
    }
