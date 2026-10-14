"""Arm D-max: the largest full context each reader CAN actually hold.

WHY THIS EXISTS WHEN ARM D IS REFUSED
-------------------------------------
`arms/baselines.py` refuses Arm D because S's full corpus is 122,324 tokens and
the best VRAM ceiling is 66,160. That refusal is correct, but it throws away the
one comparison a paper about retrieval depth most wants: *what happens when you
stop retrieving and just hand the reader the conversation?*

This arm answers that on the hardware you have. It keeps the maximal SUFFIX of
the conversation that fits the reader's ceiling and nothing else. Measured
against the real corpus:

    gemma  26 sessions of 48     phi  17 of 48     qwen   8 of 48

THE LABEL IS THE WHOLE POINT
----------------------------
This is `full_context_max`, NEVER `full_context`. The two differ by whether the
cut happened, and that difference is the entire finding. The `b_mode` is its own
string so a pooled analysis cannot confuse a cut corpus with a whole one — the
same reason `ConditionKey` carries `b_mode` at all.

It is also NOT reader-balanced: each reader holds a different number of sessions,
so the three are three points on a curve, not three replicates. Never average
across readers for this arm.

DISTRACTION OR A THROWAWAY? — THE MEASUREMENT THAT MAKES IT READABLE
---------------------------------------------------------------------
S questions hold a median of 1.896 gold sessions inside 47.7. Keeping the last
8 or 26 sessions will, for many questions, cut the gold evidence out. Then a bad
score means "we deleted the answer", not "the reader got distracted", and the
arm is measuring the wrong thing entirely.

So the arm reports **evidence survival** alongside accuracy:

    survival = 1.0, accuracy low  ->  GENUINE DISTRACTION. The evidence was
                                      present and the reader still missed it.
    survival < 1.0, accuracy low  ->  CONFOUNDED. The cut removed the answer.
                                      This number is not a distraction result.

`max_context_verdict()` refuses to call it a distraction result in the second
case. Without that check the arm produces a confident, publishable, meaningless
number - which is worse than no arm, because it is indistinguishable from a real
finding once it is in a table.

WHY THE MOST RECENT SESSIONS AND NOT THE FIRST
----------------------------------------------
Recency is the standard strong baseline in dialogue systems, and it is the
policy a deployed system with a context cap would actually use. Keeping the
OLDEST sessions instead would answer a question nobody asks. Both are one
parameter (`Recency.MOST_RECENT` / `Recency.OLDEST`) because the choice is an
experimental decision, not a technical one, and a silent default would hide it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from msat.arms.baselines import full_context_feasible
from msat.arms.render import Memory, RenderedPrompt, format_memories, render_template
from msat.store.responses import ConditionKey

MAX_CONTEXT_B_MODE = "full_context_max"
"""Distinct from `full_context`. A cut corpus and a whole one are different
conditions and must never share a cache key, a ledger cell, or a figure column."""

DEFAULT_RESERVE_TOKENS = 512
"""Room for the decode.

`ReaderLimit.max_prompt_tokens` bounds the PROMPT. KV cache keeps growing while
the answer is generated, so a prompt that exactly fills the ceiling OOMs one
token into the completion. `decoding.yaml` declares `max_tokens: 512`; pass that
in rather than hard-coding, but never pass 0.
"""


class Recency(StrEnum):
    WHICH_END_TO_KEEP = "which_end_to_keep"

    MOST_RECENT = "most_recent"
    OLDEST = "oldest"


@dataclass(frozen=True, slots=True)
class MaxContextFit:
    """How much conversation this reader can actually be shown."""

    reader: str
    ceiling: int
    mean_session_tokens: float
    n_sessions_available: int
    n_sessions_kept: int
    kept_tokens: int
    reserve_tokens: int
    dropped_for_decode: int = 0
    """Sessions shed because the render came in OVER the estimate.

    Non-zero means the corpus has heavier sessions than its mean implies, and
    the answer is measured after the fact rather than trusted in advance.
    """

    @property
    def corpus_fraction(self) -> float:
        if not self.n_sessions_available:
            return 0.0
        return self.n_sessions_kept / self.n_sessions_available

    @property
    def budget(self) -> int:
        """The `B` this cell carries. Real tokens, never a sentinel - this arm is
        budgeted by construction, and an infinite B here would be a lie."""
        return self.kept_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "ceiling": self.ceiling,
            "mean_session_tokens": round(self.mean_session_tokens, 1),
            "n_sessions_available": self.n_sessions_available,
            "n_sessions_kept": self.n_sessions_kept,
            "kept_tokens": self.kept_tokens,
            "reserve_tokens": self.reserve_tokens,
            "dropped_for_decode": self.dropped_for_decode,
            "corpus_fraction": round(self.corpus_fraction, 4),
        }


def fit_to_ceiling(
    *,
    reader: str,
    ceiling: int,
    mean_session_tokens: float,
    n_sessions_available: int,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> MaxContextFit:
    """How many sessions fit, after leaving room to decode.

    An ESTIMATE, and labelled one. Session lengths vary, so the render is
    re-checked against the ceiling afterwards and any overflow is recorded in
    `dropped_for_decode` rather than being discovered as an OOM at hour 14.
    """
    if mean_session_tokens <= 0:
        raise ValueError("mean_session_tokens must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens cannot be negative")
    usable = max(0, ceiling - reserve_tokens)
    n_keep = int(usable // mean_session_tokens)
    n_keep = max(0, min(n_keep, n_sessions_available))
    return MaxContextFit(
        reader=reader,
        ceiling=ceiling,
        mean_session_tokens=mean_session_tokens,
        n_sessions_available=n_sessions_available,
        n_sessions_kept=n_keep,
        kept_tokens=int(n_keep * mean_session_tokens),
        reserve_tokens=reserve_tokens,
    )


def select_sessions(
    memories: Sequence[Memory], n_keep: int, recency: Recency = Recency.MOST_RECENT
) -> tuple[list[Memory], int]:
    """Keep `n_keep` memories by the chosen policy. Returns (kept, n_dropped).

    `memories` must already be in CONVERSATIONAL order; this does not re-sort.
    Re-sorting by `n_tokens` or by memory id here would silently change which
    evidence survives, and the survival measurement below would still report a
    number, which is the failure this whole module exists to prevent.
    """
    if n_keep < 0:
        raise ValueError("n_keep cannot be negative")
    if n_keep >= len(memories):
        return list(memories), 0
    ordered = list(memories)
    kept = ordered[-n_keep:] if n_keep and recency is Recency.MOST_RECENT else ordered[:n_keep]
    return kept, len(memories) - len(kept)


# ── the measurement that makes the arm readable ────────────────────────────────


def evidence_survival(kept: Sequence[Memory], gold_ids: Sequence[str]) -> float:
    """Fraction of gold sessions that survived the cut.

    A question with NO gold sessions (an abstention item) scores 1.0 rather than
    0/0. There is no evidence to lose, so nothing was destroyed - and reporting
    NaN would drop those items from every average and quietly shrink the
    denominator.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    have = {m.source_question_id for m in kept}
    return len(gold & have) / len(gold)


def max_context_verdict(
    accuracy: float,
    survival: float,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Is a low score distraction, or is it a cut that deleted the answer?

    `tolerance` exists because survival is a ratio of small integers. 3/3 is
    1.0 but floating-point arithmetic can make it 0.9999999999999998, and an
    arm that refuses to call itself a distraction result because of the last bit
    of a float is worse than no verdict at all.
    """
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


# ── the arm ────────────────────────────────────────────────────────────────────


def fit_to_memories(
    fit: MaxContextFit,
    memories: Sequence[Memory],
    *,
    ceil: float = 1.0,
) -> MaxContextFit:
    """Shrink the fit until the RENDERED memories really fit.

    `fit_to_ceiling` divides by a MEAN session length, so it is a starting point
    and can be optimistic: a conversation whose later sessions are longer than
    average will overflow. This walks the kept suffix down until the real
    `n_tokens` sum fits, and records how many sessions that cost in
    `dropped_for_decode`.

    Verifying after the fact rather than trusting the estimate is the whole
    reason this step exists. An OOM at hour 14 costs a run; a session dropped
    here costs one row in a report.
    """
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
    """Arm D-max. The maximal suffix of the conversation that fits."""

    id: str = "D-max"
    isolates: str = "max full context (VRAM-capped)"
    template: str = ""
    prompt_template_name: str = ""
    recency: Recency = Recency.MOST_RECENT

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        *,
        fit: MaxContextFit,
        gold_ids: Sequence[str] = (),
        qid: str = "",
        **kwargs: Any,
    ) -> RenderedPrompt:
        if not self.template:
            raise ValueError("arm D-max has no template")
        # Re-verify against the real session lengths before rendering.
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
    limits: dict[str, int],
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> dict[str, Any]:
    """What every reader can be shown, side by side.

    Printed by `msat build-plan --max-context` so the cap is visible BEFORE the
    run rather than discovered in the results table.
    """
    mean_session = corpus_tokens / sessions_median if sessions_median else 0.0
    fits = {
        r: fit_to_ceiling(
            reader=r,
            ceiling=ceiling,
            mean_session_tokens=mean_session,
            n_sessions_available=int(sessions_median),
            reserve_tokens=reserve_tokens,
        )
        for r, ceiling in limits.items()
    }
    verdicts = {
        r: full_context_feasible(corpus_tokens, context_window=2**31, vram_ceiling=c)
        for r, c in limits.items()
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
            "VRAM ceiling. It is NOT full context and must never be reported as such: "
            "each reader holds a different number of sessions, so the three are three "
            "points on a curve, not replicates."
        ),
    }
