"""Per-question K-guard and B-guard (PRD §R3.1-§R3.5).

A cell is **degenerate** when the condition it asks for is already satisfied
without doing the thing being measured. Two ways that happens:

    K-guard   K exceeds the corpus, so "retrieve the top K" returns the whole
              conversation whatever K says
    B-guard   the budget exceeds the corpus, so the context is full whatever B says

Both are checked **per question**, never corpus-wide.

WHY PER-QUESTION AND NOT A PERCENTILE
-------------------------------------
This is the project's most expensive lesson, and the reason it is a module rather
than a line. A corpus-wide percentile answers "is K large for this dataset?" The
question is "is K large for THIS conversation?" A 40-session conversation at
K=32 is 80% of the corpus - a percentile guard at p95 waves it through - while a
5-session conversation at the same K is degenerate outright. One design ships K=64
as a headline condition and discovers at week 6 that it was full-context the whole
time.

TIE-BREAK, STATED BECAUSE IT IS A CHOICE
----------------------------------------
`K == n_units` is **NOT** degenerate: retrieving everything the corpus holds is a
real condition, and it is the most informative cell in the grid. It is only
degenerate if the condition would be *uninformative*, which means exceeding the
available material. `>=` would throw away the boundary cell and with it the
ceiling of the curve.

WHY THE B-GUARD USES THE ACTUAL PROMPT
---------------------------------------
Not the declared `B`. The prompt is `min(units_available, unit_budget)` and only
the first of those is about the corpus. Checking `B >= corpus_tokens` marks a cell
degenerate for having a large budget while its K is small enough that the prompt
never reaches it. `test_b_guard_checks_the_prompt_not_the_budget` pins this: the
bug shipped once, silently, in `design.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

Unit = Literal["session", "chunk", "fact"]
UNITS: Final[tuple[Unit, ...]] = ("session", "chunk", "fact")

#: Why a cell was rejected, as a stable code. Reported per cell so a reader can
#: tell a corpus-size degeneracy from a budget degeneracy from a missing question.
OK: Final = "OK"
K_EXCEEDS_CORPUS: Final = "K_EXCEEDS_CORPUS"
PROMPT_FULL_CONTEXT: Final = "PROMPT_FULL_CONTEXT"
EMPTY_CORPUS: Final = "EMPTY_CORPUS"
MISSING_QUESTION: Final = "MISSING_QUESTION"


@dataclass(frozen=True, slots=True)
class QuestionStats:
    """What one question's corpus actually holds. Measured, never assumed."""

    question_id: str
    units: Mapping[str, int]
    """Count per retrieval `unit`. A 40-session corpus is 40 sessions but maybe
    300 chunks and 900 facts - and the K-guard must be evaluated in the unit the
    cell actually retrieves (PRD §R1.6)."""
    corpus_tokens: int
    """Total tokens across the whole corpus for this question."""

    def n_units(self, unit: Unit) -> int:
        return int(self.units.get(unit, 0))


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    ok: bool
    code: str
    reason: str
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def degenerate(self) -> bool:
        return not self.ok


def k_guard(
    k: int,
    stats: QuestionStats | None,
    unit: Unit = "session",
) -> GuardVerdict:
    """PRD §R3.1/R3.2: is K reachable for this question, in this unit?"""
    if stats is None:
        return GuardVerdict(
            False,
            MISSING_QUESTION,
            f"question {k!r}: no corpus stats, so the condition cannot be shown to exist",
            {"K": k, "unit": unit},
        )
    available = stats.n_units(unit)
    if available == 0:
        return GuardVerdict(
            False,
            EMPTY_CORPUS,
            f"{stats.question_id}: corpus holds 0 {unit}s, so K={k} retrieves nothing",
            {"K": k, "unit": unit, "available": 0},
        )
    if k > available:
        return GuardVerdict(
            False,
            K_EXCEEDS_CORPUS,
            f"{stats.question_id}: K={k} exceeds the {available} {unit}s this question has, "
            f"so the condition is full-context regardless of K",
            {"K": k, "unit": unit, "available": available, "excess": k - available},
        )
    # k == available is NOT degenerate. Retrieving everything the corpus holds is
    # a real, and the most informative, condition.
    return GuardVerdict(
        True,
        OK,
        f"{stats.question_id}: K={k} of {available} {unit}s",
        {"K": k, "unit": unit, "available": available},
    )


def prompt_tokens(
    k: int,
    b: float,
    stats: QuestionStats,
    unit: Unit = "session",
    *,
    mean_unit_tokens: float | None = None,
) -> float:
    """Tokens the rendered prompt would occupy, under the declared B semantics.

    For `per_memory_tokens` this is `min(available, K) * min(mean_unit, B)`. For an
    absolute total it is `min(content(K), B)`.

    `mean_unit_tokens` is REQUIRED for the per-memory form and cannot be defaulted:
    deriving it from the corpus means dividing total tokens by unit count, which is
    a corpus average - and on a corpus of unevenly sized sessions that is not the
    per-memory budget the arm will actually apply. Guessing it here would make the
    guard disagree with the renderer, which is the one thing a guard must never do.
    """
    available = stats.n_units(unit)
    usable = min(float(k), float(available))
    if mean_unit_tokens is None:
        return usable * float(b)
    return usable * min(float(mean_unit_tokens), float(b))


def b_guard(
    b: float,
    stats: QuestionStats,
    k: int,
    unit: Unit = "session",
    *,
    b_mode: str = "per_memory_tokens",
    mean_unit_tokens: float | None = None,
) -> GuardVerdict:
    """PRD §R3.4: would the prompt be full-context whatever B says?

    Evaluated on the PROMPT, not on `B`. A large budget behind a small K does not
    fill the context, and rejecting on the budget alone throws away valid cells.
    """
    if stats.corpus_tokens <= 0:
        return GuardVerdict(
            False,
            EMPTY_CORPUS,
            f"{stats.question_id}: corpus holds 0 tokens, so no context condition exists",
            {"B": b, "corpus_tokens": stats.corpus_tokens},
        )
    if b_mode == "per_memory_tokens":
        if mean_unit_tokens is None:
            raise ValueError(
                "b_mode='per_memory_tokens' requires mean_unit_tokens; deriving it "
                "from the corpus would make the guard disagree with the renderer"
            )
        rendered = prompt_tokens(k, b, stats, unit, mean_unit_tokens=mean_unit_tokens)
    else:
        # Absolute total: the prompt is the budget, capped by what the corpus holds.
        rendered = min(float(stats.corpus_tokens), float(b))
    if rendered >= stats.corpus_tokens:
        return GuardVerdict(
            False,
            PROMPT_FULL_CONTEXT,
            f"{stats.question_id}: the prompt would be {rendered:,.0f} tokens against a "
            f"{stats.corpus_tokens:,}-token corpus, so the context is full whatever B says",
            {
                "B": b,
                "K": k,
                "prompt_tokens": rendered,
                "corpus_tokens": stats.corpus_tokens,
            },
        )
    return GuardVerdict(
        True,
        OK,
        f"{stats.question_id}: prompt {rendered:,.0f} of {stats.corpus_tokens:,} tokens",
        {"B": b, "K": k, "prompt_tokens": rendered, "corpus_tokens": stats.corpus_tokens},
    )


def cell_verdict(
    k: int,
    b: float,
    stats: QuestionStats | None,
    unit: Unit = "session",
    *,
    b_mode: str = "per_memory_tokens",
    mean_unit_tokens: float | None = None,
) -> GuardVerdict:
    """Both guards for one (question, K, B).

    ORDER MATTERS for the reported reason. K is checked first because if K exceeds
    the corpus the context is already full and the B question is moot - reporting
    "budget degenerate" there would blame the wrong factor.

    But the CROSSED cases are both real and must not be collapsed:
      K fine, B degenerate  -> the budget is the problem
      B fine, K degenerate  -> the depth is the problem
    `test_crossed_cases_report_the_right_factor` pins that they stay distinct.
    """
    kv = k_guard(k, stats, unit)
    if not kv.ok:
        return kv
    if stats is None:  # pragma: no cover - k_guard already rejects this
        return kv
    return b_guard(b, stats, k, unit, b_mode=b_mode, mean_unit_tokens=mean_unit_tokens)


def degeneracy_fraction(
    k: int,
    b: float,
    corpus: Sequence[QuestionStats],
    unit: Unit = "session",
    *,
    b_mode: str = "per_memory_tokens",
    mean_unit_tokens: float | None = None,
) -> float:
    """PRD §R3.3: fraction of questions for which this (K, B) is degenerate.

    Reported per cell, not as a single project-level number, because the useful
    question is "which cells are mostly degenerate", and averaging over the grid
    hides a cell that is degenerate for every question.

    Questions with no stats count as degenerate. A condition that cannot be shown
    to exist is not a condition that can be counted as valid.
    """
    if not corpus:
        return 1.0
    degenerate = sum(
        cell_verdict(k, b, stats, unit, b_mode=b_mode, mean_unit_tokens=mean_unit_tokens).degenerate
        for stats in corpus
    )
    return degenerate / len(corpus)


def corpus_summary(corpus: Sequence[QuestionStats], unit: Unit = "session") -> dict[str, object]:
    """What the corpus holds, in this unit — the input every guard is evaluated against.

    Reported alongside the verdicts so a reader can see WHY a cell was rejected
    without re-deriving the counts. The `min_units` figure is the K ceiling for
    the smallest question in the corpus, which is the binding constraint: a
    grid sized against the MEAN conversation is degenerate on the shortest one,
    and that is the K=64 lesson again.
    """
    if not corpus:
        return {"n_questions": 0, "unit": unit}
    counts = [s.n_units(unit) for s in corpus]
    tokens = [s.corpus_tokens for s in corpus]
    return {
        "n_questions": len(corpus),
        "unit": unit,
        "units_per_question": {
            "min": min(counts),
            "median": sorted(counts)[len(counts) // 2],
            "max": max(counts),
        },
        "corpus_tokens": {
            "min": min(tokens),
            "median": sorted(tokens)[len(tokens) // 2],
            "max": max(tokens),
        },
        # The largest K any question in this corpus can support.
        "max_safe_k": min(counts),
    }
