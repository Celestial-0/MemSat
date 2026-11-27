"""Per-question K-guard and B-guard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

Unit = Literal["session", "chunk", "fact"]
UNITS: Final[tuple[Unit, ...]] = ("session", "chunk", "fact")

OK: Final = "OK"
K_EXCEEDS_CORPUS: Final = "K_EXCEEDS_CORPUS"
PROMPT_FULL_CONTEXT: Final = "PROMPT_FULL_CONTEXT"
EMPTY_CORPUS: Final = "EMPTY_CORPUS"
MISSING_QUESTION: Final = "MISSING_QUESTION"


@dataclass(frozen=True, slots=True)
class QuestionStats:
    question_id: str
    units: Mapping[str, int]
    corpus_tokens: int

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


def condition_verdict(
    k: int,
    b: float,
    stats: QuestionStats | None,
    unit: Unit = "session",
    *,
    b_mode: str = "per_memory_tokens",
    mean_unit_tokens: float | None = None,
) -> GuardVerdict:
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
    if not corpus:
        return 1.0
    degenerate = sum(
        condition_verdict(
            k, b, stats, unit, b_mode=b_mode, mean_unit_tokens=mean_unit_tokens
        ).degenerate
        for stats in corpus
    )
    return degenerate / len(corpus)


def corpus_summary(corpus: Sequence[QuestionStats], unit: Unit = "session") -> dict[str, object]:
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
        "max_safe_k": min(counts),
    }
