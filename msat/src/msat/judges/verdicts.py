"""What a judge said about one item.

DELIBERATELY SMALL
------------------
`CORRECT` and `INCORRECT`, and nothing else. A third category like `UNSCORABLE`
would be tempting - a judge can refuse, or return garbage - but it changes Cohen's
κ's expected agreement in a way that makes two runs with different refusal rates
incomparable. Unscorable items are therefore EXCLUDED PAIRWISE, and the number
excluded is reported, because dropping 20% of items silently is how a reliability
figure gets better every time the data gets worse.

The three metrics P6 requires side by side (R8.1) share this record, so EM, token-F1
and the model-based evaluator can be compared per item rather than in three
separate aggregates that cannot be lined up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

ItemKey = tuple[str, str]
"""`(condition_key, qid)`. Defined here because every judge computation is indexed
by it, and defining it once keeps `kappa.py` and `audit.py` agreeing on what an
item is."""


class Verdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


class UnscorableError(ValueError):
    """A verdict that is neither correct nor incorrect.

    Raised rather than coerced. `None` in a verdict list means either a bug or a
    refusal, and the two want opposite handling: a bug should stop the run, a
    refusal should be excluded and counted. Coercing to INCORRECT would charge the
    judge for declining, which is the R8.4 distinction this project keeps drawing.
    """


def as_verdict(value: Any) -> Verdict:
    """Coerce a judge's raw output to a `Verdict`, or refuse.

    Text matching is deliberately narrow. "the answer looks correct" and "probably
    right" are NOT converted to CORRECT: a judge that hedges is a judge whose
    reliability we are trying to measure, and reading its hedges as verdicts
    manufactures agreement.
    """
    if isinstance(value, Verdict):
        return value
    if isinstance(value, bool):
        return Verdict.CORRECT if value else Verdict.INCORRECT
    if isinstance(value, int) and value in (0, 1):
        # `bool` is a subclass of `int`, so the branch above has to come first.
        return Verdict.CORRECT if value == 1 else Verdict.INCORRECT
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"correct", "true", "yes", "1"}:
            return Verdict.CORRECT
        if v in {"incorrect", "false", "no", "0"}:
            return Verdict.INCORRECT
    raise UnscorableError(
        f"{value!r} is not a verdict. A judge must emit exactly 'correct' or "
        f"'incorrect'. Hedged output is not a verdict: reading it as one "
        f"manufactures agreement and inflates kappa."
    )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judge's call on one item."""

    qid: str
    condition_key: str
    judge_id: str
    verdict: Verdict
    pass_index: int = 1
    """R8.6: which pass produced this. The same deterministic judge run twice must
    be able to produce two records for one item, and a record that cannot say which
    pass it came from cannot be compared."""
    latency_ms: float | None = None
    raw: str = ""

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.condition_key, self.qid, self.judge_id, self.pass_index)


@dataclass
class VerdictSet:
    """Every verdict for one run, indexed for the things we compute on them."""

    verdicts: list[JudgeVerdict] = field(default_factory=list)

    def add(self, v: JudgeVerdict) -> None:
        self.verdicts.append(v)

    def judges(self) -> list[str]:
        return sorted({v.judge_id for v in self.verdicts})

    def for_judge(self, judge_id: str, *, pass_index: int = 1) -> dict[tuple[str, str], Verdict]:
        """`(condition_key, qid) -> verdict` for one judge and one pass."""
        return {
            (v.condition_key, v.qid): v.verdict
            for v in self.verdicts
            if v.judge_id == judge_id and v.pass_index == pass_index
        }

    def items(self) -> list[tuple[str, str]]:
        return sorted({(v.condition_key, v.qid) for v in self.verdicts})

    def by_pass(self, pass_index: int) -> VerdictSet:
        return VerdictSet([v for v in self.verdicts if v.pass_index == pass_index])

    def disagreements(self, judge_a: str, judge_b: str) -> list[tuple[str, str]]:
        a = self.for_judge(judge_a)
        b = self.for_judge(judge_b)
        shared = sorted(set(a) & set(b))
        return [k for k in shared if a[k] is not b[k]]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "condition_key": v.condition_key,
                "qid": v.qid,
                "judge_id": v.judge_id,
                "verdict": str(v.verdict),
                "pass_index": v.pass_index,
            }
            for v in self.verdicts
        ]

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> VerdictSet:
        out = cls()
        for r in records:
            out.add(
                JudgeVerdict(
                    qid=str(r["qid"]),
                    condition_key=str(r["condition_key"]),
                    judge_id=str(r["judge_id"]),
                    verdict=as_verdict(r.get("verdict", r.get("label"))),
                    pass_index=int(r.get("pass_index", 1)),
                    latency_ms=r.get("latency_ms"),
                    raw=str(r.get("raw", "")),
                )
            )
        return out


def accuracy(verdicts: Mapping[tuple[str, str], Verdict]) -> float | None:
    """Fraction judged CORRECT. None when there is nothing to average."""
    if not verdicts:
        return None
    return sum(1 for v in verdicts.values() if v is Verdict.CORRECT) / len(verdicts)
