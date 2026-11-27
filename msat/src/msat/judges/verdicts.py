"""What a judge said about one item."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

ItemKey = tuple[str, str]


class Verdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


class UnscorableError(ValueError):
    """A verdict that is neither correct nor incorrect."""


def as_verdict(value: Any) -> Verdict:
    if isinstance(value, Verdict):
        return value
    if isinstance(value, bool):
        return Verdict.CORRECT if value else Verdict.INCORRECT
    if isinstance(value, int) and value in (0, 1):
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
    qid: str
    condition_key: str
    judge_id: str
    verdict: Verdict
    pass_index: int = 1
    latency_ms: float | None = None
    raw: str = ""

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.condition_key, self.qid, self.judge_id, self.pass_index)


@dataclass
class VerdictSet:
    verdicts: list[JudgeVerdict] = field(default_factory=list)

    def add(self, v: JudgeVerdict) -> None:
        self.verdicts.append(v)

    def judges(self) -> list[str]:
        return sorted({v.judge_id for v in self.verdicts})

    def for_judge(self, judge_id: str, *, pass_index: int = 1) -> dict[tuple[str, str], Verdict]:
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
    if not verdicts:
        return None
    return sum(1 for v in verdicts.values() if v is Verdict.CORRECT) / len(verdicts)
