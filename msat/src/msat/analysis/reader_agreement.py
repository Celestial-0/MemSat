"""Where the depth response has a different sign for different readers."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REFERENCE_FLIP_RATE: float = 0.31
REFERENCE_SOURCE = "2606.21807"

MIN_POINTS_FOR_SLOPE: int = 3

FLAT_SLOPE: float = 1e-9

FLAT = 0
RISES = 1
FALLS = -1

SIGN_NAME = {RISES: "rises", FALLS: "falls", FLAT: "flat"}


class UnsupportedComparisonError(ValueError):
    """Raised when two readers are compared over different K supports."""


@dataclass(frozen=True, slots=True)
class ConditionScore:
    reader: str
    k: int
    b: float
    scale: str
    score: float
    arm: str = "A"
    n_questions: int = 0


@dataclass(frozen=True, slots=True)
class DepthResponse:
    reader: str
    b: float
    scale: str
    arm: str
    ks: tuple[int, ...]
    slope: float | None
    sign: int
    reader_strength: float | None = None

    @property
    def n_points(self) -> int:
        return len(self.ks)

    @property
    def sign_name(self) -> str:
        return SIGN_NAME[self.sign]

    @property
    def defined(self) -> bool:
        return self.sign != FLAT

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "b": self.b,
            "scale": self.scale,
            "arm": self.arm,
            "ks": list(self.ks),
            "n_points": self.n_points,
            "slope": round(self.slope, 6) if self.slope is not None else None,
            "sign": self.sign,
            "sign_name": self.sign_name,
            "reader_strength": self.reader_strength,
        }


def depth_response(
    points: Sequence[tuple[int, float]],
    *,
    reader: str,
    b: float,
    scale: str,
    arm: str = "A",
    reader_strength: float | None = None,
    min_points: int = MIN_POINTS_FOR_SLOPE,
    flat: float = FLAT_SLOPE,
) -> DepthResponse:
    usable = sorted((int(k), float(s)) for k, s in points if int(k) > 0)
    ks = tuple(k for k, _ in usable)

    if len(usable) < min_points:
        return DepthResponse(
            reader=reader,
            b=b,
            scale=scale,
            arm=arm,
            ks=ks,
            slope=None,
            sign=FLAT,
            reader_strength=reader_strength,
        )

    xs = [math.log2(k) for k, _ in usable]
    ys = [s for _, s in usable]
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0.0:
        return DepthResponse(
            reader=reader,
            b=b,
            scale=scale,
            arm=arm,
            ks=ks,
            slope=None,
            sign=FLAT,
            reader_strength=reader_strength,
        )

    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sxx
    sign = RISES if slope > flat else FALLS if slope < -flat else FLAT
    return DepthResponse(
        reader=reader,
        b=b,
        scale=scale,
        arm=arm,
        ks=ks,
        slope=slope,
        sign=sign,
        reader_strength=reader_strength,
    )


def responses_from_conditions(
    conditions: Iterable[ConditionScore] | None = None,
    *,
    min_points: int = MIN_POINTS_FOR_SLOPE,
) -> list[DepthResponse]:
    target = conditions or ()
    buckets: dict[tuple[str, float, str, str], dict[int, list[float]]] = {}
    for c in target:
        key = (c.reader, c.b, c.scale, c.arm)
        buckets.setdefault(key, {}).setdefault(c.k, []).append(c.score)

    out: list[DepthResponse] = []
    for (reader, b, scale, arm), by_k in sorted(buckets.items()):
        points = [(k, statistics.fmean(v)) for k, v in sorted(by_k.items())]
        out.append(
            depth_response(
                points,
                reader=reader,
                b=b,
                scale=scale,
                arm=arm,
                min_points=min_points,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class PairComparison:
    reader_a: str
    reader_b: str
    b: float
    scale: str
    arm: str
    sign_a: int
    sign_b: int
    slope_a: float | None
    slope_b: float | None
    ks: tuple[int, ...]
    support: str

    @property
    def flipped(self) -> bool:
        return self.sign_a != FLAT and self.sign_b != FLAT and self.sign_a != self.sign_b

    @property
    def verdict(self) -> str:
        if self.sign_a == FLAT and self.sign_b == FLAT:
            return "agree"
        if self.sign_a == FLAT or self.sign_b == FLAT:
            return "flat_involved"
        return "flip" if self.flipped else "agree"

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader_a": self.reader_a,
            "reader_b": self.reader_b,
            "b": self.b,
            "scale": self.scale,
            "arm": self.arm,
            "sign_a": SIGN_NAME[self.sign_a],
            "sign_b": SIGN_NAME[self.sign_b],
            "slope_a": round(self.slope_a, 6) if self.slope_a is not None else None,
            "slope_b": round(self.slope_b, 6) if self.slope_b is not None else None,
            "ks": list(self.ks),
            "support": self.support,
            "verdict": self.verdict,
        }


def _support_kind(a: DepthResponse, b: DepthResponse) -> tuple[str, tuple[int, ...]]:
    ka, kb = set(a.ks), set(b.ks)
    if ka == kb:
        return "common", tuple(sorted(ka & kb))
    if not (ka & kb):
        return "truncated_both", ()
    if kb < ka:
        return "truncated_b", tuple(sorted(ka & kb))
    if ka < kb:
        return "truncated_a", tuple(sorted(ka & kb))
    return "partial_overlap", tuple(sorted(ka & kb))


@dataclass
class ReaderAgreementReport:
    comparisons: list[PairComparison] = field(default_factory=list)
    unsupported: list[PairComparison] = field(default_factory=list)

    responses: list[DepthResponse] = field(default_factory=list)
    strengths: Mapping[str, float] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.comparisons)

    @property
    def n_flips(self) -> int:
        return sum(1 for c in self.comparisons if c.flipped)

    @property
    def flip_rate(self) -> float | None:
        return self.n_flips / self.n_pairs if self.n_pairs else None

    @property
    def n_agree(self) -> int:
        return sum(1 for c in self.comparisons if c.verdict == "agree")

    @property
    def n_flat_involved(self) -> int:
        return sum(1 for c in self.comparisons if c.verdict == "flat_involved")

    @property
    def net_delta(self) -> int | None:
        flips = [c for c in self.comparisons if c.flipped]
        if not flips:
            return 0 if self.n_pairs else None
        return sum(1 for c in flips if c.sign_a == RISES) - sum(
            1 for c in flips if c.sign_a == FALLS
        )

    @property
    def directional_balance(self) -> float | None:
        flips = [c for c in self.comparisons if c.flipped]
        if not flips:
            return None
        up = sum(1 for c in flips if c.sign_a == RISES)
        return abs(up - (self.n_flips - up)) / self.n_flips

    @property
    def unsupported_rate(self) -> float:
        total = len(self.comparisons) + len(self.unsupported)
        return len(self.unsupported) / total if total else 0.0

    def by_scale(self) -> dict[str, int]:
        return _count(self.comparisons, lambda c: c.scale)

    def by_budget(self) -> dict[float, int]:
        return _count(self.comparisons, lambda c: c.b)

    def flips_by_scale(self) -> dict[str, int]:
        return _count([c for c in self.comparisons if c.flipped], lambda c: c.scale)

    def flips_by_budget(self) -> dict[float, int]:
        return _count([c for c in self.comparisons if c.flipped], lambda c: c.b)

    def flips_by_pair(self) -> dict[str, int]:
        return _count(
            [c for c in self.comparisons if c.flipped],
            lambda c: f"{c.reader_a}|{c.reader_b}",
        )

    def by_strength_direction(self) -> dict[str, int]:
        out = {
            "stronger_rises": 0,
            "stronger_falls": 0,
            "equal_strength": 0,
            "unknown": 0,
        }
        for c in self.comparisons:
            if not c.flipped:
                continue
            sa = self.strengths.get(c.reader_a)
            sb = self.strengths.get(c.reader_b)
            if sa is None or sb is None:
                out["unknown"] += 1
            elif abs(sa - sb) < 1e-12:
                out["equal_strength"] += 1
            elif (sa > sb) == (c.sign_a == RISES):
                out["stronger_rises"] += 1
            else:
                out["stronger_falls"] += 1
        return out

    def headline(self) -> str:
        if not self.n_pairs:
            if self.unsupported:
                return (
                    f"No comparable reader pairs: all {len(self.unsupported)} "
                    f"candidate comparisons had different K supports. The depth "
                    f"response is not attributable across readers on this hardware."
                )
            return "No reader pairs to compare."
        rate = self.flip_rate or 0.0
        text = (
            f"Depth-response sign differs for {self.n_flips}/{self.n_pairs} reader "
            f"pairs ({rate:.1%}); {self.n_agree} agree, {self.n_flat_involved} "
            f"involve a flat reader."
        )
        if self.unsupported:
            text += (
                f" {len(self.unsupported)} further comparisons were refused for "
                f"unequal K support ({self.unsupported_rate:.1%} of all candidates), "
                f"so this rate covers a narrower depth range than the full grid."
            )
        return text

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_flips": self.n_flips,
            "n_agree": self.n_agree,
            "n_flat_involved": self.n_flat_involved,
            "flip_rate": round(self.flip_rate, 6) if self.flip_rate is not None else None,
            "net_delta": self.net_delta,
            "directional_balance": (
                round(self.directional_balance, 6) if self.directional_balance is not None else None
            ),
            "n_unsupported": len(self.unsupported),
            "unsupported_rate": round(self.unsupported_rate, 6),
            "by_scale": self.by_scale(),
            "by_budget": {str(k): v for k, v in self.by_budget().items()},
            "flips_by_scale": self.flips_by_scale(),
            "flips_by_budget": {str(k): v for k, v in self.flips_by_budget().items()},
            "flips_by_pair": self.flips_by_pair(),
            "by_strength_direction": self.by_strength_direction(),
            "reference": {
                "flip_rate": REFERENCE_FLIP_RATE,
                "source": REFERENCE_SOURCE,
                "note": (
                    "a comparison point, not a threshold. Their setting is KV-cache "
                    "compression on a different panel, so agreement would be a "
                    "coincidence and disagreement would not refute anything."
                ),
            },
            "headline": self.headline(),
            "comparisons": [c.as_dict() for c in self.comparisons],
            "unsupported": [c.as_dict() for c in self.unsupported],
        }


def _count(comparisons: Sequence[PairComparison], key: Any) -> dict[Any, int]:
    out: dict[Any, int] = {}
    for c in comparisons:
        k = key(c)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def compare_readers(
    responses: Sequence[DepthResponse],
    *,
    strengths: Mapping[str, float] | None = None,
    on_unsupported: str = "refuse",
) -> ReaderAgreementReport:
    if on_unsupported not in ("refuse", "skip", "count"):
        raise ValueError(f"on_unsupported must be refuse|skip|count, got {on_unsupported!r}")

    by_key: dict[tuple[float, str, str], dict[str, DepthResponse]] = {}
    for r in responses:
        by_key.setdefault((r.b, r.scale, r.arm), {})[r.reader] = r

    report = ReaderAgreementReport(responses=list(responses))
    seen_unsupported = False

    for (b, scale, arm), by_reader in sorted(by_key.items()):
        readers = sorted(by_reader)
        for ra, rb in itertools.combinations(readers, 2):
            a, c = by_reader[ra], by_reader[rb]
            support, shared_ks = _support_kind(a, c)
            pair = PairComparison(
                reader_a=ra,
                reader_b=rb,
                b=b,
                scale=scale,
                arm=arm,
                sign_a=a.sign,
                sign_b=c.sign,
                slope_a=a.slope,
                slope_b=c.slope,
                ks=shared_ks,
                support=support,
            )
            if support != "common":
                seen_unsupported = True
                report.unsupported.append(pair)
                continue
            if a.n_points != len(shared_ks) or c.n_points != len(shared_ks):
                report.unsupported.append(
                    PairComparison(
                        reader_a=ra,
                        reader_b=rb,
                        b=b,
                        scale=scale,
                        arm=arm,
                        sign_a=a.sign,
                        sign_b=c.sign,
                        slope_a=a.slope,
                        slope_b=c.slope,
                        ks=shared_ks,
                        support="point_count_mismatch",
                    )
                )
                continue
            report.comparisons.append(pair)

    report.strengths = dict(strengths or {})

    if seen_unsupported and on_unsupported == "refuse":
        raise UnsupportedComparisonError(
            f"{len(report.unsupported)} reader-pair comparison(s) had unequal K "
            f"supports and were refused. Comparing them would answer two different "
            f"questions and silently truncate the factorial (R0.3, N8). "
            f"Pass on_unsupported='skip' to proceed - they stay in "
            f"report.unsupported and in unsupported_rate either way."
        )

    return report


def synthetic_conditions() -> list[ConditionScore]:
    out: list[ConditionScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0, 2048.0):
            for i, k in enumerate((1, 2, 4, 8, 16, 32)):
                gemma = 0.30 + 0.045 * i
                qwen = (0.34 - 0.020 * i) if b <= 512.0 else (0.30 + 0.030 * i)
                phi = 0.40
                for reader, score in (("gemma", gemma), ("qwen", qwen), ("phi", phi)):
                    out.append(
                        ConditionScore(
                            reader=reader,
                            k=k,
                            b=b,
                            scale=scale,
                            arm="A",
                            score=max(0.0, min(1.0, score)),
                            n_questions=500,
                        )
                    )
    return out


SYNTHETIC_STRENGTHS = {"gemma": 0.78, "qwen": 0.80, "phi": 0.72}
