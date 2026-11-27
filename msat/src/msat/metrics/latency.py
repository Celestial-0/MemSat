"""Latency reporting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
SKEW_RATIO = 2.0


@dataclass(frozen=True, slots=True)
class LatencySample:
    latency_ms: float
    queue_wait_ms: float | None = None
    ttft_ms: float | None = None
    server_time_ms: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"negative latency {self.latency_ms}")
        for name in ("queue_wait_ms", "ttft_ms", "server_time_ms"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"negative {name} {v}")
        if (
            self.ttft_ms is not None
            and self.queue_wait_ms is not None
            and self.ttft_ms < self.queue_wait_ms
        ):
            raise ValueError(
                f"ttft_ms {self.ttft_ms} is below queue_wait_ms {self.queue_wait_ms}. "
                f"TTFT is measured FROM the end of the queue wait, so the queue wait "
                f"cannot exceed it."
            )

    @property
    def queue_excluded(self) -> float:
        return self.latency_ms - (self.queue_wait_ms or 0.0)

    @property
    def valid(self) -> bool:
        return self.latency_ms > 0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


_quantile = quantile


@dataclass
class LatencySummary:
    label: str
    n: int
    n_invalid: int = 0
    quantiles: dict[str, float] = field(default_factory=dict)
    iqr: float = 0.0
    mean: float = 0.0
    queue_excluded_mean: float | None = None
    queue_excluded_p50: float | None = None
    skewed: bool = False
    caveat: str = ""

    @property
    def p99_reliable(self) -> bool:
        return self.n >= 100

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "n_invalid": self.n_invalid,
            "quantiles_ms": {k: round(v, 2) for k, v in self.quantiles.items()},
            "iqr_ms": round(self.iqr, 2),
            "mean_ms": round(self.mean, 2),
            "queue_excluded_mean_ms": (
                round(self.queue_excluded_mean, 2) if self.queue_excluded_mean is not None else None
            ),
            "queue_excluded_p50_ms": (
                round(self.queue_excluded_p50, 2) if self.queue_excluded_p50 is not None else None
            ),
            "skewed": self.skewed,
            "p99_reliable": self.p99_reliable,
            "caveat": self.caveat,
        }


def summarise_latency(samples: Sequence[LatencySample], *, label: str = "all") -> LatencySummary:
    valid = [s for s in samples if s.valid]
    n_invalid = len(samples) - len(valid)
    if not valid:
        return LatencySummary(
            label=label,
            n=0,
            n_invalid=n_invalid,
            caveat="no valid latency samples; every summary field is a placeholder",
        )

    vals = [s.latency_ms for s in valid]
    q = {f"p{int(x * 100)}": _quantile(vals, x) for x in QUANTILES}
    mean = sum(vals) / len(vals)
    p50 = q["p50"]
    skewed = mean > SKEW_RATIO * p50 > 0

    qex = [s.queue_excluded for s in valid]
    caveat = ""
    if skewed:
        caveat = (
            f"mean is {mean / p50:.1f}x p50. The distribution is right-skewed, so the "
            f"mean is not a representative latency - quote the quantiles."
        )
    elif n_invalid:
        caveat = f"{n_invalid} zero/invalid sample(s) excluded."

    return LatencySummary(
        label=label,
        n=len(valid),
        n_invalid=n_invalid,
        quantiles=q,
        iqr=q["p75"] - q["p25"],
        mean=mean,
        queue_excluded_mean=sum(qex) / len(qex),
        queue_excluded_p50=_quantile(qex, 0.5),
        skewed=skewed,
        caveat=caveat,
    )


def latency_report(
    samples: Sequence[LatencySample], *, by_label: dict[str, Sequence[LatencySample]] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": summarise_latency(samples).as_dict()}
    if by_label:
        out["by_reader"] = {
            k: summarise_latency(v, label=k).as_dict() for k, v in sorted(by_label.items())
        }

    waits = [s.queue_wait_ms for s in samples if s.queue_wait_ms is not None]
    if waits and len(waits) == len(samples):
        mean_wait = sum(waits) / len(waits)
        mean_total = sum(s.latency_ms for s in samples) / len(samples)
        share = mean_wait / mean_total if mean_total else 0.0
        out["queue_wait"] = {
            "recorded": True,
            "mean_ms": round(mean_wait, 2),
            "p50_ms": round(_quantile(waits, 0.5), 2),
            "share_of_latency": round(share, 4),
            "recording_concurrency": share < 0.05,
            "note": (
                "Queue wait is our scheduling (PRD §R5.3), not the reader. When it is "
                "a large share, `latency_ms` measures this harness. Use "
                "`queue_excluded_*` to compare readers."
            ),
        }
    else:
        out["queue_wait"] = {
            "recorded": False,
            "note": (
                "Not every sample carries a queue wait, so no share can be computed. "
                "Latency and TTFT are reported but cannot be separated."
            ),
        }
    return out
