"""Bounding the hardware confound instead of assuming it away."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from msat.metrics.normalisation import Normaliser, exact_match, token_f1
from msat.store import Condition, ResponseStore, StoredResponse


@dataclass
class ComparedPair:
    key: str
    condition: dict[str, Any]
    a_text: str
    b_text: str
    a_tokens: int
    b_tokens: int
    em_a: float
    em_b: float
    f1_a: float
    f1_b: float

    @property
    def em_delta(self) -> float:
        return self.em_b - self.em_a

    @property
    def f1_delta(self) -> float:
        return self.f1_b - self.f1_a

    @property
    def metric_moved(self) -> bool:
        return self.em_delta != 0.0 or self.f1_delta != 0.0


@dataclass
class DeltaReport:
    label_a: str
    label_b: str
    n_requested: int
    n_compared: int
    n_missing_in_a: int = 0
    n_missing_in_b: int = 0
    n_text_identical: int = 0
    n_normalised_identical: int = 0
    n_metric_equal: int = 0
    n_pairs_moved_total: int = 0
    em_a: float = 0.0
    em_b: float = 0.0
    f1_a: float = 0.0
    f1_b: float = 0.0
    worst_pairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def em_delta(self) -> float:
        return self.em_b - self.em_a

    @property
    def f1_delta(self) -> float:
        return self.f1_b - self.f1_a

    @property
    def n_pairs_metric_moved(self) -> int:
        return self.n_pairs_moved_total

    def as_dict(self) -> dict[str, Any]:
        n = max(self.n_compared, 1)
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_requested": self.n_requested,
            "n_compared": self.n_compared,
            "n_missing_in_a": self.n_missing_in_a,
            "n_missing_in_b": self.n_missing_in_b,
            "text_identical": self.n_text_identical,
            "normalised_identical": self.n_normalised_identical,
            "metric_equal": self.n_metric_equal,
            "pairs_with_moved_metric": self.n_pairs_moved_total,
            "frac_text_identical": round(self.n_text_identical / n, 4),
            "frac_normalised_identical": round(self.n_normalised_identical / n, 4),
            "frac_metric_equal": round(self.n_metric_equal / n, 4),
            "em_a": round(self.em_a, 4),
            "em_b": round(self.em_b, 4),
            "em_delta": round(self.em_delta, 6),
            "f1_a": round(self.f1_a, 4),
            "f1_b": round(self.f1_b, 4),
            "f1_delta": round(self.f1_delta, 6),
        }


def compare_stores(
    store_a: ResponseStore,
    store_b: ResponseStore,
    keys: Sequence[str],
    *,
    gold_by_key: Mapping[str, str],
    norm: Normaliser,
    label_a: str = "A",
    label_b: str = "B",
    max_worst: int = 10,
) -> DeltaReport:
    report = DeltaReport(label_a=label_a, label_b=label_b, n_requested=len(keys), n_compared=0)
    pairs: list[ComparedPair] = []
    for key in keys:
        a: StoredResponse | None = store_a.get(key)
        b: StoredResponse | None = store_b.get(key)
        if a is None:
            report.n_missing_in_a += 1
        if b is None:
            report.n_missing_in_b += 1
        if a is None or b is None:
            continue

        report.n_compared += 1
        if a.text == b.text:
            report.n_text_identical += 1
        if norm.normalise(a.text) == norm.normalise(b.text):
            report.n_normalised_identical += 1

        g = gold_by_key.get(key)
        if g is None:
            continue
        em_a = exact_match(a.text, g, norm)
        em_b = exact_match(b.text, g, norm)
        f1_a = token_f1(a.text, g, norm)
        f1_b = token_f1(b.text, g, norm)
        if em_a == em_b and f1_a == f1_b:
            report.n_metric_equal += 1
        pairs.append(
            ComparedPair(
                key=key,
                condition=dict(a.condition),
                a_text=a.text,
                b_text=b.text,
                a_tokens=a.prompt_tokens,
                b_tokens=b.prompt_tokens,
                em_a=em_a,
                em_b=em_b,
                f1_a=f1_a,
                f1_b=f1_b,
            )
        )

    if pairs:
        n = len(pairs)
        report.em_a = sum(p.em_a for p in pairs) / n
        report.em_b = sum(p.em_b for p in pairs) / n
        report.f1_a = sum(p.f1_a for p in pairs) / n
        report.f1_b = sum(p.f1_b for p in pairs) / n
        moved = [p for p in pairs if p.metric_moved]
        report.worst_pairs = [
            {
                "key": p.key,
                "em_delta": round(p.em_delta, 4),
                "f1_delta": round(p.f1_delta, 4),
                "a": p.a_text[:120],
                "b": p.b_text[:120],
            }
            for p in sorted(moved, key=lambda p: (-abs(p.em_delta), p.key))[:max_worst]
        ]
        report.n_pairs_moved_total = len(moved)
    return report


def select_probe_keys(
    store: ResponseStore, *, n: int, seed: int, prefer: str | None = None
) -> list[str]:
    keys = sorted(r.key for r in store.iter_all())
    if prefer:
        keys = [k for k in keys if prefer in k] or keys
    if len(keys) <= n:
        return keys
    return sorted(random.Random(seed).sample(keys, n))


def confound_verdict(
    report: DeltaReport,
    *,
    mde_log_odds: float | None,
    mde_pp_at_p50: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"observed": report.as_dict()}
    out["n_pairs_with_drifted_metric"] = report.n_pairs_metric_moved
    n_moved = report.n_pairs_metric_moved
    out["aggregate_delta_is_trustworthy"] = n_moved <= max(1, report.n_compared // 20)
    if not out["aggregate_delta_is_trustworthy"]:
        out["verdict"] = (
            f"TOO MANY PAIRS MOVED ({n_moved} of {report.n_compared}) to summarise as an "
            f"aggregate. Individual results differ between boxes even where the means "
            f"nearly cancel; the per-pair figure above is the honest one."
        )
        return out
    if mde_pp_at_p50 is None:
        out["verdict"] = "MDE unknown - cannot judge the drift against a target."
        return out
    drift_pp = abs(report.em_delta) * 100.0
    out["drift_pp"] = round(drift_pp, 4)
    out["mde_pp_at_p50"] = mde_pp_at_p50
    out["mde_log_odds"] = mde_log_odds
    if drift_pp < mde_pp_at_p50:
        out["verdict"] = (
            f"NEGLIGIBLE: {drift_pp:.3f}pp of EM drift is below the {mde_pp_at_p50:.3f}pp "
            f"the design can resolve, so it cannot flip a conclusion this design draws."
        )
    else:
        out["verdict"] = (
            f"CONFOUNDED: {drift_pp:.3f}pp of EM drift exceeds the {mde_pp_at_p50:.3f}pp "
            f"MDE. A cross-box comparison cannot be distinguished from a real effect, so "
            f"the two must be corrected for or reported as separate."
        )
    return out


def probe_plan(
    store: ResponseStore, *, n: int, seed: int, prefer: str | None = None
) -> dict[str, Any]:
    keys = select_probe_keys(store, n=n, seed=seed, prefer=prefer)
    total = 0
    found = 0
    for k in keys:
        r = store.get(k)
        if r is not None:
            total += r.prompt_tokens
            found += 1
    return {
        "n_keys": len(keys),
        "n_resolved": found,
        "prompt_tokens": total,
        "seed": seed,
        "prefer": prefer,
        "note": (
            "Re-run these exact cache keys on the second box and diff the two stores "
            "with `msat hardware-delta`. Deterministic selection is the point: a "
            "random probe makes the two runs incomparable."
        ),
    }


def condition_of(record: StoredResponse) -> Condition | None:
    try:
        return Condition.from_dict(record.condition)
    except (KeyError, TypeError, ValueError):
        return None
