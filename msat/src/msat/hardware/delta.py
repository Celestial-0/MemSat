"""Bounding the hardware confound instead of assuming it away.

THE PROBLEM
-----------
Response cache keys include the model, the request, the decoding parameters and
the schema version. They do NOT include the hardware - and they cannot, because
the same checkpoint on two boxes is genuinely the same condition.

That matters the moment one arm is generated somewhere else. Arm D needs an
80 GB card and everything else fits on a 4060, so "Arm D on the A100" and "Arm A
on the 4060" become entries in one store with one table. At `temperature=0` the
two boxes should agree, and mostly they do - but KV accumulation order depends on
batch composition, and on a 122k prompt that is not a rounding error you can hand
wave. A few tenths of a percentage point is enough to flip the sign of a
"full context does not beat retrieval" claim at the margin.

The claim is the whole paper. Unmeasured hardware drift under it is not
acceptable, and neither is running the whole grid twice on rented hardware.

THE HARNESS
-----------
Re-run a small, DETERMINISTIC probe of conditions on the second box, then compare
the two stores. ~100 cells is ~4M prompt tokens: a few minutes of rented time,
against a 292-hour grid.

FOUR LAYERS, AND THE ORDER MATTERS
----------------------------------
    text identical          do the raw strings match at all
    normalised identical    do they match after R1.2 normalisation
    metric equal            does the SCORE match
    metric delta            by how much does the score move

Reporting only the first layer is the trap. Two boxes that produce different
whitespace score identically, and a reader told "0% identical" concludes the
comparison is broken. The layer that actually threatens a result is the last one.

THE VERDICT IS MEASURED AGAINST THE MDE, NOT INVENTED
------------------------------------------------------
There is no threshold for "acceptable hardware drift" anywhere in this project,
and inventing one would be arbitrary. There IS a threshold that matters: the
smallest effect the design can detect (`mde_interaction`, ~0.006 log-odds). If
|metric delta| is below it, the hardware cannot flip the claims being made. If
it is above, the design cannot distinguish "this box" from "that box" and the
comparison is confounded regardless of how small both numbers look.

Reusing the power machinery means the confound is judged against what the study
can actually resolve, which is the only comparison that means anything.

ONE CODE PATH, TWO USES
-----------------------
`compare_stores` is symmetric. Comparing box A to box B measures hardware drift;
comparing a store to a COPY OF ITSELF measures backend nondeterminism (R8.6).
The self-comparison must return exactly zero - that is the harness's own proof it
is not measuring itself wrong.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from msat.metrics.normalisation import Normaliser, exact_match, token_f1
from msat.store.responses import ConditionKey, ResponseStore, StoredResponse


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
    """How far two boxes drift, at every layer."""

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
    """Pairs whose SCORE changed, however slightly.

    Tracked separately from `em_delta`, because two pairs swapping 1 and 0
    cancel in the mean while both individual results moved. When this is large,
    the aggregate is not the honest summary.
    """
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
    """Compare the same cache keys across two stores.

    `gold_by_key` maps CACHE KEY -> gold answer, not qid -> gold. A `ConditionKey`
    is `(arm, K, B, b_mode, scale, reader)` and carries NO question id, so a qid
    lookup here would miss every single time and return an all-zeros report that
    reads as "no drift measured" - the most dangerous possible failure for a
    tool whose entire job is measuring drift. The caller resolves keys to gold.

    A key with no gold entry is compared for TEXT only: it counts toward the
    identity ratios but not toward the metric means. Dropping it entirely would
    understate drift; scoring it as a zero on both boxes would fold a missing
    answer into what is supposed to be a hardware measurement.
    """
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
    """A deterministic sample of cache keys.

    Deterministic because the probe has to be re-runnable: the whole point is to
    run the SAME cells on both boxes, and a random probe makes the two runs
    incomparable. Sorted before sampling so the order does not depend on the
    filesystem.
    """
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
    """Is the drift below what the design can resolve?

    No arbitrary threshold is introduced. The comparison is between the observed
    drift and the smallest effect the study claims to detect, because that is the
    only boundary that means anything: a drift smaller than the MDE cannot change
    any conclusion the design is entitled to draw.
    """
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
    """What to re-run on the second box, and what it will cost.

    Reported in PROMPT TOKENS, because that is what the rental is priced and
    scheduled by. A probe described only in cells gives no way to decide whether
    it fits the evening.
    """
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


def condition_of(record: StoredResponse) -> ConditionKey | None:
    try:
        return ConditionKey.from_dict(record.condition)
    except (KeyError, TypeError, ValueError):
        return None
