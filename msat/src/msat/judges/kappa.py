"""Cohen's kappa between judges — PRD §R8.2.

    "Compute Cohen's kappa and compare against the cautionary bar from
    2511.02246 (kappa = 0.118)."

THE NUMBER, AND WHY IT IS THE RIGHT ONE
--------------------------------------
Percent agreement between two judges is misleading in exactly the situation we
care about: when the task is mostly "incorrect", two judges who say "incorrect"
every time agree 95% of the time and are worthless. Kappa corrects for agreement
expected by chance alone, which is why a majority-class collapse to 0.118 rather
than to 1.0.

WHY THE AGREEMENT MATRIX IS STORED AND NOT JUST COMPUTED
--------------------------------------------------------
kappa is one number derived from four cells. Reporting kappa alone means a reader
cannot see WHY it is low — and the two common causes call for opposite responses.
High disagreement concentrated on the rare CORRECT cell is a judge that misses
things (calibration). High disagreement on INCORRECT means the judges are
essentially random and the panel is not measuring anything. The matrix is the
diagnosis; kappa is the symptom.

WHEN KAPPA IS UNDEFINED, IT IS REPORTED AS UNDEFINED
----------------------------------------------------
If expected agreement is 1.0 — every item got the same verdict from both judges —
the denominator of kappa is zero and the statistic does not exist. This is common:
a judge panel that is perfectly aligned on a trivially hard item. Returning 0.0, or
1.0, or crashing, all misreport it. `AgreementResult.undefined_reason` says so and
`kappa` is None.

PAIREWISE EXCLUSION, AND THE COUNT OF WHAT WAS DROPPED
------------------------------------------------------
An item one judge declined to score cannot go in the matrix. Dropping it silently
makes kappa look better as refusals rise, so `n_excluded` is reported beside every
kappa and a run that dropped a fifth of its items says so in words.

WITH MORE THAN TWO JUDGES
-------------------------
Cohen's kappa is for two raters. With N judges this returns the N(N-1)/2 PAIRWISE
values rather than inventing a multi-rater statistic nobody asked for. Reporting
the minimum and maximum pairwise kappa alongside the mean is more informative than
one number, because a wide spread means one judge is an outlier and the panel
should be re-thought.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from msat.judges.verdicts import ItemKey, Verdict

CAUTIONARY_KAPPA = 0.118
"""2511.02246. Below this, agreement is no better than chance-corrected noise and
the panel cannot support a headline number."""

EXCLUDED_WARN_FRACTION = 0.10
"""More than this pairwise means the kappa is computed on a minority of the data
and the report has to say so in words."""


@dataclass
class AgreementResult:
    """One pairwise agreement computation, with everything that explains it."""

    label_a: str
    label_b: str
    matrix: dict[tuple[Verdict, Verdict], int] = field(default_factory=dict)
    n: int = 0
    n_excluded: int = 0
    excluded_reasons: dict[str, int] = field(default_factory=dict)
    p_observed: float | None = None
    p_expected: float | None = None
    kappa: float | None = None
    undefined_reason: str = ""

    @property
    def fraction_excluded(self) -> float:
        total = self.n + self.n_excluded
        return self.n_excluded / total if total else 0.0

    def cells(self) -> dict[str, int]:
        """The matrix as plain strings, for JSON and parquet."""
        return {
            f"{a}|{b}": n
            for (a, b), n in sorted(
                self.matrix.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
            )
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_compared": self.n,
            "n_excluded": self.n_excluded,
            "fraction_excluded": round(self.fraction_excluded, 4),
            "excluded_reasons": dict(self.excluded_reasons),
            "agreement_matrix": self.cells(),
            "p_observed": None if self.p_observed is None else round(self.p_observed, 6),
            "p_expected": None if self.p_expected is None else round(self.p_expected, 6),
            "kappa": None if self.kappa is None else round(self.kappa, 6),
            "undefined_reason": self.undefined_reason,
        }


def cohen_kappa(
    a: Mapping[ItemKey, Verdict],
    b: Mapping[ItemKey, Verdict],
    *,
    label_a: str = "A",
    label_b: str = "B",
    excluded: Mapping[ItemKey, str] | None = None,
) -> AgreementResult:
    """Cohen's kappa over the items BOTH judges scored.

    `excluded` maps item -> reason for items dropped from the matrix. Passing it is
    optional only because an empty exclusion set is the common case; when it is not
    empty the count is part of the result.
    """
    res = AgreementResult(label_a=label_a, label_b=label_b)
    excluded = dict(excluded or {})

    for verdict in Verdict:
        for other in Verdict:
            res.matrix[(verdict, other)] = 0

    shared = sorted(set(a) & set(b))
    counted = 0
    for key in shared:
        if key in excluded:
            reason = excluded[key]
            res.excluded_reasons[reason] = res.excluded_reasons.get(reason, 0) + 1
            res.n_excluded += 1
            continue
        res.matrix[(a[key], b[key])] += 1
        counted += 1
    res.n = counted

    # Keys only one judge scored are excluded too, and counted. Silently ignoring
    # them would let a judge improve kappa by scoring fewer items.
    only_one = len(set(a) ^ set(b))
    if only_one:
        res.n_excluded += only_one
        res.excluded_reasons["scored by only one judge"] = (
            res.excluded_reasons.get("scored by only one judge", 0) + only_one
        )

    if res.n == 0:
        res.undefined_reason = "no items were scored by both judges"
        return res

    n = float(res.n)
    diag = sum(res.matrix[(v, v)] for v in Verdict)
    res.p_observed = diag / n

    # p_e = sum over cells of (row marginal / n) * (column marginal / n)
    row = {v: sum(res.matrix[(v, o)] for o in Verdict) for v in Verdict}
    col = {v: sum(res.matrix[(o, v)] for o in Verdict) for v in Verdict}
    res.p_expected = sum((row[v] / n) * (col[v] / n) for v in Verdict)

    denom = 1.0 - res.p_expected
    if abs(denom) < 1e-12:
        res.undefined_reason = (
            f"expected agreement is {res.p_expected:.4f} - every item received the "
            f"same verdict - so kappa's denominator is zero and the statistic does "
            f"not exist"
        )
        return res
    res.kappa = (res.p_observed - res.p_expected) / denom
    return res


def kappa_verdict(result: AgreementResult) -> dict[str, Any]:
    """Compare one kappa to the 2511.02246 cautionary bar, in words.

    The bar is a NUMBER in this module, not in a comment, and the comparison is the
    deliverable rather than the statistic. A kappa of 0.30 means something quite
    different for a panel scoring 3% correct than for one scoring 60%, so the base
    rate is reported next to it.
    """
    k = result.kappa
    out: dict[str, Any] = {
        "pair": f"{result.label_a} vs {result.label_b}",
        "kappa": None if k is None else round(k, 6),
        "cautionary_bar": CAUTIONARY_KAPPA,
        "n_compared": result.n,
        "fraction_excluded": round(result.fraction_excluded, 4),
    }
    if k is None:
        out["verdict"] = f"UNDEFINED: {result.undefined_reason}"
        return out
    if result.fraction_excluded > EXCLUDED_WARN_FRACTION:
        out["verdict"] = (
            f"{k:.3f} clears the {CAUTIONARY_KAPPA} bar, but {result.fraction_excluded:.0%} "
            f"of items were excluded pairwise, so the figure is computed on a minority "
            f"of the data."
        )
    elif k < CAUTIONARY_KAPPA:
        out["verdict"] = (
            f"BELOW the {CAUTIONARY_KAPPA} bar (2511.02246). Agreement is no better than "
            f"chance-corrected; the panel cannot support a headline number."
        )
    else:
        out["verdict"] = f"{k:.3f} clears the {CAUTIONARY_KAPPA} bar."
    return out


def base_rates(verdicts_by_judge: Mapping[str, Mapping[ItemKey, Verdict]]) -> dict[str, float]:
    """Fraction CORRECT per judge — the context kappa alone does not carry.

    Two panels can have the same kappa with opposite failure modes: one is 95%
    INCORRECT and aligned on being wrong, the other is balanced. Reporting the rate
    beside kappa is what tells them apart.
    """
    out: dict[str, float] = {}
    for judge, verdicts in verdicts_by_judge.items():
        if not verdicts:
            continue
        out[judge] = sum(1 for v in verdicts.values() if v is Verdict.CORRECT) / len(verdicts)
    return out


def panel_agreement(
    verdicts_by_judge: Mapping[str, Mapping[ItemKey, Verdict]],
    *,
    excluded: Mapping[ItemKey, str] | None = None,
) -> dict[str, Any]:
    """Every pairwise kappa for the panel, with min/mean/max.

    With two judges this is one value. With three it is three, and the spread is the
    finding: a wide range means one judge is an outlier, and averaging it away would
    hide the panel that needs re-thinking.
    """
    labels = sorted(verdicts_by_judge)
    pairs: list[AgreementResult] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            pairs.append(
                cohen_kappa(
                    verdicts_by_judge[la],
                    verdicts_by_judge[lb],
                    label_a=la,
                    label_b=lb,
                    excluded=excluded,
                )
            )

    defined = [p.kappa for p in pairs if p.kappa is not None]
    out: dict[str, Any] = {
        "judges": labels,
        "pairs": [p.as_dict() for p in pairs],
        "verdicts": [kappa_verdict(p) for p in pairs],
        "base_rate_correct": base_rates(verdicts_by_judge),
        "n_pairs": len(pairs),
    }
    if defined:
        out["kappa_min"] = round(min(defined), 6)
        out["kappa_mean"] = round(sum(defined) / len(defined), 6)
        out["kappa_max"] = round(max(defined), 6)
        out["spread"] = round(max(defined) - min(defined), 6)
        out["spread_note"] = (
            "A wide spread means one judge is an outlier. Averaging it away would hide "
            "the panel that needs re-thinking."
        )
    else:
        out["kappa_min"] = None
        out["kappa_mean"] = None
        out["kappa_max"] = None
        out["spread"] = None
    out["panel_usable"] = bool(defined) and min(defined) >= CAUTIONARY_KAPPA
    return out


def agreement_matrix_table(
    verdicts_by_judge: Mapping[str, Mapping[ItemKey, Verdict]],
) -> dict[str, dict[str, dict[str, int]]]:
    """The N x N matrix of raw agreement counts, for the figure.

    Counts, not rates: a rate over a different number of scored items per judge is
    not comparable, and the counts make the denominators visible.
    """
    labels = sorted(verdicts_by_judge)
    out: dict[str, dict[str, dict[str, int]]] = {}
    for la in labels:
        out[la] = {}
        for lb in labels:
            r = cohen_kappa(verdicts_by_judge[la], verdicts_by_judge[lb], label_a=la, label_b=lb)
            out[la][lb] = r.cells()
    return out


def format_kappa_table(agreement: Mapping[str, Any]) -> str:
    """A fixed-width table for the CLI. Column widths are fixed because a number
    whose position moves between rows is a number nobody reads twice."""
    lines = [f"  {'pair':<28}{'kappa':>9}{'bar':>8}{'n':>8}{'excluded':>10}"]
    for v in agreement.get("verdicts", []):
        k = "n/a" if v["kappa"] is None else f"{v['kappa']:.3f}"
        lines.append(
            f"  {v['pair']:<28}{k:>9}{v['cautionary_bar']:>8.3f}"
            f"{v['n_compared']:>8}{v['fraction_excluded']:>9.0%}"
        )
    return "\n".join(lines)
