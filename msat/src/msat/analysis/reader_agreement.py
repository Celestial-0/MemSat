"""R9.12 - where the depth response has a different SIGN for different readers.

WHY THIS IS A HEADLINE AND NOT A ROBUSTNESS FOOTNOTE
----------------------------------------------------
PRD R9.12, citing 2606.21807: that paper found compression gain **decreases** with
reader baseline in 9 of 10 settings and **flips 31% of pairwise rankings**. A sign
flip means the answer to "does more retrieved depth help?" is *reader-dependent* -
so a depth curve measured at one reader is not a property of retrieval, and the
paper's own central claim cannot generalise across readers until this is reported.

THE AGGREGATE HIDES IT, AND WE HAVE BEEN BURNED THREE TIMES ALREADY
--------------------------------------------------------------------
This is the same shape as R8.6's two-pass replication, where 2609.38021's "479 vs
475" net difference concealed 8 rows whose verdict flipped; and as the
hardware-delta pairs that cancelled; and as gate branches that cancelled.

Two readers disagreeing about sign is a **count**, not a mean. Report the mean
difference between the readers' depth slopes and it is near zero whenever the
disagreements balance - which is exactly what happens when half the settings run one
way and half the other. The mean would then read "readers agree", and 31% of
pairwise rankings would be invisible.

So `n_flips` and `frac_flipped` are the headline here, and any signed aggregate is
carried beside them as context only. `n_flips` is not a sufficient statistic
either - see `directional_balance` - but it is the one that cannot be zero while
the effect is real.

THE COMPARISON IS RESTRICTED TO COMMON SUPPORT, AND THE DROPS ARE REPORTED
-------------------------------------------------------------------------
`configs/readers.yaml` documents a hard fact: on an 8 GB card the three families
cannot all serve the deepest cells (gemma ~66k tokens, qwen ~21k, phi ~44k). So the
readers do **not** share a K support, and a "K-curve sign" computed over gemma's
K = [1, 2, 4, 8, 16, 32] and qwen's K = [1, 2, 4] are answers to *different
questions*.

Dropping the cells where support differs would make that truncation invisible -
precisely the "silently-truncated factorial" that R0.3 and N8 forbid. So
`unsupported_cells` is a reported field with its own count, never an absent one, and
`compare_readers` refuses by default rather than quietly narrowing.

WHAT A SIGN MEANS
-----------------
`+1` deeper retrieval raises the score, `-1` it lowers it, `0` the response is flat
or undefined. **A flat response is not a flip and is never rounded to a sign.** Two
readers that are both flat agree; a reader that is flat and one that is positive do
not "flip" - there is no ranking to flip. Forcing a sign onto a zero would inflate
the flip rate with non-events, which is the mirror image of the bug this module
exists to prevent.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: 2606.21807's measured flip rate. A REFERENCE to compare against, never a
#: pass/fail threshold: their setting is KV-cache compression on a different task
#: panel, so agreeing with 31% would be a coincidence and disagreeing would not
#: refute anything. Treating it as a threshold would invite tuning the detector to
#: reproduce a number from another paper.
REFERENCE_FLIP_RATE: float = 0.31
REFERENCE_SOURCE = "2606.21807"

#: A slope through fewer points than this is not a trend. Two points always have a
#: slope, and reporting it would make every reader-pair at K = [1, 2] "agree" or
#: "flip" on the strength of one measurement.
MIN_POINTS_FOR_SLOPE: int = 3

#: |slope| below this counts as flat. At 2 points per doubling, this is a ~1pp move
#: over a doubling of depth - inside the noise of a 500-question cell, and calling it
#: a directional response would manufacture flips.
FLAT_SLOPE: float = 1e-9

FLAT = 0
RISES = 1
FALLS = -1

SIGN_NAME = {RISES: "rises", FALLS: "falls", FLAT: "flat"}


class UnsupportedComparisonError(ValueError):
    """Two readers were compared over different K supports.

    Refused by default because the alternative is a silent truncation, and a
    truncated factorial that looks complete is the specific failure R0.3/N8 exist
    to prevent. Pass `on_unsupported="skip"` only alongside a report that surfaces
    what was skipped.
    """


@dataclass(frozen=True, slots=True)
class CellScore:
    """One reader's mean score at one cell."""

    reader: str
    k: int
    b: float
    scale: str
    score: float
    arm: str = "A"
    n_questions: int = 0


@dataclass(frozen=True, slots=True)
class DepthResponse:
    """How one reader responds to retrieval depth, at one (B, scale, arm)."""

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
    """Slope of score against `log2(K)` for one reader at one cell.

    **`log2(K)`, not raw K.** R9.1's model is in log2 space, and a slope computed in
    raw K space weights each doubling by a different factor, so two readers whose
    supports differ - which they do, see the module docstring - would get slopes on
    different scales and their SIGNS would not be comparable.

    Only strictly positive K enters: `log2(0)` is undefined, and a K of 0 is not a
    retrieval condition.
    """
    usable = sorted((int(k), float(s)) for k, s in points if int(k) > 0)
    ks = tuple(k for k, _ in usable)

    if len(usable) < min_points:
        # FLAT, and `slope` stays None. Reporting a slope through two points would
        # let a single noisy cell decide a sign.
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
        # Every observation at the same K: there is no trend to measure. Distinct
        # `ks` normally makes this unreachable, but a duplicated K with different
        # scores would land here and dividing would raise rather than answer.
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


def responses_from_cells(
    cells: Iterable[CellScore], *, min_points: int = MIN_POINTS_FOR_SLOPE
) -> list[DepthResponse]:
    """One `DepthResponse` per (reader, b, scale, arm).

    Duplicate (reader, K, b, scale, arm) observations are averaged rather than both
    kept. They should not exist - `score_store` emits one row per (condition, qid) -
    but if they did, two rows at the same K would make `xs` have repeats and weight
    that K twice.
    """
    buckets: dict[tuple[str, float, str, str], dict[int, list[float]]] = {}
    for c in cells:
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


# ── the comparison ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PairComparison:
    """One (reader, reader, b, scale, arm) verdict."""

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
    """`common` | `truncated_a` | `truncated_b` | `truncated_both`."""

    @property
    def flipped(self) -> bool:
        """Signs disagree, and both are actually signs.

        A flat response has no ranking to flip. Counting `flat` vs `rises` as a flip
        would inflate the rate with non-events and make the number look alarming for
        a reason that has nothing to do with reader dependence.
        """
        return self.sign_a != FLAT and self.sign_b != FLAT and self.sign_a != self.sign_b

    @property
    def verdict(self) -> str:
        if self.sign_a == FLAT and self.sign_b == FLAT:
            # Both flat: no disagreement. Distinct from one-flat, which is a
            # missing ranking rather than an agreement - and lumping the two
            # together would mean `n_agree` silently excluded every pair of readers
            # that is genuinely indifferent to depth.
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
    """Classify whether the two readers' K supports coincide, and take the shared one."""
    ka, kb = set(a.ks), set(b.ks)
    if ka == kb:
        return "common", tuple(sorted(ka & kb))
    if not (ka & kb):
        return "truncated_both", ()
    if kb < ka:
        return "truncated_b", tuple(sorted(ka & kb))
    if ka < kb:
        return "truncated_a", tuple(sorted(ka & kb))
    # Partial overlap, neither a subset: worse than a truncation, because there is no
    # single reader whose support to call the narrower one.
    return "partial_overlap", tuple(sorted(ka & kb))


@dataclass
class ReaderAgreementReport:
    """R9.12's answer, with the parts that are easy to forget reported anyway."""

    comparisons: list[PairComparison] = field(default_factory=list)
    unsupported: list[PairComparison] = field(default_factory=list)
    """Comparisons refused because the readers did not share a K support. REPORTED,
    never dropped in silence - R0.3/N8 forbid a silently-truncated factorial."""

    responses: list[DepthResponse] = field(default_factory=list)
    strengths: Mapping[str, float] = field(default_factory=dict)
    """reader id -> `reader_strength` (R7.6). Drives `by_strength_direction`, the
    direct test of 2606.21807's claim. A FIELD rather than a closure, because a
    breakdown that silently finds no strength data returns zeros, and zeros read as
    a finding rather than as a gap in the input."""

    # ── the headline ──
    @property
    def n_pairs(self) -> int:
        return len(self.comparisons)

    @property
    def n_flips(self) -> int:
        return sum(1 for c in self.comparisons if c.flipped)

    @property
    def flip_rate(self) -> float | None:
        """Share of comparable pairs whose depth response disagrees in SIGN.

        `None`, not `0.0`, when nothing is comparable. "No pairs" and "no flips" are
        different states and must not print the same.
        """
        return self.n_flips / self.n_pairs if self.n_pairs else None

    @property
    def n_agree(self) -> int:
        return sum(1 for c in self.comparisons if c.verdict == "agree")

    @property
    def n_flat_involved(self) -> int:
        """Pairs where at least one reader was flat. Excluded from the flip rate,
        and counted here so the exclusion is visible rather than assumed."""
        return sum(1 for c in self.comparisons if c.verdict == "flat_involved")

    @property
    def net_delta(self) -> int | None:
        """Signed context only. RISES - FALLS over the *flipped* pairs.

        Zero here is the expected outcome for a real effect: the flips cancel. It is
        reported so a reader can see the cancellation rather than infer stability
        from the aggregate - the R8.6 lesson in one line.
        """
        flips = [c for c in self.comparisons if c.flipped]
        if not flips:
            return 0 if self.n_pairs else None
        return sum(1 for c in flips if c.sign_a == RISES) - sum(
            1 for c in flips if c.sign_a == FALLS
        )

    @property
    def directional_balance(self) -> float | None:
        """How lopsided the flips are. `n_flips` alone cannot tell these apart:

            10 flips, all A-rises/B-falls   -> B is systematically worse at depth
            10 flips, 5 each way            -> the two readers are near-symmetric

        Same headline count, different findings, different fixes. `None` when there
        are no flips, because the balance of nothing is not zero.
        """
        flips = [c for c in self.comparisons if c.flipped]
        if not flips:
            return None
        up = sum(1 for c in flips if c.sign_a == RISES)
        return abs(up - (self.n_flips - up)) / self.n_flips

    @property
    def unsupported_rate(self) -> float:
        """Share of would-be comparisons that had to be refused.

        On the 8 GB card this is expected to be material: qwen cannot serve the
        deepest cells. A high number here is a fact about the hardware, and it means
        the flip rate is measured over a narrower depth range than the full grid.
        """
        total = len(self.comparisons) + len(self.unsupported)
        return len(self.unsupported) / total if total else 0.0

    # ── strata ──
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
        """Flips split by whether the STRONGER reader is the one that improves.

        This is the direct test of 2606.21807's claim that compression gain
        *decreases* with reader baseline. It is a count, not a correlation: with
        three readers there are three pairs, and a correlation over three points
        would be reported as a finding.

        `unknown` is a reported bucket, not a zero. With no strength data supplied
        every flip would otherwise land in `equal_strength` and read as "strength
        makes no difference", which is the opposite of what is being tested.
        """
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
        """One sentence a reader can act on, or one sentence saying why not."""
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
    """Compare every reader pair, cell by cell. R9.12.

    `on_unsupported` is `"refuse"` (default), `"skip"` or `"count"`. All three
    RECORD the unsupported comparisons in `report.unsupported`; they differ only in
    whether the default is an error. There is no option that discards them, because
    a discarded comparison is a hole in the factorial and nobody would see it.

    `strengths` maps reader id -> `reader_strength` (R7.6) and drives
    `by_strength_direction`, the direct test of 2606.21807's "gain decreases with
    reader baseline".
    """
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
            # A shared K support is not enough: the SIGN itself must be measured on
            # enough points, or it is flat-by-default and every pair with a real
            # response would read as "flat_involved".
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


def synthetic_cells() -> list[CellScore]:
    """A panel whose flip behaviour is the one 2606.21807 describes.

    Shaped deliberately: gemma RISES with depth at every budget, qwen FALLS at half
    of them and rises at the rest, phi is flat throughout. A mean-based summary of
    this panel would land near zero while a third of the pairs genuinely disagree.
    """
    out: list[CellScore] = []
    for scale in ("S", "M"):
        for b in (128.0, 512.0, 2048.0):
            for i, k in enumerate((1, 2, 4, 8, 16, 32)):
                gemma = 0.30 + 0.045 * i
                # qwen flips direction with budget: falls at 128, rises at 2048.
                qwen = (0.34 - 0.020 * i) if b <= 512.0 else (0.30 + 0.030 * i)
                phi = 0.40
                for reader, score in (("gemma", gemma), ("qwen", qwen), ("phi", phi)):
                    out.append(
                        CellScore(
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
