"""R8.6 — two passes of the same deterministic system under the same judge.

    "Replicate 2609.38021's protocol. Their LongMemEval-S result (479 vs 475 / 500,
    8 flipped rows) is the reference bound. Our judge-noise floor is measured, not
    speculative."

THE SUBTLETY THE PROTOCOL EXISTS TO EXPOSE
------------------------------------------
479 correct in pass 1 and 475 in pass 2 is a difference of 4. **The number of rows
whose verdict changed is 8.**

Those two numbers are not the same and the difference is the whole point:

    6 items went correct -> wrong, 2 items went wrong -> correct
    479 - 6 + 2 = 475       and 6 + 2 = 8 flipped rows

Flips in opposite directions cancel in the aggregate. Reporting only the net
difference makes a judge that is unstable on 1.6% of items look like one that moved
by 0.8%, and halving the reported instability understates it exactly when it is
large enough to matter. `n_flips` is the headline; `net_delta` is context.

WHY THIS IS A JUDGE-NOISE FLOOR AND NOT A BUG REPORT
----------------------------------------------------
The protocol runs the SAME deterministic system twice under the SAME judge. Any
flip is the judge or the backend, not the manipulation. That number is therefore the
threshold below which no result of this study means anything: an effect smaller than
the judge's own flip rate is not an effect.

It is the same reasoning as R8.5's queue-wait split and as the hardware-delta
cancelling pairs: an aggregate can hide instability, and the per-item count is the
honest one.

DETERMINISM IS ASSUMED AND CHECKED
----------------------------------
Two passes of a deterministic system should be identical. If they are not, either
the system is not deterministic or the judge is not - and the report says which by
naming `identical`. A flip count that is compared against a reference bound is only
meaningful if the protocol was actually the one described.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from msat.judges.verdicts import ItemKey, Verdict

# 2609.38021's LongMemEval-S result, the reference bound for this protocol.
REFERENCE_N = 500
REFERENCE_PASS1_CORRECT = 479
REFERENCE_PASS2_CORRECT = 475
REFERENCE_FLIPS = 8


@dataclass
class FlipReport:
    """One two-pass comparison of a single judge."""

    judge_id: str
    n_items: int = 0
    n_excluded: int = 0
    correct_pass1: int = 0
    correct_pass2: int = 0
    n_flips: int = 0
    correct_to_wrong: int = 0
    wrong_to_correct: int = 0
    flips: list[tuple[ItemKey, str]] = field(default_factory=list)
    """`(item_key, direction)` per flipped row.

    The direction is recorded per item rather than inferred from a counter, because
    `correct_to_wrong` is a total and the flipped rows are not ordered by direction.
    """
    judge_ids_differ: bool = False
    """True if the two passes were not run under the SAME judge, which invalidates
    the whole comparison: a difference between two different judges is a panel
    disagreement, not a replication."""

    @property
    def net_delta(self) -> int:
        """The number a careless report would quote."""
        return self.correct_pass2 - self.correct_pass1

    @property
    def identical(self) -> bool:
        return self.n_flips == 0

    @property
    def flip_rate(self) -> float:
        return self.n_flips / self.n_items if self.n_items else 0.0

    @property
    def directional_balance(self) -> float | None:
        """|correct->wrong - wrong->correct| / total flips.

        Near 0 means flips go both ways and cancel; near 1 means the judge has a
        drift in one direction, which is a different problem from being unstable.
        `n_flips` alone cannot tell these apart, and the two have different fixes:
        a balanced judge needs batching or a reproducibility fix, a drifting judge
        needs its cache or its ordering examined.
        """
        if self.n_flips == 0:
            return None
        return abs(self.correct_to_wrong - self.wrong_to_correct) / self.n_flips

    def as_dict(self) -> dict[str, Any]:
        bal = self.directional_balance
        return {
            "judge_id": self.judge_id,
            "n_items": self.n_items,
            "n_excluded": self.n_excluded,
            "correct_pass1": self.correct_pass1,
            "correct_pass2": self.correct_pass2,
            "net_delta": self.net_delta,
            "n_flips": self.n_flips,
            "flip_rate": round(self.flip_rate, 6),
            "correct_to_wrong": self.correct_to_wrong,
            "wrong_to_correct": self.wrong_to_correct,
            "directional_balance": None if bal is None else round(bal, 6),
            "identical": self.identical,
            "judge_ids_differ": self.judge_ids_differ,
            "flips": [[*k, d] for k, d in self.flips],
        }

    @property
    def flipped_keys(self) -> list[ItemKey]:
        return [k for k, _ in self.flips]


def two_pass_replication(
    pass1: Mapping[ItemKey, Verdict],
    pass2: Mapping[ItemKey, Verdict],
    *,
    judge_id: str = "judge-1",
    pass1_judge: str | None = None,
    pass2_judge: str | None = None,
    excluded: Mapping[ItemKey, str] | None = None,
) -> FlipReport:
    """Compare two passes. `n_flips` counts rows whose verdict CHANGED.

    An item scored in only ONE pass cannot be replicated, so it is counted in
    `n_excluded` rather than dropped. The same reasoning as pairwise exclusion in
    `kappa`: a judge that quietly skips items makes the flip rate look better, and
    the number that went missing has to be visible.
    """
    rep = FlipReport(judge_id=judge_id)
    rep.judge_ids_differ = bool(pass1_judge and pass2_judge and pass1_judge != pass2_judge)
    excluded = dict(excluded or {})
    rep.n_excluded += len(set(pass1) ^ set(pass2))

    for key in sorted(set(pass1) & set(pass2)):
        if key in excluded:
            rep.n_excluded += 1
            continue
        a, b = pass1[key], pass2[key]
        rep.n_items += 1
        if a is Verdict.CORRECT:
            rep.correct_pass1 += 1
        if b is Verdict.CORRECT:
            rep.correct_pass2 += 1
        if a is b:
            continue
        rep.n_flips += 1
        if a is Verdict.CORRECT:
            rep.correct_to_wrong += 1
            rep.flips.append((key, "correct_to_wrong"))
        else:
            rep.wrong_to_correct += 1
            rep.flips.append((key, "wrong_to_correct"))
    return rep


def flip_verdict(report: FlipReport) -> dict[str, Any]:
    """The judge-noise floor, against 2609.38021's reference bound."""
    ref_rate = REFERENCE_FLIPS / REFERENCE_N
    out: dict[str, Any] = {
        "observed": report.as_dict(),
        "reference": {
            "source": "2609.38021 LongMemEval-S",
            "n": REFERENCE_N,
            "pass1_correct": REFERENCE_PASS1_CORRECT,
            "pass2_correct": REFERENCE_PASS2_CORRECT,
            "net_delta": REFERENCE_PASS2_CORRECT - REFERENCE_PASS1_CORRECT,
            "n_flips": REFERENCE_FLIPS,
            "flip_rate": round(ref_rate, 6),
            "note": (
                f"The reference moved {REFERENCE_PASS2_CORRECT - REFERENCE_PASS1_CORRECT} "
                f"in aggregate but flipped {REFERENCE_FLIPS} rows. This is the exact "
                f"separation the protocol is for."
            ),
        },
    }
    if report.judge_ids_differ:
        out["verdict"] = (
            "INVALID: the two passes ran under DIFFERENT judges. A difference between "
            "two judges is a panel disagreement, not a replication of one judge."
        )
        return out
    if report.n_items == 0:
        out["verdict"] = "No items were scored in both passes."
        return out
    if report.identical:
        out["verdict"] = (
            "IDENTICAL across two passes of the same deterministic system. The "
            "judge contributes no measurable instability."
        )
        return out
    ratio = report.flip_rate / ref_rate if ref_rate else float("inf")
    out["flip_rate_ratio_vs_reference"] = round(ratio, 4)
    if ratio <= 1.0:
        out["verdict"] = (
            f"{report.n_flips} flips over {report.n_items} items "
            f"({report.flip_rate:.2%}), at or below the {ref_rate:.2%} reference bound. "
            f"The judge-noise floor is no worse than 2609.38021's."
        )
    else:
        out["verdict"] = (
            f"{report.n_flips} flips over {report.n_items} items "
            f"({report.flip_rate:.2%}), ABOVE the {ref_rate:.2%} reference bound. Any "
            f"result smaller than this is inside the judge's own noise and is not a "
            f"result."
        )
    return out


def replicate_across_panel(
    pass1_by_judge: Mapping[str, Mapping[ItemKey, Verdict]],
    pass2_by_judge: Mapping[str, Mapping[ItemKey, Verdict]],
) -> dict[str, Any]:
    """Run the protocol for every judge and summarise the floor.

    The floor is the WORST judge, not the mean: a single unstable judge in the panel
    sets the noise floor for every number the panel produces, because any item it
    scores can move.
    """
    reports = []
    for judge in sorted(set(pass1_by_judge) & set(pass2_by_judge)):
        reports.append(
            two_pass_replication(
                pass1_by_judge[judge],
                pass2_by_judge[judge],
                judge_id=judge,
                pass1_judge=judge,
                pass2_judge=judge,
            )
        )
    verdicts = [flip_verdict(r) for r in reports]
    rates = [r.flip_rate for r in reports if r.n_items]
    return {
        "protocol": "R8.6 - two passes of the same deterministic system under the same judge",
        "judges": [r.as_dict() for r in reports],
        "verdicts": verdicts,
        "worst_flip_rate": round(max(rates), 6) if rates else None,
        "best_flip_rate": round(min(rates), 6) if rates else None,
        "noise_floor_rule": (
            "Any effect smaller than the WORST judge's flip rate is inside the judge's "
            "own noise. The floor is set by the worst judge, not the mean."
        ),
    }


def reference_bound() -> dict[str, Any]:
    """2609.38021's numbers, as a constant rather than a prose claim.

    The `note` is not decoration: the aggregate moved by 4 while 8 rows flipped, and
    a reader who sees only the net delta will conclude the judge was half as unstable
    as it was.
    """
    return {
        "source": "2609.38021",
        "scale": "LongMemEval-S",
        "n": REFERENCE_N,
        "pass1_correct": REFERENCE_PASS1_CORRECT,
        "pass2_correct": REFERENCE_PASS2_CORRECT,
        "net_delta": REFERENCE_PASS2_CORRECT - REFERENCE_PASS1_CORRECT,
        "n_flips": REFERENCE_FLIPS,
        "flip_rate": round(REFERENCE_FLIPS / REFERENCE_N, 6),
        "note": (
            f"Moved {REFERENCE_PASS2_CORRECT - REFERENCE_PASS1_CORRECT} in aggregate but "
            f"flipped {REFERENCE_FLIPS} rows. This is the exact separation the protocol "
            f"is for, and the reason n_flips is the headline rather than net_delta."
        ),
    }


def flipped_items(
    report: FlipReport, texts: Mapping[ItemKey, Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Per-item detail for the flipped rows.

    R8.3: verdicts and disagreements are retained per item, never reduced to an
    aggregate. A flip count without the items that flipped cannot be investigated,
    and a flip on an abstention item is a different finding from a flip on a
    scored one.
    """
    out: list[dict[str, Any]] = []
    for key, direction in report.flips:
        row: dict[str, Any] = {
            "condition_key": key[0],
            "qid": key[1],
            "direction": direction,
        }
        row.update(dict(texts.get(key, {})))
        out.append(row)
    return out


def format_flip_table(report: Mapping[str, Any]) -> str:
    lines = [f"  {'judge':<18}{'items':>8}{'p1':>7}{'p2':>7}{'net':>7}{'flips':>8}{'rate':>9}"]
    for j in report.get("judges", []):
        lines.append(
            f"  {j['judge_id']:<18}{j['n_items']:>8}{j['correct_pass1']:>7}"
            f"{j['correct_pass2']:>7}{j['net_delta']:>+7}{j['n_flips']:>8}"
            f"{j['flip_rate']:>9.2%}"
        )
    ref = report.get("verdicts", [{}])[0].get("reference") if report.get("verdicts") else None
    if ref:
        lines.append(
            f"  {'reference ' + ref['source']:<18}{ref['n']:>8}{ref['pass1_correct']:>7}"
            f"{ref['pass2_correct']:>7}{ref['net_delta']:>+7}{ref['n_flips']:>8}"
            f"{ref['flip_rate']:>9.2%}"
        )
    return "\n".join(lines)


def judged_items_for_pass(
    verdicts: Sequence[Mapping[str, Any]], judge_id: str, pass_index: int
) -> dict[ItemKey, Verdict]:
    """Pull one judge's one pass out of a flat record list."""
    return {
        (str(r["condition_key"]), str(r["qid"])): Verdict(str(r["verdict"]))
        for r in verdicts
        if str(r["judge_id"]) == judge_id and int(r.get("pass_index", 1)) == pass_index
    }
