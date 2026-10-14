"""Arm B — holding LENGTH constant while COUNT varies (PRD §R9.1, §R9.10).

WHY ARM B EXISTS AT ALL
-----------------------
The primary model (R9.1) treats `length` as a continuous token count, not a binary
arm indicator. So the A-by-G crossed grid varies length and count together, and any
depth effect could be either "more memories" or "more tokens". Arm B holds the
token count fixed and varies only how many memories fill it, which is the only way
to separate the two.

THE THREE THINGS THIS ARM MUST NOT DO
-------------------------------------
1. **Pad from anywhere but `5_filler_sess`.** There is nothing outside the corpus
   to pad beyond. Padding with the target's own unretrieved sessions would inject
   real context that a budgeted arm would never have shown, and the arm would stop
   being a control.

2. **Pad with gold.** R9.10 is emphatic: constructing insufficient-evidence
   conditions by *deleting* passages leaks the label through memory size - a
   paragraph-counting classifier reaches AUROC 0.979 on MuSiQue. Any padding that
   touches gold is doubly invalid: it is the size leak, and it makes the arm partly
   correct for reasons unrelated to the manipulation.
   `assert_no_gold_leakage` is an ASSERTION, not a warning.

3. **Miss the token target silently.** A 40% budget that lands at 55% because the
   filler ran out is not a 40% arm. The target is checked within +/-2% and a miss
   raises, because a padding arm that quietly over-pads is measuring a different B
   than the one in the condition key.

WHY THE BUDGET IS RELATIVE AND NOT ABSOLUTE
-------------------------------------------
`length` is a token count, and S and M differ by 9.957x in session count while
their sessions differ by <1% in size. An absolute budget would mean one thing on S
and something else on M, and the depth-by-scale interaction (R9.11) would be
measuring the budget's drift rather than the scale's effect. `corpus_fraction`
keeps it meaning the same thing on both.

B3: `5_filler_sess.json` IS MISSING. This module therefore REFUSES rather than
substituting anything, and `FillerSource.load` names the file it wanted. A
fabricated filler set would be invisible in every downstream number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msat.arms.render import Memory

FILLER_FILENAME = "5_filler_sess.json"
"""Arm B's ONLY legal source of padding. Named here, in code, so the set of places
that could invent padding is a set of size one."""

TOKEN_TARGET_TOLERANCE = 0.02
"""PRD `arm_b.j2`: the token target is asserted within +/-2%."""


class FillerMissingError(RuntimeError):
    """The one legal padding source does not exist. Blocker B3.

    Raised rather than degraded. The alternative - padding from the target's own
    unretrieved sessions - injects context a budgeted arm would never have shown,
    and nothing downstream would detect it.
    """


class GoldLeakageError(RuntimeError):
    """Padding or retrieval touched a gold session. R9.10.

    This invalidates the ARM, not just the run: the condition is now partly
    satisfied for a reason unrelated to the manipulation, so the size->label
    classifier that makes the arm illegal is no longer the only way to recover the
    condition from the prompt.
    """


class TokenTargetMissedError(RuntimeError):
    """Padding could not reach the declared fraction of the corpus.

    A padding arm that lands at 55% when it declared 40% is measuring a different
    B than the one in its condition key, and the cache key will not notice.
    """


@dataclass(frozen=True, slots=True)
class FillerSource:
    """The one legal source of padding text."""

    sessions: tuple[str, ...]
    path: Path

    @classmethod
    def load(cls, path: Path) -> FillerSource:
        import json

        path = Path(path)
        if path.name != FILLER_FILENAME:
            raise FillerMissingError(
                f"arm B pads from {FILLER_FILENAME} and nothing else; {path.name} was offered"
            )
        if not path.exists():
            raise FillerMissingError(
                f"{path} is missing (blocker B3). Arm B has exactly one legal source "
                f"of padding text and it is absent. Substituting the target's own "
                f"unretrieved sessions would inject real context a budgeted arm "
                f"would never have shown, and nothing downstream would detect it. "
                f"Supply {FILLER_FILENAME} or disable arm B."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else raw.get("sessions", [])
        out: list[str] = []
        for r in records:
            out.append(r if isinstance(r, str) else str(r.get("text", "")))
        if not out:
            raise FillerMissingError(f"{path} declares no sessions; arm B cannot pad")
        return cls(sessions=tuple(out), path=path)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


@dataclass(frozen=True, slots=True)
class PaddingBudget:
    """A corpus-relative token target for one question."""

    corpus_tokens: int
    fraction: float
    tolerance: float = TOKEN_TARGET_TOLERANCE

    @property
    def target_tokens(self) -> int:
        return int(self.corpus_tokens * self.fraction)

    @property
    def lower(self) -> int:
        return int(self.target_tokens * (1.0 - self.tolerance))

    @property
    def upper(self) -> int:
        return int(self.target_tokens * (1.0 + self.tolerance))


def pad_to_length(
    memories: Sequence[Memory],
    budget: PaddingBudget,
    filler: FillerSource,
    *,
    gold_sessions: Sequence[str] = (),
    chars_per_token: float = 4.0,
) -> tuple[list[Memory], dict[str, Any]]:
    """Pad `memories` to the target length, from the filler and nowhere else.

    Returns the padded list and a report. The report is part of the return value
    rather than a log line because the fraction actually reached, the pad count and
    the token total all belong in the run's manifests: without them a reader
    cannot tell a 40% arm from a 38% one.
    """
    assert_no_gold_leakage(memories, gold_sessions, what="retrieved")
    gold = set(gold_sessions)

    used = sum(m.n_tokens for m in memories)
    target = budget.target_tokens
    out = list(memories)
    n_pad = 0
    for i, text in enumerate(filler.sessions):
        if used >= budget.lower:
            break
        sid = f"{filler.path.stem}#pad{i}"
        if sid in gold:
            continue
        tokens = max(1, int(len(text) / chars_per_token))
        out.append(
            Memory(
                memory_id=sid,
                session_id=sid,
                source_question_id="__filler__",
                text=text,
                n_tokens=tokens,
            )
        )
        used += tokens
        n_pad += 1

    total = sum(m.n_tokens for m in out)
    achieved = total / budget.corpus_tokens if budget.corpus_tokens else 0.0
    if not (budget.lower <= total <= budget.upper):
        raise TokenTargetMissedError(
            f"arm B targeted {budget.fraction:.0%} of a {budget.corpus_tokens:,}-token "
            f"corpus = {target:,} tokens (+/-{budget.tolerance:.0%} -> "
            f"[{budget.lower:,}, {budget.upper:,}]) but reached {total:,} "
            f"({achieved:.1%}) using {n_pad} pad session(s) from {filler.path.name}. "
            f"The filler is too small for this question. Padding more anyway would "
            f"silently measure a different B than the condition key declares."
        )
    return out, {
        "corpus_tokens": budget.corpus_tokens,
        "target_tokens": target,
        "achieved_tokens": total,
        "achieved_fraction": round(achieved, 6),
        "n_retrieved": len(memories),
        "n_padded": n_pad,
        "filler_path": str(filler.path),
        "token_tolerance": budget.tolerance,
    }


def assert_no_gold_leakage(
    memories: Sequence[Memory], gold_sessions: Sequence[str], *, what: str
) -> None:
    """Raise if any memory belongs to a gold session. R9.10.

    Checks SESSIONS, not memory ids, and not `is_gold` flags - a flag is what this
    whole arm is trying not to trust. A gold session split into three chunks
    contributes three colliding ids to a naive check and slips through; matching on
    `session_id` cannot.
    """
    gold = set(gold_sessions)
    if not gold:
        return
    hits = sorted({m.session_id for m in memories if m.session_id in gold})
    if hits:
        raise GoldLeakageError(
            f"arm B: {len(hits)} gold session(s) reached the {what} set "
            f"(e.g. {hits[:3]}). R9.10 - the label leaks through memory size and a "
            f"paragraph-counting classifier separates the conditions. This arm is "
            f"invalid, not merely noisy."
        )


def leakage_features(memories: Sequence[Memory]) -> dict[str, float]:
    """The size features R9.10's classifier is built from.

    Emitted for every arm condition so `msat analyze` can fit
    `features -> condition label` and assert the separation fails. Four features
    rather than one because a single feature can be matched while the combination
    still separates, and the MuSiQue result was 0.979 AUROC on obvious counts.
    """
    if not memories:
        return {
            "n_memories": 0.0,
            "total_tokens": 0.0,
            "mean_tokens": 0.0,
            "max_tokens": 0.0,
            "n_over_mean": 0.0,
        }
    toks = [m.n_tokens for m in memories]
    mean = sum(toks) / len(toks)
    return {
        "n_memories": float(len(toks)),
        "total_tokens": float(sum(toks)),
        "mean_tokens": mean,
        "max_tokens": float(max(toks)),
        "n_over_mean": float(sum(1 for t in toks if t > mean)),
    }


def size_leakage_matrix(
    features_by_condition: Mapping[tuple[str, ...], Mapping[str, float]],
) -> dict[str, Any]:
    """Per-cell mean size features, so a size->condition split is visible raw.

    Deliberately does not compute AUROC. Fitting the classifier is P9's job and
    the assertion belongs there; this only guarantees the features EXIST and are
    constant enough within a cell to be meaningless on their own - which is exactly
    what R9.10 requires before the classifier is trusted.
    """
    by_arm: dict[str, dict[str, list[float]]] = {}
    for condition, feats in features_by_condition.items():
        arm = str(condition[0])
        bucket = by_arm.setdefault(arm, {})
        for name, value in feats.items():
            bucket.setdefault(name, []).append(float(value))

    per_arm: dict[str, Any] = {}
    for arm, fields in by_arm.items():
        per_arm[arm] = {
            name: {
                "mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "n": len(vals),
            }
            for name, vals in sorted(fields.items())
        }

    # The one thing that can be checked without a classifier: if arm B's mean
    # `total_tokens` is far from arm A's, length is not held constant and the
    # manipulation did not happen.
    warnings: list[str] = []
    if "A" in per_arm and "B" in per_arm:
        a = per_arm["A"]["total_tokens"]["mean"]
        b = per_arm["B"]["total_tokens"]["mean"]
        if a > 0 and abs(a - b) / a > TOKEN_TARGET_TOLERANCE:
            warnings.append(
                f"arm B mean total_tokens {b:,.0f} differs from arm A {a:,.0f} by "
                f"{abs(a - b) / a:.1%}, beyond the +/-2% target. Length is not held "
                f"constant, so the count-vs-length separation this arm exists for is "
                f"not being measured."
            )
    return {"per_arm": per_arm, "warnings": warnings}
