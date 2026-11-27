"""Arm B, holding length constant while memory count varies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msat.arms.render import Memory

FILLER_FILENAME = "5_filler_sess.json"

TOKEN_TARGET_TOLERANCE = 0.02


class FillerMissingError(RuntimeError):
    """Raised when the padding source is missing; padding never falls back to target sessions."""


class GoldLeakageError(RuntimeError):
    """Raised when padding or retrieval touches a gold session, which invalidates the arm."""


class TokenTargetMissedError(RuntimeError):
    """Raised when padding misses the declared token target."""


@dataclass(frozen=True, slots=True)
class FillerSource:
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
