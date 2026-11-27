"""The answer-normalisation function."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, cast


class UnknownNormalisationError(ValueError):
    """A config step names an operation this module does not implement."""


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    op: str
    params: dict[str, Any]

    def apply(self, text: str) -> str:
        if self.op == "unicode_normalise":
            form = str(self.params.get("form", "NFKC"))
            if form not in ("NFC", "NFD", "NFKC", "NFKD"):
                raise UnknownNormalisationError(
                    f"step {self.id!r}: unicode_normalise form must be one of "
                    "NFC, NFD, NFKC, NFKD - got " + repr(form)
                )
            return unicodedata.normalize(cast("Any", form), text)
        if self.op == "lowercase":
            return text.lower()
        if self.op == "strip":
            return text.strip()
        if self.op == "collapse_whitespace":
            return re.sub(r"\s+", str(self.params.get("replacement", " ")), text)
        if self.op == "regex_substitute":
            return re.sub(
                str(self.params["pattern"]),
                str(self.params.get("replacement", "")),
                text,
                flags=_re_flags(self.params.get("flags", "")),
            )
        raise UnknownNormalisationError(
            f"step {self.id!r}: unknown op {self.op!r}. Add it to "
            "`msat.metrics.normalisation.Step.apply` or remove it from "
            "`configs/prompts/normalisation.yaml` - do not leave it unimplemented, "
            "because a skipped step changes the score silently."
        )


_FLAG_MAP: Final[dict[str, int]] = {
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "UNICODE": re.UNICODE,
    "U": re.UNICODE,
}


def _re_flags(spec: str) -> int:
    flags = 0
    for token in str(spec).replace("|", " ").split():
        key = token.strip().upper()
        if not key:
            continue
        if key not in _FLAG_MAP:
            raise UnknownNormalisationError(
                f"unknown regex flag {token!r}; expected one of " + ", ".join(sorted(_FLAG_MAP))
            )
        flags |= _FLAG_MAP[key]
    return flags


@dataclass(frozen=True, slots=True)
class Normaliser:
    steps: tuple[Step, ...]
    apply_to: tuple[str, ...] = ("prediction", "gold")

    @classmethod
    def from_config(cls, spec: dict[str, Any]) -> Normaliser:
        block = spec.get("normalisation") or spec
        steps = []
        for raw in block.get("steps", []):
            params = {k: v for k, v in raw.items() if k not in ("id", "op")}
            steps.append(Step(id=str(raw["id"]), op=str(raw["op"]), params=params))
        if not steps:
            raise ValueError(
                "normalisation config declares no steps; R1.2 requires an explicit, "
                "documented normalisation"
            )
        return cls(
            steps=tuple(steps),
            apply_to=tuple(block.get("apply_to", ("prediction", "gold"))),
        )

    def normalise(self, text: str) -> str:
        out = text
        for step in self.steps:
            out = step.apply(out)
        return out


def exact_match(prediction: str, gold: str, norm: Normaliser) -> float:
    return float(norm.normalise(prediction) == norm.normalise(gold))


def token_f1(prediction: str, gold: str, norm: Normaliser) -> float:
    pred_tokens = norm.normalise(prediction).split()
    gold_tokens = norm.normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    shared = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(
    prediction: str,
    gold: str | tuple[str, ...],
    norm: Normaliser,
) -> dict[str, float]:
    golds = (gold,) if isinstance(gold, str) else tuple(gold)
    if not golds:
        return {"exact_match": 0.0, "token_f1": 0.0, "n_gold": 0.0}
    return {
        "exact_match": max(exact_match(prediction, g, norm) for g in golds),
        "token_f1": max(token_f1(prediction, g, norm) for g in golds),
        "n_gold": float(len(golds)),
    }
