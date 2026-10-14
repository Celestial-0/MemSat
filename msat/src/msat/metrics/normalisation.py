"""The answer-normalisation function (PRD §R1.2, §R8.1) — P2.3.

`configs/prompts/normalisation.yaml` declares an ORDERED list of steps. This
module executes it. Without this, R1.2's "own documented normalisation" is a
promise in a YAML file and nothing enforces it, which means the first person to
score an answer can silently invent their own rules.

WHY DECLARATIVE, NOT A FUNCTION
-------------------------------
A normaliser written in Python would be neither auditable nor reviewable. Loaded
from config it can be read exactly as it behaves, it lands in the `config_hash`
so changing a rule is a visible, reviewed change, and it cannot execute arbitrary
code from the config directory.

ORDER IS SEMANTIC
-----------------
Steps apply top to bottom and the order changes results. Removing articles before
lower-casing, or stripping punctuation before articles, changes the score on
exactly the items that are hardest to get right - which are the ones the
experiment's conclusions rest on. The sequence in the config is therefore data,
not an implementation detail, and it is preserved verbatim.

WHAT THIS DOES NOT DO
---------------------
It does not decide correctness. PRD §R8.1 requires EM, token-F1 and the official
model-based evaluator to be reported SIDE BY SIDE, never one alone, because no
single string metric is trusted by itself. A single string metric decides nothing;
the judge panel and the human sample carry the result. These functions exist to
make the string metrics reproducible, not to make them sufficient.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, cast


class UnknownNormalisationError(ValueError):
    """A config step names an operation this module does not implement.

    Raised rather than skipped. A normaliser that silently ignores a step is a
    normaliser that quietly computes something other than what the config says,
    and the failure is invisible in the results.
    """


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
    """An ordered pipeline built from `normalisation.yaml`."""

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
    """PRD §R8.1 EM: normalised strings must be identical."""
    return float(norm.normalise(prediction) == norm.normalise(gold))


def token_f1(prediction: str, gold: str, norm: Normaliser) -> float:
    """PRD §R8.1 token-F1: multiset overlap on normalised strings.

    SQuAD's formulation: shared tokens over (prediction length + gold length),
    which penalises both over- and under-generation. An empty input scores 0 on
    both sides, so an abstention that says nothing cannot look like a match.
    """
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
    """All string metrics at once, plus `max` for multi-gold.

    The config declares `multi_gold.strategy: any_match` - a gold that is a LIST
    means several acceptable answers, and the score is the max over alternatives,
    not the mean. Averaging would let a two-option gold beat a one-option gold
    purely by being scored against a half-match, which is not what "any of these
    is correct" means.
    """
    golds = (gold,) if isinstance(gold, str) else tuple(gold)
    if not golds:
        return {"exact_match": 0.0, "token_f1": 0.0, "n_gold": 0.0}
    return {
        "exact_match": max(exact_match(prediction, g, norm) for g in golds),
        "token_f1": max(token_f1(prediction, g, norm) for g in golds),
        "n_gold": float(len(golds)),
    }
