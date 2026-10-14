"""Arm B label-leakage assertion (PRD §R9.10).

THE PROBLEM
-----------
Arm B constructs "insufficient evidence" conditions. 2609.32269 showed that
doing this by DELETING passages leaks the label through memory size: a classifier
that only counts paragraphs reaches **AUROC 0.979** on MuSiQue. If the conditions
are distinguishable by size alone, then any accuracy difference between them is
partly a size artefact rather than an evidence effect, and the count-vs-length
claim is not supported.

WHAT THIS DOES
--------------
Fits a classifier from *(memory size features -> condition label)* and asserts it
**cannot** separate the conditions. If it can, the arm is invalid and the run is
flagged.

WHY THIS IS IN NUMPY AND NOT scikit-learn
------------------------------------------
`scikit-learn` was in `pyproject.toml` for exactly one reason — this check. It
was removed on 2026-10-12 for three reasons that turned out to matter:

1. It is **not used anywhere else**. Declared, required by the PRD's library
   table, and never imported.
2. The classifier is a **logistic regression**, and this package already contains
   one: `design.fit_logit_cluster_robust`.
3. It was **blocking two phases** on the authoring host (Smart App Control
   refuses two of its 69 extension modules), so a dependency that was not yet
   used was preventing P7 and P9 from running at all.

`roc_auc_score` is a rank statistic — Mann-Whitney U — which is ten lines. The
whole check is now ~120 lines with no new dependency, and it is testable against
hand-computed values.

A NOTE ON WHAT AUROC 0.979 MEANS HERE
-------------------------------------
The threshold is a CONFIG value, not a constant, because the threshold depends on
how many cells the arm has. With 2 conditions, a weak classifier can score high
by chance. `tests/test_no_leakage.py` pins the null: on shuffled labels the
measured AUROC must sit near 0.5, and the assertion is meaningless if that null
does not hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from msat.design import fit_logit_cluster_robust

#: Above this, memory size predicts the condition and Arm B is invalid.
#:
#: The prior art is AUROC 0.979 for a paragraph-counting classifier on MuSiQue.
#: 0.70 is deliberately far below that: this is a smoke alarm, not a replication
#: target. It lives in config so the choice is reviewable and appears in the
#: `config_hash`.
DEFAULT_AUROC_THRESHOLD: Final = 0.70

#: Feature names. Kept explicit and few: every extra feature is another chance
#: for the classifier to find a spurious cue, and a leakage detector with many
#: features detects leakage that is not there.
SIZE_FEATURES: Final[tuple[str, ...]] = (
    "n_memories",
    "n_tokens",
    "n_turns",
    "mean_tokens_per_memory",
)


@dataclass(frozen=True, slots=True)
class LeakageFeatures:
    """Per-item size features for one condition instance."""

    label: int
    """1 = the padded / full-length condition, 0 = the short condition."""
    n_memories: int
    n_tokens: int
    n_turns: int

    def as_vector(self) -> list[float]:
        mean_tokens = self.n_tokens / max(self.n_memories, 1)
        return [
            float(self.n_memories),
            float(self.n_tokens),
            float(self.n_turns),
            mean_tokens,
        ]


def roc_auc(y_true: Any, scores: Any) -> float:
    """AUROC via the Mann-Whitney U statistic, with ties counted as half.

    Equal to the probability that a randomly chosen positive scores above a
    randomly chosen negative. Hand-computable, which is the point: the test
    suite checks this against a fixture where the answer is obvious.

    Accepts any array-like and coerces internally. Callers pass numpy arrays from
    the classifier and plain lists from the hand-checked fixtures, and making
    both convert by hand at each call site is a place for a bug to hide.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Average rank for tied scores (1-based), which is what makes ties
        # contribute exactly 0.5 rather than being counted arbitrarily.
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class LeakageReport:
    auroc: float
    threshold: float
    n_items: int
    n_positive: int
    null_auroc: float
    """AUROC on shuffled labels. If this is not near 0.5, the measurement is
    meaningless — so it is reported beside the result, always."""
    passed: bool
    features: tuple[str, ...] = SIZE_FEATURES
    per_feature_auroc: dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "auroc": round(self.auroc, 6),
            "threshold": self.threshold,
            "n_items": self.n_items,
            "n_positive": self.n_positive,
            "null_auroc": round(self.null_auroc, 6),
            "per_feature_auroc": {k: round(v, 6) for k, v in self.per_feature_auroc.items()},
            "passed": self.passed,
            "features": list(self.features),
            "verdict": self.verdict,
        }


def assert_no_leakage(
    items: Sequence[LeakageFeatures],
    *,
    threshold: float = DEFAULT_AUROC_THRESHOLD,
    seed: int = 0,
    n_null: int = 20,
) -> LeakageReport:
    """Fit the size-only classifier and assert it cannot separate the conditions.

    Raises nothing: the caller decides what a failure means (PRD §R9.10 says the
    ARM is invalid and the run is flagged, not that the process dies). Returning
    a structured report keeps the decision with the experiment, where it belongs.
    """
    if len(items) < 4:
        raise ValueError("need at least 4 items to fit and test a size classifier")

    y = np.array([it.label for it in items], dtype=float)
    x = np.array([it.as_vector() for it in items], dtype=float)

    # Standardise. Raw counts span 10^2 to 10^5, so without this the n_tokens
    # column alone dominates the fit and the other features are ignored — which
    # would make the per-feature breakdown below meaningless.
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    xs = (x - means) / stds
    xs = np.column_stack([np.ones(len(xs)), xs])

    clusters = np.arange(len(items))
    _, se = fit_logit_cluster_robust(y, xs, clusters)
    beta = _fit_coefficients(y, xs)

    per_feature: dict[str, float] = {}
    for j, name in enumerate(SIZE_FEATURES):
        column = xs[:, j + 1]
        if np.allclose(column, column[0]):
            per_feature[name] = 0.5  # constant feature: no information
            continue
        per_feature[name] = roc_auc(y.astype(int), column)

    full_scores = xs @ beta
    auroc = roc_auc(y.astype(int), full_scores)
    _ = se

    # The null. A leakage detector that reports 0.9 on random labels is not
    # detecting leakage, and reporting that number beside the result is the only
    # way a reader can tell the difference.
    rng = np.random.default_rng(seed)
    nulls = [roc_auc(rng.permutation(y.astype(int)), full_scores) for _ in range(n_null)]
    null_auroc = float(np.mean(nulls))

    passed = auroc < threshold and abs(null_auroc - 0.5) < 0.1

    if passed:
        verdict = (
            f"PASS: memory size does not predict the condition "
            f"(AUROC {auroc:.3f} < {threshold}). Arm B is valid."
        )
    elif auroc >= threshold:
        verdict = (
            f"FAIL: memory size predicts the condition (AUROC {auroc:.3f} >= "
            f"{threshold}). Arm B is INVALID — the accuracy difference is partly a "
            f"size artefact. 2609.32269 reached AUROC 0.979 doing exactly this."
        )
    else:
        verdict = (
            f"INCONCLUSIVE: the null is not at chance (shuffled-label AUROC "
            f"{null_auroc:.3f}, expected ~0.5). The measurement is not trustworthy "
            f"even though the observed AUROC is {auroc:.3f}."
        )

    return LeakageReport(
        auroc=auroc,
        threshold=threshold,
        n_items=len(items),
        n_positive=int(y.sum()),
        null_auroc=null_auroc,
        passed=passed,
        per_feature_auroc=per_feature,
        verdict=verdict,
    )


def _fit_coefficients(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Plain logistic MLE for the leak detector.

    Deliberately the simplest thing that works. A leakage detector wants to be
    *sensitive* — it should catch a leak if one exists — so an underpowered
    classifier would be the dangerous failure. A convex logistic fit on
    standardised features has no local optima to get stuck in, so there is nothing
    clever to buy here.
    """
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(200):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        hessian = (x.T * w) @ x + 1e-8 * np.eye(x.shape[1])
        gradient = x.T @ (y - p)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta
