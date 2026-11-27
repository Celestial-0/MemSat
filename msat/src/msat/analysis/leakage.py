"""Arm B label-leakage assertion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from msat.design import fit_logit_cluster_robust

DEFAULT_AUROC_THRESHOLD: Final = 0.70

SIZE_FEATURES: Final[tuple[str, ...]] = (
    "n_memories",
    "n_tokens",
    "n_turns",
    "mean_tokens_per_memory",
)


@dataclass(frozen=True, slots=True)
class LeakageFeatures:
    # 1 marks the padded condition, 0 the short one.
    label: int
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
    if len(items) < 4:
        raise ValueError("need at least 4 items to fit and test a size classifier")

    y = np.array([it.label for it in items], dtype=float)
    x = np.array([it.as_vector() for it in items], dtype=float)

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
            per_feature[name] = 0.5
            continue
        per_feature[name] = roc_auc(y.astype(int), column)

    full_scores = xs @ beta
    auroc = roc_auc(y.astype(int), full_scores)
    _ = se

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
