"""Analysis layer.

`leakage` implements PRD §R9.10's Arm B label-leakage assertion in numpy — it
replaced the `scikit-learn` dependency that existed solely for it.
"""

from __future__ import annotations

from msat.analysis.leakage import (
    DEFAULT_AUROC_THRESHOLD,
    SIZE_FEATURES,
    LeakageFeatures,
    LeakageReport,
    assert_no_leakage,
    roc_auc,
)

__all__ = [
    "DEFAULT_AUROC_THRESHOLD",
    "SIZE_FEATURES",
    "LeakageFeatures",
    "LeakageReport",
    "assert_no_leakage",
    "roc_auc",
]
