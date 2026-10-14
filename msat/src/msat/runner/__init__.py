"""Runner layer.

Four concerns, deliberately separate, because the spine is very hard to debug when
they are fused into one class:

  `schedule`  seeded, stratified-interleaved order        (PRD §R5.1)
  `wal`       append-only journal, fsynced per line       (PRD §R5.2)
  `ledger`    per-cell completion and the exclusion log   (PRD §R5.4, §R5.5)
  `executor`  the loop that joins them under a budget     (PRD §R5.3, §N4)

Each has a different failure mode and is tested on its own.
"""

from __future__ import annotations

from msat.runner.executor import (
    BudgetExceededError,
    Executor,
    Generator,
    Render,
    RunConfig,
    RunResult,
    make_schedule,
    reconcile,
    status_report,
)
from msat.runner.ledger import (
    EXCLUDED_FINISH_REASONS,
    CellLedger,
    CellState,
    RunLedger,
    classify_finish,
    is_excluded,
)
from msat.runner.schedule import (
    ScheduledUnit,
    build_schedule,
    schedule_digest,
    verify_not_sequential_by_k,
)
from msat.runner.wal import Wal, WalEntry, WalState, build_resume_state, read_wal

__all__ = [
    "EXCLUDED_FINISH_REASONS",
    "BudgetExceededError",
    "CellLedger",
    "CellState",
    "Executor",
    "Generator",
    "Render",
    "RunConfig",
    "RunLedger",
    "RunResult",
    "ScheduledUnit",
    "Wal",
    "WalEntry",
    "WalState",
    "build_resume_state",
    "build_schedule",
    "classify_finish",
    "is_excluded",
    "make_schedule",
    "read_wal",
    "reconcile",
    "schedule_digest",
    "status_report",
    "verify_not_sequential_by_k",
]
