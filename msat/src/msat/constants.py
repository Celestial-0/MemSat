"""Process exit codes and command metadata for the `msat` CLI.

These are declared once, here, so the Makefile, the tests, and the phase gates
can all assert against a single source. A silently-degenerate condition must be
impossible (PRD §G3): every failure mode gets a *distinguishable* non-zero code
rather than a generic 1.
"""

from __future__ import annotations

from typing import Final

# ── Exit codes ─────────────────────────────────────────────────────────────────
EXIT_OK: Final = 0
"""Command succeeded."""

EXIT_FAILURE: Final = 1
"""Generic, unclassified failure."""

EXIT_CONFIG_INVALID: Final = 2
"""`validate-config` found a missing or null required field (PRD §R1.5, §R7.5)."""

EXIT_DEGENERATE_CELL: Final = 3
"""`build-plan` rejected the spec: a (K, B) cell is degenerate (PRD §R3.2)."""

EXIT_BUDGET_EXCEEDED: Final = 4
"""Planned cost exceeds `budget.max_usd` (PRD §R0.5, §N8)."""

EXIT_DRIFT: Final = 5
"""A committed artifact did not reproduce byte-for-byte (PRD §R2.4)."""

EXIT_UNDERPOWERED: Final = 6
"""The K x B interaction is not estimable and no reduced design was accepted
(PRD §R0.3). Raised instead of launching a silently-truncated factorial."""

EXIT_NO_GO: Final = 7
"""The GO / NO-GO gate failed all four branches (Architecture §P5)."""

EXIT_ASSERTION_FAILED: Final = 8
"""A scientific assertion failed — e.g. the Arm B leakage check (PRD §R9.10),
Arm E not collapsing (PRD §R4.3), or the Arm B +/-2% token target."""

EXIT_NOT_IMPLEMENTED: Final = 9
"""The command is registered but its phase has not been built yet."""

EXIT_LEAKAGE_DETECTED: Final = 10
"""Reserved for PRD §R9.10: memory size predicts the condition label."""


EXIT_BUDGET_UNSET: Final = 10
"""`budget.max_usd` has not been set (PRD §R0.5, §N8; blocker B7).

Distinct from `EXIT_BUDGET_EXCEEDED` because "we do not know the ceiling" and
"the estimate is over the ceiling" call for completely different conversations.
An unset ceiling must never be treated as infinite.
"""


#: Judge-noise levels the power sweep reports at, as `(judge_flip, sigma_run)`.
#:
#: `judge_flip` is the fraction of verdicts the grader records INCORRECTLY, which
#: attenuates the observed effect by roughly `1 - 2 * flip`. It is a
#: misclassification model, not a variance inflation — see `SimSpec.judge_flip`.
#:
#: 0.00 — perfect measurement. An upper bound, never a forecast.
#: 0.02 — seed variance only; the grader is essentially exact.
#: 0.06 — mild disagreement.
#: 0.12 — the territory implied by kappa = 0.118 (2511.02246) and by
#:         2609.38021 flipping 8 of 500 verdicts between two passes of the SAME
#:         deterministic system under the SAME judge.
#:
#: A single MDE would flatter the design. Reporting the sweep is what turns "we
#: are well powered" into "we are well powered *conditional on measurement being
#: this good*, and here is exactly how good it has to be".
JUDGE_NOISE_GRID: Final[tuple[tuple[float, float], ...]] = (
    (0.00, 0.0),
    (0.02, 0.3),
    (0.06, 0.6),
    (0.12, 1.0),
)


# ── Phase ownership ────────────────────────────────────────────────────────────# Which build phase owns each command. Surfaced by `msat --version`-style output
# and by the stub messages, so nobody mistakes "not built yet" for "broken".
PHASE_OF_COMMAND: Final[dict[str, str]] = {
    "validate-config": "P2",
    "design-power": "P2",
    "ceiling": "P1",
    "data-lock": "P1",
    "build-plan": "P3",
    "run": "P4",
    "status": "P4",
    "replay": "P4",
    "score": "P6",
    "judge-audit": "P6",
    "analyze": "P9",
    "verify": "P9",
    "figures": "P10",
    "paper": "P10",
}

# ── PRD §N3 — commands that must never touch the network ────────────────────────
# `verify` / `analyze` / `figures` / `paper` are named in PRD §N3. `score` is
# added because PRD §G4 states it reads Storage only, and `status`, `replay`,
# `ceiling --verify`, and `data-lock --verify` because they read committed
# artifacts or the content-addressed store. Asserted by `tests/test_cli.py`.
OFFLINE_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "verify",
        "analyze",
        "figures",
        "paper",
        "score",
        "status",
        "replay",
        "ceiling",  # offline in `--verify` mode; enforced in Phase 1
        "data-lock",  # offline in `--verify` mode; enforced in Phase 1
    }
)
