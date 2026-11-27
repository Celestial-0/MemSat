"""Process exit codes and command metadata for the `msat` CLI."""

from __future__ import annotations

from typing import Final

EXIT_OK: Final = 0

EXIT_FAILURE: Final = 1

EXIT_CONFIG_INVALID: Final = 2

EXIT_DEGENERATE_CELL: Final = 3

EXIT_BUDGET_EXCEEDED: Final = 4

EXIT_DRIFT: Final = 5

EXIT_UNDERPOWERED: Final = 6

EXIT_NO_GO: Final = 7

EXIT_ASSERTION_FAILED: Final = 8

EXIT_NOT_IMPLEMENTED: Final = 9

EXIT_LEAKAGE_DETECTED: Final = 10


EXIT_BUDGET_UNSET: Final = 10
"""An unset budget is unknown, never infinite."""


# Judge-flip levels as `(judge_flip, sigma_run)`; flip attenuates effects by `1 - 2 * flip`.
JUDGE_NOISE_GRID: Final[tuple[tuple[float, float], ...]] = (
    (0.00, 0.0),
    (0.02, 0.3),
    (0.06, 0.6),
    (0.12, 1.0),
)


PHASE_OF_COMMAND: Final[dict[str, str]] = {
    "validate-config": "P2",
    "estimate-power": "P2",
    "ceiling": "P1",
    "data-lock": "P1",
    "build-manifest": "P3",
    "run": "P4",
    "status": "P4",
    "score": "P6",
    "judge-audit": "P6",
    "analyze": "P9",
    "figures": "P10",
    "paper": "P10",
    "report-reader-capacity": "P1",
    "hardware-delta": "P5",
    "vram": "P6",
}

OFFLINE_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "validate-config",
        "estimate-power",
        "ceiling",
        "data-lock",
        "build-manifest",
        "status",
        "score",
        "judge-audit",
        "analyze",
        "figures",
        "paper",
        "report-reader-capacity",
        "hardware-delta",
        "vram",
    }
)
