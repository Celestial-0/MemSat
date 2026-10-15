"""Judges, kappa and the R8.6 audit. Evaluation reads Storage, never the live model.

`panel` is the part that RUNS the panel; `verdicts`, `kappa`, `audit`,
`human_sample` and `report` are the parts that reason about what it produced.
Re-exported here so a caller needs one import, matching how `arms/` and
`analysis/` expose their own surface.
"""

from msat.judges.panel import (
    JUDGE_ABSTENTION_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
    JudgeBackendUnavailableError,
    JudgeDecoding,
    JudgeIndependenceError,
    JudgeItem,
    JudgePanel,
    JudgeReadiness,
    JudgeSequentialError,
    JudgeSpec,
    JudgeVramError,
    PanelHealth,
    PanelKappaBelowBarError,
    PanelRun,
    PanelSpecError,
    format_health,
    judge_item,
    judge_panel_from_config,
    model_family,
    panel_health,
    panel_readiness,
    read_templates,
    run_panel,
)

__all__ = [
    "JUDGE_ABSTENTION_PROMPT_TEMPLATE",
    "JUDGE_PROMPT_TEMPLATE",
    "JudgeBackendUnavailableError",
    "JudgeDecoding",
    "JudgeIndependenceError",
    "JudgeItem",
    "JudgePanel",
    "JudgeReadiness",
    "JudgeSequentialError",
    "JudgeSpec",
    "JudgeVramError",
    "PanelHealth",
    "PanelKappaBelowBarError",
    "PanelRun",
    "PanelSpecError",
    "format_health",
    "judge_item",
    "judge_panel_from_config",
    "model_family",
    "panel_health",
    "panel_readiness",
    "read_templates",
    "run_panel",
]