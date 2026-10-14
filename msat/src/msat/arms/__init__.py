"""Arms of the crossed design (PRD §R4).

Arms are DATA: `configs/arms.yaml` names a template, `configs/prompts/*.j2` holds
the text, and `ArmRegistry` dispatches. Adding an arm needs no pipeline change.

`render(question, condition) -> RenderedPrompt` is PURE — no globals, no network,
no file I/O, no clock. See `render.py` for why that is a correctness property
rather than a style rule.

`negative` holds Arm E, the negative control, which GATES the whole experiment.
"""

from __future__ import annotations

from msat.arms.negative import (
    ControlConfig,
    ControlOutcome,
    PooledMemory,
    Verdict,
    draw_control_memories,
    evaluate,
    go_no_go,
    render_control,
)
from msat.arms.render import (
    Arm,
    ArmRegistry,
    Memory,
    RenderedPrompt,
    SimpleArm,
    TemplateError,
    build_registry,
    format_memories,
    render_template,
)

__all__ = [
    "Arm",
    "ArmRegistry",
    "ControlConfig",
    "ControlOutcome",
    "Memory",
    "PooledMemory",
    "RenderedPrompt",
    "SimpleArm",
    "TemplateError",
    "Verdict",
    "build_registry",
    "draw_control_memories",
    "evaluate",
    "format_memories",
    "go_no_go",
    "render_control",
    "render_template",
]
