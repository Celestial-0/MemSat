"""Figures. PRD §R10.2 - diagrams are code, never hand-drawn SVG.

Emitted as SVG built from arithmetic rather than through a plotting library, so
that the figure's guarantees are checkable on the emitted SOURCE (see
`svg.py`) and not only on a rasterised image.
"""

from __future__ import annotations

from .centrepiece import (
    CEILING_DASH,
    PREVIEW_WARNING,
    Cell,
    FigureInputMissing,
    build_from_disk,
    centrepiece,
    preview_report,
    write_centrepiece,
)
from .svg import Figure, Panel, Series, Sheet

__all__ = [
    "CEILING_DASH",
    "PREVIEW_WARNING",
    "Cell",
    "Figure",
    "FigureInputMissing",
    "Panel",
    "Series",
    "Sheet",
    "build_from_disk",
    "centrepiece",
    "preview_report",
    "write_centrepiece",
]
