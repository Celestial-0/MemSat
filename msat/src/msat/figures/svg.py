"""Minimal SVG emitter, plus the parsers that inspect what it emitted.

WHY THIS EXISTS RATHER THAN A PLOTTING LIBRARY
----------------------------------------------
`cli.py`'s docstring records that this started as "Plotly -> kaleido" and was
wrong until P7 caught it. The practical reason the arithmetic approach wins:

**The figure's main guarantee is checkable on SVG source, and is not checkable
on a rasterised image.** PRD §R9.6 requires the ceiling to be a bound LINE and
never a shaded region. That is a property of the element census - "zero filled
areas, four polylines, two of them dashed" - and a rasterised PNG cannot be
census-checked without decoding pixels and guessing at anti-aliasing. So the
figure is emitted as text, and the guarantees are unit tests over that text.

The second reason is that `is_well_formed`, `parse`, `polylines`, `rects` and
`shaded_regions` below are the *verifiers*, and a plotting library gives no
handle on its own output. Emitting the SVG means the checks read the same bytes
a reviewer would open.

Coordinates are emitted from `fmt`, which is fixed-precision, so the same input
produces the same bytes. That is what makes `test_rendering_is_byte_stable` and
a meaningful git diff on a figure both possible.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import escape, unescape
from typing import Any

SVG_NS = "http://www.w3.org/2000/svg"

#: Rect class marking a legend swatch. The ONE rect shape that is allowed to be
#: filled - see `shaded_regions`, where every other filled rect is the failure
#: mode this whole module exists to detect.
LEGEND_BOX = "legend-box"

#: Distinct hues for one line per B. Chosen so adjacent entries stay separable in
#: greyscale too, which matters because the figure may be printed.
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2")


def fmt(value: float) -> str:
    """Two decimals, trailing zeros stripped.

    `repr(0.1 + 0.2)` is `0.30000000000000004`. Accumulated coordinate
    arithmetic would put that in the file, and two renderings of the same data
    would differ in the last digit - so `test_rendering_is_byte_stable` would
    pass by luck and a diff on a regenerated figure would be unreadable.
    """
    s = f"{value:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


@dataclass
class Series:
    """One polyline, optionally dashed, optionally marked.

    `dash` is `""` for a measurement and the ceiling dash for the bound. Storing
    it as a field rather than inferring it from the label is what lets
    `test_an_empty_ceiling_yields_no_ceiling_series` ask "is any line dashed?"
    without parsing strings.
    """

    label: str
    points: list[tuple[float, float]]
    colour: str = PALETTE[0]
    dash: str = ""
    marker: bool = True


@dataclass
class Figure:
    """A standalone single-panel chart. The minimal thing `test_labels_are_xml_escaped`
    exercises; `Sheet` is the real product."""

    title: str
    series: list[Series] = field(default_factory=list)
    width: int = 720
    height: int = 440
    xlabel: str = ""
    ylabel: str = ""

    def render(self) -> str:
        """A single-panel chart, as a one-panel `Sheet`."""
        ys = [y for s in self.series for _, y in s.points]
        return Sheet(
            title=self.title,
            panels=[
                Panel(
                    panel_title=self.title,
                    series=list(self.series),
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    y_range=(0.0, max(1.0, max(ys) * 1.05) if ys else 1.0),
                    width=self.width,
                    height=self.height,
                )
            ],
        ).render()


@dataclass
class Panel:
    """One axes. Nested as its own `<svg>` so the census can count them."""

    panel_title: str
    series: list[Series] = field(default_factory=list)
    subtitle: str = ""
    xlabel: str = ""
    ylabel: str = ""
    annotations: list[str] = field(default_factory=list)
    y_range: tuple[float, float] = (0.0, 1.0)
    x_range: tuple[float, float] | None = None
    width: int = 460
    height: int = 360


@dataclass
class Sheet:
    """Several panels side by side under one title."""

    title: str = ""
    panels: list[Panel] = field(default_factory=list)
    subtitle: str = ""

    def render(self) -> str:
        if not self.panels:
            raise ValueError("a Sheet needs at least one panel")
        gap = 40
        head = 64 if self.title or self.subtitle else 8
        width = sum(p.width for p in self.panels) + gap * (len(self.panels) - 1)
        height = self.panels[0].height + head + 24

        out = [
            f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
            # No background <rect>. It would be a filled rect, `rects()` would
            # return it, and `test_the_only_rects_are_legend_boxes` would fail
            # with "every rect must be a legend box" - a white background is not
            # a shaded data region, but the census cannot tell the difference.
            # SVG's default is transparent, which renders white on any page.
        ]
        if self.title:
            out.append(_text(self.title, 14, 26, size=17, weight="bold"))
        if self.subtitle:
            out.append(_text(self.subtitle, 14, 46, size=11, fill="#555555"))

        for i, panel in enumerate(self.panels):
            x = i * (panel.width + gap)
            out.append(_render_panel(panel, x, head))

        out.append("</svg>")
        return "\n".join(out) + "\n"


# ── internals ─────────────────────────────────────────────────────────────────

_PAD_L, _PAD_R, _PAD_T = 56, 18, 34


def _pad_b(panel: Panel) -> int:
    """Bottom padding grows with the annotation block.

    A nested `<svg>` CLIPS to its viewBox. Annotations drawn past the panel edge
    do not overflow gracefully - they vanish, and the corpus_fraction range and
    the ceiling's BOUND label are exactly the text the figure is required to
    carry (R9.6, G12). Silent clipping would look like a figure that simply
    omitted them.
    """
    return 46 + 14 * len(panel.annotations)


def _render_panel(panel: Panel, ox: int, oy: int) -> str:
    pw, ph = panel.width, panel.height
    pad_b = _pad_b(panel)
    px0, px1 = ox + _PAD_L, ox + pw - _PAD_R
    py0, py1 = oy + _PAD_T, oy + ph - pad_b

    xs = [p[0] for s in panel.series for p in s.points]
    if panel.x_range is not None:
        x0, x1 = panel.x_range
    elif xs:
        x0, x1 = min(xs), max(xs)
    else:
        x0, x1 = 0.0, 1.0
    y0, y1 = panel.y_range
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y1 = y0 + 1.0

    def sx(v: float) -> float:
        return px0 + (v - x0) / (x1 - x0) * (px1 - px0)

    def sy(v: float) -> float:
        return py1 - (v - y0) / (y1 - y0) * (py1 - py0)

    # Nested <svg> - counted by `parse` as one panel element.
    out = [
        f'<svg x="{ox}" y="{oy}" width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}">',
        _text(panel.panel_title, _PAD_L, 20, size=13, weight="bold"),
    ]
    if panel.subtitle:
        out.append(_text(panel.subtitle, _PAD_L, 33, size=10, fill="#555555"))

    # Axes: lines only. No fills anywhere in here - see `shaded_regions`.
    out.append(
        f'<line x1="{fmt(px0)}" y1="{fmt(py1)}" x2="{fmt(px1)}" y2="{fmt(py1)}" '
        f'stroke="#333333" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{fmt(px0)}" y1="{fmt(py0)}" x2="{fmt(px0)}" y2="{fmt(py1)}" '
        f'stroke="#333333" stroke-width="1"/>'
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y0 + frac * (y1 - y0)
        yy = sy(yv)
        out.append(
            f'<line x1="{fmt(px0)}" y1="{fmt(yy)}" x2="{fmt(px1)}" y2="{fmt(yy)}" '
            f'stroke="#e4e4e4" stroke-width="1"/>'
        )
        out.append(_text(fmt(yv), px0 - 8, yy + 4, size=9, fill="#666666", anchor="end"))
    for xv in _ticks(x0, x1):
        out.append(_text(fmt(xv), sx(xv), py1 + 16, size=9, fill="#666666", anchor="middle"))

    for series in panel.series:
        if len(series.points) < 2:
            continue
        coords = " ".join(f"{fmt(sx(x))},{fmt(sy(y))}" for x, y in series.points)
        dash = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        out.append(
            f'<polyline class="series" points="{coords}" fill="none" '
            f'stroke="{series.colour}" stroke-width="2"{dash}/>'
        )
        if series.marker:
            for x, y in series.points:
                out.append(_marker(sx(x), sy(y), series.colour))

    # Legend. Swatches are `<rect class="legend-box">` - the single permitted
    # filled rect, and explicitly not a shaded data region.
    ly = py0 + 4
    for series in panel.series:
        if not series.points:
            continue
        dash = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        out.append(
            f'<line x1="{fmt(px0 + 4)}" y1="{fmt(ly + 5)}" x2="{fmt(px0 + 22)}" '
            f'y2="{fmt(ly + 5)}" stroke="{series.colour}" stroke-width="2"{dash}/>'
        )
        out.append(
            f'<rect class="{LEGEND_BOX}" x="{fmt(px0 + 26)}" y="{fmt(ly)}" '
            f'width="9" height="9" fill="{series.colour}"/>'
        )
        out.append(_text(series.label, px0 + 39, ly + 9, size=9, fill="#333333"))
        ly += 13

    if panel.xlabel:
        out.append(_text(panel.xlabel, (px0 + px1) / 2, py1 + 33, size=11, anchor="middle"))
    for i, note in enumerate(panel.annotations):
        out.append(_text(note, px0, py1 + 48 + i * 14, size=9, fill="#444444"))

    out.append("</svg>")
    return "\n".join(out)


def _marker(cx: float, cy: float, colour: str) -> str:
    """A square marker as a `<path>`.

    Deliberately NOT a `<rect>`: `test_the_only_rects_are_legend_boxes` counts
    rect elements to decide whether anything is shaded, and a marker drawn as a
    rect would make that count meaningless.
    """
    r = 3
    d = (
        f"M{fmt(cx - r)},{fmt(cy - r)} "
        f"L{fmt(cx + r)},{fmt(cy - r)} "
        f"L{fmt(cx + r)},{fmt(cy + r)} "
        f"L{fmt(cx - r)},{fmt(cy + r)} Z"
    )
    return f'<path d="{d}" fill="{colour}" stroke="#ffffff" stroke-width="0.8"/>'


def _ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 1.0 if span <= 6 else (2.0 if span <= 14 else 5.0)
    ticks, v = [], -(-lo // step) * step
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _text(
    content: str,
    x: float,
    y: float,
    *,
    size: int = 11,
    fill: str = "#111111",
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    esc = escape(str(content), quote=True)
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{esc}</text>'
    )


# ── parsers: the verifiers, over the same bytes a reviewer would open ──────────

_RE_RECT = re.compile(r"<rect\b[^>]*/?>", re.IGNORECASE)
_RE_POLYLINE = re.compile(r'<polyline\b[^>]*\bpoints="([^"]*)"', re.IGNORECASE)
_RE_TEXT = re.compile(r"<text\b[^>]*>(.*?)</text>", re.IGNORECASE | re.DOTALL)


def is_well_formed(svg: str) -> bool:
    """Parse, don't eyeball. A malformed SVG renders as a blank box in some
    viewers and a MarkupError in others, and the difference is invisible until
    someone opens the file."""
    try:
        ET.fromstring(svg)
    except ET.ParseError:
        return False
    return True


def parse(svg: str) -> dict[str, int]:
    """Element census by tag name, so a caller can assert exact counts."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return {}
    counts: dict[str, int] = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def rects(svg: str) -> list[str]:
    """Every `<rect>` tag, verbatim."""
    return _RE_RECT.findall(svg)


def polylines(svg: str) -> list[str]:
    """Point strings, one per `<polyline>`. Returned raw so a caller can count
    points without re-parsing coordinates."""
    return _RE_POLYLINE.findall(svg)


def iter_texts(svg: str) -> list[str]:
    """Visible text, unescaped. Uses the regex rather than a DOM walk so that
    text containing markup-ish characters still comes back whole."""
    return [unescape(m) for m in _RE_TEXT.findall(svg)]


def shaded_regions(svg: str) -> int:
    """Count filled rects that are NOT legend swatches.

    PRD §R9.6: the ceiling is a bound LINE, never a shaded region. A filled band
    under the measured curve reads as "the area the method achieved", which
    recasts the result as a fraction of a bound and makes the crossover - the one
    thing the figure exists to show - unlocatable.

    A detector that only ever returns 0 is unfalsifiable, so the negative case
    is covered by `test_a_figure_that_did_shade_would_be_detected` and the false
    positive by `test_the_legend_box_is_not_mistaken_for_a_band`.
    """
    n = 0
    for tag in rects(svg):
        if f'class="{LEGEND_BOX}"' in tag:
            continue
        if re.search(r'\bfill\s*=\s*"(?!none)', tag, re.IGNORECASE):
            n += 1
    return n


def __getattr__(name: str) -> Any:  # pragma: no cover - defensive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
