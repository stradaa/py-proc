"""Aspect-aware box packing in figure coordinates.

Camera frames have fixed aspect ratios, so a gridspec cell that does not match
a frame's aspect wastes the difference as white space — which is exactly what
makes a composite replay video feel small. These helpers instead allot a *box*
(a (left, bottom, width, height) rect in figure coordinates) and return the
largest rects of the required aspect that fit inside it, so each axes is created
already at the frame's aspect and the image fills it edge to edge.

`fig_ar` is the figure's own width/height in inches: box rects are fractions of
the figure, so converting between a rect ratio and a physical aspect ratio needs
it. Aspect ratios here are always physical width/height.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def box_aspect(box: Box, fig_ar: float) -> float:
    """Physical width/height of `box`."""
    return (box[2] / max(box[3], 1e-9)) * fig_ar


def fit(box: Box, ar: float, fig_ar: float) -> Box:
    """Largest rect of physical aspect `ar` centered inside `box`."""
    x, y, w, h = box
    b_ar = box_aspect(box, fig_ar)
    if b_ar > ar:                 # box too wide: keep height, shrink width
        nw, nh = w * ar / b_ar, h
    else:                         # box too tall: keep width, shrink height
        nw, nh = w, h * b_ar / ar
    return (x + (w - nw) / 2, y + (h - nh) / 2, nw, nh)


def _offset(free: float, align: str) -> float:
    """Where to put `free` leftover space: 'start' = all after, 'end' = all
    before, 'center' = split. 'start' is left/bottom, 'end' is right/top."""
    if align == "start":
        return 0.0
    if align == "end":
        return free
    return free / 2


def pack_col(box: Box, ars: Sequence[float], fig_ar: float,
             gap: float = 0.012, align: str = "center",
             cross_align: str = "center") -> List[Box]:
    """Stack rects of the given aspects vertically inside `box`, all one width.

    The shared width is the largest that lets every rect keep its aspect and
    still fit the box height (gaps included). `align` places the leftover
    height ('end' pins the stack to the box's top) and `cross_align` the
    leftover width ('start' pins it to the box's left).
    """
    n = len(ars)
    if n == 0:
        return []
    x, y, w, h = box
    total_gap = gap * (n - 1)
    # A rect of width ww needs height ww * fig_ar / ar in figure coordinates, so
    # the stack height is ww * fig_ar * sum(1/ar) + gaps: invert that for ww.
    w_by_h = (h - total_gap) / max(fig_ar * sum(1.0 / a for a in ars), 1e-9)
    ww = min(w, w_by_h)
    heights = [ww * fig_ar / a for a in ars]
    top = y + h - _offset(h - (sum(heights) + total_gap), align)
    left = x + _offset(w - ww, cross_align)
    out = []
    for hh in heights:
        top -= hh
        out.append((left, top, ww, hh))
        top -= gap
    return out


def pack_row(box: Box, ars: Sequence[float], fig_ar: float,
             gap: float = 0.008, align: str = "center",
             cross_align: str = "center") -> List[Box]:
    """Lay rects of the given aspects out horizontally inside `box`, one height.

    `align` places the leftover width ('start' pins the row to the box's left)
    and `cross_align` the leftover height ('end' pins it to the box's top).
    """
    n = len(ars)
    if n == 0:
        return []
    x, y, w, h = box
    total_gap = gap * (n - 1)
    h_by_w = (w - total_gap) * fig_ar / max(sum(ars), 1e-9)
    hh = min(h, h_by_w)
    widths = [hh * a / fig_ar for a in ars]
    left = x + _offset(w - (sum(widths) + total_gap), align)
    bottom = y + _offset(h - hh, cross_align)
    out = []
    for ww in widths:
        out.append((left, bottom, ww, hh))
        left += ww + gap
    return out


def inset(box: Box, left: float = 0.0, bottom: float = 0.0,
          right: float = 0.0, top: float = 0.0) -> Box:
    """Shrink `box` by the given figure-fraction margins (room for tick labels)."""
    x, y, w, h = box
    return (x + left, y + bottom, max(w - left - right, 1e-3),
            max(h - bottom - top, 1e-3))
