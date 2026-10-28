"""Named figure layouts for the replay MP4.

Every layout composes the same three pieces — camera images, the cursor/target
recreation panel, and the analog trace strip — but arranges them differently.
They all follow the same rules: the cursor/target recreation is the visual
center and the largest single panel, the camera images are given as much area
as possible, and the time series is demoted to a thin supporting strip.

Placement is by explicit figure-coordinate boxes fed through packing.py rather
than by a gridspec, because every camera has a fixed aspect ratio: packing each
image at its own aspect is what removes the white gutters a uniform grid leaves
around it, and those gutters are most of the difference between a small-looking
and a large-looking video.

Add a layout by writing a `_layout_<name>(c, win, cams)` builder — `c` is the
Canvas helper below — and registering it in LAYOUTS. `build_layout()` returns a
ReplayComposite whose `update(t_rel_s)` advances every piece to one instant,
which is all the exporter and the snapshot tool need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from pyReplay import adjust, packing
from pyReplay.cursor_panel import CursorPanel
from pyReplay.trace_panel import TRACE_ORDER, TracePanel
from pyReplay.window import ReplayWindow

# Preferred display order; unknown cameras keep their own order after these.
CAMERA_ORDER = ["Cam Top", "Cam Left", "Cam Rear", "Cam Eye"]

# Room the cursor panel needs outside its square for ticks/labels/title, as a
# fraction of the figure.
CURSOR_PAD = dict(left=0.028, bottom=0.050, right=0.004, top=0.030)


class CameraArt:
    """One camera image axis: brightness-corrected, oriented, name overlaid."""

    def __init__(self, win: ReplayWindow, name: str, ax, *, scale: float = 1.0):
        self.win = win
        self.name = name
        self.ax = ax
        self.adjust = adjust.resolve(name)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("#444444")
        # The axes box is already at the frame's aspect (see packing.fit), so
        # aspect="auto" lets the image fill it exactly with no gutters.
        self.im = ax.imshow(self._prep(first_frame(win, name)), cmap="gray",
                            vmin=0, vmax=255, aspect="auto", animated=True)
        suffix = " (180°)" if adjust.is_rotated_180(name) else ""
        ax.text(0.012, 0.98, name + suffix, transform=ax.transAxes,
                fontsize=11 * scale, color="white", fontweight="bold",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="#000000", ec="none",
                          alpha=0.5))

    def _prep(self, frame):
        return adjust.orient(self.name, self.adjust.apply(frame))

    def update(self, t_ns: int):
        track = self.win.cameras.get(self.name)
        frame = track.frame_at(t_ns) if track else None
        if frame is not None:
            self.im.set_data(self._prep(frame))


class ClockArt:
    """Time / trial-state readout, for layouts with room for one."""

    def __init__(self, win: ReplayWindow, ax, *, scale: float = 1.0):
        self.win = win
        ax.set_axis_off()
        self.t_text = ax.text(0.5, 0.66, "", transform=ax.transAxes,
                              fontsize=20 * scale, fontweight="bold",
                              ha="center", va="center", family="monospace")
        self.state_text = ax.text(0.5, 0.30, "", transform=ax.transAxes,
                                  fontsize=12 * scale, color="#555555",
                                  ha="center", va="center")

    def update(self, t_rel_s: float):
        self.t_text.set_text(f"{t_rel_s:.2f} s")
        tgt = self.win.active_target(self.win.rel_to_ns(t_rel_s))
        self.state_text.set_text("trial — target on" if tgt is not None
                                 else "inter-trial")


@dataclass
class ReplayComposite:
    """A built figure plus every per-frame updatable piece in it."""

    fig: Figure
    win: ReplayWindow
    cameras: List[CameraArt] = field(default_factory=list)
    cursor: Optional[CursorPanel] = None
    traces: Optional[TracePanel] = None
    clock: Optional[ClockArt] = None

    def update(self, t_rel_s: float):
        t_ns = self.win.rel_to_ns(t_rel_s)
        for cam in self.cameras:
            cam.update(t_ns)
        for part in (self.cursor, self.traces, self.clock):
            if part is not None:
                part.update(t_rel_s)


def first_frame(win: ReplayWindow, name: str) -> np.ndarray:
    track = win.cameras.get(name)
    if track and track.frames:
        return track.frames[0]
    return np.zeros((2, 2), np.uint8)


def camera_aspect(win: ReplayWindow, name: str) -> float:
    """Frame width/height in display orientation (180° rotation preserves it)."""
    h, w = first_frame(win, name).shape[:2]
    return w / max(h, 1)


def order_cameras(names: Sequence[str]) -> List[str]:
    """Cameras in the preferred display order, unknown ones appended."""
    known = [n for n in CAMERA_ORDER if n in names]
    return known + [n for n in names if n not in known]


class Canvas:
    """A figure plus box-based placement helpers shared by every layout."""

    def __init__(self, fig: Figure, win: ReplayWindow):
        self.fig = fig
        self.win = win
        w_in, h_in = fig.get_size_inches()
        self.fig_ar = w_in / h_in
        # Boxes the last camera_row/camera_col actually used, so a layout can
        # place the cursor panel against the real edge of a packed camera group
        # instead of guessing at a height that depends on the frame aspects.
        self.last_boxes: List = []

    # --- cameras ---
    def cameras(self, names: Sequence[str], boxes, *, scale: float = 1.0):
        """One CameraArt per (name, box), each fitted to its frame aspect."""
        arts = []
        for name, box in zip(names, boxes):
            rect = packing.fit(box, camera_aspect(self.win, name), self.fig_ar)
            arts.append(CameraArt(self.win, name, self.fig.add_axes(rect),
                                  scale=scale))
        return arts

    def camera_col(self, names, box, *, gap=0.012, scale=1.0, align="center",
                   cross_align="center"):
        ars = [camera_aspect(self.win, n) for n in names]
        boxes = packing.pack_col(box, ars, self.fig_ar, gap, align, cross_align)
        self.last_boxes = boxes
        return self.cameras(names, boxes, scale=scale)

    def camera_row(self, names, box, *, gap=0.008, scale=1.0, align="center",
                   cross_align="center"):
        ars = [camera_aspect(self.win, n) for n in names]
        boxes = packing.pack_row(box, ars, self.fig_ar, gap, align, cross_align)
        self.last_boxes = boxes
        return self.cameras(names, boxes, scale=scale)

    # --- cursor panel ---
    def cursor(self, box, *, scale=1.0, labels=True, title="Cursor & target"):
        """The square cursor panel, centered in `box` with room for its labels."""
        pad = packing.inset(box, **CURSOR_PAD) if labels else box
        rect = packing.fit(pad, 1.0, self.fig_ar)
        return CursorPanel(self.win, self.fig.add_axes(rect), scale=scale,
                           labels=labels, title=title)

    # --- traces ---
    def traces(self, box, scroll_s, *, orientation="row", gap=0.035, scale=1.0):
        """The trace strip: `box` split evenly into one cell per channel."""
        n = max(len([c for c, _ in TRACE_ORDER if c in self.win.analog]), 1)
        x, y, w, h = box
        if orientation == "row":
            cw = (w - gap * (n - 1)) / n
            rects = [(x + i * (cw + gap), y, cw, h) for i in range(n)]
        else:
            ch = (h - gap * (n - 1)) / n
            rects = [(x, y + h - (i + 1) * ch - i * gap, w, ch) for i in range(n)]
        return TracePanel(self.win, self.fig, rects=rects, scroll_s=scroll_s,
                          orientation=orientation, compact=True, scale=scale)

    def clock(self, box, *, scale=1.0):
        return ClockArt(self.win, self.fig.add_axes(box), scale=scale)


# --------------------------------------------------------------------------
# Layouts.  All boxes are (left, bottom, width, height) in figure coordinates.
# --------------------------------------------------------------------------

def _layout_stage(c: Canvas, win, cams, opts) -> ReplayComposite:
    """Cameras in two flanking columns, big cursor panel between them, traces
    as a short full-width strip underneath."""
    cursor = c.cursor((0.28, 0.20, 0.44, 0.79), scale=1.35)
    half = (len(cams) + 1) // 2
    arts = c.camera_col(cams[:half], (0.010, 0.20, 0.265, 0.79))
    arts += c.camera_col(cams[half:], (0.725, 0.20, 0.265, 0.79))
    traces = c.traces((0.035, 0.055, 0.945, 0.105), opts["scroll_s"])
    return ReplayComposite(c.fig, win, arts, cursor, traces)


def _layout_bigtop(c: Canvas, win, cams, opts) -> ReplayComposite:
    """One camera row spanning the full width at the top, then the cursor panel
    centered below it, the clock on its left and the trace stack on its right."""
    arts = c.camera_row(cams, (0.008, 0.300, 0.984, 0.690), scale=1.05,
                        cross_align="end")
    floor = min((b[1] for b in c.last_boxes), default=0.70) - 0.015
    cursor = c.cursor((0.290, 0.015, 0.420, floor - 0.015), scale=1.3)
    clock = c.clock((0.045, 0.290, 0.195, 0.110), scale=1.15)
    traces = c.traces((0.775, 0.150, 0.200, floor - 0.235), opts["scroll_s"],
                      orientation="stacked", gap=0.045)
    return ReplayComposite(c.fig, win, arts, cursor, traces, clock)


def _layout_cross(c: Canvas, win, cams, opts) -> ReplayComposite:
    """The widest camera as a band over the cursor panel, the rest flanking it,
    traces as a short full-width strip underneath."""
    band = _widest(c, cams)
    rest = [n for n in cams if n != band]
    split = (len(rest) + 1) // 2

    arts = (c.camera_row([band], (0.250, 0.755, 0.500, 0.240), cross_align="end")
            if band else [])
    band_floor = min((b[1] for b in c.last_boxes), default=0.99) if band else 0.99
    cursor = c.cursor((0.285, 0.030, 0.430, band_floor - 0.035), scale=1.3)
    arts += c.camera_col(rest[:split], (0.008, 0.030, 0.265, 0.960),
                         align="end")
    arts += c.camera_col(rest[split:], (0.727, 0.480, 0.265, 0.512),
                         align="end")
    traces = c.traces((0.760, 0.075, 0.225, 0.330), opts["scroll_s"],
                      orientation="stacked", gap=0.045)
    return ReplayComposite(c.fig, win, arts, cursor, traces)


def _layout_focus(c: Canvas, win, cams, opts) -> ReplayComposite:
    """Maximum emphasis on the recreation: a very large cursor panel centered, a
    narrow trace stack on its left, cameras stacked on its right, and the widest
    camera as a band along the bottom."""
    band = _widest(c, cams)
    side = [n for n in cams if n != band]

    cursor = c.cursor((0.230, 0.200, 0.520, 0.790), scale=1.45)
    arts = c.camera_col(side, (0.745, 0.200, 0.248, 0.790), align="end")
    if band:
        arts += c.camera_row([band], (0.240, 0.018, 0.520, 0.180),
                             cross_align="start")
    traces = c.traces((0.030, 0.360, 0.165, 0.480), opts["scroll_s"],
                      orientation="stacked", gap=0.055)
    return ReplayComposite(c.fig, win, arts, cursor, traces)


def _widest(c: Canvas, cams: Sequence[str]) -> Optional[str]:
    """The camera with the widest frame aspect — the one that suits a band (on
    AlexRig that is Cam Eye, ~2.4:1 against the others' 1.25:1)."""
    if not cams:
        return None
    return max(cams, key=lambda n: camera_aspect(c.win, n))


LAYOUTS: Dict[str, Callable] = {
    "stage": _layout_stage,
    "bigtop": _layout_bigtop,
    "cross": _layout_cross,
    "focus": _layout_focus,
}

DEFAULT_LAYOUT = "cross"


def build_layout(name: str, win: ReplayWindow, *,
                 figsize=(16.0, 9.0), scroll_s: float = 2.0,
                 camera_names: Optional[Sequence[str]] = None,
                 facecolor: str = "white") -> ReplayComposite:
    """Build the named layout for `win` and return its ReplayComposite."""
    try:
        builder = LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown layout '{name}'; choose from {sorted(LAYOUTS)}")
    cams = order_cameras(list(camera_names if camera_names is not None else win.cameras))
    fig = Figure(figsize=figsize, facecolor=facecolor)
    return builder(Canvas(fig, win), win, cams, {"scroll_s": scroll_s})
