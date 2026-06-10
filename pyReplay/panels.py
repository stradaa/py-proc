"""Matplotlib signal panels: joystick 2D trajectory + scrolling analog traces.

Layout: a square joystick-trajectory panel on the left (pyCheck aesthetic) and
the analog NIDAQ channels stacked on the right as scrolling traces with a fixed
central playhead. All limits are fixed so the panels can be blitted (see
blit.py); only the trail/cursor/target and trace lines change per frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from pyReplay.window import ReplayWindow

# Channels rendered as traces, in order, with colors.
_TRACE_ORDER = [
    ("fiducial", "#4c78a8"),
    ("photodiode", "#e45756"),
    ("reward", "#54a24b"),
]

_TRAIL_S = 1.0          # seconds of cursor history drawn as a trail


class SignalPanels:
    """Owns the matplotlib Figure for the joystick + trace panels."""

    def __init__(self, win: ReplayWindow, scroll_s: float = 2.0,
                 fig: Optional[Figure] = None, host=None):
        """fig/host let the panels build into an existing figure region (used by
        the MP4 export, which puts a camera row above the same panels). When both
        are None, the panels own a standalone figure (used by the live player)."""
        self.win = win
        self.scroll_s = scroll_s
        self.fig = fig if fig is not None else Figure(figsize=(9, 4.2), facecolor="white")

        traces = [(n, c) for n, c in _TRACE_ORDER if n in win.analog]
        nrows = max(len(traces), 1)
        if host is None:
            gs = self.fig.add_gridspec(nrows, 2, width_ratios=[1.0, 1.4],
                                       wspace=0.28, hspace=0.18,
                                       left=0.07, right=0.97, top=0.92, bottom=0.12)
        else:
            gs = host.subgridspec(nrows, 2, width_ratios=[1.0, 1.4],
                                  wspace=0.3, hspace=0.25)

        self.ax_joy = self.fig.add_subplot(gs[:, 0])
        self.animated: List = []
        self._build_joystick()

        self.trace_lines: Dict[str, object] = {}
        self.trace_axes: Dict[str, object] = {}
        self._build_traces(gs, traces)

    # ---- joystick panel (cursor space [0,1], with task target) ----
    def _build_joystick(self):
        ax = self.ax_joy
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Cursor & target")
        ax.set_xlabel("cursor x (norm)")
        ax.set_ylabel("cursor y (norm)")
        ax.grid(True, alpha=0.2)

        # static center gate (part of the blit background)
        if self.win.center_gate_radius > 0:
            ax.add_patch(Circle((0.5, 0.5), self.win.center_gate_radius,
                                fill=False, ec="#999999", ls="--", lw=1.0, alpha=0.6))

        # active target (hidden until a trial target is active)
        self.target_patch = Circle((0.5, 0.5), 0.0, color="#4caf50", alpha=0.25,
                                   visible=False)
        ax.add_patch(self.target_patch)
        # cursor trail + current cursor
        (self.trail_line,) = ax.plot([], [], color="#1f77b4", lw=2.0, alpha=0.9)
        (self.cursor_pt,) = ax.plot([], [], "o", color="black", ms=8, zorder=6)
        for art in (self.target_patch, self.trail_line, self.cursor_pt):
            self.animated.append(art)

    # ---- trace panels ----
    def _build_traces(self, gs, traces):
        bottom = traces[-1][0] if traces else None
        for row, (name, color) in enumerate(traces):
            ax = self.fig.add_subplot(gs[row, 1])
            track = self.win.analog[name]
            if len(track.values):
                vlo, vhi = float(track.values.min()), float(track.values.max())
                pad = 0.05 * (vhi - vlo + 1e-9)
            else:
                vlo, vhi, pad = 0.0, 1.0, 0.05
            ax.set_xlim(-self.scroll_s, self.scroll_s)
            ax.set_ylim(vlo - pad, vhi + pad)
            ax.set_ylabel(name, fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.axvline(0.0, color="#111111", lw=1.2, alpha=0.7)  # static playhead
            if name == bottom:
                ax.set_xlabel("time from playhead (s)")
            else:
                ax.tick_params(labelbottom=False)
            (line,) = ax.plot([], [], color=color, lw=1.2)
            self.trace_lines[name] = line
            self.trace_axes[name] = ax
            self.animated.append(line)

    def set_scroll(self, scroll_s: float):
        """Change the trace scroll-window half-width (invalidates blit bg)."""
        self.scroll_s = scroll_s
        for ax in self.trace_axes.values():
            ax.set_xlim(-scroll_s, scroll_s)

    # ---- per-frame update ----
    def update(self, t_rel_s: float):
        win = self.win
        t_ns = win.rel_to_ns(t_rel_s)

        # cursor trail + current cursor (normalized [0,1] task space)
        idx = win.joystick_index_at(t_ns)
        if idx >= 0 and len(win.cursor_x):
            lo_ns = t_ns - int(_TRAIL_S * 1e9)
            j0 = int(np.searchsorted(win.joystick_t_ns, lo_ns, side="left"))
            self.trail_line.set_data(win.cursor_x[j0:idx + 1],
                                     win.cursor_y[j0:idx + 1])
            self.cursor_pt.set_data([win.cursor_x[idx]], [win.cursor_y[idx]])

        # active target
        tgt = win.active_target(t_ns)
        if tgt is not None and tgt.radius > 0:
            self.target_patch.set_center((tgt.x, tgt.y))
            self.target_patch.set_radius(tgt.radius)
            self.target_patch.set_color(tgt.color)
            self.target_patch.set_alpha(0.25)
            self.target_patch.set_visible(True)
        else:
            self.target_patch.set_visible(False)

        # traces (scroll ±scroll_s around now; playhead fixed at x=0)
        lo_ns = t_ns - int(self.scroll_s * 1e9)
        hi_ns = t_ns + int(self.scroll_s * 1e9)
        for name, line in self.trace_lines.items():
            ts, vals = win.analog[name].slice(lo_ns, hi_ns)
            line.set_data((ts - t_ns) / 1e9, vals)
