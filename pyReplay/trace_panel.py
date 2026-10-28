"""Scrolling analog traces (fiducial / photodiode / reward) with a fixed playhead.

Split out of panels.py so a layout can place the traces anywhere and at any
size. `orientation` picks a vertical stack (one channel per row) or a single
row of side-by-side channels; `compact` strips the traces down to a label and a
line so they read as a small supporting strip rather than a main panel — which
is what the video layouts want, since the time series is deliberately not the
focus.

Limits are fixed so only the line data changes per frame (blittable).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from matplotlib.figure import Figure

from pyReplay.window import ReplayWindow

# Channels rendered as traces, in order, with colors.
TRACE_ORDER = [
    ("fiducial", "#4c78a8"),
    ("photodiode", "#e45756"),
    ("reward", "#54a24b"),
]


class TracePanel:
    """The analog trace strip built into a host gridspec cell."""

    def __init__(self, win: ReplayWindow, fig: Figure, host=None, *,
                 rects=None, scroll_s: float = 2.0, orientation: str = "stacked",
                 compact: bool = False, scale: float = 1.0,
                 channels: Sequence[Tuple[str, str]] = None):
        """Built either into a gridspec cell (`host`) or at explicit figure-
        coordinate rects (`rects`, one per channel — see layouts.py)."""
        self.win = win
        self.fig = fig
        self.scroll_s = scroll_s
        self.compact = compact
        self.scale = scale
        self.animated: List = []
        self.lines: Dict[str, object] = {}
        self.axes: Dict[str, object] = {}

        chans = [(n, c) for n, c in (channels or TRACE_ORDER) if n in win.analog]
        n = max(len(chans), 1)
        if rects is not None:
            cells = list(rects)[:len(chans)]
        elif orientation == "row":
            gs = host.subgridspec(1, n, wspace=0.18 if compact else 0.25)
            cells = [gs[0, i] for i in range(len(chans))]
        else:
            gs = host.subgridspec(n, 1, hspace=0.30 if compact else 0.25)
            cells = [gs[i, 0] for i in range(len(chans))]
        self._build(chans, cells, orientation)

    def _add_axes(self, cell):
        """`cell` is a gridspec SubplotSpec or a (l, b, w, h) figure rect."""
        if isinstance(cell, tuple):
            return self.fig.add_axes(cell)
        return self.fig.add_subplot(cell)

    def _build(self, chans, cells, orientation: str):
        s = self.scale
        last = len(chans) - 1
        for i, ((name, color), cell) in enumerate(zip(chans, cells)):
            ax = self._add_axes(cell)
            track = self.win.analog[name]
            if len(track.values):
                vlo, vhi = float(track.values.min()), float(track.values.max())
                pad = 0.05 * (vhi - vlo + 1e-9)
            else:
                vlo, vhi, pad = 0.0, 1.0, 0.05
            ax.set_xlim(-self.scroll_s, self.scroll_s)
            ax.set_ylim(vlo - pad, vhi + pad)
            ax.grid(True, alpha=0.15)
            ax.axvline(0.0, color="#111111", lw=1.0 * s, alpha=0.7)  # playhead

            if self.compact:
                # Label inside the axes; no ticks. Keeps the strip thin.
                ax.set_yticks([])
                ax.text(0.012, 0.86, name, transform=ax.transAxes,
                        fontsize=8 * s, color=color, fontweight="bold",
                        ha="left", va="top")
                ax.tick_params(labelsize=7 * s)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            else:
                ax.set_ylabel(name, fontsize=9 * s)
                ax.tick_params(labelsize=8 * s)

            bottom_edge = (orientation == "row") or (i == last)
            if bottom_edge:
                ax.set_xlabel("time from playhead (s)", fontsize=8 * s)
            else:
                ax.tick_params(labelbottom=False)

            (line,) = ax.plot([], [], color=color, lw=1.1 * s)
            self.lines[name] = line
            self.axes[name] = ax
            self.animated.append(line)

    def set_scroll(self, scroll_s: float):
        """Change the scroll half-width (invalidates any blit background)."""
        self.scroll_s = scroll_s
        for ax in self.axes.values():
            ax.set_xlim(-scroll_s, scroll_s)

    def update(self, t_rel_s: float):
        t_ns = self.win.rel_to_ns(t_rel_s)
        lo_ns = t_ns - int(self.scroll_s * 1e9)
        hi_ns = t_ns + int(self.scroll_s * 1e9)
        for name, line in self.lines.items():
            ts, vals = self.win.analog[name].slice(lo_ns, hi_ns)
            line.set_data((ts - t_ns) / 1e9, vals)
