"""The cursor / target recreation panel (normalized [0,1] task space).

This is the panel the replay video is built around: the reconstructed cursor,
its recent trail, the active trial target and the static center gate. It owns a
single matplotlib Axes so any layout can place it wherever it likes and size it
freely (see layouts.py); `scale` grows the marker/line/font sizes together so a
large hero panel does not end up with a tiny cursor dot on it.

All limits are fixed, so only the trail/cursor/target artists change per frame
and the panel stays blittable (see blit.py).
"""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.patches import Circle

from pyReplay.window import ReplayWindow

_TRAIL_S = 1.0          # seconds of cursor history drawn as a trail


class CursorPanel:
    """Cursor + target recreation drawn into one Axes."""

    def __init__(self, win: ReplayWindow, ax, *, scale: float = 1.0,
                 trail_s: float = _TRAIL_S, title: str = "Cursor & target",
                 labels: bool = True):
        self.win = win
        self.ax = ax
        self.scale = scale
        self.trail_s = trail_s
        self.animated: List = []
        self._build(title, labels)

    def _build(self, title: str, labels: bool):
        ax, s = self.ax, self.scale
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        if title:
            ax.set_title(title, fontsize=11 * s, fontweight="bold", pad=6 * s)
        if labels:
            ax.set_xlabel("cursor x (norm)", fontsize=9 * s)
            ax.set_ylabel("cursor y (norm)", fontsize=9 * s)
            ax.tick_params(labelsize=8 * s)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, alpha=0.2)

        # static center gate (part of the blit background)
        if self.win.center_gate_radius > 0:
            ax.add_patch(Circle((0.5, 0.5), self.win.center_gate_radius,
                                fill=False, ec="#999999", ls="--",
                                lw=1.0 * s, alpha=0.6))

        # active target (hidden until a trial target is active)
        self.target_patch = Circle((0.5, 0.5), 0.0, color="#4caf50", alpha=0.25,
                                   visible=False)
        ax.add_patch(self.target_patch)
        self.target_ring = Circle((0.5, 0.5), 0.0, fill=False, ec="#2e7d32",
                                  lw=1.8 * s, alpha=0.9, visible=False)
        ax.add_patch(self.target_ring)
        # cursor trail + current cursor
        (self.trail_line,) = self.ax.plot([], [], color="#1f77b4",
                                          lw=2.0 * s, alpha=0.9)
        (self.cursor_pt,) = self.ax.plot([], [], "o", color="black",
                                         ms=8 * s, zorder=6)
        self.animated += [self.target_patch, self.target_ring,
                          self.trail_line, self.cursor_pt]

    def update(self, t_rel_s: float):
        win = self.win
        t_ns = win.rel_to_ns(t_rel_s)

        idx = win.joystick_index_at(t_ns)
        if idx >= 0 and len(win.cursor_x):
            lo_ns = t_ns - int(self.trail_s * 1e9)
            j0 = int(np.searchsorted(win.joystick_t_ns, lo_ns, side="left"))
            self.trail_line.set_data(win.cursor_x[j0:idx + 1],
                                     win.cursor_y[j0:idx + 1])
            self.cursor_pt.set_data([win.cursor_x[idx]], [win.cursor_y[idx]])

        tgt = win.active_target(t_ns)
        if tgt is not None and tgt.radius > 0:
            for patch in (self.target_patch, self.target_ring):
                patch.set_center((tgt.x, tgt.y))
                patch.set_radius(tgt.radius)
                patch.set_visible(True)
            self.target_patch.set_color(tgt.color)
            self.target_patch.set_alpha(0.25)
            self.target_ring.set_edgecolor("#2e7d32")
        else:
            self.target_patch.set_visible(False)
            self.target_ring.set_visible(False)
