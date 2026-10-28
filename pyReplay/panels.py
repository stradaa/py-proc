"""The live player's signal panels: cursor/target panel + scrolling traces.

Layout: a square cursor-recreation panel on the left (see cursor_panel.py) and
the analog NIDAQ channels stacked on the right as scrolling traces with a fixed
central playhead (see trace_panel.py). This composite is what the interactive
player embeds; the MP4 exporter places the same two pieces itself, in one of
the arrangements in layouts.py.
"""

from __future__ import annotations

from typing import Optional

from matplotlib.figure import Figure

from pyReplay.cursor_panel import CursorPanel
from pyReplay.trace_panel import TracePanel
from pyReplay.window import ReplayWindow


class SignalPanels:
    """Cursor panel + trace panel side by side in one figure region."""

    def __init__(self, win: ReplayWindow, scroll_s: float = 2.0,
                 fig: Optional[Figure] = None, host=None):
        """fig/host let the panels build into an existing figure region. When
        both are None, the panels own a standalone figure (the live player)."""
        self.win = win
        self.fig = fig if fig is not None else Figure(figsize=(9, 4.2), facecolor="white")

        if host is None:
            gs = self.fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4],
                                       wspace=0.28,
                                       left=0.07, right=0.97, top=0.92, bottom=0.12)
        else:
            gs = host.subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.3)

        self.ax_joy = self.fig.add_subplot(gs[0, 0])
        self.cursor = CursorPanel(win, self.ax_joy)
        self.traces = TracePanel(win, self.fig, gs[0, 1], scroll_s=scroll_s)
        self.animated = self.cursor.animated + self.traces.animated

    # --- attributes the player reads ---
    @property
    def scroll_s(self) -> float:
        return self.traces.scroll_s

    @property
    def trace_axes(self):
        return self.traces.axes

    @property
    def trace_lines(self):
        return self.traces.lines

    def set_scroll(self, scroll_s: float):
        self.traces.set_scroll(scroll_s)

    def update(self, t_rel_s: float):
        self.cursor.update(t_rel_s)
        self.traces.update(t_rel_s)
