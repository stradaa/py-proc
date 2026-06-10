"""Minimal blit manager for fast matplotlib redraws (matplotlib docs pattern).

Caches the static background once and only redraws the animated artists each
frame, which keeps the joystick + trace panels smooth at playback rates.

Requires that axis limits and any static artists do NOT change between frames;
only the animated artists' data may change.
"""

from __future__ import annotations

from typing import Iterable, List


class BlitManager:
    def __init__(self, canvas, animated_artists: Iterable = ()):
        self.canvas = canvas
        self._bg = None
        self._artists: List = []
        for art in animated_artists:
            self.add_artist(art)
        self._cid = canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event):
        cv = self.canvas
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        fig = self.canvas.figure
        for art in self._artists:
            fig.draw_artist(art)

    def update(self):
        cv = self.canvas
        if self._bg is None:
            self._on_draw(None)
        else:
            cv.restore_region(self._bg)
            self._draw_animated()
            cv.blit(cv.figure.bbox)
        cv.flush_events()
