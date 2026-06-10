"""Standalone PyQt6 replay player window.

Cameras are laid out across the top as native QLabel/QPixmap views (fast); the
joystick + analog trace panels sit below in an embedded, blitted matplotlib
canvas. A QTimer drives real-time playback; the toolbar offers play/pause, a
seek slider, a speed multiplier, and the trace scroll-window width.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from pyReplay.blit import BlitManager
from pyReplay.panels import SignalPanels
from pyReplay.window import CameraTrack, ReplayWindow

_SLIDER_TICKS = 2000
_FPS = 30


class CameraView(QLabel):
    """A QLabel that holds a grayscale frame and rescales it on resize."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(160, 130)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#000; color:#aaa;")
        self.setText(name)

    def set_frame(self, frame: np.ndarray | None):
        if frame is None:
            return
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape
        img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(img.copy())
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        self._rescale()
        super().resizeEvent(event)


class ReplayPlayer(QWidget):
    def __init__(self, win: ReplayWindow, scroll_s: float = 2.0):
        super().__init__()
        self.win = win
        self.cur_s = win.start_s
        self.playing = False
        self.speed = 1.0

        self.setWindowTitle(f"pyReplay — {win.source}  [{win.start_s:.1f}-{win.stop_s:.1f}s]")
        self.resize(1280, 820)

        root = QVBoxLayout(self)
        root.addLayout(self._build_cameras(), stretch=3)

        self.panels = SignalPanels(win, scroll_s=scroll_s)
        self.canvas = FigureCanvasQTAgg(self.panels.fig)
        root.addWidget(self.canvas, stretch=2)
        self.blit = BlitManager(self.canvas, self.panels.animated)

        root.addLayout(self._build_controls())

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / _FPS))
        self.timer.timeout.connect(self._on_tick)

        self.canvas.draw()
        self._render(self.cur_s)

    # ---- UI construction ----
    def _build_cameras(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.cam_views: Dict[str, CameraView] = {}
        for name in self.win.cameras:
            view = CameraView(name)
            self.cam_views[name] = view
            col = QVBoxLayout()
            title = QLabel(name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(title)
            col.addWidget(view, stretch=1)
            row.addLayout(col)
        return row

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _SLIDER_TICKS)
        self.slider.valueChanged.connect(self._on_slider)
        bar.addWidget(self.slider, stretch=1)

        self.time_label = QLabel()
        self.time_label.setMinimumWidth(140)
        bar.addWidget(self.time_label)

        bar.addWidget(QLabel("Speed"))
        self.speed_box = QComboBox()
        for s in ("0.25", "0.5", "1.0", "2.0", "4.0"):
            self.speed_box.addItem(f"{s}x", float(s))
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentIndexChanged.connect(self._on_speed)
        bar.addWidget(self.speed_box)

        bar.addWidget(QLabel("Scroll ±s"))
        self.scroll_box = QDoubleSpinBox()
        self.scroll_box.setRange(0.25, 30.0)
        self.scroll_box.setSingleStep(0.5)
        self.scroll_box.setValue(self.panels.scroll_s)
        self.scroll_box.valueChanged.connect(self._on_scroll)
        bar.addWidget(self.scroll_box)

        self.export_btn = QPushButton("Export MP4")
        self.export_btn.clicked.connect(self._on_export)
        bar.addWidget(self.export_btn)
        return bar

    # ---- playback ----
    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            if self.cur_s >= self.win.stop_s - 1e-3:
                self.cur_s = self.win.start_s
            self.timer.start()
        else:
            self.timer.stop()

    def _on_tick(self):
        self.cur_s += (1.0 / _FPS) * self.speed
        if self.cur_s >= self.win.stop_s:
            self.cur_s = self.win.stop_s
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("Play")
        self._render(self.cur_s)

    def _on_slider(self, value: int):
        if self.playing:
            return
        frac = value / _SLIDER_TICKS
        self.cur_s = self.win.start_s + frac * self.win.duration_s
        self._render(self.cur_s)

    def _on_speed(self):
        self.speed = float(self.speed_box.currentData())

    def _on_scroll(self, value: float):
        self.panels.set_scroll(value)
        self.canvas.draw()          # refresh blit background for new xlim
        self._render(self.cur_s)

    def _on_export(self):
        if self.playing:
            self._toggle_play()
        default = f"replay_{int(self.win.start_s)}_{int(self.win.stop_s)}s.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export MP4", default, "MP4 (*.mp4)")
        if not path:
            return
        from pyReplay.export import export_mp4

        dlg = QProgressDialog("Rendering MP4…", "Cancel", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setValue(0)

        def prog(frac):
            if dlg.wasCanceled():
                raise RuntimeError("export cancelled")
            dlg.setValue(int(frac * 100))
            QApplication.processEvents()

        self.export_btn.setEnabled(False)
        try:
            export_mp4(self.win, path, scroll_s=self.panels.scroll_s, progress=prog)
        except RuntimeError:
            pass  # user cancelled
        finally:
            dlg.close()
            self.export_btn.setEnabled(True)

    # ---- rendering ----
    def _render(self, t_rel_s: float):
        t_ns = self.win.rel_to_ns(t_rel_s)
        for name, view in self.cam_views.items():
            track: CameraTrack = self.win.cameras[name]
            view.set_frame(track.frame_at(t_ns))

        self.panels.update(t_rel_s)
        self.blit.update()

        frac = (t_rel_s - self.win.start_s) / max(self.win.duration_s, 1e-9)
        self.slider.blockSignals(True)
        self.slider.setValue(int(frac * _SLIDER_TICKS))
        self.slider.blockSignals(False)
        self.time_label.setText(f"{t_rel_s:7.2f}s / {self.win.stop_s:.2f}s")
