"""Figure preview with zoom and pan.

The browser on the right of the window shows generated PNGs.  Plots are often
small, so the view supports wheel zoom around the cursor, drag panning and a
small toolbar (fit / 1:1 / +- / zoom readout).  ``ImagePreview`` keeps the
``set_image`` / ``clear_image`` API the rest of the GUI already uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPainter, QPixmap, QShortcut, QTransform
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


MIN_SCALE = 0.05
MAX_SCALE = 40.0
ZOOM_STEP = 1.25


class ZoomableImageView(QGraphicsView):
    """Graphics view holding one pixmap, zoomable with the wheel."""

    scaleChanged = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self._fit_mode = True
        self._has_pixmap = False
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # -- content --------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_pixmap = not pixmap.isNull()
        self.fit()

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self._has_pixmap = False

    def has_pixmap(self) -> bool:
        return self._has_pixmap

    # -- zoom -----------------------------------------------------------
    def current_scale(self) -> float:
        return float(self.transform().m11())

    def fit(self) -> None:
        self._fit_mode = True
        if not self._has_pixmap:
            return
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self.scaleChanged.emit(self.current_scale())

    def reset_zoom(self) -> None:
        """Show the figure at its native pixel size."""
        if not self._has_pixmap:
            return
        self._fit_mode = False
        self.setTransform(QTransform())
        self.scaleChanged.emit(self.current_scale())

    def zoom_by(self, factor: float) -> None:
        if not self._has_pixmap:
            return
        target = self.current_scale() * factor
        if target < MIN_SCALE or target > MAX_SCALE:
            return
        self._fit_mode = False
        self.scale(factor, factor)
        self.scaleChanged.emit(self.current_scale())

    # -- events ---------------------------------------------------------
    def wheelEvent(self, event: Any) -> None:
        if not self._has_pixmap:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP)
        event.accept()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.fit()
        event.accept()


class ImagePreview(QWidget):
    """Toolbar + zoomable view, with a placeholder when nothing is loaded."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.view = ZoomableImageView()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.scaleChanged.connect(self._on_scale_changed)

        self.placeholder = QLabel("No image selected")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet("color:#888;")

        layout.addWidget(self._build_toolbar())
        layout.addWidget(self.view, 1)
        layout.addWidget(self.placeholder, 1)
        self.view.hide()

        self._bind_shortcuts()

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(3)
        self.zoom_out_btn = self._tool_button("−", "Zoom out (wheel down)", lambda: self.view.zoom_by(1.0 / ZOOM_STEP))
        self.zoom_in_btn = self._tool_button("+", "Zoom in (wheel up)", lambda: self.view.zoom_by(ZOOM_STEP))
        self.fit_btn = self._tool_button("Fit", "Fit the figure to the panel (double-click, Ctrl+0)", self.view.fit)
        self.one_to_one_btn = self._tool_button("1:1", "Show at native pixel size (Ctrl+1)", self.view.reset_zoom)
        self.zoom_label = QLabel("--")
        self.zoom_label.setStyleSheet("color:#888;")
        for widget in (self.zoom_out_btn, self.zoom_in_btn, self.fit_btn, self.one_to_one_btn, self.zoom_label):
            row.addWidget(widget)
        row.addStretch(1)
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color:#888;")
        row.addWidget(self.path_label)
        return bar

    def _tool_button(self, text: str, tip: str, slot: Any) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.clicked.connect(slot)
        return button

    def _bind_shortcuts(self) -> None:
        for keys, slot in (
            ("Ctrl+0", self.view.fit),
            ("Ctrl+1", self.view.reset_zoom),
            ("Ctrl++", lambda: self.view.zoom_by(ZOOM_STEP)),
            ("Ctrl+=", lambda: self.view.zoom_by(ZOOM_STEP)),
            ("Ctrl+-", lambda: self.view.zoom_by(1.0 / ZOOM_STEP)),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # -- API used by the main window ------------------------------------
    def set_image(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear_image(f"Could not load image:\n{image_path}")
            return
        self.view.set_pixmap(pixmap)
        self.path_label.setText(f"{pixmap.width()}x{pixmap.height()} px")
        self.placeholder.hide()
        self.view.show()

    def clear_image(self, text: str = "No image selected") -> None:
        self.view.clear()
        self.view.hide()
        self.path_label.setText("")
        self.zoom_label.setText("--")
        self.placeholder.setText(text)
        self.placeholder.show()

    def _on_scale_changed(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")
