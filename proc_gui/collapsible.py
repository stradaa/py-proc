"""Collapsible splitter panels.

Every top-level panel in the GUI is wrapped in a :class:`CollapsiblePanel`,
which adds a thin title bar carrying a collapse marker.  Collapsing a panel
shrinks it to a narrow labelled strip (or to its title bar, inside a vertical
splitter) so the figure preview can take the freed space; clicking the strip
gives the panel its previous size back.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


STRIP_PX = 24          # width of a collapsed horizontal panel
MIN_DONOR_PX = 60      # never shrink a neighbour below this when expanding
QWIDGETSIZE_MAX = 16777215


class VerticalLabel(QLabel):
    """QLabel that paints its text rotated, for the collapsed strip."""

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.translate(self.width(), 0)
        painter.rotate(90)
        painter.drawText(
            0,
            0,
            self.height(),
            self.width(),
            int(Qt.AlignmentFlag.AlignCenter),
            self.text(),
        )
        painter.end()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(hint.height(), hint.width())

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.height(), hint.width())


class CollapsiblePanel(QWidget):
    """Wrap a splitter child so it can be minimised out of the way.

    ``orientation`` is the orientation of the splitter the panel lives in:
    a panel in a horizontal splitter collapses sideways to a labelled strip,
    one in a vertical splitter collapses down onto its title bar.

    When ``content`` is a :class:`QGroupBox` its title is moved onto the panel
    header so the title is not drawn twice.
    """

    collapsedChanged = pyqtSignal(bool)

    def __init__(
        self,
        content: QWidget,
        *,
        key: str,
        title: str = "",
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not title and isinstance(content, QGroupBox):
            title = content.title()
            content.setTitle("")
        self.key = key
        self.title = title or key
        self._orientation = orientation
        self._content = content
        self._collapsed = False
        self._expanded_size = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = _PanelHeader(self.title, self.toggle)
        self._strip = QWidget()
        strip_layout = QVBoxLayout(self._strip)
        strip_layout.setContentsMargins(0, 2, 0, 2)
        strip_layout.setSpacing(2)
        self._strip_button = QToolButton()
        self._strip_button.setAutoRaise(True)
        self._strip_button.setText("▸")
        self._strip_button.setToolTip(f"Show {self.title}")
        self._strip_button.clicked.connect(self.expand)
        self._strip_label = VerticalLabel(self.title)
        self._strip_label.setStyleSheet("color:#888;")
        self._strip_label.setToolTip(f"Show {self.title}")
        strip_layout.addWidget(self._strip_button)
        strip_layout.addWidget(self._strip_label, 1)
        self._strip.hide()

        outer.addWidget(self._header)
        outer.addWidget(self._content, 1)
        outer.addWidget(self._strip, 1)

    # -- state ---------------------------------------------------------
    def is_collapsed(self) -> bool:
        return self._collapsed

    def expanded_size(self) -> int:
        """Splitter size the panel returns to when expanded."""
        if not self._collapsed:
            current = self._splitter_size()
            if current:
                self._expanded_size = current
        return self._expanded_size

    def set_expanded_size(self, size: int) -> None:
        if size > 0:
            self._expanded_size = size

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def expand(self) -> None:
        self.set_collapsed(False)

    def collapse(self) -> None:
        self.set_collapsed(True)

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        if collapsed:
            self._expanded_size = self._splitter_size() or self._expanded_size
            self._content.hide()
            if self._orientation == Qt.Orientation.Horizontal:
                self._header.hide()
                self._strip.show()
                self.setMinimumWidth(STRIP_PX)
                self.setMaximumWidth(STRIP_PX)
            else:
                self.setMaximumHeight(self._header.sizeHint().height())
        else:
            self._strip.hide()
            self._header.show()
            self._content.show()
            self.setMinimumWidth(0)
            self.setMaximumWidth(QWIDGETSIZE_MAX)
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            QTimer.singleShot(0, self._restore_splitter_size)
        self._collapsed = collapsed
        self._header.set_collapsed(collapsed)
        self.collapsedChanged.emit(collapsed)

    # -- splitter bookkeeping ------------------------------------------
    def _splitter(self) -> Optional[QSplitter]:
        parent = self.parentWidget()
        return parent if isinstance(parent, QSplitter) else None

    def _splitter_size(self) -> int:
        splitter = self._splitter()
        if splitter is None:
            return 0
        index = splitter.indexOf(self)
        if index < 0:
            return 0
        return splitter.sizes()[index]

    def _restore_splitter_size(self) -> None:
        """Give the panel its pre-collapse size back, taking the space from
        the roomiest neighbour so the other panels keep their sizes."""
        splitter = self._splitter()
        if splitter is None or self._expanded_size <= 0:
            return
        sizes = splitter.sizes()
        index = splitter.indexOf(self)
        if index < 0 or len(sizes) < 2:
            return
        delta = self._expanded_size - sizes[index]
        if delta <= 0:
            return
        donor = max(
            range(len(sizes)),
            key=lambda i: -1 if i == index else sizes[i],
        )
        if sizes[donor] - delta < MIN_DONOR_PX:
            delta = max(0, sizes[donor] - MIN_DONOR_PX)
        if delta <= 0:
            return
        sizes[donor] -= delta
        sizes[index] += delta
        splitter.setSizes(sizes)


class _PanelHeader(QWidget):
    """Title bar with the collapse marker; double-click toggles as well."""

    def __init__(self, title: str, on_toggle: Any) -> None:
        super().__init__()
        self._on_toggle = on_toggle
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)
        self.button = QToolButton()
        self.button.setAutoRaise(True)
        self.button.setText("▾")
        self.button.setToolTip(f"Minimise {title}")
        self.button.clicked.connect(on_toggle)
        self.label = QLabel(title)
        font = self.label.font()
        font.setBold(True)
        self.label.setFont(font)
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self._title = title

    def set_collapsed(self, collapsed: bool) -> None:
        self.button.setText("▸" if collapsed else "▾")
        self.button.setToolTip(
            f"Show {self._title}" if collapsed else f"Minimise {self._title}"
        )

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self._on_toggle()
        event.accept()
