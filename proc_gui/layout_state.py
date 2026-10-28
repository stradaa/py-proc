"""Persistence for the proc_gui window layout.

Window geometry, splitter positions, which panels are minimised and the active
tab are written to :class:`QSettings` on close and restored on the next launch,
so a session comes back arranged the way it was left.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import QByteArray, QSettings
from PyQt6.QtWidgets import QMainWindow, QSplitter

from proc_gui.collapsible import CollapsiblePanel


ORGANISATION = "PesaranLab"
APPLICATION = "ProcGui"
PREFIX = "ProcGui/layout"
VERSION = 1


def settings() -> QSettings:
    return QSettings(ORGANISATION, APPLICATION)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    if value is None:
        return default
    return bool(value)


class LayoutState:
    """Registry of the widgets whose arrangement survives between sessions."""

    def __init__(self) -> None:
        self._splitters: dict[str, QSplitter] = {}
        self._splitter_defaults: dict[str, QByteArray] = {}
        self._panels: dict[str, CollapsiblePanel] = {}
        self._values: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]] = {}
        self._window_default: Optional[QByteArray] = None

    # -- registration ---------------------------------------------------
    def add_splitter(self, name: str, splitter: QSplitter) -> None:
        """Register a splitter.  Call after its default sizes are set: the
        current state becomes the layout that ``reset`` returns to."""
        self._splitters[name] = splitter
        self._splitter_defaults[name] = splitter.saveState()

    def add_panel(self, panel: CollapsiblePanel) -> None:
        self._panels[panel.key] = panel

    def add_value(self, name: str, getter: Callable[[], Any], setter: Callable[[Any], None]) -> None:
        """Register any other scalar worth remembering (e.g. the active tab)."""
        self._values[name] = (getter, setter)

    def panels(self) -> list[CollapsiblePanel]:
        return list(self._panels.values())

    # -- save / restore -------------------------------------------------
    def save(self, window: QMainWindow) -> None:
        store = settings()
        store.setValue(f"{PREFIX}/version", VERSION)
        store.setValue(f"{PREFIX}/geometry", window.saveGeometry())
        store.setValue(f"{PREFIX}/maximised", window.isMaximized())
        for name, splitter in self._splitters.items():
            store.setValue(f"{PREFIX}/splitter/{name}", splitter.saveState())
        for key, panel in self._panels.items():
            store.setValue(f"{PREFIX}/panel/{key}/collapsed", panel.is_collapsed())
            store.setValue(f"{PREFIX}/panel/{key}/size", panel.expanded_size())
        for name, (getter, _setter) in self._values.items():
            try:
                store.setValue(f"{PREFIX}/value/{name}", getter())
            except Exception:
                pass
        store.sync()

    def restore(self, window: QMainWindow) -> bool:
        """Apply the stored layout.  Returns False when there is nothing saved
        (first launch, or after a reset)."""
        store = settings()
        if store.value(f"{PREFIX}/version") is None:
            return False
        if self._window_default is None:
            self._window_default = window.saveGeometry()

        # Panels first: collapsing constrains a panel's size, so the splitter
        # states saved alongside them only make sense once that is applied.
        for key, panel in self._panels.items():
            # Collapse first: collapsing records the current size, so the saved
            # one has to be applied afterwards to survive.
            panel.set_collapsed(_as_bool(store.value(f"{PREFIX}/panel/{key}/collapsed"), False))
            try:
                panel.set_expanded_size(int(store.value(f"{PREFIX}/panel/{key}/size", 0)))
            except (TypeError, ValueError):
                pass

        for name, splitter in self._splitters.items():
            state = store.value(f"{PREFIX}/splitter/{name}")
            if isinstance(state, QByteArray):
                splitter.restoreState(state)

        geometry = store.value(f"{PREFIX}/geometry")
        if isinstance(geometry, QByteArray):
            window.restoreGeometry(geometry)
        if _as_bool(store.value(f"{PREFIX}/maximised"), False):
            window.showMaximized()

        for name, (_getter, setter) in self._values.items():
            value = store.value(f"{PREFIX}/value/{name}")
            if value is None:
                continue
            try:
                setter(value)
            except Exception:
                pass
        return True

    def reset(self, window: QMainWindow) -> None:
        """Forget the stored layout and put the window back to its defaults."""
        store = settings()
        store.remove(PREFIX)
        store.sync()
        for panel in self._panels.values():
            panel.set_collapsed(False)
        for name, splitter in self._splitters.items():
            default = self._splitter_defaults.get(name)
            if default is not None:
                splitter.restoreState(default)
        if self._window_default is not None:
            window.restoreGeometry(self._window_default)
        window.showMaximized()
