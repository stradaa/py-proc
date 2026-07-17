"""Reconstruct cursor position (normalized [0,1] task space) from joystick.

Thin wrapper over the shared canonical implementation in ``pyjoystick.cursor``
so the replay overlay and the pyCheck metrics layer cannot drift. See that
module for the direct / cumulative / blend control-law definitions.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from pyjoystick.cursor import apply_direction_influence, reconstruct_cursor

__all__ = ["apply_direction_influence", "reconstruct"]


def reconstruct(
    joystick_t_ns: np.ndarray,
    joystick_x: np.ndarray,
    joystick_y: np.ndarray,
    cfg: Dict,
    trial_starts_ns: Sequence[int] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cursor_x, cursor_y) in [0,1], aligned to joystick samples."""
    if len(joystick_x) == 0:
        return np.empty(0), np.empty(0)
    t_s = np.asarray(joystick_t_ns, dtype=float) / 1e9
    trial_start_times_s = [float(s) / 1e9 for s in trial_starts_ns]
    return reconstruct_cursor(t_s, joystick_x, joystick_y, cfg, trial_start_times_s)
