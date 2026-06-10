"""Reconstruct cursor position (normalized [0,1] task space) from joystick.

The joystick panel overlays the task target, which is specified in normalized
[0,1] task coordinates. Raw joystick deflection is in [-1,1], so we map it to
cursor space exactly as the task does. Logic mirrors
pyCheck/joystick_validation.py (reconstruct_cursor / apply_direction_influence).

- "direct" mode: cursor = 0.5 + influence(joystick) * direct_range (stateless).
- cumulative/velocity mode: integrate joystick over time, resetting to center
  (0.5, 0.5) at each trial boundary.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def apply_direction_influence(jx: np.ndarray, jy: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    up = float(cfg.get("up_influence_pct", 100.0)) / 100.0
    down = float(cfg.get("down_influence_pct", 100.0)) / 100.0
    left = float(cfg.get("left_influence_pct", 100.0)) / 100.0
    right = float(cfg.get("right_influence_pct", 100.0)) / 100.0
    jx = np.asarray(jx, dtype=float).copy()
    jy = np.asarray(jy, dtype=float).copy()
    jx[jx > 0] *= right
    jx[jx < 0] *= left
    jy[jy > 0] *= up
    jy[jy < 0] *= down
    return jx, jy


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

    jx, jy = apply_direction_influence(joystick_x, joystick_y, cfg)
    mode = str(cfg.get("control_mode", "direct")).lower()

    if mode == "direct":
        direct_range = float(cfg.get("direct_range", 0.45))
        cx = np.clip(0.5 + jx * direct_range, 0.0, 1.0)
        cy = np.clip(0.5 + jy * direct_range, 0.0, 1.0)
        return cx, cy

    # cumulative / velocity integration, reset to center at each trial start
    speed = float(cfg.get("cumulative_speed", 0.70))
    zero_drift = bool(cfg.get("zero_drift_mode", True))
    buffer = float(cfg.get("zero_drift_buffer", 0.05))
    t_s = joystick_t_ns.astype(float) / 1e9
    dt = np.diff(t_s, prepend=t_s[0])
    resets = set(int(np.searchsorted(joystick_t_ns, s, side="left")) for s in trial_starts_ns)

    cx = np.empty(len(jx)); cy = np.empty(len(jy))
    px = py = 0.5
    for i in range(len(jx)):
        if i in resets or i == 0:
            px = py = 0.5
        else:
            ix, iy = float(jx[i]), float(jy[i])
            if zero_drift and np.hypot(ix, iy) < buffer:
                ix = iy = 0.0
            px = min(max(px + ix * speed * dt[i], 0.0), 1.0)
            py = min(max(py + iy * speed * dt[i], 0.0), 1.0)
        cx[i] = px; cy[i] = py
    return cx, cy
