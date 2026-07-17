"""Canonical joystick->cursor reconstruction (normalized [0,1] task space).

This mirrors the Rust task's per-frame cursor law in
``rust/joystick_task/src/state.rs`` (Trial::step, the control_mode dispatch).
Keep the three branches here in lockstep with that file.

Raw joystick deflection is in [-1, 1]; the cursor lives in [0, 1] with 0.5 the
center. ``apply_direction_influence`` applies the per-direction gain the task
applies before either mapping.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def apply_direction_influence(
    jx: np.ndarray, jy: np.ndarray, cfg: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale raw joystick per quadrant by the {up,down,left,right}_influence_pct."""
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


def _reset_indices(
    t_s: np.ndarray, trial_start_times_s: Sequence[float]
) -> set:
    """Sample indices at (or just after) each trial-start time; where the
    cursor is re-centered to 0.5, matching the task's reset_cursor_each_trial."""
    if len(trial_start_times_s) == 0:
        return set()
    idx = np.searchsorted(t_s, np.asarray(trial_start_times_s, dtype=float), side="left")
    return {int(i) for i in idx if 0 <= i < len(t_s)}


def reconstruct_cursor(
    t_s: np.ndarray,
    jx: np.ndarray,
    jy: np.ndarray,
    cfg: Dict,
    trial_start_times_s: Sequence[float] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cursor_x, cursor_y) in [0,1], aligned to the joystick samples.

    ``t_s`` is the joystick sample time in seconds (any monotonic clock; only
    diffs are used). ``trial_start_times_s`` (same clock) are re-center points
    honored by the stateful modes when ``reset_cursor_each_trial`` is set.
    """
    jx = np.asarray(jx, dtype=float)
    jy = np.asarray(jy, dtype=float)
    if jx.size == 0:
        return np.empty(0), np.empty(0)

    ijx, ijy = apply_direction_influence(jx, jy, cfg)
    mode = str(cfg.get("control_mode", "direct")).lower()
    direct_range = float(cfg.get("direct_range", 0.45))

    # --- direct: stateless absolute position ---
    if mode == "direct":
        cx = np.clip(0.5 + ijx * direct_range, 0.0, 1.0)
        cy = np.clip(0.5 + ijy * direct_range, 0.0, 1.0)
        return cx, cy

    # --- stateful modes: cumulative (velocity) and blend (leaky integrator) ---
    is_blend = mode == "blend"
    speed = float(cfg.get("cumulative_speed", 0.70))
    zero_drift = bool(cfg.get("zero_drift_mode", True))
    buffer = float(cfg.get("zero_drift_buffer", 0.05))
    blend = float(np.clip(cfg.get("position_velocity_blend", 0.0), 0.0, 1.0)) if is_blend else 0.0

    reset_each_trial = bool(cfg.get("reset_cursor_each_trial", True))
    resets = _reset_indices(np.asarray(t_s, dtype=float), trial_start_times_s) if reset_each_trial else set()

    t = np.asarray(t_s, dtype=float)
    dt = np.diff(t, prepend=t[0])

    cx = np.empty(jx.size)
    cy = np.empty(jy.size)
    px = py = 0.5
    for i in range(jx.size):
        if i == 0 or i in resets:
            px = py = 0.5
            cx[i] = px
            cy[i] = py
            continue

        # Velocity core (same zero-drift deadband as cumulative mode).
        vx = float(ijx[i])
        vy = float(ijy[i])
        if zero_drift and np.hypot(vx, vy) < buffer:
            vx = vy = 0.0
        px = px + vx * speed * dt[i]
        py = py + vy * speed * dt[i]

        # Blend: relax toward the direct (stick-proportional) target with
        # relax = clamp((1-b)/b * dt, 0, 1). b<=0 -> relax=1 (== direct),
        # b>=1 -> relax=0 (== cumulative). Uses the (non-deadbanded) influenced
        # joystick for the target, matching the Rust `px = 0.5 + bjx*range`.
        if is_blend:
            if blend <= 0.0:
                relax = 1.0
            elif blend >= 1.0:
                relax = 0.0
            else:
                relax = min(max((1.0 - blend) / blend * dt[i], 0.0), 1.0)
            tx = 0.5 + float(ijx[i]) * direct_range
            ty = 0.5 + float(ijy[i]) * direct_range
            px = px + (tx - px) * relax
            py = py + (ty - py) * relax

        px = min(max(px, 0.0), 1.0)
        py = min(max(py, 0.0), 1.0)
        cx[i] = px
        cy[i] = py

    return cx, cy
