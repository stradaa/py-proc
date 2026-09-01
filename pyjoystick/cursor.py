"""Canonical joystick->cursor reconstruction (normalized [0,1] task space).

This mirrors the Rust task's per-frame cursor law in
``rust/joystick_task/src/state.rs`` (Trial::step, the control_mode dispatch).
Keep the three branches here in lockstep with that file.

Raw joystick deflection is in [-1, 1]; the cursor lives in [0, 1] with 0.5 the
center. ``apply_direction_influence`` applies the per-direction gain the task
applies before either mapping.

The stateful modes (cumulative/blend) integrate the stick, so any difference
between the ROS joystick stream we replay and the stream the task itself
integrated accumulates without bound when ``reset_cursor_each_trial`` is off.
The task logs its own cursor on every ``target_entry`` / ``hold_start`` /
``target_exit`` event; pass those as ``cursor_anchors`` and the drift is
corrected against them (see ``apply_cursor_anchors``).
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


def apply_cursor_anchors(
    t_s: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cursor_anchors: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct an integrated cursor against the task's own logged positions.

    ``cursor_anchors`` are ``(time_s, cursor_x, cursor_y)`` triples on the same
    clock as ``t_s``. The residual at each anchor is interpolated linearly
    between anchors (held constant outside their span), so the corrected trace
    passes exactly through every anchor and stays continuous in between.
    """
    anchors = np.asarray(cursor_anchors, dtype=float)
    if anchors.size == 0 or cx.size == 0:
        return cx, cy
    anchors = anchors[np.all(np.isfinite(anchors), axis=1)]
    if anchors.size == 0:
        return cx, cy
    # Anchors outside the sample span would clamp onto the first/last sample and
    # inject a residual measured at the wrong time; drop them instead.
    t_s = np.asarray(t_s, dtype=float)
    anchors = anchors[(anchors[:, 0] >= t_s[0]) & (anchors[:, 0] <= t_s[-1])]
    if anchors.size == 0:
        return cx, cy
    anchors = anchors[np.argsort(anchors[:, 0])]

    idx = np.clip(np.searchsorted(t_s, anchors[:, 0], side="left"), 0, len(t_s) - 1)
    # One anchor per sample index (keep the last), so np.interp gets strictly
    # increasing sample points.
    idx, keep = np.unique(idx[::-1], return_index=True)
    keep = len(anchors) - 1 - keep
    err_x = anchors[keep, 1] - cx[idx]
    err_y = anchors[keep, 2] - cy[idx]

    # Interpolate over sample index, not time: the joystick stream carries
    # duplicate timestamps (~18% of samples), which np.interp would silently
    # mishandle as a non-increasing xp. Sampling is near-uniform, so index and
    # time are interchangeable here.
    samples = np.arange(len(cx), dtype=float)
    anchor_idx = idx.astype(float)
    return (
        cx + np.interp(samples, anchor_idx, err_x),
        cy + np.interp(samples, anchor_idx, err_y),
    )


def reconstruct_cursor(
    t_s: np.ndarray,
    jx: np.ndarray,
    jy: np.ndarray,
    cfg: Dict,
    trial_start_times_s: Sequence[float] = (),
    cursor_anchors: Sequence[Tuple[float, float, float]] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cursor_x, cursor_y) in [0,1], aligned to the joystick samples.

    ``t_s`` is the joystick sample time in seconds (any monotonic clock; only
    diffs are used). ``trial_start_times_s`` (same clock) are re-center points
    honored by the stateful modes when ``reset_cursor_each_trial`` is set.
    ``cursor_anchors`` are ``(time_s, cursor_x, cursor_y)`` triples logged by
    the task itself; when given, the stateful modes integrate unclamped and the
    accumulated drift is removed against them before the final clamp.
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
    t = np.asarray(t_s, dtype=float)
    resets = _reset_indices(t, trial_start_times_s) if reset_each_trial else set()

    # With anchors the integration must run unclamped: clamping at [0,1] is a
    # lossy nonlinearity that would bake the drift we are about to remove into
    # the state. The clamp is applied once, after the correction.
    anchored = len(cursor_anchors) > 0
    dt = np.diff(t, prepend=t[0])

    if anchored and not is_blend and not resets:
        # Pure velocity integration is a cumulative sum once the clamp is gone.
        vx, vy = ijx.copy(), ijy.copy()
        if zero_drift:
            idle = np.hypot(vx, vy) < buffer
            vx[idle] = 0.0
            vy[idle] = 0.0
        vx[0] = vy[0] = 0.0
        cx = 0.5 + np.cumsum(vx * speed * dt)
        cy = 0.5 + np.cumsum(vy * speed * dt)
    else:
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

            if not anchored:
                px = min(max(px, 0.0), 1.0)
                py = min(max(py, 0.0), 1.0)
            cx[i] = px
            cy[i] = py

    if anchored:
        cx, cy = apply_cursor_anchors(t, cx, cy, cursor_anchors)
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
    return cx, cy
