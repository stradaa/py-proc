"""Per-trial joystick kinematic primitives, on the task/behave clock.

Building blocks for the cross-day learning metrics (see
`cross_day_kinematics.py`): event-time lookup, cursor speed profiles,
submovement counting, SPARC smoothness, initial reach-direction error,
normalized reach paths, and engagement signals. Everything works off a
`JoystickDataset` (reconstructed cursor + `joystick_task_s`) and attempt event
`time_perf_counter` values — never the recorder clock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

try:
    from .joystick_validation import select_attempt_event
except ImportError:  # pragma: no cover - script-mode fallback
    from joystick_validation import select_attempt_event


# Window for the "initial" reach direction, and a sanity cap on reach duration
# (longer reaches are disengaged/idle, not movements we want kinematics from).
DIR_WINDOW_S = 0.15
MAX_REACH_S = 5.0


def event_perf(attempt: Dict[str, Any], name: str, which: str = "first") -> float:
    """`time_perf_counter` of an attempt event, or NaN."""
    event = select_attempt_event(attempt, name, which)
    if event is None:
        return float("nan")
    try:
        return float(event["time_perf_counter"])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def interp_xy(dataset: Any, t: float) -> Tuple[float, float]:
    return (
        float(np.interp(t, dataset.joystick_task_s, dataset.cursor_x)),
        float(np.interp(t, dataset.joystick_task_s, dataset.cursor_y)),
    )


def reach_speed(dataset: Any, t0: float, t1: float):
    """Cursor speed profile over [t0, t1] resampled to a uniform grid.

    Returns (speed, fs) or None if the window is too short/long.
    """
    if not (np.isfinite(t0) and np.isfinite(t1)) or (t1 - t0) < 0.05 or (t1 - t0) > MAX_REACH_S:
        return None
    mask = (dataset.joystick_task_s >= t0) & (dataset.joystick_task_s <= t1)
    if int(np.count_nonzero(mask)) < 4:
        return None
    n = max(8, int((t1 - t0) * 200.0))
    grid = np.linspace(t0, t1, n)
    xg = np.interp(grid, dataset.joystick_task_s, dataset.cursor_x)
    yg = np.interp(grid, dataset.joystick_task_s, dataset.cursor_y)
    fs = (n - 1) / (t1 - t0)
    speed = np.hypot(np.gradient(xg, 1.0 / fs), np.gradient(yg, 1.0 / fs))
    return speed, fs


def submovement_count(speed: np.ndarray) -> float:
    """Number of speed-profile peaks (submovements); lower = more ballistic."""
    if speed is None or len(speed) < 5:
        return float("nan")
    peak = float(np.max(speed))
    if peak <= 0:
        return float("nan")
    peaks, _ = find_peaks(speed, height=0.1 * peak, prominence=0.1 * peak)
    return float(max(1, len(peaks)))


def sparc(speed: np.ndarray, fs: float, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015).

    Higher (closer to 0) = smoother; more negative = jerkier.
    """
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 4 or not np.any(speed > 0):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    if mag.max() <= 0:
        return float("nan")
    mag = mag / mag.max()
    sel = freq <= fc
    freq, mag = freq[sel], mag[sel]
    inx = np.where(mag >= amp_th)[0]
    if len(inx) < 2:
        return float("nan")
    freq = freq[inx[0]:inx[-1] + 1]
    mag = mag[inx[0]:inx[-1] + 1]
    if freq[-1] - freq[0] <= 0:
        return float("nan")
    df = np.diff(freq) / (freq[-1] - freq[0])
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(mag) ** 2)))


def initial_direction_error_deg(dataset: Any, t_move: float,
                                target_x: float, target_y: float) -> float:
    """Angle (deg) between the first ~150 ms of cursor motion and the target."""
    if not (np.isfinite(t_move) and np.isfinite(target_x) and np.isfinite(target_y)):
        return float("nan")
    x0, y0 = interp_xy(dataset, t_move)
    x1, y1 = interp_xy(dataset, t_move + DIR_WINDOW_S)
    move = np.array([x1 - x0, y1 - y0])
    goal = np.array([target_x - x0, target_y - y0])
    n_move, n_goal = np.linalg.norm(move), np.linalg.norm(goal)
    if n_move < 1e-3 or n_goal < 1e-6:
        return float("nan")
    cos = np.clip(float(np.dot(move, goal) / (n_move * n_goal)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def normalized_path(dataset: Any, t0: float, t1: float, n: int = 20):
    """Reach path resampled onto n evenly-spaced time points, or None."""
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0 or (t1 - t0) > MAX_REACH_S:
        return None
    if int(np.count_nonzero((dataset.joystick_task_s >= t0) & (dataset.joystick_task_s <= t1))) < 3:
        return None
    grid = np.linspace(t0, t1, n)
    return (
        np.interp(grid, dataset.joystick_task_s, dataset.cursor_x),
        np.interp(grid, dataset.joystick_task_s, dataset.cursor_y),
    )


def all_event_perfs(behav_result: Dict[str, Any]) -> List[float]:
    """Every event `time_perf_counter` across all attempts (for session span)."""
    attempts = behav_result.get("attempts") or []
    if not attempts and isinstance(behav_result.get("final_attempt"), dict):
        attempts = [behav_result["final_attempt"]]
    out: List[float] = []
    for att in attempts:
        if not isinstance(att, dict):
            continue
        for ev in att.get("events", []) or []:
            v = ev.get("time_perf_counter")
            if v is not None:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
    return out


def has_idle(behav_result: Dict[str, Any]) -> bool:
    """True if the trial timed out idle (disengagement signal)."""
    if "idle" in str(behav_result.get("final_outcome", "")).lower():
        return True
    attempts = behav_result.get("attempts") or []
    if not attempts and isinstance(behav_result.get("final_attempt"), dict):
        attempts = [behav_result["final_attempt"]]
    for att in attempts:
        if isinstance(att, dict):
            for ev in att.get("events", []) or []:
                if str(ev.get("name", "")).lower() == "ignored_idle_timeout":
                    return True
    return False
