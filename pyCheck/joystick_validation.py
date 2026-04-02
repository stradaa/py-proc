from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import yaml
from scipy.io import loadmat


@dataclass
class TrialSegment:
    trial_index: int
    task_time_s: np.ndarray
    rec_time_ms: np.ndarray
    joystick_x: np.ndarray
    joystick_y: np.ndarray
    cursor_x: np.ndarray
    cursor_y: np.ndarray
    target_x: float
    target_y: float
    target_radius: float
    events_task_s: Dict[str, float]
    event_points: List[Dict[str, Any]]
    attempt: Dict[str, Any]
    alltrial: Dict[str, Any]


EVENT_STYLES = [
    ("JoystickTargetOn", "target_on", "#4c78a8"),
    ("JoystickFirstMovement", "first_move", "#f58518"),
    ("JoystickTargetEntry", "entry_first", "#e45756"),
    ("JoystickTargetEntryFinal", "entry_last", "#ff9da6"),
    ("JoystickTargetExit", "exit_first", "#9467bd"),
    ("JoystickTargetExitFinal", "exit_last", "#c5b0d5"),
    ("JoystickHoldStart", "hold_start_first", "#54a24b"),
    ("JoystickHoldStartFinal", "hold_start_last", "#88d27a"),
    ("JoystickHoldBreak", "hold_break_first", "#b279a2"),
    ("JoystickHoldBreakFinal", "hold_break_last", "#d4a6c8"),
    ("JoystickHoldComplete", "hold_complete", "#2ca02c"),
    ("JoystickReward", "reward", "#17becf"),
    ("End", "trial_end", "#222222"),
]

ATTEMPT_EVENT_COLORS = {
    "target_on": "#4c78a8",
    "first_joystick_movement": "#f58518",
    "target_entry": "#e45756",
    "target_exit": "#9467bd",
    "hold_start": "#54a24b",
    "hold_break": "#b279a2",
    "hold_complete": "#2ca02c",
    "reward_triggered": "#17becf",
    "bonus_reward_triggered": "#72b7b2",
    "success": "#222222",
    "fail": "#d62728",
}


@dataclass
class JoystickDataset:
    root: Path
    day: str
    rec: str
    w_offset_s: float
    w_slope: float
    joystick_task_s: np.ndarray
    joystick_rec_ms: np.ndarray
    joystick_x: np.ndarray
    joystick_y: np.ndarray
    cursor_x: np.ndarray
    cursor_y: np.ndarray
    all_trials: Dict[str, Any]
    behav_results: List[Dict[str, Any]]
    task_configs: List[Dict[str, Any]]

    def rec_ms_to_task_s(self, rec_ms: float) -> float:
        return ((rec_ms / 1e3) - self.w_offset_s) / self.w_slope

    def task_s_to_rec_ms(self, task_s: float) -> float:
        return 1e3 * (self.w_offset_s + self.w_slope * task_s)


def load_joystick_dataset(repo_root: str | Path, day: str, rec: str = "001") -> JoystickDataset:
    root = Path(repo_root).resolve()
    rec_dir = root / day / rec
    bag_mat = rec_dir / f"rec{rec}.bag" / "mat"

    all_trials_raw = loadmat(root / day / "mat" / "AllTrials.mat", simplify_cells=True)["AllTrials"]
    w = np.asarray(loadmat(rec_dir / f"rec{rec}.w_alignment.mat", simplify_cells=True)["w_drift_ros"], dtype=float).ravel()
    joystick = loadmat(bag_mat / "joystick.mat", simplify_cells=True)
    joystick_task_s = (
        np.asarray(joystick["header_stamp_sec"], dtype=float).ravel()
        + np.asarray(joystick["header_stamp_nanosec"], dtype=float).ravel() * 1e-9
    )
    joystick_x = np.asarray(joystick["x"], dtype=float).ravel()
    joystick_y = np.asarray(joystick["y"], dtype=float).ravel()
    joystick_rec_ms = 1e3 * (w[0] + w[1] * joystick_task_s)

    ev = loadmat(rec_dir / f"rec{rec}.ev.mat", simplify_cells=True)
    behav_results = [_parse_jsonish(v) for v in _as_list(ev["behav_results"])]
    task_configs = [_parse_jsonish(v) for v in _as_list(ev["trial_config"])]
    all_trials = _filter_all_trials_for_rec(all_trials_raw, int(rec))

    ref_cfg = task_configs[0]
    cursor_x, cursor_y = reconstruct_cursor(joystick_x, joystick_y, joystick_task_s, ref_cfg)

    return JoystickDataset(
        root=root,
        day=day,
        rec=rec,
        w_offset_s=float(w[0]),
        w_slope=float(w[1]),
        joystick_task_s=joystick_task_s,
        joystick_rec_ms=joystick_rec_ms,
        joystick_x=joystick_x,
        joystick_y=joystick_y,
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        all_trials=all_trials,
        behav_results=behav_results,
        task_configs=task_configs,
    )


def _filter_all_trials_for_rec(all_trials: Dict[str, Any], rec: int) -> Dict[str, Any]:
    rec_values = np.asarray(all_trials["Rec"], dtype=float).ravel()
    keep = rec_values == float(rec)
    filtered: Dict[str, Any] = {}
    for key, value in all_trials.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            filtered[key] = arr
        elif arr.shape[0] == len(rec_values):
            filtered[key] = arr[keep]
        else:
            filtered[key] = value
    return filtered


def reconstruct_cursor(
    joystick_x: np.ndarray,
    joystick_y: np.ndarray,
    task_time_s: np.ndarray,
    task_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    control_mode = str(task_config.get("control_mode", "direct")).lower()
    jx, jy = apply_direction_influence(joystick_x, joystick_y, task_config)

    if control_mode == "direct":
        direct_range = float(task_config.get("direct_range", 0.45))
        cursor_x = 0.5 + jx * direct_range
        cursor_y = 0.5 + jy * direct_range
        return np.clip(cursor_x, 0.0, 1.0), np.clip(cursor_y, 0.0, 1.0)

    cumulative_speed = float(task_config.get("cumulative_speed", 0.70))
    zero_drift_mode = bool(task_config.get("zero_drift_mode", True))
    zero_drift_buffer = float(task_config.get("zero_drift_buffer", 0.05))
    cursor_x = np.empty_like(jx, dtype=float)
    cursor_y = np.empty_like(jy, dtype=float)
    cursor_x[0] = 0.5
    cursor_y[0] = 0.5
    dt = np.diff(task_time_s, prepend=task_time_s[0])

    for i in range(1, len(jx)):
        jx_i = float(jx[i])
        jy_i = float(jy[i])
        if zero_drift_mode and np.hypot(jx_i, jy_i) < zero_drift_buffer:
            jx_i = 0.0
            jy_i = 0.0
        cursor_x[i] = np.clip(cursor_x[i - 1] + jx_i * cumulative_speed * dt[i], 0.0, 1.0)
        cursor_y[i] = np.clip(cursor_y[i - 1] + jy_i * cumulative_speed * dt[i], 0.0, 1.0)

    return cursor_x, cursor_y


def apply_direction_influence(
    joystick_x: np.ndarray,
    joystick_y: np.ndarray,
    task_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    up = float(task_config.get("up_influence_pct", 100.0)) / 100.0
    down = float(task_config.get("down_influence_pct", 100.0)) / 100.0
    left = float(task_config.get("left_influence_pct", 100.0)) / 100.0
    right = float(task_config.get("right_influence_pct", 100.0)) / 100.0

    jx = np.asarray(joystick_x, dtype=float).copy()
    jy = np.asarray(joystick_y, dtype=float).copy()
    jx[jx > 0] *= right
    jx[jx < 0] *= left
    jy[jy > 0] *= up
    jy[jy < 0] *= down
    return jx, jy


def get_trial_segment(
    dataset: JoystickDataset,
    trial_index: int,
    pre_s: float = 0.15,
    post_s: float = 0.25,
) -> TrialSegment:
    alltrial = get_trial_row(dataset.all_trials, trial_index)
    attempt = _get_final_attempt(dataset.behav_results[trial_index])
    events_task_s = extract_trial_event_times_task_s(dataset, alltrial)

    target_on_event = select_attempt_event(attempt, "target_on", "first")
    end_name = "success" if str(attempt.get("outcome", "")).lower() == "success" else "fail"
    end_event = select_attempt_event(attempt, end_name, "last")
    if target_on_event is None or end_event is None:
        raise ValueError(f"Missing target_on or end event for trial {trial_index + 1}")

    t_start = float(target_on_event["time_perf_counter"]) - pre_s
    t_end = float(end_event["time_perf_counter"]) + post_s
    mask = (dataset.joystick_task_s >= t_start) & (dataset.joystick_task_s <= t_end)
    if not np.any(mask):
        raise ValueError(f"No joystick samples found for trial {trial_index + 1}")

    target_position = attempt.get("target_position", {}) if isinstance(attempt, dict) else {}
    target_x = float(target_position.get("x_norm", np.nan))
    target_y = float(target_position.get("y_norm", np.nan))
    target_radius = float(attempt.get("target_radius_ratio", np.nan)) if isinstance(attempt, dict) else np.nan

    return TrialSegment(
        trial_index=trial_index,
        task_time_s=dataset.joystick_task_s[mask],
        rec_time_ms=dataset.joystick_rec_ms[mask],
        joystick_x=dataset.joystick_x[mask],
        joystick_y=dataset.joystick_y[mask],
        cursor_x=dataset.cursor_x[mask],
        cursor_y=dataset.cursor_y[mask],
        target_x=target_x,
        target_y=target_y,
        target_radius=target_radius,
        events_task_s=events_task_s,
        event_points=attempt.get("events", []) if isinstance(attempt, dict) else [],
        attempt=attempt,
        alltrial=alltrial,
    )


def extract_trial_event_times_task_s(dataset: JoystickDataset, alltrial: Dict[str, Any]) -> Dict[str, float]:
    start_on_rec_ms = float(alltrial["StartOn"])
    events: Dict[str, float] = {"StartOn": dataset.rec_ms_to_task_s(start_on_rec_ms)}
    for key in (
        "disStartOn",
        "End",
        "JoystickTargetOn",
        "JoystickFirstMovement",
        "JoystickTargetEntry",
        "JoystickTargetEntryFinal",
        "JoystickTargetExit",
        "JoystickTargetExitFinal",
        "JoystickHoldStart",
        "JoystickHoldStartFinal",
        "JoystickHoldBreak",
        "JoystickHoldBreakFinal",
        "JoystickHoldComplete",
        "JoystickReward",
    ):
        value = float(alltrial.get(key, np.nan))
        events[key] = np.nan if np.isnan(value) else dataset.rec_ms_to_task_s(start_on_rec_ms + value)
    return events


def _timeseries_reference_time(segment: TrialSegment) -> Tuple[float, str]:
    dis_start_on = float(segment.events_task_s.get("disStartOn", np.nan))
    if np.isfinite(dis_start_on):
        return dis_start_on, "disStartOn"

    target_on_event = select_attempt_event(segment.attempt, "target_on", "first")
    if target_on_event is not None:
        return float(target_on_event["time_perf_counter"]), "target_on"

    return float(segment.task_time_s[0]), "trial_start"


def _assign_event_label_lanes(event_times: Sequence[float], min_spacing_s: float = 0.09, max_lanes: int = 4) -> List[int]:
    lane_last_x = np.full(max_lanes, -np.inf, dtype=float)
    lanes: List[int] = []
    for x in event_times:
        lane = 0
        while lane < max_lanes and x - lane_last_x[lane] < min_spacing_s:
            lane += 1
        lane = min(lane, max_lanes - 1)
        lane_last_x[lane] = x
        lanes.append(int(lane))
    return lanes


def _preferred_event_label_lane(event_name: str) -> Optional[int]:
    bottom_names = {"target_on", "target_entry", "hold_complete"}
    top_names = {"first_joystick_movement", "hold_start", "success", "fail"}
    if event_name in bottom_names:
        return 0
    if event_name in top_names:
        return 1
    return None


def _preferred_event_label_x_offset(event_name: str) -> float:
    left_names = {"target_on", "target_entry", "hold_complete"}
    right_names = {"first_joystick_movement", "hold_start", "success", "fail"}
    if event_name in left_names:
        return -0.018
    if event_name in right_names:
        return 0.018
    return 0.0


def validate_trial_alignment(dataset: JoystickDataset, trial_index: int) -> List[Dict[str, Any]]:
    alltrial = get_trial_row(dataset.all_trials, trial_index)
    attempt = _get_final_attempt(dataset.behav_results[trial_index])
    events_task_s = extract_trial_event_times_task_s(dataset, alltrial)
    rows: List[Dict[str, Any]] = []

    pairs = [
        ("JoystickTargetOn", "target_on", "first"),
        ("JoystickFirstMovement", "first_joystick_movement", "first"),
        ("JoystickTargetEntry", "target_entry", "first"),
        ("JoystickTargetEntryFinal", "target_entry", "last"),
        ("JoystickTargetExit", "target_exit", "first"),
        ("JoystickTargetExitFinal", "target_exit", "last"),
        ("JoystickHoldStart", "hold_start", "first"),
        ("JoystickHoldStartFinal", "hold_start", "last"),
        ("JoystickHoldBreak", "hold_break", "first"),
        ("JoystickHoldBreakFinal", "hold_break", "last"),
        ("JoystickHoldComplete", "hold_complete", "first"),
        ("JoystickReward", "reward_triggered", "first"),
        ("End", "success" if int(alltrial["Success"]) == 1 else "fail", "first"),
    ]

    for field_name, event_name, which in pairs:
        trial_event_time = events_task_s.get(field_name, np.nan)
        task_event = select_attempt_event(attempt, event_name, which)
        if task_event is None or np.isnan(trial_event_time):
            continue

        row = {
            "trial": trial_index + 1,
            "field_name": field_name,
            "event_name": event_name,
            "which": which,
            "trial_time_s": float(trial_event_time),
            "behav_time_s": float(task_event["time_perf_counter"]),
            "time_error_ms": 1e3 * float(trial_event_time - float(task_event["time_perf_counter"])),
        }

        if "cursor_x" in task_event and "cursor_y" in task_event:
            sample_index = nearest_index(dataset.joystick_task_s, trial_event_time)
            row["nearest_sample_dt_ms"] = 1e3 * float(dataset.joystick_task_s[sample_index] - trial_event_time)
            row["reconstructed_cursor_x"] = float(np.interp(trial_event_time, dataset.joystick_task_s, dataset.cursor_x))
            row["reconstructed_cursor_y"] = float(np.interp(trial_event_time, dataset.joystick_task_s, dataset.cursor_y))
            row["logged_cursor_x"] = float(task_event["cursor_x"])
            row["logged_cursor_y"] = float(task_event["cursor_y"])
            row["cursor_error"] = float(
                np.hypot(
                    row["reconstructed_cursor_x"] - row["logged_cursor_x"],
                    row["reconstructed_cursor_y"] - row["logged_cursor_y"],
                )
            )
        rows.append(row)

    return rows


def validate_day_alignment(dataset: JoystickDataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_trials = len(np.asarray(dataset.all_trials["Trial"]).ravel())
    for i in range(n_trials):
        rows.extend(validate_trial_alignment(dataset, i))
    return rows


def sweep_cursor_shift_ms(
    dataset: JoystickDataset,
    rows: Sequence[Dict[str, Any]],
    shift_ms_values: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    if shift_ms_values is None:
        shift_ms_values = np.arange(-600.0, 600.1, 5.0)

    out: List[Dict[str, float]] = []
    cursor_rows = [r for r in rows if "logged_cursor_x" in r]
    for shift_ms in shift_ms_values:
        errors = []
        for row in cursor_rows:
            t_shifted = row["trial_time_s"] + shift_ms * 1e-3
            cx = float(np.interp(t_shifted, dataset.joystick_task_s, dataset.cursor_x))
            cy = float(np.interp(t_shifted, dataset.joystick_task_s, dataset.cursor_y))
            errors.append(np.hypot(cx - row["logged_cursor_x"], cy - row["logged_cursor_y"]))
        errors_np = np.asarray(errors, dtype=float)
        out.append(
            {
                "shift_ms": float(shift_ms),
                "cursor_error_mean": float(np.nanmean(errors_np)),
                "cursor_error_median": float(np.nanmedian(errors_np)),
            }
        )
    return out


def plot_trial_trajectory(segment: TrialSegment, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(segment.cursor_x, segment.cursor_y, color="#1f77b4", linewidth=2.0, alpha=0.9)

    target = plt.Circle((segment.target_x, segment.target_y), segment.target_radius, color="#4caf50", alpha=0.2)
    ax.add_patch(target)
    target_on_event = select_attempt_event(segment.attempt, "target_on", "first")
    end_name = "success" if str(segment.attempt.get("outcome", "")).lower() == "success" else "fail"
    end_event = select_attempt_event(segment.attempt, end_name, "last")
    target_on_time = float(target_on_event["time_perf_counter"]) if target_on_event is not None else segment.task_time_s[0]
    end_time = float(end_event["time_perf_counter"]) if end_event is not None else segment.task_time_s[-1]
    target_on_idx = nearest_index(segment.task_time_s, target_on_time)
    end_idx = nearest_index(segment.task_time_s, end_time)
    ax.scatter(
        segment.cursor_x[target_on_idx],
        segment.cursor_y[target_on_idx],
        color="black",
        s=60,
        label="cursor @ target_on",
        zorder=6,
    )
    ax.scatter(
        segment.cursor_x[end_idx],
        segment.cursor_y[end_idx],
        color="red",
        s=60,
        label="cursor @ trial_end",
        zorder=6,
    )

    for labeled_event in iter_labeled_attempt_events(segment.attempt):
        idx = nearest_index(segment.task_time_s, labeled_event["time_perf_counter"])
        ax.scatter(
            segment.cursor_x[idx],
            segment.cursor_y[idx],
            s=28,
            color=labeled_event["color"],
            edgecolor="black",
            linewidth=0.35,
            zorder=5,
        )
        if labeled_event["name"] not in {"target_on", "success", "fail"}:
            ax.text(
                float(segment.cursor_x[idx]) + 0.01,
                float(segment.cursor_y[idx]) + 0.01,
                labeled_event["label"],
                fontsize=8,
                color=labeled_event["color"],
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Trial {segment.trial_index + 1}: reconstructed cursor trajectory")
    ax.set_xlabel("cursor_x (task-region normalized)")
    ax.set_ylabel("cursor_y (task-region normalized)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trial_timeseries(segment: TrialSegment, out_path: str | Path) -> None:
    reference_time, reference_label = _timeseries_reference_time(segment)
    t_rel = segment.task_time_s - reference_time
    dis_start_on_time = float(segment.events_task_s.get("disStartOn", np.nan))

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    axes[0].plot(t_rel, segment.joystick_x, label="joystick_x", color="#1f77b4")
    axes[0].plot(t_rel, segment.joystick_y, label="joystick_y", color="#ff7f0e")
    axes[0].legend(loc="best")
    axes[0].set_ylabel("joystick")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t_rel, segment.cursor_x, label="cursor_x", color="#2ca02c")
    axes[1].plot(t_rel, segment.cursor_y, label="cursor_y", color="#d62728")
    axes[1].axhline(segment.target_x, color="#2ca02c", linestyle="--", alpha=0.5, label="target_x")
    axes[1].axhline(segment.target_y, color="#d62728", linestyle="--", alpha=0.5, label="target_y")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("cursor")
    axes[1].set_xlabel(f"time from {reference_label} (task seconds)")
    axes[1].grid(True, alpha=0.2)

    if np.isfinite(dis_start_on_time):
        dis_start_on_x = dis_start_on_time - reference_time
        for ax in axes:
            ax.axvline(dis_start_on_x, color="#111111", alpha=0.9, linestyle=":", linewidth=1.6, zorder=3)
        axes[0].text(
            dis_start_on_x,
            0.06,
            "disStartOn",
            rotation=90,
            color="#111111",
            fontsize=8,
            ha="center",
            va="bottom",
            transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.5},
        )

    text_transform = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    labeled_events = iter_labeled_attempt_events(segment.attempt)
    event_x = [float(labeled_event["time_perf_counter"]) - reference_time for labeled_event in labeled_events]
    lanes = _assign_event_label_lanes(event_x)
    lane_y = [0.72, 0.96, 0.56, 0.84]
    for labeled_event, x, lane in zip(labeled_events, event_x, lanes):
        preferred_lane = _preferred_event_label_lane(labeled_event["name"])
        if preferred_lane is not None:
            lane = preferred_lane
        x_text = x + _preferred_event_label_x_offset(labeled_event["name"])
        for ax in axes:
            ax.axvline(x, color=labeled_event["color"], alpha=0.7, linestyle="--", linewidth=1.2)
        axes[0].text(
            x_text,
            lane_y[lane],
            labeled_event["label"],
            rotation=90,
            color=labeled_event["color"],
            fontsize=8,
            ha="center",
            va="bottom",
            transform=text_transform,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.6},
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_alignment_summary(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> None:
    time_errors = np.array([r["time_error_ms"] for r in rows], dtype=float)
    cursor_errors = np.array([r["cursor_error"] for r in rows if "cursor_error" in r], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(time_errors, bins=30, color="#4c78a8")
    axes[0].set_title("AllTrials vs behav_result event time error")
    axes[0].set_xlabel("error (ms)")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(cursor_errors, bins=30, color="#f58518")
    axes[1].set_title("reconstructed vs logged cursor error")
    axes[1].set_xlabel("distance in normalized units")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_shift_sweep(shift_rows: Sequence[Dict[str, float]], out_path: str | Path) -> None:
    shift_ms = np.array([r["shift_ms"] for r in shift_rows], dtype=float)
    err_mean = np.array([r["cursor_error_mean"] for r in shift_rows], dtype=float)
    err_median = np.array([r["cursor_error_median"] for r in shift_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(shift_ms, err_mean, label="mean cursor error", color="#4c78a8")
    ax.plot(shift_ms, err_median, label="median cursor error", color="#f58518")
    ax.set_xlabel("constant shift applied to processed event time (ms)")
    ax.set_ylabel("cursor error (normalized units)")
    ax.set_title("Joystick timestamp shift sweep")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_validation_report(
    repo_root: str | Path,
    day: str,
    rec: str = "001",
    out_dir: Optional[str | Path] = None,
    sample_trials: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    dataset = load_joystick_dataset(repo_root, day, rec)
    out_path = Path(out_dir) if out_dir is not None else Path(repo_root) / "pyCheck" / "output" / day
    out_path.mkdir(parents=True, exist_ok=True)

    rows = validate_day_alignment(dataset)
    plot_alignment_summary(rows, out_path / "alignment_summary.png")
    shift_rows = sweep_cursor_shift_ms(dataset, rows)
    plot_shift_sweep(shift_rows, out_path / "cursor_shift_sweep.png")

    if sample_trials is None:
        break_trials = sorted({r["trial"] for r in rows if r["event_name"] == "hold_break"})
        sample_trials = [1, 2, 4, break_trials[0] if break_trials else 1, len(np.asarray(dataset.all_trials["Trial"]).ravel())]

    chosen = []
    for trial in sample_trials:
        if trial < 1:
            continue
        idx = int(trial) - 1
        if idx in chosen:
            continue
        chosen.append(idx)
        segment = get_trial_segment(dataset, idx)
        plot_trial_trajectory(segment, out_path / f"trial_{trial:03d}_trajectory.png")
        plot_trial_timeseries(segment, out_path / f"trial_{trial:03d}_timeseries.png")

    summary = summarize_validation_rows(rows)
    best_shift = min(shift_rows, key=lambda r: r["cursor_error_median"])
    summary["best_constant_shift_ms_for_cursor"] = float(best_shift["shift_ms"])
    summary["best_shift_cursor_error_median"] = float(best_shift["cursor_error_median"])
    summary["output_dir"] = str(out_path)
    summary["sample_trials"] = [i + 1 for i in chosen]
    with open(out_path / "validation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_trial_tokens(trial_tokens: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not trial_tokens:
        return None
    out: List[int] = []
    for token in trial_tokens:
        if "," in token:
            parts = [p.strip() for p in token.split(",") if p.strip()]
            parsed = parse_trial_tokens(parts)
            if parsed:
                out.extend(parsed)
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            step = 1 if end_i >= start_i else -1
            out.extend(list(range(start_i, end_i + step, step)))
        else:
            out.append(int(token))
    deduped = []
    for trial in out:
        if trial not in deduped:
            deduped.append(trial)
    return deduped


def render_trial_replay_video(
    repo_root: str | Path,
    day: str,
    rec: str,
    trial_numbers: Sequence[int],
    out_path: str | Path,
    fps: int = 30,
    playback_speed: float = 1.0,
    figsize: Tuple[float, float] = (7.5, 7.5),
) -> Path:
    dataset = load_joystick_dataset(repo_root, day, rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trial_indices = [int(t) - 1 for t in trial_numbers]
    fig, ax = plt.subplots(figsize=figsize)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000)

    with writer.saving(fig, str(out_path), dpi=140):
        for trial_index in trial_indices:
            segment = get_trial_segment(dataset, trial_index, pre_s=0.2, post_s=0.3)
            render_trial_segment(segment, fig, ax, writer, fps=fps, playback_speed=playback_speed)

    plt.close(fig)
    return out_path


def render_trial_segment(
    segment: TrialSegment,
    fig: plt.Figure,
    ax: plt.Axes,
    writer: FFMpegWriter,
    fps: int,
    playback_speed: float,
) -> None:
    playback_speed = max(1e-3, float(playback_speed))
    dt_frame = playback_speed / float(fps)
    target_on_event = select_attempt_event(segment.attempt, "target_on", "first")
    end_name = "success" if str(segment.attempt.get("outcome", "")).lower() == "success" else "fail"
    end_event = select_attempt_event(segment.attempt, end_name, "last")
    target_on_time = float(target_on_event["time_perf_counter"]) if target_on_event is not None else segment.task_time_s[0]
    end_time = float(end_event["time_perf_counter"]) if end_event is not None else segment.task_time_s[-1]
    frame_times = np.arange(segment.task_time_s[0], segment.task_time_s[-1] + dt_frame, dt_frame)
    labeled_events = iter_labeled_attempt_events(segment.attempt)

    for t_now in frame_times:
        ax.clear()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("cursor_x (task-region normalized)")
        ax.set_ylabel("cursor_y (task-region normalized)")
        ax.set_title(f"Trial {segment.trial_index + 1}")

        if t_now >= target_on_time:
            target = plt.Circle((segment.target_x, segment.target_y), segment.target_radius, color="#4caf50", alpha=0.2)
            ax.add_patch(target)

        visible_mask = segment.task_time_s <= t_now
        if np.count_nonzero(visible_mask) >= 2:
            ax.plot(segment.cursor_x[visible_mask], segment.cursor_y[visible_mask], color="#1f77b4", linewidth=2.2)

        cx = float(np.interp(t_now, segment.task_time_s, segment.cursor_x))
        cy = float(np.interp(t_now, segment.task_time_s, segment.cursor_y))
        ax.scatter([cx], [cy], s=110, color="#d62728", edgecolor="black", linewidth=0.8, zorder=5)

        shown_events = [ev for ev in labeled_events if ev["time_perf_counter"] <= t_now]
        for ev in shown_events:
            ex = float(np.interp(ev["time_perf_counter"], segment.task_time_s, segment.cursor_x))
            ey = float(np.interp(ev["time_perf_counter"], segment.task_time_s, segment.cursor_y))
            ax.scatter([ex], [ey], s=28, color=ev["color"], edgecolor="black", linewidth=0.4, zorder=4)

        current_event = shown_events[-1]["label"] if shown_events else "pre_target"
        status = (
            f"time from target_on: {t_now - target_on_time:+.3f} s\n"
            f"trial outcome: {segment.attempt.get('outcome', '')}\n"
            f"current event: {current_event}"
        )
        ax.text(
            0.02,
            0.98,
            status,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        if t_now >= end_time:
            ax.text(
                0.98,
                0.98,
                "trial end",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=11,
                color="#222222",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

        fig.tight_layout()
        writer.grab_frame()


def summarize_validation_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    time_errors = np.array([r["time_error_ms"] for r in rows], dtype=float)
    cursor_errors = np.array([r["cursor_error"] for r in rows if "cursor_error" in r], dtype=float)
    nearest_dt = np.array([r["nearest_sample_dt_ms"] for r in rows if "nearest_sample_dt_ms" in r], dtype=float)
    return {
        "n_events_checked": int(len(rows)),
        "n_cursor_events_checked": int(len(cursor_errors)),
        "time_error_ms_mean": float(np.nanmean(time_errors)) if len(time_errors) else np.nan,
        "time_error_ms_median": float(np.nanmedian(time_errors)) if len(time_errors) else np.nan,
        "time_error_ms_max_abs": float(np.nanmax(np.abs(time_errors))) if len(time_errors) else np.nan,
        "nearest_sample_dt_ms_median_abs": float(np.nanmedian(np.abs(nearest_dt))) if len(nearest_dt) else np.nan,
        "nearest_sample_dt_ms_max_abs": float(np.nanmax(np.abs(nearest_dt))) if len(nearest_dt) else np.nan,
        "cursor_error_mean": float(np.nanmean(cursor_errors)) if len(cursor_errors) else np.nan,
        "cursor_error_median": float(np.nanmedian(cursor_errors)) if len(cursor_errors) else np.nan,
        "cursor_error_max": float(np.nanmax(cursor_errors)) if len(cursor_errors) else np.nan,
    }


def get_trial_row(all_trials: Dict[str, Any], trial_index: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in all_trials.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                row[key] = value[trial_index]
            elif value.ndim >= 2 and value.shape[0] > trial_index:
                row[key] = value[trial_index]
            else:
                row[key] = value
        else:
            row[key] = value
    return row


def select_attempt_event(attempt: Dict[str, Any], event_name: str, which: str) -> Optional[Dict[str, Any]]:
    events = [ev for ev in attempt.get("events", []) if str(ev.get("name", "")).lower() == event_name.lower()]
    if not events:
        return None
    return events[0] if which == "first" else events[-1]


def iter_labeled_attempt_events(attempt: Dict[str, Any]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    labeled = []
    for event in attempt.get("events", []):
        name = str(event.get("name", ""))
        counts[name] = counts.get(name, 0) + 1
        short_name = name.replace("first_joystick_movement", "first_move").replace("reward_triggered", "reward")
        label = short_name if counts[name] == 1 else f"{short_name}_{counts[name]}"
        labeled.append(
            {
                "name": name,
                "label": label,
                "time_perf_counter": float(event["time_perf_counter"]),
                "color": ATTEMPT_EVENT_COLORS.get(name, "#666666"),
            }
        )
    return labeled


def nearest_index(times: np.ndarray, target_time: float) -> int:
    return int(np.argmin(np.abs(times - target_time)))


def _get_final_attempt(behav_result: Dict[str, Any]) -> Dict[str, Any]:
    final_attempt = behav_result.get("final_attempt")
    if isinstance(final_attempt, dict):
        return final_attempt
    attempts = behav_result.get("attempts", [])
    return attempts[-1] if attempts else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _parse_jsonish(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            parsed = yaml.safe_load(value)
            return parsed if isinstance(parsed, dict) else {}
    return {}
