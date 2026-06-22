from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    from .joystick_validation import _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event
except ImportError:
    from joystick_validation import _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event


def _load_all_trials(repo_root: Path, day: str) -> Dict[str, np.ndarray]:
    all_trials = loadmat(repo_root / day / "mat" / "AllTrials.mat", simplify_cells=True)["AllTrials"]
    return {key: np.asarray(value) for key, value in all_trials.items()}


def _flat(data: Dict[str, np.ndarray], key: str, dtype=float) -> np.ndarray:
    value = np.asarray(data[key]).ravel()
    if dtype is object:
        return value.astype(object)
    return value.astype(dtype)


def _parse_int_token(token: str) -> List[int]:
    token = token.strip()
    if not token:
        return []
    if "-" in token:
        start_s, end_s = token.split("-", 1)
        start_i = int(start_s)
        end_i = int(end_s)
        step = 1 if end_i >= start_i else -1
        return list(range(start_i, end_i + step, step))
    return [int(token)]


def _parse_rec_spec(spec: str) -> List[int]:
    recs: List[int] = []
    for token in spec.split(","):
        recs.extend(_parse_int_token(token))
    return sorted(set(recs))


def _discover_recs(repo_root: Path, day: str) -> List[int]:
    data = _load_all_trials(repo_root, day)
    return sorted(np.unique(_flat(data, "Rec", int)).tolist())


def _normalize_day_rec_selection(spec: str | Sequence[str]) -> str:
    if isinstance(spec, str):
        return spec.strip()
    return " ".join(str(part).strip() for part in spec if str(part).strip()).strip()


def _parse_day_rec_selection(spec: str | Sequence[str], repo_root: Path) -> Tuple[str, List[int]]:
    text = _normalize_day_rec_selection(spec)
    if not text:
        raise ValueError("Empty --day-recs selection.")
    if ":" in text:
        day, rec_spec = text.split(":", 1)
        day = day.strip()
        rec_spec = rec_spec.strip()
        recs = _parse_rec_spec(rec_spec) if rec_spec else _discover_recs(repo_root, day)
    else:
        day = text
        recs = _discover_recs(repo_root, day)
    if not recs:
        raise ValueError(f"No recordings selected for day {day}.")
    return day, recs


def _default_out_dir(repo_root: Path) -> Path:
    return repo_root / "claude" / "figures" / "cross_day_beh"


def _is_requested_task(task_value: object, requested: set[str]) -> bool:
    if not requested:
        return True
    return str(task_value).strip().lower() in requested


def _first_event_after(events: Sequence[Dict[str, Any]], event_name: str, time_s: float) -> Optional[Dict[str, Any]]:
    matches = [
        ev for ev in events
        if str(ev.get("name", "")).lower() == event_name.lower() and float(ev.get("time_perf_counter", np.nan)) >= time_s
    ]
    if not matches:
        return None
    return min(matches, key=lambda ev: float(ev["time_perf_counter"]))


def _compute_path_efficiency(
    dataset: Any,
    target_on_s: float,
    hold_complete_s: float,
    target_x: float,
    target_y: float,
) -> float:
    if not (np.isfinite(target_on_s) and np.isfinite(hold_complete_s) and hold_complete_s > target_on_s):
        return float("nan")

    mask = (dataset.joystick_task_s >= target_on_s) & (dataset.joystick_task_s <= hold_complete_s)
    xs = np.asarray(dataset.cursor_x[mask], dtype=float)
    ys = np.asarray(dataset.cursor_y[mask], dtype=float)
    ts = np.asarray(dataset.joystick_task_s[mask], dtype=float)

    start_x = float(np.interp(target_on_s, dataset.joystick_task_s, dataset.cursor_x))
    start_y = float(np.interp(target_on_s, dataset.joystick_task_s, dataset.cursor_y))
    end_x = float(np.interp(hold_complete_s, dataset.joystick_task_s, dataset.cursor_x))
    end_y = float(np.interp(hold_complete_s, dataset.joystick_task_s, dataset.cursor_y))

    if xs.size == 0:
        xs = np.array([start_x, end_x], dtype=float)
        ys = np.array([start_y, end_y], dtype=float)
        ts = np.array([target_on_s, hold_complete_s], dtype=float)
    else:
        if ts[0] > target_on_s:
            xs = np.insert(xs, 0, start_x)
            ys = np.insert(ys, 0, start_y)
        else:
            xs[0] = start_x
            ys[0] = start_y
        if ts[-1] < hold_complete_s:
            xs = np.append(xs, end_x)
            ys = np.append(ys, end_y)
        else:
            xs[-1] = end_x
            ys[-1] = end_y

    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    straight_line = float(np.hypot(target_x - start_x, target_y - start_y))
    if not np.isfinite(straight_line) or straight_line <= 1e-6:
        return float("nan")
    return path_length / straight_line


def _build_day_summary(
    repo_root: Path,
    day: str,
    recs: Sequence[int],
    requested_task_types: set[str],
) -> Dict[str, Any]:
    successful_durations_s: List[float] = []
    successful_target_radii: List[float] = []
    hold_break_flags: List[float] = []
    first_entry_success_flags: List[float] = []
    path_efficiencies: List[float] = []
    successful_trials = 0
    included_trials = 0

    for rec in recs:
        rec_str = f"{int(rec):03d}"
        dataset = load_joystick_dataset(repo_root, day, rec_str)
        n_trials = len(np.asarray(dataset.all_trials["Trial"]).ravel())

        for trial_index in range(n_trials):
            alltrial = get_trial_row(dataset.all_trials, trial_index)
            if not _is_requested_task(alltrial.get("PyTaskType", ""), requested_task_types):
                continue

            included_trials += 1
            success = int(round(float(alltrial.get("Success", 0)))) == 1
            if success:
                successful_trials += 1

            attempt = _get_final_attempt(dataset.behav_results[trial_index])
            events = attempt.get("events", []) if isinstance(attempt, dict) else []
            target_on_event = select_attempt_event(attempt, "target_on", "first")
            entry_event = select_attempt_event(attempt, "target_entry", "first")
            hold_complete_event = select_attempt_event(attempt, "hold_complete", "first")

            if entry_event is not None:
                entry_time = float(entry_event["time_perf_counter"])
                hold_break_event = _first_event_after(events, "hold_break", entry_time)
                hold_break_before_complete = (
                    hold_break_event is not None
                    and (
                        hold_complete_event is None
                        or float(hold_break_event["time_perf_counter"]) < float(hold_complete_event["time_perf_counter"])
                    )
                )
                hold_break_flags.append(float(hold_break_before_complete))

                exit_event = _first_event_after(events, "target_exit", entry_time)
                first_entry_success = (
                    success
                    and hold_complete_event is not None
                    and (exit_event is None or float(exit_event["time_perf_counter"]) >= float(hold_complete_event["time_perf_counter"]))
                    and (hold_break_event is None or float(hold_break_event["time_perf_counter"]) >= float(hold_complete_event["time_perf_counter"]))
                )
                first_entry_success_flags.append(float(first_entry_success))

            if not success or target_on_event is None or hold_complete_event is None:
                continue

            target_on_s = float(target_on_event["time_perf_counter"])
            hold_complete_s = float(hold_complete_event["time_perf_counter"])
            if np.isfinite(target_on_s) and np.isfinite(hold_complete_s) and hold_complete_s > target_on_s:
                successful_durations_s.append(hold_complete_s - target_on_s)

            target_radius = float(attempt.get("target_radius_ratio", np.nan))
            if np.isfinite(target_radius):
                successful_target_radii.append(target_radius)

            target_position = attempt.get("target_position", {}) if isinstance(attempt, dict) else {}
            target_x = float(target_position.get("x_norm", np.nan))
            target_y = float(target_position.get("y_norm", np.nan))
            efficiency = _compute_path_efficiency(dataset, target_on_s, hold_complete_s, target_x, target_y)
            if np.isfinite(efficiency):
                path_efficiencies.append(efficiency)

    return {
        "day": day,
        "selected_recs": [int(rec) for rec in recs],
        "n_included_trials": int(included_trials),
        "n_successful_trials": int(successful_trials),
        "n_trials_with_entry": int(len(first_entry_success_flags)),
        "n_successful_durations": int(len(successful_durations_s)),
        "n_path_efficiency_trials": int(len(path_efficiencies)),
        "median_successful_trial_duration_s": float(np.nanmedian(successful_durations_s)) if successful_durations_s else float("nan"),
        "mean_successful_target_radius": float(np.nanmean(successful_target_radii)) if successful_target_radii else float("nan"),
        "hold_break_rate_after_entry": float(np.nanmean(hold_break_flags)) if hold_break_flags else float("nan"),
        "first_entry_success_rate": float(np.nanmean(first_entry_success_flags)) if first_entry_success_flags else float("nan"),
        "median_path_efficiency": float(np.nanmedian(path_efficiencies)) if path_efficiencies else float("nan"),
    }


def _write_summary_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "day",
        "selected_recs",
        "n_included_trials",
        "n_successful_trials",
        "n_trials_with_entry",
        "n_successful_durations",
        "n_path_efficiency_trials",
        "median_successful_trial_duration_s",
        "mean_successful_target_radius",
        "hold_break_rate_after_entry",
        "first_entry_success_rate",
        "median_path_efficiency",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["selected_recs"] = ",".join(f"{int(rec):03d}" for rec in row["selected_recs"])
            writer.writerow(formatted)


def _load_summary_csv(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    with open(out_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, Any]] = []
        for raw in reader:
            rows.append(
                {
                    "day": str(raw.get("day", "")).strip(),
                    "selected_recs": [int(token) for token in str(raw.get("selected_recs", "")).split(",") if token.strip()],
                    "n_included_trials": int(float(raw.get("n_included_trials", 0) or 0)),
                    "n_successful_trials": int(float(raw.get("n_successful_trials", 0) or 0)),
                    "n_trials_with_entry": int(float(raw.get("n_trials_with_entry", 0) or 0)),
                    "n_successful_durations": int(float(raw.get("n_successful_durations", 0) or 0)),
                    "n_path_efficiency_trials": int(float(raw.get("n_path_efficiency_trials", 0) or 0)),
                    "median_successful_trial_duration_s": float(raw.get("median_successful_trial_duration_s", "nan")),
                    "mean_successful_target_radius": float(raw.get("mean_successful_target_radius", "nan")),
                    "hold_break_rate_after_entry": float(raw.get("hold_break_rate_after_entry", "nan")),
                    "first_entry_success_rate": float(raw.get("first_entry_success_rate", "nan")),
                    "median_path_efficiency": float(raw.get("median_path_efficiency", "nan")),
                }
            )
        return rows


def _merge_summary_rows(existing_rows: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {str(row["day"]): dict(row) for row in existing_rows if str(row.get("day", "")).strip()}
    for row in new_rows:
        merged[str(row["day"])] = dict(row)
    return [merged[day] for day in sorted(merged)]


def _plot_cross_day_metrics(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)

    duration = np.asarray([row["median_successful_trial_duration_s"] for row in rows], dtype=float)
    radius = np.asarray([row["mean_successful_target_radius"] for row in rows], dtype=float)
    hold_break_rate = 100.0 * np.asarray([row["hold_break_rate_after_entry"] for row in rows], dtype=float)
    first_entry_success = 100.0 * np.asarray([row["first_entry_success_rate"] for row in rows], dtype=float)
    path_efficiency = np.asarray([row["median_path_efficiency"] for row in rows], dtype=float)
    successful_trials = np.asarray([row["n_successful_trials"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax_duration, ax_rates, ax_path, ax_successful = axes.ravel()

    finite_radius = np.isfinite(radius)
    if np.any(np.isfinite(duration)):
        ax_duration.plot(x, duration, color="#1f4e79", linewidth=1.8, alpha=0.7, zorder=1)
        scatter = ax_duration.scatter(
            x,
            duration,
            c=radius if np.any(finite_radius) else np.full_like(duration, 0.0),
            cmap="viridis",
            s=90,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        if np.any(finite_radius):
            cbar = fig.colorbar(scatter, ax=ax_duration, fraction=0.046, pad=0.04)
            cbar.set_label("Mean successful target radius")
    ax_duration.set_ylabel("Seconds")
    ax_duration.set_title("Median successful trial duration by day")
    ax_duration.grid(alpha=0.2)

    ax_rates.plot(x, hold_break_rate, marker="o", color="#b279a2", linewidth=2, label="Hold-break rate after first entry")
    ax_rates.plot(x, first_entry_success, marker="o", color="#2ca25f", linewidth=2, label="First-entry success rate")
    ax_rates.set_ylabel("Rate (%)")
    ax_rates.set_ylim(0, 105)
    ax_rates.set_title("Post-entry stability and first-entry success")
    ax_rates.grid(alpha=0.2)
    ax_rates.legend(loc="best", frameon=False)

    ax_path.plot(x, path_efficiency, marker="o", color="#e67e22", linewidth=2)
    ax_path.set_ylabel("Path / straight-line")
    ax_path.set_title("Median path efficiency by day")
    ax_path.grid(alpha=0.2)

    ax_successful.bar(x, successful_trials, color="#4c78a8", alpha=0.85, width=0.7)
    ax_successful.set_ylabel("Successful trials")
    ax_successful.set_title("Successful trials by day")
    ax_successful.grid(axis="y", alpha=0.2)

    for ax in [ax_duration, ax_rates, ax_path, ax_successful]:
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=25, ha="right")
        ax.set_xlabel("Day")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_cross_day_plots(
    repo_root: str | Path,
    selections: Sequence[str | Sequence[str]],
    out_dir: str | Path | None = None,
    task_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir is not None else _default_out_dir(repo_root_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    requested_task_types = {str(task).strip().lower() for task in (task_types or []) if str(task).strip()}

    parsed = [_parse_day_rec_selection(spec, repo_root_path) for spec in selections]
    rows = [
        _build_day_summary(repo_root_path, day=day, recs=recs, requested_task_types=requested_task_types)
        for day, recs in parsed
    ]
    rows.sort(key=lambda row: row["day"])

    figure_path = out_dir_path / "cross_day_metrics.png"
    metrics_json_path = out_dir_path / "cross_day_summary_metrics.json"
    metrics_csv_path = out_dir_path / "cross_day_summary_metrics.csv"

    _plot_cross_day_metrics(rows, figure_path)
    merged_rows = _merge_summary_rows(_load_summary_csv(metrics_csv_path), rows)
    _write_summary_csv(merged_rows, metrics_csv_path)
    with open(metrics_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "selections": [{"day": row["day"], "selected_recs": row["selected_recs"]} for row in rows],
                "task_types": sorted(requested_task_types),
                "figure_path": str(figure_path),
                "days": rows,
                "csv_days": merged_rows,
            },
            fh,
            indent=2,
        )

    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics JSON: {metrics_json_path}")
    print(f"Saved metrics CSV: {metrics_csv_path}")

    result = {
        "figure_path": str(figure_path),
        "metrics_json_path": str(metrics_json_path),
        "metrics_csv_path": str(metrics_csv_path),
        "days": rows,
        "csv_days": merged_rows,
    }

    # Companion learning-kinematics figure (separate file, so the headline figure
    # above stays uncluttered). Guarded: never let it break the core summary.
    try:
        from .cross_day_kinematics import generate_cross_day_kinematics

        kin = generate_cross_day_kinematics(
            repo_root_path, selections, out_dir_path, task_types
        )
        result["kinematics_figure_path"] = kin["figure_path"]
        result["kinematics_metrics_json_path"] = kin["metrics_json_path"]
        result["kinematics_metrics_csv_path"] = kin["metrics_csv_path"]
    except Exception as exc:  # noqa: BLE001
        print(f"cross_day_kinematics: skipped ({exc})")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-day joystick learning plots from selected days and recordings")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--day-recs",
        action="append",
        nargs="+",
        required=True,
        help="Repeatable day selection like 260408:1-3,5 or 260409 for all recs on that day.",
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--task-types", nargs="*", default=None)
    args = parser.parse_args()

    generate_cross_day_plots(
        repo_root=args.repo_root,
        selections=args.day_recs,
        out_dir=args.out_dir,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
