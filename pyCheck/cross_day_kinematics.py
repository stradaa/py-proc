"""Cross-day joystick *learning* metrics, written to a separate figure.

This complements `cross_day_plots.py` (throughput / duration / path efficiency /
post-entry stability) with the finer-grained motor-learning signals — movement
initiation, aiming, smoothness, variability, engagement, and direction-resolved
success — that distinguish "planning got better" from "execution got smoother"
from "behaviour got more consistent".

Everything is computed on the **task/behave clock**, exactly like the existing
cross-day metrics: event times come from each attempt's `time_perf_counter`
(already a behave-clock value on this single-machine rig) and the cursor is the
reconstruction from `joystick.mat`. None of it touches the recorder-clock
display path that drifts on recent days. Output is a standalone
`cross_day_kinematics.png` plus a JSON/CSV, so the original figure stays
uncluttered.

Per-trial primitives live in `kinematics_features.py`; the figure is rendered by
`cross_day_kinematics_plots.py`. This module does per-day aggregation and
orchestration.

Metrics per day (medians/means across the selected recs' trials):
  * Movement-time decomposition: reaction (target_on→first_move), transport
    (first_move→first entry), acquisition (entry→hold_complete).
  * Reaction-time variability (CV).
  * Initial reach-direction error (deg).
  * Submovement count and SPARC smoothness.
  * Trajectory variability & endpoint scatter (cross-trial spread, per target).
  * Engagement: attempts/trial, idle-timeout rate, trials attempted per minute.
  * Direction-resolved success-rate heatmap (day × target).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .cross_day_plots import _parse_day_rec_selection, _default_out_dir, _is_requested_task
    from .cross_day_kinematics_plots import plot_cross_day_kinematics
    from .joystick_validation import (
        _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event,
    )
    from . import kinematics_features as kf
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_plots import _parse_day_rec_selection, _default_out_dir, _is_requested_task
    from cross_day_kinematics_plots import plot_cross_day_kinematics
    from joystick_validation import (
        _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event,
    )
    import kinematics_features as kf


# Scalar metric columns (the direction heatmap lives only in the JSON).
_KIN_FIELDS = [
    "day", "selected_recs", "n_included_trials",
    "median_reaction_time_ms", "median_transport_time_ms",
    "median_acquisition_time_ms", "reaction_time_cv",
    "median_initial_direction_error_deg", "median_submovement_count",
    "median_sparc", "trajectory_variability", "endpoint_scatter",
    "mean_attempts_per_trial", "idle_timeout_rate", "trials_per_min",
]


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def build_day_kinematics(
    repo_root: Path,
    day: str,
    recs: Sequence[int],
    requested_task_types: set[str],
) -> Dict[str, Any]:
    rt: List[float] = []
    transport: List[float] = []
    acquisition: List[float] = []
    dir_err: List[float] = []
    submv: List[float] = []
    sparc: List[float] = []
    attempts_per_trial: List[float] = []
    idle_flags: List[float] = []

    paths_by_dir: Dict[int, List[tuple]] = {}
    endpoints_by_dir: Dict[int, List[tuple]] = {}
    success_by_dir: Dict[int, List[float]] = {}
    xy_by_dir: Dict[int, List[tuple]] = {}

    rec_trial_counts: List[int] = []
    rec_spans_s: List[float] = []
    included = 0

    for rec in recs:
        rec_str = f"{int(rec):03d}"
        try:
            dataset = load_joystick_dataset(repo_root, day, rec_str)
        except Exception as exc:  # noqa: BLE001
            print(f"  kinematics: could not load {day}/{rec_str} ({exc}); skipping")
            continue
        n_trials = len(np.asarray(dataset.all_trials["Trial"]).ravel())

        rec_trials = 0
        rec_perfs: List[float] = []
        rec_target_on: List[float] = []

        for trial_index in range(n_trials):
            alltrial = get_trial_row(dataset.all_trials, trial_index)
            if not _is_requested_task(alltrial.get("PyTaskType", ""), requested_task_types):
                continue
            included += 1
            rec_trials += 1

            behav_result = dataset.behav_results[trial_index]
            attempt = _get_final_attempt(behav_result)
            success = int(round(float(alltrial.get("Success", 0)))) == 1

            t_on = kf.event_perf(attempt, "target_on")
            t_move = kf.event_perf(attempt, "first_joystick_movement")
            t_entry = kf.event_perf(attempt, "target_entry")
            t_hc = kf.event_perf(attempt, "hold_complete")

            to_ev = select_attempt_event(attempt, "target_on", "first")
            tpos = attempt.get("target_position", {}) if isinstance(attempt, dict) else {}
            target_x = float(tpos.get("x_norm", to_ev.get("target_x", np.nan) if to_ev else np.nan))
            target_y = float(tpos.get("y_norm", to_ev.get("target_y", np.nan) if to_ev else np.nan))
            tidx = (int(to_ev["target_index"])
                    if to_ev and to_ev.get("target_index") is not None else -1)

            # Engagement / session-span bookkeeping
            apc = behav_result.get("trial_attempt_count")
            if apc is not None:
                attempts_per_trial.append(float(apc))
            idle_flags.append(1.0 if kf.has_idle(behav_result) else 0.0)
            rec_perfs.extend(kf.all_event_perfs(behav_result))
            if np.isfinite(t_on):
                rec_target_on.append(t_on)

            # Movement-time decomposition
            if np.isfinite(t_on) and np.isfinite(t_move) and t_move > t_on:
                rt.append(t_move - t_on)
            if np.isfinite(t_move) and np.isfinite(t_entry) and t_entry > t_move:
                transport.append(t_entry - t_move)
            if success and np.isfinite(t_entry) and np.isfinite(t_hc) and t_hc > t_entry:
                acquisition.append(t_hc - t_entry)

            # Aiming
            de = kf.initial_direction_error_deg(dataset, t_move, target_x, target_y)
            if np.isfinite(de):
                dir_err.append(de)

            # Smoothness / submovements / variability over the reach
            reach_start = t_move if np.isfinite(t_move) else t_on
            if np.isfinite(reach_start) and np.isfinite(t_entry) and t_entry > reach_start:
                speed = kf.reach_speed(dataset, reach_start, t_entry)
                if speed is not None:
                    submv.append(kf.submovement_count(speed[0]))
                    sparc.append(kf.sparc(speed[0], speed[1]))
                if success and tidx >= 0:
                    path = kf.normalized_path(dataset, reach_start, t_entry)
                    if path is not None:
                        paths_by_dir.setdefault(tidx, []).append(path)
                    endpoints_by_dir.setdefault(tidx, []).append(kf.interp_xy(dataset, t_entry))

            if tidx >= 0:
                success_by_dir.setdefault(tidx, []).append(1.0 if success else 0.0)
                if np.isfinite(target_x) and np.isfinite(target_y):
                    xy_by_dir.setdefault(tidx, []).append((target_x, target_y))

        if rec_trials and rec_target_on and rec_perfs:
            span = max(rec_perfs) - min(rec_target_on)
            if span > 0:
                rec_trial_counts.append(rec_trials)
                rec_spans_s.append(span)

    # Cross-trial variability (within each target direction, then pooled)
    trajectory_variability = _pooled_path_variability(paths_by_dir)
    endpoint_scatter = _pooled_endpoint_scatter(endpoints_by_dir)

    total_span = float(np.sum(rec_spans_s))
    trials_per_min = (float(np.sum(rec_trial_counts)) / (total_span / 60.0)
                      if total_span > 0 else float("nan"))

    rt_arr = np.asarray(rt, dtype=float)
    rt_cv = (float(np.std(rt_arr) / np.mean(rt_arr))
             if rt_arr.size and np.mean(rt_arr) > 0 else float("nan"))

    dir_success = {str(k): float(np.mean(v)) for k, v in success_by_dir.items() if v}
    dir_xy = {str(k): [float(np.median([p[0] for p in v])),
                       float(np.median([p[1] for p in v]))]
              for k, v in xy_by_dir.items() if v}

    return {
        "day": day,
        "selected_recs": [int(r) for r in recs],
        "n_included_trials": int(included),
        "median_reaction_time_ms": 1e3 * _median(rt),
        "median_transport_time_ms": 1e3 * _median(transport),
        "median_acquisition_time_ms": 1e3 * _median(acquisition),
        "reaction_time_cv": rt_cv,
        "median_initial_direction_error_deg": _median(dir_err),
        "median_submovement_count": _median(submv),
        "median_sparc": _median(sparc),
        "trajectory_variability": trajectory_variability,
        "endpoint_scatter": endpoint_scatter,
        "mean_attempts_per_trial": _mean(attempts_per_trial),
        "idle_timeout_rate": _mean(idle_flags),
        "trials_per_min": trials_per_min,
        "dir_success": dir_success,
        "dir_xy": dir_xy,
    }


def _pooled_path_variability(paths_by_dir: Dict[int, List[tuple]]) -> float:
    vals: List[float] = []
    weights: List[int] = []
    for paths in paths_by_dir.values():
        if len(paths) < 3:
            continue
        xs = np.array([p[0] for p in paths])
        ys = np.array([p[1] for p in paths])
        per_point = np.sqrt(np.var(xs, axis=0) + np.var(ys, axis=0))
        vals.append(float(np.mean(per_point)))
        weights.append(len(paths))
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def _pooled_endpoint_scatter(endpoints_by_dir: Dict[int, List[tuple]]) -> float:
    vals: List[float] = []
    weights: List[int] = []
    for eps in endpoints_by_dir.values():
        if len(eps) < 3:
            continue
        pts = np.asarray(eps, dtype=float)
        centroid = pts.mean(axis=0)
        vals.append(float(np.mean(np.hypot(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1]))))
        weights.append(len(eps))
    return float(np.average(vals, weights=weights)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# CSV (cumulative, keyed by day) + orchestration
# --------------------------------------------------------------------------- #

def _row_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: row.get(k) for k in _KIN_FIELDS}
    out["selected_recs"] = ",".join(f"{int(r):03d}" for r in row["selected_recs"])
    return out


def _write_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        with open(out_path, "r", newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                merged[str(raw.get("day", "")).strip()] = dict(raw)
    for row in rows:
        merged[str(row["day"])] = _row_for_csv(row)
    ordered = [merged[d] for d in sorted(merged) if d]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_KIN_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in _KIN_FIELDS})
    return ordered


def generate_cross_day_kinematics(
    repo_root: str | Path,
    selections: Sequence[str | Sequence[str]],
    out_dir: str | Path | None = None,
    task_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir is not None else _default_out_dir(repo_root_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    requested = {str(t).strip().lower() for t in (task_types or []) if str(t).strip()}

    parsed = [_parse_day_rec_selection(spec, repo_root_path) for spec in selections]
    rows = [build_day_kinematics(repo_root_path, day=day, recs=recs, requested_task_types=requested)
            for day, recs in parsed]
    rows.sort(key=lambda r: r["day"])

    figure_path = out_dir_path / "cross_day_kinematics.png"
    json_path = out_dir_path / "cross_day_kinematics_metrics.json"
    csv_path = out_dir_path / "cross_day_kinematics.csv"

    plot_cross_day_kinematics(rows, figure_path)
    csv_rows = _write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"days": rows, "task_types": sorted(requested),
                   "figure_path": str(figure_path)}, fh, indent=2)

    print(f"Saved figure: {figure_path}")
    print(f"Saved kinematics JSON: {json_path}")
    print(f"Saved kinematics CSV: {csv_path}")
    return {
        "figure_path": str(figure_path),
        "metrics_json_path": str(json_path),
        "metrics_csv_path": str(csv_path),
        "days": rows,
        "csv_days": csv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-day joystick learning kinematics from selected days/recs")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--day-recs", action="append", nargs="+", required=True,
                        help="Repeatable day selection like 260408:1-3,5 or 260409 for all recs.")
    parser.add_argument("--out-dir")
    parser.add_argument("--task-types", nargs="*", default=None)
    args = parser.parse_args()

    generate_cross_day_kinematics(
        repo_root=args.repo_root,
        selections=args.day_recs,
        out_dir=args.out_dir,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
