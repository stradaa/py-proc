"""Cross-day Fitts'-law bit-rate (throughput) metric, on the task/behave clock.

This is the field-standard way to report cursor-control performance as a
single number comparable across conditions that vary target radius or
hold-cue duration, and — the real motivation — across control modalities
(joystick now, brain-control later): ISO 9241-9 "throughput" in bits/s,
computed from Fitts' index of difficulty divided by movement time.

Per successful attempt:
  * Amplitude A = distance from the cursor position at movement onset
    (`first_joystick_movement`) to the target center, in an aspect-corrected
    pixel-equivalent space (see `pixel_scale`).
  * Movement time MT = first_movement -> first `target_entry` ("transport"
    time only — excludes reaction time *and* the hold/dwell tax entirely,
    which is what makes this metric independent of the cued hold duration).
  * A signed 1-D deviation of the first `target_entry` point (the initial
    ballistic landing) from the target center, projected onto the movement
    axis. This — not the final hold-satisfied position, which is trivially
    inside the target radius by construction in this dwell-select task — is
    the correct analogue of a classic Fitts "selection point" for the
    ISO 9241-9 effective-width correction.

Per condition bin (pooled across attempts sharing a bin within a day):
  * Effective width We = 4.133 * SD(deviations) (Shannon-Welford / MacKenzie
    correction for the subject's actual aiming accuracy, not the nominal
    target radius).
  * Effective index of difficulty IDe = log2(mean(A)/We + 1).
  * Throughput = IDe / MT, per attempt; day-level scalar = mean of per-bin
    average throughput (equal-per-bin weighting, so the interleaved-random
    grid trials don't swamp the well-populated center-out directions).
  * When enough distinct IDe bins are estimable, also the classic Fitts
    regression (MT vs IDe across bins) -> 1/slope, the more standard,
    externally-comparable "index of performance."

Two condition families, reported separately (`schedule_phase`-tagged, same
tagging the Rust executor already writes — see `co_structured.py`):
  * Center-out ring (`schedule_phase == "peripheral"`) is the **headline**
    metric — 8 fixed radial targets, well-populated per day, the closest
    analogue to standard BCI-field center-out throughput reporting.
  * Everything else (grid / random / sequence / center attempts) is a
    **secondary/diagnostic** throughput, useful for QC but not the number to
    lead with, since most individual grid targets get too few repeats per
    day for a trustworthy effective-width estimate.

Like path efficiency (`cross_day_plots.py`), this is **control-law
sensitive**: cursor reconstruction (`pyjoystick/cursor.py`) differs by
`control_mode` (direct/cumulative/blend), so A, MT, and the entry-point
deviation are all downstream of that reconstruction.

Per-trial primitives reuse `kinematics_features.py` (`event_perf`,
`interp_xy`); the figure is rendered by `fitts_throughput_plots.py`. This
module does per-day aggregation and orchestration, mirroring
`cross_day_kinematics.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .cross_day_plots import _parse_day_rec_selection, _default_out_dir, _is_requested_task, _summarize_modes
    from .fitts_throughput_plots import plot_cross_day_throughput
    from .joystick_validation import (
        _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event,
    )
    from . import kinematics_features as kf
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_plots import _parse_day_rec_selection, _default_out_dir, _is_requested_task, _summarize_modes
    from fitts_throughput_plots import plot_cross_day_throughput
    from joystick_validation import (
        _get_final_attempt, get_trial_row, load_joystick_dataset, select_attempt_event,
    )
    import kinematics_features as kf


# Rig display is fixed at 1920x1080 (confirmed) across the days this compares.
# Only the aspect ratio matters for the Fitts' math (A/W ratios, and We which
# is itself a distance in the same units, are scale-invariant) — an assumed
# resolution sharing the true aspect ratio gives identical IDe/throughput to
# the real one, so this only needs revisiting if the rig's display ratio ever
# changes (e.g. a different monitor) or multiple rigs with different ratios
# get compared cross-day.
ASSUMED_DISPLAY_W_PX = 1920.0
ASSUMED_DISPLAY_H_PX = 1080.0

# Shannon-Welford / MacKenzie (1992) effective-width constant, per ISO 9241-9.
EFFECTIVE_WIDTH_K = 4.133

# Minimum distinct condition bins (each meeting min_attempts_per_bin) needed
# before a Fitts regression (MT vs IDe) is attempted; below this the fit is a
# guess, not an estimate, so we report NaN instead.
MIN_REGRESSION_BINS = 4

# Scalar metric columns (per-target-bin QC dicts live only in the JSON).
_FITTS_FIELDS = [
    "day", "selected_recs", "control_mode", "position_velocity_blend", "control_mode_mixed",
    "n_included_trials",
    "n_co_attempts_included", "n_co_bins_estimable",
    "co_throughput_bits_s", "co_throughput_regression_bits_s", "co_regression_r2", "co_median_ide_bits",
    "n_grid_attempts_included", "n_grid_bins_estimable",
    "grid_throughput_bits_s", "grid_throughput_regression_bits_s", "grid_regression_r2", "grid_median_ide_bits",
]


@dataclass
class FittsSample:
    A_px: float
    W_nominal_px: float
    MT_s: float
    dev_signed_px: float


@dataclass
class BinResult:
    n: int
    A_mean_px: float
    We_px: float
    IDe_bits: float
    MT_mean_s: float
    per_attempt_throughput: List[float]


def pixel_scale(task_config: Dict[str, Any]) -> tuple[float, float]:
    """(scale_x, scale_y): aspect-corrected pixel-equivalent scale of the task
    region, mirroring the Rust task's own `region_bounds_ratios`/
    `region_min_dim` (rust/joystick_task/src/state.rs:524-550)."""
    width_ratio = float(task_config.get("task_region_width", 0.5))
    height_ratio = float(task_config.get("task_region_height", 0.67))
    return width_ratio * ASSUMED_DISPLAY_W_PX, height_ratio * ASSUMED_DISPLAY_H_PX


def attempt_fitts_sample(
    dataset: Any,
    attempt: Dict[str, Any],
    target_x: float,
    target_y: float,
    target_radius_ratio: float,
    scale_x: float,
    scale_y: float,
) -> Optional[FittsSample]:
    t_move = kf.event_perf(attempt, "first_joystick_movement")
    t_entry = kf.event_perf(attempt, "target_entry")
    if not (np.isfinite(t_move) and np.isfinite(t_entry) and t_entry > t_move):
        return None

    entry_ev = select_attempt_event(attempt, "target_entry", "first")
    if entry_ev is None or entry_ev.get("cursor_x") is None or entry_ev.get("cursor_y") is None:
        return None

    x0, y0 = kf.interp_xy(dataset, t_move)
    move_x, move_y = (target_x - x0) * scale_x, (target_y - y0) * scale_y
    A = float(np.hypot(move_x, move_y))
    if A < 1e-6:
        return None
    axis_x, axis_y = move_x / A, move_y / A

    xe, ye = float(entry_ev["cursor_x"]), float(entry_ev["cursor_y"])
    dev_x, dev_y = (xe - target_x) * scale_x, (ye - target_y) * scale_y
    dev_signed = float(dev_x * axis_x + dev_y * axis_y)

    W_nom = 2.0 * float(target_radius_ratio) * min(scale_x, scale_y)
    return FittsSample(A_px=A, W_nominal_px=W_nom, MT_s=float(t_entry - t_move), dev_signed_px=dev_signed)


def pool_bin(samples: Sequence[FittsSample]) -> Optional[BinResult]:
    if not samples:
        return None
    A = np.array([s.A_px for s in samples], dtype=float)
    dev = np.array([s.dev_signed_px for s in samples], dtype=float)
    mt = np.array([s.MT_s for s in samples], dtype=float)
    We = float(EFFECTIVE_WIDTH_K * np.std(dev, ddof=1)) if len(samples) >= 2 else float("nan")
    A_mean = float(np.mean(A))
    IDe = float(np.log2(A_mean / We + 1.0)) if np.isfinite(We) and We > 1e-6 else float("nan")
    per_attempt_tp = [IDe / m for m in mt if m > 1e-6] if np.isfinite(IDe) else []
    return BinResult(
        n=len(samples), A_mean_px=A_mean, We_px=We, IDe_bits=IDe,
        MT_mean_s=float(np.mean(mt)), per_attempt_throughput=per_attempt_tp,
    )


def _distance_quantile_bins(samples: Sequence[FittsSample], n_bins: int) -> Dict[int, Optional[BinResult]]:
    if len(samples) < n_bins:
        return {}
    amplitudes = np.array([s.A_px for s in samples], dtype=float)
    edges = np.quantile(amplitudes, np.linspace(0.0, 1.0, n_bins + 1))
    buckets: Dict[int, List[FittsSample]] = {i: [] for i in range(n_bins)}
    for sample, a in zip(samples, amplitudes):
        idx = min(max(int(np.searchsorted(edges, a, side="right") - 1), 0), n_bins - 1)
        buckets[idx].append(sample)
    return {i: pool_bin(v) for i, v in buckets.items()}


def _summarize_condition(bins: Dict[Any, Optional[BinResult]], min_attempts_per_bin: int) -> Dict[str, Any]:
    qualifying = {
        k: b for k, b in bins.items()
        if b is not None and b.n >= min_attempts_per_bin and np.isfinite(b.IDe_bits)
    }
    bin_means = [float(np.mean(b.per_attempt_throughput)) for b in qualifying.values() if b.per_attempt_throughput]
    ide_values = [b.IDe_bits for b in qualifying.values()]

    throughput_regression_bits_s = float("nan")
    regression_r2 = float("nan")
    if len(qualifying) >= MIN_REGRESSION_BINS and len(set(np.round(ide_values, 6))) >= 2:
        ide_arr = np.asarray(ide_values, dtype=float)
        mt_arr = np.asarray([b.MT_mean_s for b in qualifying.values()], dtype=float)
        slope, intercept = np.polyfit(ide_arr, mt_arr, 1)
        if abs(slope) > 1e-9:
            pred = slope * ide_arr + intercept
            ss_res = float(np.sum((mt_arr - pred) ** 2))
            ss_tot = float(np.sum((mt_arr - np.mean(mt_arr)) ** 2))
            regression_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
            throughput_regression_bits_s = float(1.0 / slope)

    return {
        "n_bins_estimable": len(qualifying),
        "throughput_bits_s": float(np.mean(bin_means)) if bin_means else float("nan"),
        "throughput_regression_bits_s": throughput_regression_bits_s,
        "regression_r2": regression_r2,
        "median_ide_bits": float(np.median(ide_values)) if ide_values else float("nan"),
    }


def _bin_result_to_dict(b: Optional[BinResult]) -> Optional[Dict[str, Any]]:
    if b is None:
        return None
    return {
        "n": b.n,
        "A_mean_px": b.A_mean_px,
        "We_px": b.We_px,
        "IDe_bits": b.IDe_bits,
        "MT_mean_s": b.MT_mean_s,
        "throughput_bits_s": float(np.mean(b.per_attempt_throughput)) if b.per_attempt_throughput else float("nan"),
    }


def build_day_throughput(
    repo_root: Path,
    day: str,
    recs: Sequence[int],
    requested_task_types: set[str],
    min_attempts_per_bin: int = 8,
    n_grid_distance_bins: int = 4,
) -> Dict[str, Any]:
    co_samples_by_target: Dict[int, List[FittsSample]] = {}
    grid_samples: List[FittsSample] = []
    grid_samples_by_target: Dict[int, List[FittsSample]] = {}
    included = 0
    control_modes: set[str] = set()
    blend_values: set[float] = set()

    for rec in recs:
        rec_str = f"{int(rec):03d}"
        try:
            dataset = load_joystick_dataset(repo_root, day, rec_str)
        except Exception as exc:  # noqa: BLE001
            print(f"  fitts_throughput: could not load {day}/{rec_str} ({exc}); skipping")
            continue
        control_modes.add(str(dataset.control_mode))
        blend_values.add(round(float(dataset.position_velocity_blend), 4))
        n_trials = len(np.asarray(dataset.all_trials["Trial"]).ravel())

        for trial_index in range(n_trials):
            alltrial = get_trial_row(dataset.all_trials, trial_index)
            if not _is_requested_task(alltrial.get("PyTaskType", ""), requested_task_types):
                continue
            if int(round(float(alltrial.get("Success", 0)))) != 1:
                continue
            included += 1

            attempt = _get_final_attempt(dataset.behav_results[trial_index])
            if not isinstance(attempt, dict):
                continue

            to_ev = select_attempt_event(attempt, "target_on", "first")
            tpos = attempt.get("target_position", {}) if isinstance(attempt, dict) else {}
            target_x = float(tpos.get("x_norm", to_ev.get("target_x", np.nan) if to_ev else np.nan))
            target_y = float(tpos.get("y_norm", to_ev.get("target_y", np.nan) if to_ev else np.nan))
            tidx = (int(to_ev["target_index"])
                    if to_ev and to_ev.get("target_index") is not None else -1)
            target_radius_ratio = float(attempt.get("target_radius_ratio", np.nan))
            if not (np.isfinite(target_x) and np.isfinite(target_y) and np.isfinite(target_radius_ratio)):
                continue

            task_config = dataset.task_configs[trial_index] if trial_index < len(dataset.task_configs) else {}
            scale_x, scale_y = pixel_scale(task_config if isinstance(task_config, dict) else {})

            sample = attempt_fitts_sample(dataset, attempt, target_x, target_y, target_radius_ratio, scale_x, scale_y)
            if sample is None:
                continue

            if str(attempt.get("schedule_phase") or "") == "peripheral":
                co_samples_by_target.setdefault(tidx, []).append(sample)
            else:
                grid_samples.append(sample)
                if tidx >= 0:
                    grid_samples_by_target.setdefault(tidx, []).append(sample)

    co_bins = {tidx: pool_bin(samples) for tidx, samples in co_samples_by_target.items()}
    co_summary = _summarize_condition(co_bins, min_attempts_per_bin)

    grid_bins_qc = {tidx: pool_bin(samples) for tidx, samples in grid_samples_by_target.items()}
    grid_dist_bins = _distance_quantile_bins(grid_samples, n_grid_distance_bins)
    grid_summary = _summarize_condition(grid_dist_bins, min_attempts_per_bin)

    control_mode_label, blend_repr, mode_mixed = _summarize_modes(control_modes, blend_values)
    return {
        "day": day,
        "selected_recs": [int(r) for r in recs],
        "control_mode": control_mode_label,
        "position_velocity_blend": blend_repr,
        "control_mode_mixed": int(mode_mixed),
        "n_included_trials": int(included),
        "n_co_attempts_included": int(sum(len(v) for v in co_samples_by_target.values())),
        "n_co_bins_estimable": co_summary["n_bins_estimable"],
        "co_throughput_bits_s": co_summary["throughput_bits_s"],
        "co_throughput_regression_bits_s": co_summary["throughput_regression_bits_s"],
        "co_regression_r2": co_summary["regression_r2"],
        "co_median_ide_bits": co_summary["median_ide_bits"],
        "n_grid_attempts_included": int(len(grid_samples)),
        "n_grid_bins_estimable": grid_summary["n_bins_estimable"],
        "grid_throughput_bits_s": grid_summary["throughput_bits_s"],
        "grid_throughput_regression_bits_s": grid_summary["throughput_regression_bits_s"],
        "grid_regression_r2": grid_summary["regression_r2"],
        "grid_median_ide_bits": grid_summary["median_ide_bits"],
        "co_bins_by_target": {str(k): _bin_result_to_dict(v) for k, v in co_bins.items()},
        "grid_bins_by_target": {str(k): _bin_result_to_dict(v) for k, v in grid_bins_qc.items()},
    }


# --------------------------------------------------------------------------- #
# CSV (cumulative, keyed by day) + orchestration
# --------------------------------------------------------------------------- #

def _row_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: row.get(k) for k in _FITTS_FIELDS}
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
        writer = csv.DictWriter(fh, fieldnames=_FITTS_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in _FITTS_FIELDS})
    return ordered


def generate_cross_day_throughput(
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
    rows = [build_day_throughput(repo_root_path, day=day, recs=recs, requested_task_types=requested)
            for day, recs in parsed]
    rows.sort(key=lambda r: r["day"])

    figure_path = out_dir_path / "cross_day_throughput.png"
    json_path = out_dir_path / "cross_day_throughput_metrics.json"
    csv_path = out_dir_path / "cross_day_throughput.csv"

    plot_cross_day_throughput(rows, figure_path)
    csv_rows = _write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"days": rows, "task_types": sorted(requested),
                   "figure_path": str(figure_path)}, fh, indent=2)

    print(f"Saved figure: {figure_path}")
    print(f"Saved throughput JSON: {json_path}")
    print(f"Saved throughput CSV: {csv_path}")
    return {
        "figure_path": str(figure_path),
        "metrics_json_path": str(json_path),
        "metrics_csv_path": str(csv_path),
        "days": rows,
        "csv_days": csv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-day Fitts'-law bit-rate throughput from selected days/recs")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--day-recs", action="append", nargs="+", required=True,
                        help="Repeatable day selection like 260408:1-3,5 or 260409 for all recs.")
    parser.add_argument("--out-dir")
    parser.add_argument("--task-types", nargs="*", default=None)
    args = parser.parse_args()

    generate_cross_day_throughput(
        repo_root=args.repo_root,
        selections=args.day_recs,
        out_dir=args.out_dir,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
