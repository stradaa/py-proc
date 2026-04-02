from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _load_all_trials(repo_root: Path, day: str) -> Dict[str, np.ndarray]:
    all_trials = loadmat(repo_root / day / "mat" / "AllTrials.mat", simplify_cells=True)["AllTrials"]
    out: Dict[str, np.ndarray] = {}
    for key, value in all_trials.items():
        out[key] = np.asarray(value)
    return out


def _flat(data: Dict[str, np.ndarray], key: str, dtype=float) -> np.ndarray:
    value = np.asarray(data[key]).ravel()
    if dtype is object:
        return value.astype(object)
    return value.astype(dtype)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _decode_json_stream(blob: str) -> List[dict]:
    decoder = json.JSONDecoder()
    out: List[dict] = []
    i = 0
    n = len(blob)
    while i < n:
        while i < n and blob[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(blob, i)
        if isinstance(obj, dict):
            out.append(obj)
        i = j
    return out


def _parse_target_config_entry(entry: object) -> List[dict]:
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, list):
        out: List[dict] = []
        for item in entry:
            out.extend(_parse_target_config_entry(item))
        return out
    if isinstance(entry, dict):
        return [entry]
    if not isinstance(entry, str) or not entry:
        return []

    try:
        parsed = ast.literal_eval(entry)
    except Exception:
        parsed = entry

    if isinstance(parsed, list):
        text = "".join(str(x) for x in parsed)
    else:
        text = str(parsed)
    try:
        return _decode_json_stream(text)
    except Exception:
        return []


def _target_center_for_trial(target_id: float, config_entry: object) -> Optional[dict]:
    if not np.isfinite(target_id):
        return None
    targets = _parse_target_config_entry(config_entry)
    if not targets:
        return None

    idx = int(round(float(target_id)))
    if any(int(round(float(tid))) == 0 for tid in [target_id]):
        zero_based = True
    else:
        zero_based = False

    if zero_based:
        target_idx = idx
    else:
        target_idx = idx - 1 if 1 <= idx <= len(targets) else idx

    if target_idx < 0 or target_idx >= len(targets):
        if 0 <= idx < len(targets):
            target_idx = idx
        else:
            return None

    target = targets[target_idx]
    try:
        return {
            "x": float(target.get("x_norm", np.nan)),
            "y": float(target.get("y_norm", np.nan)),
            "radius": float(target.get("radius_ratio", np.nan)),
            "name": str(target.get("name", target_idx)),
        }
    except Exception:
        return None


def _title_suffix(exclude_recs: List[int]) -> str:
    if not exclude_recs:
        return ""
    rec_text = ", ".join(f"rec{rec:03d}" for rec in sorted(exclude_recs))
    plural = "s" if len(exclude_recs) > 1 else ""
    return f" (excluded rec{plural}: {rec_text})"


def _save_figure(fig: plt.Figure, out_path: Path, label: str) -> None:
    print(f"Generating figure: {label}")
    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def _longest_success_streak(success: np.ndarray) -> Dict[str, float]:
    best = 0
    current = 0
    best_end = -1
    for i, value in enumerate(np.asarray(success, dtype=float)):
        if np.isfinite(value) and value >= 0.5:
            current += 1
            if current > best:
                best = current
                best_end = i
        else:
            current = 0

    if best <= 0:
        return {
            "length": 0.0,
            "start_trial_index_in_rec": float("nan"),
            "end_trial_index_in_rec": float("nan"),
            "render_trial_range": "",
        }

    start_idx = best_end - best + 2
    end_idx = best_end + 1
    return {
        "length": float(best),
        "start_trial_index_in_rec": float(start_idx),
        "end_trial_index_in_rec": float(end_idx),
        "render_trial_range": f"{start_idx}-{end_idx}",
    }


def plot_session_overview(
    data: Dict[str, np.ndarray],
    out_path: Path,
    day: str,
    exclude_recs: List[int],
) -> Dict[str, Dict[str, float]]:
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)

    recs = sorted(np.unique(rec).tolist())
    counts = np.array([np.sum(rec == r) for r in recs], dtype=float)
    success_rate = np.array([np.nanmean(success[rec == r]) for r in recs], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(recs, counts, color="#5b8def", alpha=0.9, width=0.7)
    ax.set_xlabel("Recording")
    ax.set_ylabel("Trial count")
    ax.set_title(f"{day} joystick trials by recording{_title_suffix(exclude_recs)}")
    ax.set_xticks(recs)
    ax.grid(axis="y", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(recs, 100.0 * success_rate, color="#d95f02", marker="o", linewidth=2)
    ax2.set_ylabel("Success rate (%)")
    ax2.set_ylim(0, 105)

    for bar, rate in zip(bars, success_rate):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, 100.0 * rate + 2.0, f"{100.0 * rate:.0f}%",
                 ha="center", va="bottom", fontsize=9, color="#7f2704")

    fig.tight_layout()
    _save_figure(fig, out_path, "session overview")

    summary: Dict[str, Dict[str, object]] = {}
    for r, c, s in zip(recs, counts, success_rate):
        streak = _longest_success_streak(success[rec == r])
        summary[str(r)] = {
            "trial_count": float(c),
            "success_rate": float(s),
            "longest_consecutive_success_streak": streak,
        }
    return summary


def plot_performance_over_time(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, float]:
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    duration = _flat(data, "End", float)

    order = np.arange(1, len(success) + 1, dtype=int)
    rolling_success = 100.0 * _rolling_mean(success, window=25)

    colors = {
        1: "#4c78a8",
        2: "#f58518",
        4: "#54a24b",
        5: "#e45756",
        6: "#72b7b2",
        7: "#b279a2",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True, height_ratios=[1, 1.2])

    ax1.plot(order, rolling_success, color="#1f4e79", linewidth=2.2)
    ax1.scatter(order, 100.0 * success, c=[colors.get(int(r), "#777777") for r in rec], s=14, alpha=0.55)
    ax1.set_ylabel("Success (%)")
    ax1.set_ylim(-5, 105)
    ax1.set_title(f"Performance over session{_title_suffix(exclude_recs)}")
    ax1.grid(alpha=0.2)

    for r in sorted(np.unique(rec)):
        mask = rec == r
        ax2.scatter(order[mask], duration[mask], s=18, alpha=0.7, label=f"rec{r:03d}",
                    color=colors.get(int(r), "#777777"))

    ax2.plot(order, _rolling_mean(duration, window=25), color="#222222", linewidth=2, alpha=0.9)
    ax2.set_xlabel("Trial order across day")
    ax2.set_ylabel("Trial duration (ms)")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)

    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    _save_figure(fig, out_path, "performance over time")

    return {
        "overall_success_rate": float(np.nanmean(success)),
        "median_trial_duration_ms": float(np.nanmedian(duration)),
        "n_trials": float(len(success)),
    }


def plot_target_performance(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, Dict[str, float]]:
    target = _flat(data, "Target", float)
    success = _flat(data, "Success", float)
    config_entries = _flat(data, "TargetConfigs", object)

    centers: Dict[tuple, Dict[str, float]] = {}
    for target_id, cfg, succ in zip(target, config_entries, success):
        info = _target_center_for_trial(target_id, cfg)
        if info is None:
            continue
        x = info["x"]
        y = info["y"]
        r = info["radius"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        key = (round(x, 4), round(y, 4), round(r, 4), info["name"])
        bucket = centers.setdefault(key, {
            "x": x,
            "y": y,
            "radius": r if np.isfinite(r) else 0.06,
            "name": info["name"],
            "success": [],
        })
        bucket["success"].append(float(succ))

    rows = list(centers.values())
    xs = np.array([row["x"] for row in rows], dtype=float)
    ys = np.array([row["y"] for row in rows], dtype=float)
    rs = np.array([row["radius"] for row in rows], dtype=float)
    counts = np.array([len(row["success"]) for row in rows], dtype=float)
    success_rate = np.array([np.nanmean(row["success"]) for row in rows], dtype=float)

    gx = np.linspace(0.0, 1.0, 180)
    gy = np.linspace(0.0, 1.0, 180)
    X, Y = np.meshgrid(gx, gy)
    sigma = max(0.06, float(np.nanmedian(rs[np.isfinite(rs)])) if np.isfinite(rs).any() else 0.08)
    heat_num = np.zeros_like(X, dtype=float)
    heat_den = np.zeros_like(X, dtype=float)
    for x, y, c, s in zip(xs, ys, counts, success_rate):
        weight = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2.0 * sigma ** 2)) * max(c, 1.0)
        heat_num += weight * s
        heat_den += weight
    heat = np.divide(heat_num, heat_den, out=np.full_like(heat_num, np.nan), where=heat_den > 0)

    density = np.zeros_like(X, dtype=float)
    for x, y, c in zip(xs, ys, counts):
        density += np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2.0 * sigma ** 2)) * max(c, 1.0)

    valid_success = success_rate[np.isfinite(success_rate)]
    vmin = float(np.nanmin(valid_success)) if valid_success.size else 0.0
    vmax = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.8))

    im = ax1.imshow(
        heat,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        alpha=0.82,
        aspect="equal",
    )
    for row, c, s in zip(rows, counts, success_rate):
        circle = plt.Circle((row["x"], row["y"]), max(row["radius"], 0.02),
                            edgecolor="black", facecolor="none", linewidth=1.1, alpha=0.8)
        ax1.add_patch(circle)
        ax1.text(row["x"], row["y"], f"{100.0 * s:.0f}%\n(n={int(c)})",
                 ha="center", va="center", fontsize=7.5, color="black")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Target center X (task space)")
    ax1.set_ylabel("Target center Y (task space)")
    ax1.set_title("Spatial success map")
    ax1.grid(alpha=0.15)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(f"Success rate ({100.0 * vmin:.0f}% to 100%)")

    im2 = ax2.imshow(
        density,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="Blues",
        alpha=0.88,
        aspect="equal",
    )
    for row, c, s in zip(rows, counts, success_rate):
        ax2.add_patch(plt.Circle((row["x"], row["y"]), max(row["radius"], 0.02),
                                 edgecolor="#333333", facecolor="none", linewidth=1.0, alpha=0.65))
        ax2.scatter(row["x"], row["y"], s=max(c, 1.0) * 14.0, color="#08306b", alpha=0.8)
        ax2.text(row["x"], row["y"] - 0.045, f"{row['name']}\n n={int(c)}",
                 ha="center", va="top", fontsize=7.5)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Target center X (task space)")
    ax2.set_ylabel("Target center Y (task space)")
    ax2.set_title("Spatial trial density and target usage")
    ax2.grid(alpha=0.15)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Relative trial density")

    fig.suptitle(f"Spatial target performance{_title_suffix(exclude_recs)}")
    fig.tight_layout()
    _save_figure(fig, out_path, "target performance")

    return {
        row["name"]: {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "radius": float(row["radius"]),
            "trial_count": float(c),
            "success_rate": float(s),
        }
        for row, c, s in zip(rows, counts, success_rate)
    }


def plot_success_failure_timing(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, Dict[str, float]]:
    success = _flat(data, "Success", float)
    fields = {
        "First movement": 1e-3 * _flat(data, "JoystickFirstMovement", float),
        "Target entry": 1e-3 * _flat(data, "JoystickTargetEntry", float),
        "Hold complete": 1e-3 * _flat(data, "JoystickHoldComplete", float),
        "Trial end": 1e-3 * _flat(data, "End", float),
    }

    labels = list(fields.keys())
    success_data = []
    failure_data = []
    summary: Dict[str, Dict[str, float]] = {}

    for label, arr in fields.items():
        success_vals = arr[(success == 1) & np.isfinite(arr)]
        failure_vals = arr[(success == 0) & np.isfinite(arr)]
        success_data.append(success_vals)
        failure_data.append(failure_vals)
        summary[label] = {
            "success_n": float(len(success_vals)),
            "success_median_s": float(np.nanmedian(success_vals)) if len(success_vals) else float("nan"),
            "failure_n": float(len(failure_vals)),
            "failure_median_s": float(np.nanmedian(failure_vals)) if len(failure_vals) else float("nan"),
        }

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    pos = np.arange(len(labels), dtype=float)

    def _plot_group(groups: List[np.ndarray], positions: np.ndarray, face: str, edge: str) -> None:
        valid_idx = [i for i, g in enumerate(groups) if len(g) > 0]
        if valid_idx:
            parts = ax.violinplot([groups[i] for i in valid_idx],
                                  positions=positions[valid_idx],
                                  widths=0.28,
                                  showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(face)
                body.set_edgecolor(edge)
                body.set_alpha(0.55)
            parts["cmedians"].set_color(edge)

    _plot_group(success_data, pos - 0.18, "#54a24b", "#1b5e20")
    _plot_group(failure_data, pos + 0.18, "#e45756", "#7f0000")

    for i, (svals, fvals) in enumerate(zip(success_data, failure_data)):
        if len(svals):
            ax.scatter(np.full(len(svals), pos[i] - 0.18), svals, s=8, alpha=0.15, color="#1b5e20")
        if len(fvals):
            ax.scatter(np.full(len(fvals), pos[i] + 0.18), fvals, s=8, alpha=0.15, color="#7f0000")

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency from StartOn (s)")
    ax.set_title(f"Success vs failure timing{_title_suffix(exclude_recs)}")
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="#54a24b", lw=8, alpha=0.6, label="Successful trials"),
        Line2D([0], [0], color="#e45756", lw=8, alpha=0.6, label="Failed trials"),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_path, "success vs failure timing")

    return summary


def plot_display_alignment(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, object]:
    rec = _flat(data, "Rec", int)
    display_latency_ms = _flat(data, "disStartOn", float)
    valid = np.isfinite(display_latency_ms)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.8, 7.4), height_ratios=[1.0, 1.15], sharex=False
    )

    valid_latency = display_latency_ms[valid]
    if len(valid_latency):
        bins = np.linspace(
            float(np.nanmin(valid_latency)) - 2.5,
            float(np.nanmax(valid_latency)) + 2.5,
            min(40, max(12, int(np.sqrt(len(valid_latency)) * 3))),
        )
        ax1.hist(valid_latency, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="white")
        median_latency = float(np.nanmedian(valid_latency))
        mean_latency = float(np.nanmean(valid_latency))
        ax1.axvline(median_latency, color="#d95f02", linewidth=2, label=f"median {median_latency:.1f} ms")
        ax1.axvline(mean_latency, color="#1b9e77", linewidth=1.8, linestyle="--", label=f"mean {mean_latency:.1f} ms")
        ax1.legend(frameon=False, loc="upper right")
    else:
        median_latency = float("nan")
        mean_latency = float("nan")
        ax1.text(0.5, 0.5, "No valid disStartOn values", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_xlabel("disStartOn relative to StartOn (ms)")
    ax1.set_ylabel("Trial count")
    ax1.set_title(f"Display alignment latency distribution{_title_suffix(exclude_recs)}")
    ax1.grid(axis="y", alpha=0.2)

    recs = sorted(np.unique(rec).tolist())
    rec_summary: Dict[str, Dict[str, float]] = {}
    x_positions: List[float] = []
    rec_groups: List[np.ndarray] = []
    for i, r in enumerate(recs):
        vals = display_latency_ms[(rec == r) & valid]
        if len(vals):
            rec_groups.append(vals)
            x_positions.append(float(i + 1))
        rec_summary[str(r)] = {
            "n_valid": float(len(vals)),
            "median_ms": float(np.nanmedian(vals)) if len(vals) else float("nan"),
            "mean_ms": float(np.nanmean(vals)) if len(vals) else float("nan"),
            "min_ms": float(np.nanmin(vals)) if len(vals) else float("nan"),
            "max_ms": float(np.nanmax(vals)) if len(vals) else float("nan"),
        }

    if rec_groups:
        parts = ax2.violinplot(rec_groups, positions=np.asarray(x_positions), widths=0.8, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor("#72b7b2")
            body.set_edgecolor("#1f4e79")
            body.set_alpha(0.45)
        parts["cmedians"].set_color("#1f4e79")

        for x, vals in zip(x_positions, rec_groups):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax2.scatter(np.full(len(vals), x) + jitter, vals, s=10, alpha=0.25, color="#0b3954")

    ax2.axhline(50.0, color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.9)
    ax2.set_xticks(np.arange(1, len(recs) + 1, dtype=float))
    ax2.set_xticklabels([f"rec{r:03d}" for r in recs], rotation=0)
    ax2.set_ylabel("disStartOn relative to StartOn (ms)")
    ax2.set_xlabel("Recording")
    ax2.set_title("Per-recording display alignment")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    _save_figure(fig, out_path, "display alignment")

    return {
        "n_valid_trials": float(np.sum(valid)),
        "n_missing_trials": float(np.sum(~valid)),
        "median_ms": median_latency,
        "mean_ms": mean_latency,
        "min_ms": float(np.nanmin(valid_latency)) if len(valid_latency) else float("nan"),
        "max_ms": float(np.nanmax(valid_latency)) if len(valid_latency) else float("nan"),
        "per_rec": rec_summary,
    }


def _resolve_day_paths(args: argparse.Namespace) -> Tuple[Path, str]:
    if args.day_dir is not None:
        day_dir = Path(args.day_dir).resolve()
        return day_dir.parent, day_dir.name

    if args.repo_root is None or args.day is None:
        raise ValueError("Provide either --day-dir or both --repo-root and --day.")

    return Path(args.repo_root).resolve(), str(args.day)


def _default_out_dir(repo_root: Path, day: str) -> Path:
    return repo_root / "claude" / "figures" / day / "beh"


def _filter_trial_rows(data: Dict[str, np.ndarray], keep: np.ndarray) -> Dict[str, np.ndarray]:
    filtered: Dict[str, np.ndarray] = {}
    n_rows = len(keep)
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            filtered[key] = arr
        elif arr.shape[0] == n_rows:
            filtered[key] = arr[keep]
        else:
            filtered[key] = arr
    return filtered


def generate_day_presentation_plots(
    repo_root: str | Path,
    day: str,
    out_dir: str | Path | None = None,
    exclude_recs: Optional[List[int]] = None,
    task_types: Optional[List[str]] = None,
) -> Dict[str, object]:
    repo_root_path = Path(repo_root).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir is not None else _default_out_dir(repo_root_path, day)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    exclude_recs = exclude_recs or []
    task_types = task_types or []

    print(f"Using output directory: {out_dir_path}")
    if exclude_recs:
        print("Excluding recordings:", ", ".join(f"rec{rec:03d}" for rec in sorted(exclude_recs)))
    else:
        print("Excluding recordings: none")
    if task_types:
        print("Filtering task types:", ", ".join(sorted(task_types)))
    else:
        print("Filtering task types: none")

    data = _load_all_trials(repo_root_path, day)
    if exclude_recs:
        rec = _flat(data, "Rec", int)
        keep = ~np.isin(rec, np.asarray(exclude_recs, dtype=int))
        data = _filter_trial_rows(data, keep)

    if task_types:
        requested = {str(task_type).strip().lower() for task_type in task_types if str(task_type).strip()}
        task_type_values = _flat(data, "PyTaskType", object)
        keep = np.array([str(task).strip().lower() in requested for task in task_type_values], dtype=bool)
        print(f"Trials kept after task filter: {int(np.sum(keep))} / {len(keep)}")
        data = _filter_trial_rows(data, keep)

    metrics = {
        "session_overview": plot_session_overview(
            data, out_dir_path / f"{day}_overview_by_rec.png", day, exclude_recs
        ),
        "performance_over_time": plot_performance_over_time(
            data, out_dir_path / f"{day}_performance_over_time.png", exclude_recs
        ),
        "display_alignment": plot_display_alignment(
            data, out_dir_path / f"{day}_display_alignment.png", exclude_recs
        ),
        "target_performance": plot_target_performance(
            data, out_dir_path / f"{day}_target_performance.png", exclude_recs
        ),
        "success_failure_timing": plot_success_failure_timing(
            data, out_dir_path / f"{day}_success_failure_timing.png", exclude_recs
        ),
    }

    metrics_path = out_dir_path / f"{day}_summary_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved summary metrics: {metrics_path}")

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "out_dir": str(out_dir_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-style summary plots from AllTrials.mat")
    parser.add_argument("--repo-root")
    parser.add_argument("--day")
    parser.add_argument("--day-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--exclude-recs", nargs="*", type=int, default=[])
    parser.add_argument("--task-types", nargs="*", default=None)
    args = parser.parse_args()

    repo_root, day = _resolve_day_paths(args)
    generate_day_presentation_plots(
        repo_root=repo_root,
        day=day,
        out_dir=args.out_dir,
        exclude_recs=args.exclude_recs,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
