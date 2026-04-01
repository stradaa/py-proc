from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def plot_session_overview(data: Dict[str, np.ndarray], out_path: Path) -> Dict[str, Dict[str, float]]:
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)

    recs = sorted(np.unique(rec).tolist())
    counts = np.array([np.sum(rec == r) for r in recs], dtype=float)
    success_rate = np.array([np.nanmean(success[rec == r]) for r in recs], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(recs, counts, color="#5b8def", alpha=0.9, width=0.7)
    ax.set_xlabel("Recording")
    ax.set_ylabel("Trial count")
    ax.set_title("260331 joystick trials by recording")
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        str(r): {
            "trial_count": float(c),
            "success_rate": float(s),
        }
        for r, c, s in zip(recs, counts, success_rate)
    }


def plot_performance_over_time(data: Dict[str, np.ndarray], out_path: Path) -> Dict[str, float]:
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
    ax1.set_title("Performance over session (rec003 excluded)")
    ax1.grid(alpha=0.2)

    for r in sorted(np.unique(rec)):
        mask = rec == r
        ax2.scatter(order[mask], duration[mask], s=18, alpha=0.7, label=f"rec{r:03d}",
                    color=colors.get(int(r), "#777777"))

    ax2.plot(order, _rolling_mean(duration, window=25), color="#222222", linewidth=2, alpha=0.9)
    ax2.set_xlabel("Trial order across day")
    ax2.set_ylabel("Trial duration (ms)")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "overall_success_rate": float(np.nanmean(success)),
        "median_trial_duration_ms": float(np.nanmedian(duration)),
        "n_trials": float(len(success)),
    }


def plot_target_performance(data: Dict[str, np.ndarray], out_path: Path) -> Dict[str, Dict[str, float]]:
    target = _flat(data, "Target", float)
    success = _flat(data, "Success", float)
    hold_complete = _flat(data, "JoystickHoldComplete", float)

    valid = np.isfinite(target)
    targets = sorted(np.unique(target[valid]).astype(int).tolist())
    counts = np.array([np.sum(target == t) for t in targets], dtype=float)
    success_rate = np.array([np.nanmean(success[target == t]) for t in targets], dtype=float)
    median_complete = np.array([
        np.nanmedian(hold_complete[(target == t) & np.isfinite(hold_complete)])
        if np.any((target == t) & np.isfinite(hold_complete)) else np.nan
        for t in targets
    ], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[1, 1])

    bar_colors = plt.cm.tab20(np.linspace(0, 1, len(targets)))
    ax1.bar(targets, 100.0 * success_rate, color=bar_colors, alpha=0.9)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Target-wise performance (rec003 excluded)")
    ax1.grid(axis="y", alpha=0.2)

    ax1b = ax1.twinx()
    ax1b.plot(targets, counts, color="#222222", marker="o", linewidth=1.8)
    ax1b.set_ylabel("Trial count")

    ax2.bar(targets, median_complete, color=bar_colors, alpha=0.9)
    ax2.set_xlabel("Target ID")
    ax2.set_ylabel("Median hold-complete time (ms)")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        str(t): {
            "trial_count": float(c),
            "success_rate": float(s),
            "median_hold_complete_ms": float(m) if np.isfinite(m) else float("nan"),
        }
        for t, c, s, m in zip(targets, counts, success_rate, median_complete)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-style summary plots from AllTrials.mat")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-recs", nargs="*", type=int, default=[])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all_trials(repo_root, args.day)
    if args.exclude_recs:
        rec = _flat(data, "Rec", int)
        keep = ~np.isin(rec, np.asarray(args.exclude_recs, dtype=int))
        filtered: Dict[str, np.ndarray] = {}
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                filtered[key] = arr
            elif arr.shape[0] == len(rec):
                filtered[key] = arr[keep]
            else:
                filtered[key] = arr
        data = filtered

    metrics = {
        "session_overview": plot_session_overview(data, out_dir / f"{args.day}_overview_by_rec.png"),
        "performance_over_time": plot_performance_over_time(data, out_dir / f"{args.day}_performance_over_time.png"),
        "target_performance": plot_target_performance(data, out_dir / f"{args.day}_target_performance.png"),
    }

    with open(out_dir / f"{args.day}_summary_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
