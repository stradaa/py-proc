"""day_behavior_metrics.py — Additional day-summary behavior plots.

Derived metrics that complement the timing/spatial plots in
``day_presentation_plots``. Everything here is computed from fields already
present in ``AllTrials.mat``:

- Kinematic intervals: reaction time, movement time, hold duration.
- Failure-mode breakdown: why failed trials failed (funnel stage).
- Overshoot & attempts: target re-entries and JoystickAttemptCount.
- Engagement: trial rate, inter-trial interval, reward rate over the session.

All Joystick* timestamps are milliseconds relative to per-trial ``StartOn``;
``StartOn`` itself is an absolute, per-rec monotonic clock (ms) usable for
inter-trial intervals. Trials with ``End <= 0`` are degenerate (aborted/
placeholder) and excluded from interval and engagement metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from pyCheck.plot_utils import _flat, _save_figure, _title_suffix

SUCCESS_FACE = "#54a24b"
SUCCESS_EDGE = "#1b5e20"
FAILURE_FACE = "#e45756"
FAILURE_EDGE = "#7f0000"


def _valid_mask(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Real trials only — End > 0 drops aborted/placeholder rows."""
    end = _flat(data, "End", float)
    return np.isfinite(end) & (end > 0.0)


def _clipped_overlay_hist(
    ax: plt.Axes,
    svals: np.ndarray,
    fvals: np.ndarray,
    unit: str = "s",
) -> None:
    """Overlay success/failure histograms with 99th-percentile x-clipping."""
    combined = np.concatenate([v for v in (svals, fvals) if len(v)]) if (len(svals) or len(fvals)) else np.array([])
    if combined.size == 0:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
        return

    lo = float(np.nanmin(combined))
    hi = float(np.nanpercentile(combined, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    n_outliers = int(np.sum(combined > hi))
    n_bins = int(np.clip(np.sqrt(combined.size) * 2.0, 12, 40))
    bins = np.linspace(lo, hi, n_bins + 1)

    def _clip(vals: np.ndarray) -> np.ndarray:
        return np.clip(vals, lo, hi) if len(vals) else vals

    if len(svals):
        ax.hist(_clip(svals), bins=bins, color=SUCCESS_FACE, alpha=0.55, edgecolor="white",
                label=f"success (n={len(svals)})")
        med = float(np.nanmedian(svals))
        ax.axvline(med, color=SUCCESS_EDGE, linewidth=2, label=f"median {med:.2f} {unit}")
    if len(fvals):
        ax.hist(_clip(fvals), bins=bins, color=FAILURE_FACE, alpha=0.55, edgecolor="white",
                label=f"failure (n={len(fvals)})")
        med = float(np.nanmedian(fvals))
        ax.axvline(med, color=FAILURE_EDGE, linewidth=2, linestyle="--", label=f"median {med:.2f} {unit}")
    if n_outliers:
        ax.text(0.98, 0.60, f"{n_outliers} trial(s) > {hi:.0f} {unit}\n(piled in last bin)",
                ha="right", va="top", transform=ax.transAxes, fontsize=6.5, color="#555555")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")


def _interval_summary(svals: np.ndarray, fvals: np.ndarray) -> Dict[str, float]:
    return {
        "success_n": float(len(svals)),
        "success_median_s": float(np.nanmedian(svals)) if len(svals) else float("nan"),
        "failure_n": float(len(fvals)),
        "failure_median_s": float(np.nanmedian(fvals)) if len(fvals) else float("nan"),
    }


def plot_kinematic_intervals(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, Dict[str, float]]:
    """Reaction time, movement time, and hold duration as success/failure histograms."""
    valid = _valid_mask(data)
    success = _flat(data, "Success", float)
    target_on = _flat(data, "JoystickTargetOn", float)
    first_move = _flat(data, "JoystickFirstMovement", float)
    target_entry = _flat(data, "JoystickTargetEntry", float)
    hold_start = _flat(data, "JoystickHoldStart", float)
    hold_complete = _flat(data, "JoystickHoldComplete", float)

    intervals = {
        "Reaction time\n(first movement - target on)": 1e-3 * (first_move - target_on),
        "Movement time\n(target entry - first movement)": 1e-3 * (target_entry - first_move),
        "Hold duration\n(hold complete - hold start)": 1e-3 * (hold_complete - hold_start),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    summary: Dict[str, Dict[str, float]] = {}
    for ax, (label, arr) in zip(axes, intervals.items()):
        # Negative intervals are timestamp artifacts, not real measurements.
        ok = valid & np.isfinite(arr) & (arr >= 0)
        svals = arr[ok & (success == 1)]
        fvals = arr[ok & (success == 0)]
        _clipped_overlay_hist(ax, svals, fvals)
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Trial count")
        summary[label.split("\n")[0]] = _interval_summary(svals, fvals)

    fig.suptitle(f"Kinematic intervals{_title_suffix(exclude_recs)}")
    fig.tight_layout()
    _save_figure(fig, out_path, "kinematic intervals")
    return summary


def plot_failure_modes(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, object]:
    """Stacked bar of why failed trials failed, per recording (funnel stage)."""
    valid = _valid_mask(data)
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    fin = np.isfinite
    first_move = fin(_flat(data, "JoystickFirstMovement", float))
    target_entry = fin(_flat(data, "JoystickTargetEntry", float))
    hold_complete = fin(_flat(data, "JoystickHoldComplete", float))

    fail = valid & (success == 0)
    categories = {
        "No movement": fail & ~first_move,
        "Never acquired target": fail & first_move & ~target_entry,
        "Broke hold": fail & target_entry & ~hold_complete,
        "Other": fail & target_entry & hold_complete,
    }
    colors = {
        "No movement": "#bdbdbd",
        "Never acquired target": "#f58518",
        "Broke hold": "#e45756",
        "Other": "#9467bd",
    }

    recs = sorted(np.unique(rec).tolist())
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bottoms = np.zeros(len(recs), dtype=float)
    per_rec: Dict[str, Dict[str, float]] = {str(r): {} for r in recs}
    totals: Dict[str, float] = {}
    for label, mask in categories.items():
        counts = np.array([int(np.sum(mask & (rec == r))) for r in recs], dtype=float)
        totals[label] = float(counts.sum())
        if counts.sum() > 0:
            ax.bar(range(len(recs)), counts, bottom=bottoms, color=colors[label], label=label, width=0.7)
        bottoms += counts
        for r, c in zip(recs, counts):
            per_rec[str(r)][label] = float(c)

    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels([f"rec{r:03d}" for r in recs])
    ax.set_xlabel("Recording")
    ax.set_ylabel("Failed trial count")
    ax.set_title(f"Failure modes by recording{_title_suffix(exclude_recs)}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    _save_figure(fig, out_path, "failure modes")
    return {"totals": totals, "per_rec": per_rec, "n_failures": float(np.sum(fail))}


def plot_overshoot_attempts(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, object]:
    """Target re-entry (overshoot) rate per rec, and attempts-per-trial distribution."""
    valid = _valid_mask(data)
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    target_entry = _flat(data, "JoystickTargetEntry", float)
    target_entry_final = _flat(data, "JoystickTargetEntryFinal", float)
    attempts = _flat(data, "JoystickAttemptCount", float)

    acquired = valid & np.isfinite(target_entry) & np.isfinite(target_entry_final)
    reentry = acquired & (target_entry_final > target_entry + 1e-6)

    recs = sorted(np.unique(rec).tolist())
    reentry_rate = np.array([
        float(np.sum(reentry & (rec == r))) / max(1.0, float(np.sum(acquired & (rec == r))))
        for r in recs
    ], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax1.bar(range(len(recs)), 100.0 * reentry_rate, color="#4c78a8", width=0.7)
    ax1.set_xticks(range(len(recs)))
    ax1.set_xticklabels([f"rec{r:03d}" for r in recs])
    ax1.set_ylabel("Trials with target re-entry (%)")
    ax1.set_xlabel("Recording")
    ax1.set_title("Overshoot / corrective re-entry rate")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.2)
    for i, v in enumerate(reentry_rate):
        ax1.text(i, 100.0 * v + 1.5, f"{100.0 * v:.0f}%", ha="center", va="bottom", fontsize=8)

    att_valid = attempts[valid & np.isfinite(attempts)]
    succ_valid = success[valid & np.isfinite(attempts)]
    attempt_summary: Dict[str, Dict[str, float]] = {}
    if att_valid.size:
        max_att = int(np.nanmax(att_valid))
        edges = np.arange(0.5, max_att + 1.5, 1.0)
        centers = np.arange(1, max_att + 1)
        total = np.array([float(np.sum(att_valid == c)) for c in centers], dtype=float)
        succ_n = np.array([float(np.sum((att_valid == c) & (succ_valid == 1))) for c in centers], dtype=float)
        ax2.bar(centers, total, color="#cfcfcf", width=0.8, label="all trials")
        ax2.bar(centers, succ_n, color=SUCCESS_FACE, width=0.8, label="successful")
        ax2.set_xticks(centers)
        for c, t, s in zip(centers, total, succ_n):
            attempt_summary[str(int(c))] = {
                "n_trials": float(t),
                "success_rate": float(s / t) if t > 0 else float("nan"),
            }
        _ = edges  # bins documented; centers drive the bar layout
    else:
        ax2.text(0.5, 0.5, "No attempt-count data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Attempts in trial (JoystickAttemptCount)")
    ax2.set_ylabel("Trial count")
    ax2.set_title("Attempts per trial")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Overshoot and attempts{_title_suffix(exclude_recs)}")
    fig.tight_layout()
    _save_figure(fig, out_path, "overshoot and attempts")
    return {
        "overall_reentry_rate": float(np.sum(reentry) / max(1, np.sum(acquired))),
        "per_rec_reentry_rate": {f"rec{r:03d}": float(v) for r, v in zip(recs, reentry_rate)},
        "attempts": attempt_summary,
    }


def plot_engagement(
    data: Dict[str, np.ndarray],
    out_path: Path,
    exclude_recs: List[int],
) -> Dict[str, object]:
    """Inter-trial interval, cumulative trials/reward, and per-rec reward rate."""
    valid = _valid_mask(data)
    rec = _flat(data, "Rec", int)
    start_on = _flat(data, "StartOn", float)  # absolute ms, monotonic within rec
    reward = _flat(data, "RewardReceived", float)

    recs = sorted(np.unique(rec).tolist())

    # Inter-trial-onset interval within each rec (seconds).
    itis: List[float] = []
    rate_per_rec: Dict[str, float] = {}
    for r in recs:
        m = valid & (rec == r)
        s = np.sort(start_on[m])
        if s.size >= 2:
            d = 1e-3 * np.diff(s)
            d = d[np.isfinite(d) & (d > 0)]
            itis.extend(d.tolist())
            span_min = (s[-1] - s[0]) / 60000.0
            rate_per_rec[f"rec{r:03d}"] = float(s.size / span_min) if span_min > 0 else float("nan")
        else:
            rate_per_rec[f"rec{r:03d}"] = float("nan")
    itis_arr = np.asarray(itis, dtype=float)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.4))

    # ITI histogram (clipped to drop long inter-rec breaks / pauses).
    if itis_arr.size:
        hi = float(np.nanpercentile(itis_arr, 95))
        hi = hi if (np.isfinite(hi) and hi > 0) else float(np.nanmax(itis_arr))
        clipped = np.clip(itis_arr, 0, hi)
        ax1.hist(clipped, bins=int(np.clip(np.sqrt(itis_arr.size) * 2, 12, 40)),
                 color="#72b7b2", alpha=0.85, edgecolor="white")
        med = float(np.nanmedian(itis_arr))
        ax1.axvline(med, color="#1f4e79", linewidth=2, label=f"median {med:.1f} s")
        ax1.legend(frameon=False, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No ITI data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("Inter-trial interval (s)")
    ax1.set_ylabel("Count")
    ax1.set_title("Inter-trial interval")
    ax1.grid(axis="y", alpha=0.2)

    # Cumulative trials and reward across the day (in trial order).
    order = np.arange(1, int(np.sum(valid)) + 1)
    rew_valid = reward[valid]
    ax2.plot(order, order, color="#999999", linewidth=1.6, label="trials")
    ax2.plot(order, np.cumsum(np.nan_to_num(rew_valid)), color="#d95f02", linewidth=2.2, label="rewards")
    ax2.set_xlabel("Trial order across day")
    ax2.set_ylabel("Cumulative count")
    ax2.set_title("Cumulative trials vs rewards")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    # Per-rec trial rate.
    rates = [rate_per_rec[f"rec{r:03d}"] for r in recs]
    ax3.bar(range(len(recs)), rates, color="#b279a2", width=0.7)
    ax3.set_xticks(range(len(recs)))
    ax3.set_xticklabels([f"rec{r:03d}" for r in recs])
    ax3.set_ylabel("Trials per minute")
    ax3.set_xlabel("Recording")
    ax3.set_title("Trial rate by recording")
    ax3.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Session engagement{_title_suffix(exclude_recs)}")
    fig.tight_layout()
    _save_figure(fig, out_path, "engagement")

    n_valid = int(np.sum(valid))
    return {
        "n_trials": float(n_valid),
        "median_iti_s": float(np.nanmedian(itis_arr)) if itis_arr.size else float("nan"),
        "total_reward": float(np.nansum(rew_valid)),
        "reward_rate": float(np.nansum(rew_valid) / n_valid) if n_valid else float("nan"),
        "trials_per_min_by_rec": rate_per_rec,
    }
