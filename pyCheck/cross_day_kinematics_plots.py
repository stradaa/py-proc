"""Plotting for the cross-day learning-kinematics figure.

Separated from `cross_day_kinematics.py` (computation/orchestration) to keep
each file focused and under the repo's 500-line limit. Renders the standalone
`cross_day_kinematics.png` — a 3×3 panel of motor-learning metrics plus a
direction-resolved success-rate heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _line(ax, x, values, color, label=None, marker="o"):
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        ax.plot(x, values, marker=marker, color=color, linewidth=2, label=label)


def plot_cross_day_kinematics(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)

    def col(key):
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 13.0))
    (ax_decomp, ax_rtcv, ax_dir,
     ax_submv, ax_sparc, ax_var,
     ax_engage, ax_throughput, ax_heat) = axes.ravel()

    # (1) Movement-time decomposition
    _line(ax_decomp, x, col("median_reaction_time_ms"), "#4c78a8", "Reaction (target_on→move)")
    _line(ax_decomp, x, col("median_transport_time_ms"), "#f58518", "Transport (move→entry)")
    _line(ax_decomp, x, col("median_acquisition_time_ms"), "#54a24b", "Acquisition (entry→hold)")
    ax_decomp.set_ylabel("Median time (ms)")
    ax_decomp.set_title("Movement-time decomposition")
    ax_decomp.legend(loc="best", frameon=False, fontsize=8)

    # (2) Reaction-time variability
    _line(ax_rtcv, x, col("reaction_time_cv"), "#b279a2")
    ax_rtcv.set_ylabel("RT coefficient of variation")
    ax_rtcv.set_title("Reaction-time variability (lower = more consistent)")

    # (3) Initial direction error
    _line(ax_dir, x, col("median_initial_direction_error_deg"), "#e45756")
    ax_dir.set_ylabel("Degrees")
    ax_dir.set_title("Initial reach-direction error (lower = better aim)")

    # (4) Submovement count
    _line(ax_submv, x, col("median_submovement_count"), "#9467bd")
    ax_submv.set_ylabel("Median submovements / reach")
    ax_submv.set_title("Submovement count (lower = more ballistic)")

    # (5) SPARC smoothness
    _line(ax_sparc, x, col("median_sparc"), "#1f9e89")
    ax_sparc.set_ylabel("SPARC")
    ax_sparc.set_title("Movement smoothness (higher = smoother)")

    # (6) Variability
    _line(ax_var, x, col("trajectory_variability"), "#0b3954", "Trajectory variability")
    _line(ax_var, x, col("endpoint_scatter"), "#e67e22", "Endpoint scatter")
    ax_var.set_ylabel("Cross-trial spread (norm. units)")
    ax_var.set_title("Trajectory & endpoint variability (lower = more consistent)")
    ax_var.legend(loc="best", frameon=False, fontsize=8)

    # (7) Engagement: attempts/trial + idle rate
    _line(ax_engage, x, col("mean_attempts_per_trial"), "#4c78a8", "Attempts / trial")
    ax_engage.set_ylabel("Attempts / trial", color="#4c78a8")
    ax_engage.set_title("Engagement")
    ax_engage_r = ax_engage.twinx()
    idle = 100.0 * col("idle_timeout_rate")
    _line(ax_engage_r, x, idle, "#d62728", "Idle-timeout rate", marker="s")
    ax_engage_r.set_ylabel("Idle-timeout rate (%)", color="#d62728")
    ax_engage_r.set_ylim(0, max(5.0, float(np.nanmax(idle)) * 1.2) if np.any(np.isfinite(idle)) else 5.0)

    # (8) Throughput
    _line(ax_throughput, x, col("trials_per_min"), "#2ca25f")
    ax_throughput.set_ylabel("Trials attempted / min")
    ax_throughput.set_title("Throughput")

    for ax in (ax_decomp, ax_rtcv, ax_dir, ax_submv, ax_sparc, ax_var, ax_engage, ax_throughput):
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=25, ha="right")
        ax.set_xlabel("Day")
        ax.grid(alpha=0.2)

    # (9) Direction-resolved success-rate heatmap
    _plot_direction_heatmap(ax_heat, rows, days)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_direction_heatmap(ax, rows, days) -> None:
    tidxs = sorted({int(k) for row in rows for k in row.get("dir_success", {})})
    if not tidxs:
        ax.text(0.5, 0.5, "No per-target data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Direction-resolved success rate")
        ax.axis("off")
        return

    matrix = np.full((len(tidxs), len(rows)), np.nan)
    for j, row in enumerate(rows):
        ds = row.get("dir_success", {})
        for i, t in enumerate(tidxs):
            if str(t) in ds:
                matrix[i, j] = 100.0 * ds[str(t)]

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="lower")
    ax.set_xticks(np.arange(len(days)))
    ax.set_xticklabels(days, rotation=25, ha="right")
    labels = []
    for t in tidxs:
        xy = next((row["dir_xy"].get(str(t)) for row in rows if str(t) in row.get("dir_xy", {})), None)
        labels.append(f"tgt{t}\n({xy[0]:.2f},{xy[1]:.2f})" if xy else f"tgt{t}")
    ax.set_yticks(np.arange(len(tidxs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Day")
    ax.set_title("Direction-resolved success rate (%)")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")
