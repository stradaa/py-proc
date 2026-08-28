"""Plotting for the headline cross-day summary figure.

Separated from `cross_day_plots.py` (computation/orchestration) to keep each
file focused and under the repo's 500-line limit. Renders
`cross_day_metrics.png` — trial duration, hold stability vs. hold demand, path
efficiency, and successful-trial counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from .cross_day_axes import draw_mode_transitions, style_day_xaxis
    from .cross_day_hold import resolve_reference_hold, survival_at
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_axes import draw_mode_transitions, style_day_xaxis
    from cross_day_hold import resolve_reference_hold, survival_at


def plot_cross_day_metrics(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    reference_hold_s: float | None = None,
) -> None:
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)

    duration = np.asarray([row["median_successful_trial_duration_s"] for row in rows], dtype=float)
    radius = np.asarray([row["mean_successful_target_radius"] for row in rows], dtype=float)
    path_efficiency = np.asarray([row["median_path_efficiency"] for row in rows], dtype=float)
    successful_trials = np.asarray([row["n_successful_trials"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax_duration, ax_hold, ax_path, ax_successful = axes.ravel()

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

    _plot_hold_panel(ax_hold, rows, x, reference_hold_s)

    ax_path.plot(x, path_efficiency, marker="o", color="#e67e22", linewidth=2)
    ax_path.set_ylabel("Path / straight-line")
    ax_path.set_title("Median path efficiency by day  (control-law sensitive)")
    ax_path.grid(alpha=0.2)

    ax_successful.bar(x, successful_trials, color="#4c78a8", alpha=0.85, width=0.7)
    ax_successful.set_ylabel("Successful trials")
    ax_successful.set_title("Successful trials by day")
    ax_successful.grid(axis="y", alpha=0.2)

    any_transition = False
    for ax in [ax_duration, ax_hold, ax_path, ax_successful]:
        style_day_xaxis(ax, days, x)
        any_transition = draw_mode_transitions(ax, rows) or any_transition

    if any_transition:
        fig.text(
            0.5, 0.005,
            "Dashed line = cursor control-mode change; shaded = mixed-mode day. "
            "Path efficiency (and other cursor-kinematic metrics) is control-law sensitive: "
            "steps at a transition may reflect the control law, not skill.",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hold_panel(ax, rows: Sequence[Dict[str, Any]], x: np.ndarray, reference_hold_s: float | None) -> None:
    """Hold stability against the hold the task actually demanded.

    Two rates, because neither alone is interpretable: the matched-criterion
    survival is comparable across days (same criterion everywhere), while the
    task-criterion rate answers "did she meet what was asked" and moves with the
    requirement. The requirement itself is on the right axis, so a step in
    either rate can be read against the step in demand that caused it. Detail
    lives in the companion `cross_day_hold_stability.png`.
    """
    ref_s = resolve_reference_hold(rows, reference_hold_s)
    matched = 100.0 * np.asarray([survival_at(row, ref_s) for row in rows], dtype=float)
    task_rate = 100.0 * np.asarray(
        [row.get("task_criterion_hold_success_rate", np.nan) for row in rows], dtype=float
    )
    required = np.asarray([row.get("median_required_hold_s", np.nan) for row in rows], dtype=float)

    if not np.any(np.isfinite(matched)) and not np.any(np.isfinite(task_rate)):
        ax.text(0.5, 0.5, "No hold-stability data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Hold stability by day")
        return

    ax.plot(x, matched, marker="o", markersize=4, color="#2ca25f", linewidth=2,
            label=f"Holds ≥ {ref_s:.2f}s (matched criterion)")
    ax.plot(x, task_rate, marker="s", markersize=4, color="#b279a2", linewidth=1.6, alpha=0.9,
            label="Holds meeting that day's requirement")
    ax.set_ylabel("Holds surviving per target entry (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hold stability vs. hold demand")
    ax.grid(alpha=0.2)

    ax_r = ax.twinx()
    ax_r.step(x, required, where="mid", color="#444444", linestyle="--", linewidth=1.4, alpha=0.8,
              label="Required hold (s)")
    ax_r.set_ylabel("Required hold (s)", color="#444444")
    ax_r.tick_params(axis="y", labelcolor="#444444")
    finite_required = required[np.isfinite(required)]
    if finite_required.size:
        ax_r.set_ylim(0.0, max(0.6, float(np.max(finite_required)) * 1.25))

    handles, labels = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, labels + labels_r, loc="lower left", frameon=False, fontsize=7)
