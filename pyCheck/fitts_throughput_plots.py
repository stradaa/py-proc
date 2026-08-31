"""Plotting for the cross-day Fitts'-law throughput figure.

Separated from `fitts_throughput.py` (computation/orchestration) to keep each
file focused, matching the `cross_day_kinematics.py`/`cross_day_kinematics_plots.py`
split. Renders the standalone `cross_day_throughput.png`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from .cross_day_axes import draw_mode_transitions, style_day_xaxis
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_axes import draw_mode_transitions, style_day_xaxis


def _col(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


def _line(ax, x, values, color, label=None, marker="o"):
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        ax.plot(x, values, marker=marker, color=color, linewidth=2, label=label)


def plot_cross_day_throughput(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax_co, ax_grid, ax_ide, ax_n = axes.ravel()

    _line(ax_co, x, _col(rows, "co_throughput_bits_s"), "#1f4e79", "Per-attempt avg")
    _line(ax_co, x, _col(rows, "co_throughput_regression_bits_s"), "#e67e22", "Regression slope", marker="s")
    ax_co.set_ylabel("bits/s")
    ax_co.set_title("Center-out throughput (headline)")
    ax_co.legend(loc="best", frameon=False, fontsize=8)
    ax_co.grid(alpha=0.2)

    _line(ax_grid, x, _col(rows, "grid_throughput_bits_s"), "#54a24b", "Per-attempt avg")
    _line(ax_grid, x, _col(rows, "grid_throughput_regression_bits_s"), "#b279a2", "Regression slope", marker="s")
    ax_grid.set_ylabel("bits/s")
    ax_grid.set_title("Grid throughput (secondary / diagnostic)")
    ax_grid.legend(loc="best", frameon=False, fontsize=8)
    ax_grid.grid(alpha=0.2)

    _line(ax_ide, x, _col(rows, "co_median_ide_bits"), "#1f4e79", "CO median IDe")
    _line(ax_ide, x, _col(rows, "grid_median_ide_bits"), "#54a24b", "Grid median IDe")
    ax_ide.set_ylabel("bits")
    ax_ide.set_title("Effective index of difficulty by day")
    ax_ide.legend(loc="best", frameon=False, fontsize=8)
    ax_ide.grid(alpha=0.2)

    width = 0.35
    n_co = _col(rows, "n_co_bins_estimable")
    n_grid = _col(rows, "n_grid_bins_estimable")
    if np.any(np.isfinite(n_co)) or np.any(np.isfinite(n_grid)):
        ax_n.bar(x - width / 2, np.nan_to_num(n_co), width=width, color="#1f4e79", label="CO bins (of 8)")
        ax_n.bar(x + width / 2, np.nan_to_num(n_grid), width=width, color="#54a24b", label="Grid bins")
    ax_n.axhline(4, color="#888888", linestyle="--", linewidth=1, label="Regression minimum")
    ax_n.set_ylabel("Estimable bins")
    ax_n.set_title("Bins meeting min-attempts threshold")
    ax_n.legend(loc="best", frameon=False, fontsize=8)
    ax_n.grid(axis="y", alpha=0.2)

    any_transition = False
    for ax in [ax_co, ax_grid, ax_ide, ax_n]:
        style_day_xaxis(ax, days, x)
        any_transition = draw_mode_transitions(ax, rows) or any_transition

    footer = (
        "Fitts'-law (ISO 9241-9 Shannon-Welford) throughput: IDe = log2(A/We + 1), We = 4.133*SD(entry-point "
        "deviation along movement axis), MT = first_movement→first_target_entry (dwell time excluded, so "
        "this is hold-cue-duration-independent). Center-out is the headline metric (fixed radial A/W, closest "
        "analogue to standard BCI center-out throughput reporting); grid is secondary/diagnostic. Like path "
        "efficiency, this is control-law sensitive: a control-mode step can shift throughput without any "
        "underlying change in motor skill."
    )
    if any_transition:
        footer = "Dashed line = cursor control-mode change; shaded = mixed-mode day.  " + footer
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
