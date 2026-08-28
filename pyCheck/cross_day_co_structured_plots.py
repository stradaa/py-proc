"""Plotting for the cross-day structured-CO learning figure.

Separated from `cross_day_co_structured.py` (computation/orchestration) to
keep each file focused and under the repo's 500-line limit. Renders the
standalone `cross_day_co_structured.png` — a 2x3 panel of day-over-day
structured-CO learning trends plus a direction-resolved success-rate heatmap.
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


def _line(ax, x, values, color, label=None, marker="o"):
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        ax.plot(x, values, marker=marker, color=color, linewidth=2, label=label)


def plot_cross_day_co_structured(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)

    def col(key):
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    (ax_success, ax_speed, ax_path,
     ax_pairs, ax_attempts, ax_heat) = axes.ravel()

    # (1) Structured vs random success rate
    _line(ax_success, x, 100.0 * col("structured_success_rate"), "#4c78a8", "Structured")
    _line(ax_success, x, 100.0 * col("random_success_rate"), "#8a8a8a", "Random")
    ax_success.set_ylabel("Success rate (%)")
    ax_success.set_ylim(0, 105)
    ax_success.set_title("Structured vs random success rate")
    ax_success.legend(loc="best", frameon=False, fontsize=8)

    # (2) Speed: median reach duration (successful peripheral only)
    _line(ax_speed, x, col("median_duration_s"), "#e67e22")
    ax_speed.set_ylabel("Median duration (s)")
    ax_speed.set_title("Speed (successful peripheral reaches)")

    # (3) Path efficiency (successful peripheral only)
    _line(ax_path, x, col("median_path_efficiency"), "#54a24b")
    ax_path.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_path.set_ylabel("Path length / straight-line")
    ax_path.set_title("Path efficiency (successful peripheral reaches)")

    # (4) Center->peripheral pair completion rate
    pair_rate = 100.0 * col("pair_completion_rate")
    _line(ax_pairs, x, pair_rate, "#b279a2")
    if not np.any(np.isfinite(pair_rate)):
        ax_pairs.text(
            0.5, 0.5, "No pair-adjacency data\n(pre pair-boundary executor build)",
            ha="center", va="center", transform=ax_pairs.transAxes, fontsize=8, color="#888888",
        )
    ax_pairs.set_ylabel("Pair completion rate (%)")
    ax_pairs.set_ylim(0, 105)
    ax_pairs.set_title("Center→peripheral pair completion")

    # (5) Attempt counts by day (context for the rates above)
    structured_n = col("n_structured_attempts")
    random_n = col("n_random_attempts")
    width = 0.35
    ax_attempts.bar(x - width / 2, structured_n, width=width, color="#4c78a8", alpha=0.85, label="Structured")
    ax_attempts.bar(x + width / 2, random_n, width=width, color="#8a8a8a", alpha=0.85, label="Random")
    ax_attempts.set_ylabel("Attempts")
    ax_attempts.set_title("Attempts by day")
    ax_attempts.legend(loc="best", frameon=False, fontsize=8)

    for ax in (ax_success, ax_speed, ax_path, ax_pairs, ax_attempts):
        style_day_xaxis(ax, days, x)
        ax.grid(alpha=0.2)
        draw_mode_transitions(ax, rows)

    # (6) Direction-resolved success-rate heatmap
    _plot_direction_heatmap(ax_heat, rows, days)

    fig.suptitle(
        "Structured center-out (CO) learning across days\n"
        "dashed = control-mode change; path efficiency & trajectory shape are control-law sensitive",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_direction_heatmap(ax, rows, days) -> None:
    angles = sorted({int(k) for row in rows for k in row.get("dir_success", {})})
    if not angles:
        ax.text(0.5, 0.5, "No per-direction data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Direction-resolved success rate")
        ax.axis("off")
        return

    matrix = np.full((len(angles), len(days)), np.nan)
    for j, row in enumerate(rows):
        ds = row.get("dir_success", {})
        for i, angle in enumerate(angles):
            if str(angle) in ds:
                matrix[i, j] = 100.0 * ds[str(angle)]

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="lower")
    style_day_xaxis(ax, days, xlabel=None)
    ax.set_yticks(np.arange(len(angles)))
    ax.set_yticklabels([f"{angle}°" for angle in angles], fontsize=8)
    ax.set_xlabel("Day")
    ax.set_title("Direction-resolved success rate (%)")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")
