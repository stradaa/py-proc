"""Plotting for the cross-day structured-CO learning figure.

Separated from `cross_day_co_structured.py` (computation/orchestration) to
keep each file focused and under the repo's 500-line limit. Renders the
standalone `cross_day_co_structured.png`.

Layout (3x3):
  Row 1: Success rate | Reach speed | Path efficiency
         (ring-split lines when multiple rings are present)
  Row 2: Pair completion | Attempt counts | (reserved / ring summary)
  Row 3: Direction heatmap (success) | Direction heatmap (speed) |
         Direction heatmap (path efficiency)
         Rows are ring×angle when multiple rings detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from .cross_day_axes import draw_mode_transitions, style_day_xaxis
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_axes import draw_mode_transitions, style_day_xaxis


# Ring-aware color palettes (up to 3 rings)
_RING_COLORS = {
    "ring": "#4c78a8",
    "near": "#4c78a8",
    "far": "#e45756",
    "ring1": "#54a24b",
}
_RING_COLORS_LIGHT = {
    "ring": "#4c78a8",
    "near": "#4c78a8",
    "far": "#e45756",
    "ring1": "#54a24b",
}


def _line(ax, x, values, color, label=None, marker="o"):
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        ax.plot(x, values, marker=marker, color=color, linewidth=2, label=label)


def _ring_labels_across_days(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Union of ring_labels across all days, preserving order."""
    seen: dict[str, None] = {}
    for row in rows:
        for rl in row.get("ring_labels", ["ring"]):
            seen.setdefault(rl, None)
    return list(seen)


def _col_ring_scalar(rows: Sequence[Dict[str, Any]], ring_label: str, key: str) -> np.ndarray:
    values = []
    for row in rows:
        rs = row.get("ring_scalars", {}).get(ring_label, {})
        values.append(rs.get(key, np.nan))
    return np.asarray(values, dtype=float)


def _col(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


# ---------------------------------------------------------------------------
# Direction heatmap helpers
# ---------------------------------------------------------------------------

def _all_angles(rows: Sequence[Dict[str, Any]], heatmap_key: str) -> List[int]:
    """Sorted union of all angles across all days and rings for a given heatmap field."""
    angles: set[int] = set()
    for row in rows:
        data = row.get(heatmap_key, {})
        if isinstance(data, dict):
            for ring_data in data.values():
                if isinstance(ring_data, dict):
                    angles.update(int(k) for k in ring_data)
    return sorted(angles)


def _build_heatmap_matrix(
    rows: Sequence[Dict[str, Any]],
    ring_labels: List[str],
    angles: List[int],
    heatmap_key: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, List[str]]:
    """Build (n_rows, n_days) matrix and row labels for a ring×angle heatmap."""
    row_labels: List[str] = []
    for rl in ring_labels:
        for angle in angles:
            if len(ring_labels) > 1:
                row_labels.append(f"{rl} {angle}°")
            else:
                row_labels.append(f"{angle}°")

    matrix = np.full((len(row_labels), len(rows)), np.nan)
    for j, row in enumerate(rows):
        data = row.get(heatmap_key, {})
        if not isinstance(data, dict):
            continue
        for ri, rl in enumerate(ring_labels):
            ring_data = data.get(rl, {})
            for ai, angle in enumerate(angles):
                val = ring_data.get(str(angle))
                if val is not None and np.isfinite(val):
                    matrix[ri * len(angles) + ai, j] = val * scale

    return matrix, row_labels


def _plot_heatmap(
    ax, matrix: np.ndarray, row_labels: List[str], days: List[str],
    title: str, cbar_label: str, cmap: str = "viridis",
    vmin: float | None = None, vmax: float | None = None,
    ring_labels: List[str] | None = None, n_angles: int = 0,
) -> None:
    if matrix.size == 0:
        ax.text(0.5, 0.5, f"No {title.lower()} data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
        ax.set_title(title)
        ax.axis("off")
        return

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    style_day_xaxis(ax, days, xlabel=None)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7)

    # Draw horizontal separator between rings
    if ring_labels is not None and len(ring_labels) > 1 and n_angles > 0:
        for ri in range(1, len(ring_labels)):
            ax.axhline(ri * n_angles - 0.5, color="white", linewidth=2)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_cross_day_co_structured(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)
    ring_labels = _ring_labels_across_days(rows)
    multi_ring = len(ring_labels) > 1

    fig, axes = plt.subplots(3, 3, figsize=(18.0, 13.5))
    (ax_success, ax_speed, ax_path,
     ax_pairs, ax_attempts, ax_ring_summary,
     ax_heat_success, ax_heat_speed, ax_heat_patheff) = axes.ravel()

    # --- Row 1: Scalar trends (ring-split when applicable) ---

    # (1) Success rate
    if multi_ring:
        for rl in ring_labels:
            color = _RING_COLORS.get(rl, "#4c78a8")
            _line(ax_success, x, 100.0 * _col_ring_scalar(rows, rl, "success_rate"),
                  color, f"{rl.capitalize()}")
    else:
        _line(ax_success, x, 100.0 * _col(rows, "structured_success_rate"), "#4c78a8", "Structured")
        _line(ax_success, x, 100.0 * _col(rows, "random_success_rate"), "#8a8a8a", "Random")
    ax_success.set_ylabel("Success rate (%)")
    ax_success.set_ylim(0, 105)
    ax_success.set_title("Success rate" + (" by ring" if multi_ring else " (structured vs random)"))
    ax_success.legend(loc="best", frameon=False, fontsize=8)

    # (2) Speed: median reach duration
    if multi_ring:
        for rl in ring_labels:
            color = _RING_COLORS.get(rl, "#e67e22")
            _line(ax_speed, x, _col_ring_scalar(rows, rl, "median_duration_s"),
                  color, f"{rl.capitalize()}")
    else:
        _line(ax_speed, x, _col(rows, "median_duration_s"), "#e67e22")
    ax_speed.set_ylabel("Median duration (s)")
    ax_speed.set_title("Reach speed" + (" by ring" if multi_ring else " (successful peripheral)"))
    if multi_ring:
        ax_speed.legend(loc="best", frameon=False, fontsize=8)

    # (3) Path efficiency
    if multi_ring:
        for rl in ring_labels:
            color = _RING_COLORS.get(rl, "#54a24b")
            _line(ax_path, x, _col_ring_scalar(rows, rl, "median_path_efficiency"),
                  color, f"{rl.capitalize()}")
    else:
        _line(ax_path, x, _col(rows, "median_path_efficiency"), "#54a24b")
    ax_path.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_path.set_ylabel("Path length / straight-line")
    ax_path.set_title("Path efficiency" + (" by ring" if multi_ring else " (successful peripheral)"))
    if multi_ring:
        ax_path.legend(loc="best", frameon=False, fontsize=8)

    # --- Row 2: Pairs, counts, ring summary ---

    # (4) Pair completion rate
    pair_rate = 100.0 * _col(rows, "pair_completion_rate")
    _line(ax_pairs, x, pair_rate, "#b279a2")
    if not np.any(np.isfinite(pair_rate)):
        ax_pairs.text(
            0.5, 0.5, "No pair-adjacency data\n(pre pair-boundary executor build)",
            ha="center", va="center", transform=ax_pairs.transAxes, fontsize=8, color="#888888",
        )
    ax_pairs.set_ylabel("Pair completion rate (%)")
    ax_pairs.set_ylim(0, 105)
    ax_pairs.set_title("Center\u2192peripheral pair completion")

    # (5) Attempt counts
    if multi_ring:
        width = 0.7 / max(len(ring_labels), 1)
        offsets = np.linspace(-0.35 + width / 2, 0.35 - width / 2, len(ring_labels))
        for ri, rl in enumerate(ring_labels):
            n_vals = _col_ring_scalar(rows, rl, "n_attempts")
            color = _RING_COLORS.get(rl, "#4c78a8")
            ax_attempts.bar(x + offsets[ri], np.nan_to_num(n_vals), width=width,
                            color=color, alpha=0.85, label=rl.capitalize())
        ax_attempts.legend(loc="best", frameon=False, fontsize=8)
    else:
        structured_n = _col(rows, "n_structured_attempts")
        random_n = _col(rows, "n_random_attempts")
        width = 0.35
        ax_attempts.bar(x - width / 2, structured_n, width=width, color="#4c78a8",
                        alpha=0.85, label="Structured")
        ax_attempts.bar(x + width / 2, random_n, width=width, color="#8a8a8a",
                        alpha=0.85, label="Random")
        ax_attempts.legend(loc="best", frameon=False, fontsize=8)
    ax_attempts.set_ylabel("Attempts")
    ax_attempts.set_title("Attempts by day")

    # (6) Ring summary: structured vs random success (always show, even multi-ring)
    if multi_ring:
        _line(ax_ring_summary, x, 100.0 * _col(rows, "structured_success_rate"),
              "#4c78a8", "Structured (all)")
        _line(ax_ring_summary, x, 100.0 * _col(rows, "random_success_rate"),
              "#8a8a8a", "Random")
        ax_ring_summary.set_ylabel("Success rate (%)")
        ax_ring_summary.set_ylim(0, 105)
        ax_ring_summary.set_title("Structured vs random (all rings)")
        ax_ring_summary.legend(loc="best", frameon=False, fontsize=8)
    else:
        # Single ring: show structured vs random success rate as text summary
        _line(ax_ring_summary, x, 100.0 * _col(rows, "structured_success_rate"),
              "#4c78a8", "Structured")
        _line(ax_ring_summary, x, 100.0 * _col(rows, "random_success_rate"),
              "#8a8a8a", "Random")
        ax_ring_summary.set_ylabel("Success rate (%)")
        ax_ring_summary.set_ylim(0, 105)
        ax_ring_summary.set_title("Structured vs random success")
        ax_ring_summary.legend(loc="best", frameon=False, fontsize=8)

    for ax in (ax_success, ax_speed, ax_path, ax_pairs, ax_attempts, ax_ring_summary):
        style_day_xaxis(ax, days, x)
        ax.grid(alpha=0.2)
        draw_mode_transitions(ax, rows)

    # --- Row 3: Direction-resolved heatmaps ---

    # Success heatmap
    angles_success = _all_angles(rows, "ring_dir_success")
    if angles_success:
        mat_s, labels_s = _build_heatmap_matrix(
            rows, ring_labels, angles_success, "ring_dir_success", scale=100.0)
        _plot_heatmap(ax_heat_success, mat_s, labels_s, days,
                      "Direction success rate (%)", "Success %", cmap="viridis",
                      vmin=0, vmax=100,
                      ring_labels=ring_labels if multi_ring else None,
                      n_angles=len(angles_success))
    else:
        ax_heat_success.text(0.5, 0.5, "No per-direction data", ha="center", va="center",
                             transform=ax_heat_success.transAxes, fontsize=8, color="#888888")
        ax_heat_success.set_title("Direction success rate (%)")
        ax_heat_success.axis("off")

    # Speed heatmap
    angles_speed = _all_angles(rows, "ring_dir_speed")
    if angles_speed:
        mat_sp, labels_sp = _build_heatmap_matrix(
            rows, ring_labels, angles_speed, "ring_dir_speed")
        _plot_heatmap(ax_heat_speed, mat_sp, labels_sp, days,
                      "Direction reach speed (s)", "Duration (s)", cmap="magma_r",
                      ring_labels=ring_labels if multi_ring else None,
                      n_angles=len(angles_speed))
    else:
        ax_heat_speed.text(0.5, 0.5, "No per-direction speed data", ha="center", va="center",
                           transform=ax_heat_speed.transAxes, fontsize=8, color="#888888")
        ax_heat_speed.set_title("Direction reach speed (s)")
        ax_heat_speed.axis("off")

    # Path efficiency heatmap
    angles_pe = _all_angles(rows, "ring_dir_path_eff")
    if angles_pe:
        mat_pe, labels_pe = _build_heatmap_matrix(
            rows, ring_labels, angles_pe, "ring_dir_path_eff")
        _plot_heatmap(ax_heat_patheff, mat_pe, labels_pe, days,
                      "Direction path efficiency", "Path len / straight-line", cmap="RdYlGn_r",
                      vmin=1.0,
                      ring_labels=ring_labels if multi_ring else None,
                      n_angles=len(angles_pe))
    else:
        ax_heat_patheff.text(0.5, 0.5, "No per-direction path eff data", ha="center", va="center",
                             transform=ax_heat_patheff.transAxes, fontsize=8, color="#888888")
        ax_heat_patheff.set_title("Direction path efficiency")
        ax_heat_patheff.axis("off")

    # --- Title and footer ---
    any_transition = any(draw_mode_transitions(ax, rows) for ax in [ax_heat_success])

    fig.suptitle(
        "Structured center-out (CO) learning across days"
        + (" — multi-ring" if multi_ring else "")
        + "\ndashed = control-mode change; path efficiency & trajectory shape are control-law sensitive",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
