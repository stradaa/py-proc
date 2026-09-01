"""co_structured_traj_plots.py — the per-day structured CO trajectory figure.

One panel per peripheral CO direction, laid out on a compass grid sized to the
number of directions the day actually ran (8 -> 3x3, 16 -> 5x5, 24 -> 7x7). The
inner block is merged into a single overlay of every direction's mean path.

Directions, their order, colours and the compass cell they land on all come
from ``co_structured_directions``.

One geometry note this figure has to get right: the task region is not square
(0.5 x 1920 px wide by 0.67 x 1080 px tall on the Eevee rig) and
``target_radius_ratio`` scales by the region's *min* dimension. A target is
therefore a circle on screen but an **ellipse** in the normalized [0,1]
coordinates this figure plots in; it is drawn as that ellipse, sized from
``fitts_throughput.pixel_scale``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from pyCheck.co_structured import AttemptRecord
from pyCheck.co_structured_directions import (
    Direction,
    compass_cells,
    direction_colors,
    direction_label,
    direction_order,
    group_by_direction,
)
from pyCheck.fitts_throughput import pixel_scale
from pyCheck.plot_utils import _save_figure, _title_suffix


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _target_ellipse(xy, radius: float, scale: Tuple[float, float], **kwargs) -> Ellipse:
    """Target patch in normalized coords: a screen circle is an ellipse here."""
    scale_x, scale_y = scale
    min_dim = min(scale_x, scale_y)
    return Ellipse(
        xy,
        width=2.0 * radius * min_dim / scale_x,
        height=2.0 * radius * min_dim / scale_y,
        fill=False, **kwargs,
    )


def _draw_targets(ax, records, scale, edge_color: str = "#444444") -> None:
    seen = set()
    for record in records:
        key = (round(record.target_x, 4), round(record.target_y, 4))
        if key in seen or not np.isfinite(record.target_x):
            continue
        seen.add(key)
        radius = record.target_radius if np.isfinite(record.target_radius) else 0.03
        ax.add_patch(_target_ellipse(
            (record.target_x, record.target_y), radius, scale,
            edgecolor=edge_color, linewidth=1.0, alpha=0.8,
        ))


def _draw_center(ax, gate_radius: float, scale) -> None:
    if np.isfinite(gate_radius) and gate_radius > 0:
        ax.add_patch(_target_ellipse(
            (0.5, 0.5), gate_radius, scale,
            edgecolor="#888888", linewidth=1.0, alpha=0.7, linestyle=":",
        ))
    ax.scatter([0.5], [0.5], marker="+", color="#222222", s=50, zorder=5)


def _draw_paths(ax, records, color, linewidth: float = 1.0, alpha: float = 0.35) -> None:
    for record in records:
        if record.path is None:
            continue
        ax.plot(record.path["cursor_x"], record.path["cursor_y"],
                color=color, linewidth=linewidth, alpha=alpha)


def _setup_reach_axis(ax, title: str, *, ticks: bool) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.15)
    if not ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def _average_trajectory(
    paths: List[Dict[str, np.ndarray]], n_points: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resample each path onto n_points of its own normalized time [0,1], average x/y.

    Paths are transport-only (target_on -> hold_complete), so normalized time is
    comparable across trials: 0 is target onset and 1 is acquisition for all of
    them. Including the in-target hold would make normalized time mean different
    things on different trials and smear the mean into a hooked petal.
    """
    xs_resampled: List[np.ndarray] = []
    ys_resampled: List[np.ndarray] = []
    grid = np.linspace(0.0, 1.0, n_points)
    for path in paths:
        t = path["task_time_s"]
        if len(t) < 2 or t[-1] <= t[0]:
            continue
        t_norm = (t - t[0]) / (t[-1] - t[0])
        xs_resampled.append(np.interp(grid, t_norm, path["cursor_x"]))
        ys_resampled.append(np.interp(grid, t_norm, path["cursor_y"]))
    if not xs_resampled:
        return None
    return np.mean(xs_resampled, axis=0), np.mean(ys_resampled, axis=0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_co_structured_trajectories(
    records: List[AttemptRecord],
    out_path: Path,
    day: str,
    exclude_recs: List[int],
    task_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    peripheral_success = [
        r for r in records
        if r.schedule_phase == "peripheral" and r.outcome == "success" and np.isfinite(r.target_angle_deg)
    ]
    cfg = task_config or {}
    scale = pixel_scale(cfg) if task_config else (1.0, 1.0)
    gate_radius = float(cfg.get("center_gate_radius_ratio", np.nan))

    groups = group_by_direction(peripheral_success)
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        ax.text(0.5, 0.5, "no structured center-out data", ha="center", va="center",
                color="#999999")
        fig.suptitle(f"{day} structured center-out trajectories{_title_suffix(exclude_recs)}")
        _save_figure(fig, out_path, "co_structured_trajectories")
        return {"n_peripheral_success": 0, "directions_present": []}

    order = direction_order(groups.keys())
    size, cells = compass_cells(order)
    colors = direction_colors(order)
    multi_ring = len({d[0] for d in order}) > 1

    fig = plt.figure(figsize=(3.4 * size, 3.4 * size))
    gs = fig.add_gridspec(size, size)
    axes: Dict[Tuple[int, int], Any] = {}

    for direction in order:
        cell = cells.get(direction)
        if cell is None:  # pragma: no cover - compass_cells is a bijection
            print(f"co_structured: no compass cell for {direction_label(direction, multi_ring)}, skipping")
            continue
        row, col = cell
        ax = fig.add_subplot(gs[row, col])
        axes[cell] = ax
        group = groups[direction]
        color = colors[direction]
        _draw_paths(ax, group, color)
        drawn = [r.path for r in group if r.path is not None]
        avg = _average_trajectory(drawn)
        if avg is not None:
            ax.plot(avg[0], avg[1], color=color, linewidth=2.8, alpha=1.0, zorder=5)
        _draw_targets(ax, group, scale)
        _draw_center(ax, gate_radius, scale)
        _setup_reach_axis(
            ax, f"{direction_label(direction, multi_ring)}  (n={len(drawn)})",
            ticks=(row == size - 1 or col == 0),
        )

    for row in range(size):
        for col in range(size):
            interior = 0 < row < size - 1 and 0 < col < size - 1
            if interior or (row, col) in axes:
                continue
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999999")

    ax_center = fig.add_subplot(gs[1:size - 1, 1:size - 1])
    for direction in order:
        avg = _average_trajectory([r.path for r in groups[direction] if r.path is not None])
        if avg is None:
            continue
        ax_center.plot(avg[0], avg[1], color=colors[direction], linewidth=2.0,
                       label=direction_label(direction, multi_ring))
    for group in groups.values():
        _draw_targets(ax_center, group, scale, edge_color="#bbbbbb")
    _draw_center(ax_center, gate_radius, scale)
    _setup_reach_axis(ax_center, "All directions (mean paths)", ticks=True)
    ax_center.legend(loc="upper right", fontsize=6, ncol=2)

    fig.suptitle(f"{day} structured center-out trajectories{_title_suffix(exclude_recs)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, "co_structured_trajectories")

    return {
        "n_peripheral_success": len(peripheral_success),
        "directions_present": [direction_label(d, multi_ring) for d in order],
        "grid_size": size,
    }
