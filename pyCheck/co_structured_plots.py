"""co_structured_plots.py — figures for co_structured.py's structured CO data.

Two figures, both driven by the same list of ``AttemptRecord``:

- ``plot_co_structured_trajectories`` — a 3x3 compass grid, one panel per
  peripheral CO target laid out by true screen direction (angle from the
  task center at (0.5, 0.5)), each showing every successful reach to that
  target plus its time-normalized average trajectory. The unused center
  cell overlays all 8 average trajectories for a quick direction comparison.
- ``plot_co_structured_metrics`` — three bar panels: success rate (per
  peripheral target / structured vs random / pair completion), median reach
  duration by direction, and median path efficiency by direction.

Both figures order/color peripheral targets by ``target_angle_deg`` rather
than ``target_name`` — target names are arbitrary experimenter strings, not
a reliable proxy for direction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyCheck.co_structured import AttemptRecord, RANDOM_COLOR, TARGET_PALETTE
from pyCheck.plot_utils import _save_figure, _title_suffix

# Compass-grid position (row, col) for each peripheral target's angle, in a
# 3x3 grid with (1, 1) reserved for the combined direction overlay.
_COMPASS_GRID = {
    90: (0, 1), 45: (0, 2),
    180: (1, 0), 0: (1, 2),
    225: (2, 0), 270: (2, 1), 315: (2, 2),
    135: (0, 0),
}


def _success_rate(records: List[AttemptRecord]) -> float:
    if not records:
        return np.nan
    return float(np.mean([r.outcome == "success" for r in records]))


def _draw_targets(ax, records: List[AttemptRecord], edge_color: str = "#444444") -> None:
    seen = set()
    for record in records:
        key = (round(record.target_x, 4), round(record.target_y, 4))
        if key in seen or not np.isfinite(record.target_x):
            continue
        seen.add(key)
        radius = record.target_radius if np.isfinite(record.target_radius) else 0.03
        ax.add_patch(plt.Circle(
            (record.target_x, record.target_y), radius,
            fill=False, edgecolor=edge_color, linewidth=1.0, alpha=0.8,
        ))


def _draw_paths(
    ax,
    records: List[AttemptRecord],
    color_of,
    *,
    success_only: bool = False,
    linewidth: float = 1.2,
    alpha: float = 0.75,
) -> None:
    for record in records:
        if record.path is None:
            continue
        success = record.outcome == "success"
        if success_only and not success:
            continue
        ax.plot(
            record.path["cursor_x"], record.path["cursor_y"],
            color=color_of(record),
            linewidth=linewidth if success else linewidth * 0.6,
            alpha=alpha if success else alpha * 0.4,
            linestyle="-" if success else "--",
        )


def _setup_reach_axis(ax, title: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.15)


def _average_trajectory(
    paths: List[Dict[str, np.ndarray]], n_points: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resample each path onto n_points of its own normalized time [0,1], average x/y."""
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


def _angle_key(record: AttemptRecord) -> int:
    return int(round(record.target_angle_deg)) % 360


def _target_color_by_angle(angles: List[int]) -> Dict[int, str]:
    return {angle: TARGET_PALETTE[i % len(TARGET_PALETTE)] for i, angle in enumerate(angles)}


def plot_co_structured_trajectories(
    records: List[AttemptRecord],
    out_path: Path,
    day: str,
    exclude_recs: List[int],
) -> Dict[str, object]:
    peripheral_success = [
        r for r in records
        if r.schedule_phase == "peripheral" and r.outcome == "success" and np.isfinite(r.target_angle_deg)
    ]
    groups: Dict[int, List[AttemptRecord]] = {}
    for record in peripheral_success:
        groups.setdefault(_angle_key(record), []).append(record)
    angle_order = sorted(groups.keys())
    target_color = _target_color_by_angle(angle_order)

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    used_cells = set()

    for angle, group in groups.items():
        if angle not in _COMPASS_GRID:
            continue
        row, col = _COMPASS_GRID[angle]
        used_cells.add((row, col))
        ax = axes[row, col]
        color = target_color[angle]
        _draw_paths(ax, group, lambda r: color, success_only=True, linewidth=1.0, alpha=0.35)
        avg = _average_trajectory([r.path for r in group if r.path is not None])
        if avg is not None:
            ax.plot(avg[0], avg[1], color=color, linewidth=2.8, alpha=1.0, zorder=5)
        _draw_targets(ax, group)
        ax.scatter([0.5], [0.5], marker="+", color="#222222", s=50, zorder=5)
        _setup_reach_axis(ax, f"{angle}°  (n={len(group)})")

    for row in range(3):
        for col in range(3):
            if (row, col) == (1, 1) or (row, col) in used_cells:
                continue
            axes[row, col].axis("off")
            axes[row, col].text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=axes[row, col].transAxes, fontsize=9, color="#999999",
            )

    ax_center = axes[1, 1]
    for angle in angle_order:
        group = groups[angle]
        avg = _average_trajectory([r.path for r in group if r.path is not None])
        if avg is None:
            continue
        ax_center.plot(avg[0], avg[1], color=target_color[angle], linewidth=2.0, label=f"{angle}°")
    for group in groups.values():
        _draw_targets(ax_center, group, edge_color="#bbbbbb")
    ax_center.scatter([0.5], [0.5], marker="+", color="#222222", s=50, zorder=5)
    _setup_reach_axis(ax_center, "All directions (mean paths)")
    if angle_order:
        ax_center.legend(loc="upper right", fontsize=6, ncol=2)

    fig.suptitle(f"{day} structured center-out trajectories{_title_suffix(exclude_recs)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, "co_structured_trajectories")

    return {
        "n_peripheral_success": len(peripheral_success),
        "directions_present": angle_order,
    }


def _plot_performance_panel(
    ax,
    records: List[AttemptRecord],
    angle_order: List[int],
    target_color: Dict[int, str],
) -> Dict[str, object]:
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    center = [r for r in records if r.schedule_phase == "center"]
    sequence = [r for r in records if r.schedule_phase == "sequence"]
    structured = peripheral + center + sequence
    random_inserts = [r for r in records if r.schedule_phase == "random"]
    pairs = {r.pair_id for r in records if r.pair_id >= 0}
    complete_pairs = {r.pair_id for r in records if r.pair_id >= 0 and r.pair_complete}

    labels: List[str] = []
    values: List[float] = []
    counts: List[int] = []
    colors: List[str] = []
    per_target: Dict[str, Dict[str, object]] = {}
    for angle in angle_order:
        group = [r for r in peripheral if np.isfinite(r.target_angle_deg) and _angle_key(r) == angle]
        name = next((r.target_name or f"target {r.target_index}" for r in group), f"{angle}°")
        labels.append(f"{angle}°")
        rate = _success_rate(group)
        values.append(rate)
        counts.append(len(group))
        colors.append(target_color[angle])
        per_target[name] = {"n": len(group), "success_rate": rate if np.isfinite(rate) else None}
    for label, group, color in (
        ("structured", structured, "#333333"),
        ("random", random_inserts, RANDOM_COLOR),
    ):
        labels.append(label)
        values.append(_success_rate(group))
        counts.append(len(group))
        colors.append(color)
    labels.append("pairs complete")
    values.append(len(complete_pairs) / len(pairs) if pairs else np.nan)
    counts.append(len(pairs))
    colors.append("#54a24b")

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=0.85)
    for xi, value, count in zip(x, values, counts):
        if np.isfinite(value):
            ax.text(xi, value + 0.02, f"{value:.0%}\nn={count}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("success rate")
    ax.set_title("Performance: per target / structured vs random / pairs", fontsize=10)
    ax.grid(True, axis="y", alpha=0.15)

    return {
        "n_attempts": len(records),
        "n_by_phase": {
            phase: sum(1 for r in records if r.schedule_phase == phase)
            for phase in ("center", "peripheral", "sequence", "random")
        },
        "structured_success_rate": _success_rate(structured) if structured else None,
        "random_success_rate": _success_rate(random_inserts) if random_inserts else None,
        "per_peripheral_target": per_target,
        "n_pairs": len(pairs),
        "n_complete_pairs": len(complete_pairs),
        "pair_completion_rate": (len(complete_pairs) / len(pairs)) if pairs else None,
    }


def _bar_with_iqr(ax, angle_order: List[int], values_by_angle: Dict[int, np.ndarray], target_color: Dict[int, str]):
    medians, los, his, ns = [], [], [], []
    for angle in angle_order:
        vals = values_by_angle.get(angle, np.array([]))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, med, hi = np.percentile(vals, [25, 50, 75])
        else:
            lo = med = hi = np.nan
        medians.append(med)
        los.append(med - lo if np.isfinite(med) else 0.0)
        his.append(hi - med if np.isfinite(med) else 0.0)
        ns.append(vals.size)

    x = np.arange(len(angle_order))
    colors = [target_color[angle] for angle in angle_order]
    ax.bar(x, medians, color=colors, alpha=0.85)
    ax.errorbar(x, medians, yerr=[los, his], fmt="none", ecolor="#333333", capsize=3, linewidth=1.0)
    finite_medians = [m for m in medians if np.isfinite(m)]
    pad = 0.03 * (max(finite_medians) if finite_medians else 1.0)
    for i, (med, hi, n) in enumerate(zip(medians, his, ns)):
        if np.isfinite(med):
            ax.text(i, med + hi + pad, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{angle}°" for angle in angle_order], fontsize=8)
    return medians, ns


def _plot_speed_panel(
    ax,
    peripheral_success: List[AttemptRecord],
    angle_order: List[int],
    target_color: Dict[int, str],
) -> Dict[str, object]:
    by_angle: Dict[int, np.ndarray] = {}
    for angle in angle_order:
        vals = np.asarray([r.duration_s for r in peripheral_success if _angle_key(r) == angle], dtype=float)
        by_angle[angle] = vals

    medians, ns = _bar_with_iqr(ax, angle_order, by_angle, target_color)
    ax.set_ylabel("Median duration (s)")
    ax.set_title("Speed by direction (successful reaches)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.15)

    return {
        str(angle): {
            "n": ns[i],
            "median_duration_s": medians[i] if np.isfinite(medians[i]) else None,
        }
        for i, angle in enumerate(angle_order)
    }


def _plot_path_efficiency_panel(
    ax,
    peripheral_success: List[AttemptRecord],
    angle_order: List[int],
    target_color: Dict[int, str],
) -> Dict[str, object]:
    by_angle: Dict[int, np.ndarray] = {}
    for angle in angle_order:
        vals = np.asarray(
            [r.path_efficiency for r in peripheral_success if _angle_key(r) == angle and np.isfinite(r.path_efficiency)],
            dtype=float,
        )
        by_angle[angle] = vals

    medians, ns = _bar_with_iqr(ax, angle_order, by_angle, target_color)
    finite_medians = [m for m in medians if np.isfinite(m)]
    upper = max([1.3] + [m * 1.15 for m in finite_medians])
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6, label="straight line")
    ax.set_ylim(min(0.9, *finite_medians) if finite_medians else 0.9, upper)
    ax.set_ylabel("Path length / straight-line distance")
    ax.set_title("Path efficiency by direction (successful reaches)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(loc="upper right", fontsize=7)

    return {
        str(angle): {
            "n": ns[i],
            "median_path_efficiency": medians[i] if np.isfinite(medians[i]) else None,
        }
        for i, angle in enumerate(angle_order)
    }


def plot_co_structured_metrics(
    records: List[AttemptRecord],
    out_path: Path,
    day: str,
    exclude_recs: List[int],
) -> Dict[str, object]:
    peripheral_success = [
        r for r in records
        if r.schedule_phase == "peripheral" and r.outcome == "success" and np.isfinite(r.target_angle_deg)
    ]
    angle_order = sorted({_angle_key(r) for r in peripheral_success})
    target_color = _target_color_by_angle(angle_order)

    fig, (ax_perf, ax_speed, ax_path) = plt.subplots(1, 3, figsize=(16, 5.5))

    metrics = _plot_performance_panel(ax_perf, records, angle_order, target_color)
    metrics["speed_by_direction"] = _plot_speed_panel(ax_speed, peripheral_success, angle_order, target_color)
    metrics["path_efficiency_by_direction"] = _plot_path_efficiency_panel(
        ax_path, peripheral_success, angle_order, target_color
    )

    fig.suptitle(f"{day} structured center-out metrics{_title_suffix(exclude_recs)}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, out_path, "co_structured_metrics")

    return metrics
