"""co_structured_plots.py — the per-day structured CO metrics figure.

Three bar panels driven by ``co_structured.py``'s ``AttemptRecord`` list:
success rate (per peripheral direction / structured vs random / pair
completion), median reach duration by direction, and median path efficiency by
direction.

Directions are keyed by ``(ring_label, angle)`` and the bars are grouped by
ring, because two rings are two different reach amplitudes and comparing their
durations or efficiencies against each other is meaningless. Target *names* are
arbitrary experimenter strings and are never used as a proxy for direction.

The trajectory figure lives in ``co_structured_traj_plots.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyCheck.co_structured import AttemptRecord, RANDOM_COLOR
from pyCheck.co_structured_directions import (
    Direction,
    direction_colors,
    direction_label,
    direction_order,
    group_by_direction,
    ring_order,
)
from pyCheck.plot_utils import _save_figure, _title_suffix


def _success_rate(records: Sequence[AttemptRecord]) -> float:
    if not records:
        return np.nan
    return float(np.mean([r.outcome == "success" for r in records]))


def _direction_positions(order: Sequence[Direction]) -> Tuple[Dict[Direction, float], Dict[str, float]]:
    """x position per direction, one contiguous block per ring separated by a gap.

    Returns ``(x_by_direction, block_center_by_ring)``; the block centers anchor
    the ring labels under each group.
    """
    x_by_direction: Dict[Direction, float] = {}
    block_center: Dict[str, float] = {}
    x = 0.0
    for ring in ring_order([d[0] for d in order]):
        block = [d for d in order if d[0] == ring]
        if not block:
            continue
        start = x
        for direction in block:
            x_by_direction[direction] = x
            x += 1.0
        block_center[ring] = (start + x - 1.0) / 2.0
        x += 1.0  # gap between rings
    return x_by_direction, block_center


def _annotate_rings(ax, block_center: Dict[str, float], multi_ring: bool) -> None:
    """Ring name under each block of direction bars."""
    if not multi_ring:
        return
    for ring, center in block_center.items():
        ax.annotate(ring, xy=(center, 0), xytext=(0, -36), textcoords="offset points",
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=9, color="#444444")


def _label_direction_axis(ax, order, positions, block_center, multi_ring: bool) -> None:
    ax.set_xticks([positions[d] for d in order])
    ax.set_xticklabels([f"{d[1]:.0f}\u00b0" for d in order], fontsize=8, rotation=45, ha="right")
    _annotate_rings(ax, block_center, multi_ring)


def _plot_performance_panel(
    ax,
    records: List[AttemptRecord],
    order: Sequence[Direction],
    colors: Dict[Direction, str],
    multi_ring: bool,
) -> Dict[str, object]:
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    center = [r for r in records if r.schedule_phase == "center"]
    sequence = [r for r in records if r.schedule_phase == "sequence"]
    structured = peripheral + center + sequence
    random_inserts = [r for r in records if r.schedule_phase == "random"]
    pairs = {r.pair_id for r in records if r.pair_id >= 0}
    complete_pairs = {r.pair_id for r in records if r.pair_id >= 0 and r.pair_complete}

    by_direction = group_by_direction(peripheral)
    positions, block_center = _direction_positions(order)

    xs: List[float] = []
    labels: List[str] = []
    values: List[float] = []
    counts: List[int] = []
    bar_colors: List[str] = []
    per_target: Dict[str, Dict[str, object]] = {}
    for direction in order:
        group = by_direction.get(direction, [])
        rate = _success_rate(group)
        xs.append(positions[direction])
        labels.append(direction_label(direction, multi_ring))
        values.append(rate)
        counts.append(len(group))
        bar_colors.append(colors[direction])
        per_target[direction_label(direction, multi_ring)] = {
            "n": len(group),
            "success_rate": rate if np.isfinite(rate) else None,
        }

    x = max(xs) + 2.0 if xs else 0.0
    summary_xs: List[float] = []
    for label, group, color in (
        ("structured", structured, "#333333"),
        ("random", random_inserts, RANDOM_COLOR),
        ("pairs complete", None, "#54a24b"),
    ):
        summary_xs.append(x)
        xs.append(x)
        labels.append(label)
        if group is None:
            values.append(len(complete_pairs) / len(pairs) if pairs else np.nan)
            counts.append(len(pairs))
        else:
            values.append(_success_rate(group))
            counts.append(len(group))
        bar_colors.append(color)
        x += 1.0

    ax.bar(xs, values, color=bar_colors, alpha=0.85)
    for xi, value, count in zip(xs, values, counts):
        if np.isfinite(value):
            ax.text(xi, value + 0.02, f"{value:.0%}\nn={count}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{d[1]:.0f}\u00b0" for d in order] + labels[len(order):],
        fontsize=8, rotation=45, ha="right",
    )
    _annotate_rings(ax, block_center, multi_ring)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("success rate")
    ax.set_title("Performance: per direction / structured vs random / pairs", fontsize=10)
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


def _bar_with_iqr(
    ax,
    order: Sequence[Direction],
    values_by_direction: Dict[Direction, np.ndarray],
    colors: Dict[Direction, str],
    multi_ring: bool,
):
    medians, los, his, ns = [], [], [], []
    for direction in order:
        vals = values_by_direction.get(direction, np.array([]))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, med, hi = np.percentile(vals, [25, 50, 75])
        else:
            lo = med = hi = np.nan
        medians.append(med)
        los.append(med - lo if np.isfinite(med) else 0.0)
        his.append(hi - med if np.isfinite(med) else 0.0)
        ns.append(vals.size)

    positions, block_center = _direction_positions(order)
    x = [positions[d] for d in order]
    ax.bar(x, medians, color=[colors[d] for d in order], alpha=0.85)
    ax.errorbar(x, medians, yerr=[los, his], fmt="none", ecolor="#333333", capsize=3, linewidth=1.0)
    finite_medians = [m for m in medians if np.isfinite(m)]
    pad = 0.03 * (max(finite_medians) if finite_medians else 1.0)
    for xi, med, hi, n in zip(x, medians, his, ns):
        if np.isfinite(med):
            ax.text(xi, med + hi + pad, f"n={n}", ha="center", va="bottom", fontsize=7)
    _label_direction_axis(ax, order, positions, block_center, multi_ring)
    return medians, ns, his


def _values_by_direction(records, order, attr) -> Dict[Direction, np.ndarray]:
    grouped = group_by_direction(records)
    return {
        d: np.asarray([getattr(r, attr) for r in grouped.get(d, [])], dtype=float)
        for d in order
    }


def _plot_speed_panel(ax, peripheral_success, order, colors, multi_ring) -> Dict[str, object]:
    medians, ns, his = _bar_with_iqr(
        ax, order, _values_by_direction(peripheral_success, order, "duration_s"),
        colors, multi_ring,
    )
    tops = [m + h for m, h in zip(medians, his) if np.isfinite(m)]
    ax.set_ylim(0.0, max([1.0] + [t * 1.12 for t in tops]))
    ax.set_ylabel("Median duration (s)")
    ax.set_title("Speed by direction (successful reaches)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.15)
    return {
        direction_label(d, multi_ring): {
            "n": ns[i],
            "median_duration_s": medians[i] if np.isfinite(medians[i]) else None,
        }
        for i, d in enumerate(order)
    }


def _plot_path_efficiency_panel(ax, peripheral_success, order, colors, multi_ring) -> Dict[str, object]:
    medians, ns, his = _bar_with_iqr(
        ax, order, _values_by_direction(peripheral_success, order, "path_efficiency"),
        colors, multi_ring,
    )
    tops = [m + h for m, h in zip(medians, his) if np.isfinite(m)]
    finite_medians = [m for m in medians if np.isfinite(m)]
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6, label="straight line")
    ax.set_ylim(min(0.9, *finite_medians) if finite_medians else 0.9,
                max([1.3] + [t * 1.12 for t in tops]))
    ax.set_ylabel("Path length / straight-line distance")
    ax.set_title("Path efficiency by direction (successful reaches)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(loc="upper right", fontsize=7)
    return {
        direction_label(d, multi_ring): {
            "n": ns[i],
            "median_path_efficiency": medians[i] if np.isfinite(medians[i]) else None,
        }
        for i, d in enumerate(order)
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
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    order = direction_order(group_by_direction(peripheral).keys())
    colors = direction_colors(order)
    multi_ring = len({d[0] for d in order}) > 1

    width = min(24.0, max(16.0, 2.0 + 0.42 * (len(order) + 3) * 3))
    fig, (ax_perf, ax_speed, ax_path) = plt.subplots(1, 3, figsize=(width, 7.0))

    metrics = _plot_performance_panel(ax_perf, records, order, colors, multi_ring)
    metrics["rings"] = ring_order([d[0] for d in order]) if multi_ring else []
    metrics["speed_by_direction"] = _plot_speed_panel(
        ax_speed, peripheral_success, order, colors, multi_ring)
    metrics["path_efficiency_by_direction"] = _plot_path_efficiency_panel(
        ax_path, peripheral_success, order, colors, multi_ring)

    fig.suptitle(f"{day} structured center-out metrics{_title_suffix(exclude_recs)}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, out_path, "co_structured_metrics")

    return metrics
