"""Plotting for the cross-day hold-stability figure.

Renders `cross_day_hold_stability.png`: how well Eevee holds inside the target
day to day, separated from how hard the task made that hold. Computation lives
in `cross_day_hold.py`; this file only draws.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from .cross_day_axes import draw_mode_transitions, style_day_xaxis
    from .cross_day_hold import is_estimable_at, resolve_reference_hold, survival_at, survival_curve
except ImportError:  # pragma: no cover - script-mode fallback
    from cross_day_axes import draw_mode_transitions, style_day_xaxis
    from cross_day_hold import is_estimable_at, resolve_reference_hold, survival_at, survival_curve

DEMAND_COLOR = "#1f4e79"
RADIUS_COLOR = "#e67e22"
MATCHED_COLOR = "#2ca25f"
TASK_COLOR = "#b279a2"
COST_COLOR = "#c0392b"


def _col(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


def _line(ax, x, values, color, label=None, marker="o", **kwargs):
    values = np.asarray(values, dtype=float)
    if np.any(np.isfinite(values)):
        ax.plot(x, values, marker=marker, color=color, linewidth=1.8, markersize=4, label=label, **kwargs)


def _curve_day_indices(rows: Sequence[Dict[str, Any]], max_curves: int = 8) -> List[int]:
    usable = [i for i, row in enumerate(rows) if int(row.get("n_hold_attempts", 0) or 0) > 0]
    if len(usable) <= max_curves:
        return usable
    picks = np.linspace(0, len(usable) - 1, max_curves)
    return [usable[int(round(p))] for p in picks]


def plot_cross_day_hold(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    reference_hold_s: float | None = None,
) -> float:
    """Draw the hold-stability figure; returns the criterion actually used."""
    days = [str(row["day"]) for row in rows]
    x = np.arange(len(days), dtype=float)
    ref_s = resolve_reference_hold(rows, reference_hold_s)

    required = _col(rows, "median_required_hold_s")
    required_lo = _col(rows, "min_required_hold_s")
    required_hi = _col(rows, "max_required_hold_s")
    radius = _col(rows, "mean_target_radius_ratio")
    task_rate = 100.0 * _col(rows, "task_criterion_hold_success_rate")
    matched_rate = 100.0 * np.asarray([survival_at(row, ref_s) for row in rows], dtype=float)
    excess_s = _col(rows, "mean_excess_acquisition_s")
    breaks = _col(rows, "hold_breaks_per_trial")
    holds_per_trial = _col(rows, "holds_per_trial")
    hold_phase_pct = 100.0 * _col(rows, "median_hold_phase_fraction")

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    ax_demand, ax_stability, ax_survival, ax_cost, ax_matched, ax_entries = axes.ravel()

    # (1) What the task asked for: hold requirement and target size move together
    _line(ax_demand, x, required, DEMAND_COLOR, "Required hold (median)")
    ok = np.isfinite(required_lo) & np.isfinite(required_hi)
    if np.any(ok):
        ax_demand.fill_between(
            x, required_lo, required_hi, where=ok, color=DEMAND_COLOR, alpha=0.18,
            step=None, label="Required hold (min–max)",
        )
    ax_demand.axhline(ref_s, color="#555555", linestyle=":", linewidth=1.2)
    ax_demand.text(
        0.005, ref_s, f" matched criterion {ref_s:.2f}s", transform=ax_demand.get_yaxis_transform(),
        va="bottom", ha="left", fontsize=7, color="#555555",
    )
    ax_demand.set_ylabel("Required hold (s)", color=DEMAND_COLOR)
    ax_demand.set_title("Task demand: hold requirement & target size")
    ax_demand.legend(loc="upper left", frameon=False, fontsize=7)
    ax_demand_r = ax_demand.twinx()
    _line(ax_demand_r, x, radius, RADIUS_COLOR, marker="s", linestyle="--", alpha=0.85)
    ax_demand_r.set_ylabel("Mean target radius (ratio)", color=RADIUS_COLOR)
    ax_demand_r.tick_params(axis="y", labelcolor=RADIUS_COLOR)
    ax_demand.tick_params(axis="y", labelcolor=DEMAND_COLOR)

    # (2) Stability, scored two ways
    _line(ax_stability, x, matched_rate, MATCHED_COLOR, f"Matched criterion (≥{ref_s:.2f}s)")
    _line(ax_stability, x, task_rate, TASK_COLOR, "That day's own requirement", marker="s")
    ax_stability.set_ylabel("Holds surviving (%)")
    ax_stability.set_ylim(0, 105)
    ax_stability.set_title("Hold stability per target entry")
    ax_stability.legend(loc="lower right", frameon=False, fontsize=7)

    # (3) The distribution behind panel 2, early → late
    _plot_survival_curves(ax_survival, rows, days, ref_s)

    # (4) What broken holds cost in time
    _line(ax_cost, x, excess_s, COST_COLOR)
    ax_cost.axhline(0.0, color="#999999", linewidth=0.8)
    ax_cost.set_ylabel("Mean excess acquisition time (s)", color=COST_COLOR)
    ax_cost.set_title("Cost of broken holds (entry→complete minus required hold)")
    ax_cost.tick_params(axis="y", labelcolor=COST_COLOR)
    ax_cost_r = ax_cost.twinx()
    _line(ax_cost_r, x, breaks, "#7f7f7f", marker="s", linestyle="--", alpha=0.85)
    ax_cost_r.set_ylabel("Hold breaks / trial", color="#7f7f7f")
    ax_cost_r.tick_params(axis="y", labelcolor="#7f7f7f")

    # (5) Skill at matched demand: same requirement, later day, better hold?
    _plot_matched_demand(ax_matched, rows, days, required, matched_rate, ref_s)

    # (6) How many bites at the target each trial takes
    _line(ax_entries, x, holds_per_trial, "#4c78a8")
    ax_entries.set_ylabel("Hold attempts / trial", color="#4c78a8")
    ax_entries.set_title("Re-entries needed, and share of trial spent holding")
    ax_entries.tick_params(axis="y", labelcolor="#4c78a8")
    ax_entries_r = ax_entries.twinx()
    _line(ax_entries_r, x, hold_phase_pct, "#54a24b", marker="s", linestyle="--", alpha=0.85)
    ax_entries_r.set_ylabel("Hold phase (% of successful trial)", color="#54a24b")
    ax_entries_r.tick_params(axis="y", labelcolor="#54a24b")

    any_transition = False
    for ax in (ax_demand, ax_stability, ax_cost, ax_entries):
        style_day_xaxis(ax, days, x)
        ax.grid(alpha=0.2)
        any_transition = draw_mode_transitions(ax, rows) or any_transition

    fig.suptitle(
        "Hold stability across days — task demand separated from performance\n"
        f"matched criterion = {ref_s:.2f}s (toughest hold most days in the selection can be scored at); "
        "dwell measured from each hold_start until target exit / hold break, censored at hold_complete",
        fontsize=10,
    )
    footer = (
        "Matched-criterion stability is comparable across days; the task-criterion line is not "
        "(it drops when the requirement rises even if the hold is unchanged)."
    )
    n_unscored = sum(1 for row in rows if not is_estimable_at(row, ref_s))
    if n_unscored:
        footer += (
            f" {n_unscored} day(s) demanded a hold shorter than {ref_s:.2f}s and cannot be scored"
            " at the matched criterion (gap in the green line)."
        )
    if any_transition:
        footer += " Dashed red = control-mode change; shaded = mixed-mode day."
    fig.text(
        0.5, 0.005, "\n".join(textwrap.wrap(footer, width=150)),
        ha="center", va="bottom", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return ref_s


def _plot_survival_curves(ax, rows, days, ref_s: float) -> None:
    indices = _curve_day_indices(rows)
    if not indices:
        ax.text(0.5, 0.5, "No hold data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("In-target dwell survival")
        ax.axis("off")
        return

    cmap = plt.get_cmap("viridis")
    drawn = 0
    for order, index in enumerate(indices):
        row = rows[index]
        min_informative = max(20, int(0.1 * float(row.get("n_hold_attempts", 0) or 0)))
        # Past a day's own shortest requirement the task has censored every
        # completed hold, so the curve would nose-dive on broken holds alone.
        times, survival = survival_curve(
            row, float(row.get("min_required_hold_s", np.nan)), min_informative
        )
        if times.size == 0:
            continue
        color = cmap(order / max(len(indices) - 1, 1))
        ax.step(
            times, 100.0 * survival, where="post", color=color, linewidth=1.6,
            label=f"{days[index]} (hold {float(row.get('median_required_hold_s', np.nan)):.2f}s)",
        )
        drawn += 1
    ax.axvline(ref_s, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_xlabel("In-target dwell (s)", fontsize=9)
    ax.set_ylabel("Holds still in target (%)")
    ax.set_ylim(0, 102)
    ax.set_title("In-target dwell survival (early → late)")
    ax.grid(alpha=0.2)
    if drawn:
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)


def _plot_matched_demand(ax, rows, days, required, matched_rate, ref_s: float) -> None:
    ok = np.isfinite(required) & np.isfinite(matched_rate)
    if not np.any(ok):
        ax.text(0.5, 0.5, "No matched-demand data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Stability vs. demand")
        ax.axis("off")
        return

    order = np.arange(len(rows), dtype=float)
    jitter = 0.006 * (np.arange(len(rows)) % 5 - 2)  # separate days sharing a requirement
    scatter = ax.scatter(
        required[ok] + jitter[ok], matched_rate[ok], c=order[ok], cmap="viridis",
        s=55, edgecolor="black", linewidth=0.4,
    )
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Session order (early → late)")
    tick_indices = np.linspace(0, len(days) - 1, min(5, len(days))).astype(int)
    cbar.set_ticks(order[tick_indices])
    cbar.set_ticklabels([days[i] for i in tick_indices], fontsize=6.5)
    ax.set_xlabel("Required hold on that day (s)", fontsize=9)
    ax.set_ylabel(f"Holds ≥ {ref_s:.2f}s (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Stability at matched demand (up-and-right = harder task, held anyway)")
    ax.grid(alpha=0.2)
