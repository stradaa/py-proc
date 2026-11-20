"""task_summaries.py — Which summary panels belong to which task.

A day now routinely mixes tasks (Eevee runs ``Grid (Rust)`` joystick and
``Simple Touch (Rust)`` in the same session).  Every panel in
``day_presentation_plots`` / ``day_behavior_metrics`` reads ``Joystick*`` fields,
so running them over a whole day blends tasks that share no metrics.

``generate_day_presentation_plots`` partitions trials by canonical task type and
asks this module which panels to run for each.  Panels take a small ``PanelCtx``
rather than a fixed argument list, because the joystick and touch panels need
different inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple

import numpy as np


@dataclass
class PanelCtx:
    """Everything a panel may need. ``data`` is already subset to one task."""

    data: Dict[str, np.ndarray]
    out_dir: Path
    day: str
    repo_root: Path
    exclude_recs: List[int]
    task: str

    def path(self, suffix: str) -> Path:
        return self.out_dir / f"{self.day}_{self.task}_{suffix}.png"


class Panel(NamedTuple):
    key: str          # key in the per-task metrics dict
    run: Callable[[PanelCtx], object]


# --- joystick ---------------------------------------------------------------

def _joystick_panels() -> List[Panel]:
    from pyCheck.day_behavior_metrics import (
        plot_engagement,
        plot_failure_modes,
        plot_kinematic_intervals,
        plot_overshoot_attempts,
    )
    from pyCheck.day_presentation_plots import (
        plot_performance_over_time,
        plot_session_overview,
        plot_success_failure_timing,
        plot_target_performance,
    )

    return [
        Panel("session_overview", lambda c: plot_session_overview(
            c.data, c.path("overview_by_rec"), c.day, c.exclude_recs)),
        Panel("performance_over_time", lambda c: plot_performance_over_time(
            c.data, c.path("performance_over_time"), c.exclude_recs)),
        Panel("target_performance", lambda c: plot_target_performance(
            c.data, c.path("target_performance"), c.exclude_recs)),
        Panel("success_failure_timing", lambda c: plot_success_failure_timing(
            c.data, c.path("success_failure_timing"), c.exclude_recs)),
        Panel("kinematic_intervals", lambda c: plot_kinematic_intervals(
            c.data, c.path("kinematic_intervals"), c.exclude_recs)),
        Panel("failure_modes", lambda c: plot_failure_modes(
            c.data, c.path("failure_modes"), c.exclude_recs)),
        Panel("overshoot_attempts", lambda c: plot_overshoot_attempts(
            c.data, c.path("overshoot_attempts"), c.exclude_recs)),
        Panel("engagement", lambda c: plot_engagement(
            c.data, c.path("engagement"), c.exclude_recs)),
    ]


# --- simple_touch -----------------------------------------------------------

def _simple_touch_panels() -> List[Panel]:
    from pyCheck.simple_touch_metrics import (
        plot_simple_touch_accuracy,
        plot_simple_touch_performance,
    )

    return [
        Panel("performance", lambda c: plot_simple_touch_performance(
            c.data, c.path("performance"), c.repo_root, c.day, c.exclude_recs)),
        Panel("touch_accuracy", lambda c: plot_simple_touch_accuracy(
            c.data, c.path("touch_accuracy"), c.repo_root, c.day, c.exclude_recs)),
    ]


# Keyed by canonical task type (pyjoystick.tasks.canonical_task_type), so the
# Rust delegates land on their Python task's panels.
TASK_PANEL_SETS: Dict[str, Callable[[], List[Panel]]] = {
    "joystick_intro": _joystick_panels,
    "simple_touch_task": _simple_touch_panels,
}


def panels_for(task: str) -> List[Panel]:
    """Panels registered for ``task``; empty if the task has no summary yet."""
    factory = TASK_PANEL_SETS.get(task)
    return factory() if factory is not None else []


def plot_day_overview(
    data: Dict[str, np.ndarray],
    tasks: np.ndarray,
    out_path: Path,
    day: str,
    exclude_recs: List[int],
) -> Dict[str, object]:
    """Day-level view across every task: trials per recording, coloured by task,
    with each recording's success rate on a second axis.

    Kept at the top of ``beh/`` under its historical name because
    ``proc_gui/app.py`` tests for exactly this file to decide whether a day has
    been summarised.
    """
    import matplotlib.pyplot as plt

    from pyCheck.plot_utils import _flat, _save_figure, _title_suffix

    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    recs = sorted(np.unique(rec).tolist())
    task_names = sorted(set(tasks.tolist()))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    colors = {t: palette[i % len(palette)] for i, t in enumerate(task_names)}

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bottoms = np.zeros(len(recs))
    for task in task_names:
        vals = np.array([int(np.sum((rec == r) & (tasks == task))) for r in recs], dtype=float)
        ax.bar(recs, vals, bottom=bottoms, color=colors[task], width=0.7, label=task)
        bottoms += vals
    ax.set_xlabel("Recording")
    ax.set_ylabel("Trial count")
    ax.set_xticks(recs)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(f"{day} trials by recording and task{_title_suffix(exclude_recs)}")
    ax.legend(frameon=False, fontsize=8)

    ax2 = ax.twinx()
    rates = np.array([np.nanmean(success[rec == r]) for r in recs], dtype=float)
    ax2.plot(recs, 100.0 * rates, color="#222222", marker="o", linewidth=2, alpha=0.85)
    ax2.set_ylabel("Success rate (%)")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    _save_figure(fig, out_path, "day overview")

    return {
        "tasks_present": task_names,
        "per_rec": {
            f"rec{r:03d}": {
                "trial_count": int(np.sum(rec == r)),
                "success_rate": float(np.nanmean(success[rec == r])),
                "tasks": sorted(set(tasks[rec == r].tolist())),
            }
            for r in recs
        },
    }
