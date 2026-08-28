"""cross_day_axes.py — Shared x-axis decoration for the cross-day figures.

Every cross-day figure puts one day per x tick, so they all need the same two
things: readable date ticks (dozens of YYMMDD labels in a single axis) and the
control-mode transition markers. Kept here so `cross_day_metrics_plots`,
`cross_day_kinematics_plots`, `cross_day_co_structured_plots` and
`cross_day_hold_plots` share one implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def format_day_labels(days: Sequence[str]) -> List[str]:
    """Shorten YYMMDD day names to MM/DD when every day shares a year.

    With 40+ sessions on one axis the leading year is pure noise; dropping it
    buys two characters of legibility. Mixed-year selections keep the full name
    so labels stay unambiguous.
    """
    texts = [str(day).strip() for day in days]
    if texts and all(len(t) == 6 and t.isdigit() for t in texts):
        if len({t[:2] for t in texts}) == 1:
            return [f"{t[2:4]}/{t[4:6]}" for t in texts]
    return texts


def day_label_fontsize(n_days: int) -> float:
    if n_days <= 12:
        return 8.0
    if n_days <= 25:
        return 7.0
    if n_days <= 45:
        return 6.0
    return 5.0


def style_day_xaxis(
    ax,
    days: Sequence[str],
    x: Sequence[float] | None = None,
    xlabel: str | None = "Day",
) -> None:
    """Vertical, size-scaled date ticks — the only readable option once a
    selection runs to dozens of days."""
    positions = np.arange(len(days), dtype=float) if x is None else np.asarray(x, dtype=float)
    fontsize = day_label_fontsize(len(days))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        format_day_labels(days),
        rotation=90,
        ha="center",
        va="top",
        fontsize=fontsize,
    )
    ax.tick_params(axis="x", length=2, pad=1.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)


def _mode_family(label: str) -> str:
    """'blend 0.16' -> 'blend'; keeps mixed(...) labels intact."""
    text = str(label).strip()
    return text.split(" ", 1)[0] if text else ""


def draw_mode_transitions(ax, rows: Sequence[Dict[str, Any]]) -> bool:
    """Mark where the cursor control law changed and shade 'mixed' days.

    Two tiers, because the blend ramp changes value almost daily and labelling
    every step buried the panels in overlapping red text: a change of control
    *family* (direct → blend → cumulative) gets a labelled dashed line, while a
    same-family parameter step (blend 0.16 → 0.19) gets a light unlabelled tick.
    Trends stay pooled (single line); the markers just flag where the control
    law changed so boundary steps aren't misread as skill change. Returns True
    if any marker was drawn.
    """
    labels = [str(row.get("control_mode", "")) for row in rows]
    families = [_mode_family(label) for label in labels]
    drawn = False
    for i, row in enumerate(rows):
        if int(row.get("control_mode_mixed", 0)):
            ax.axvspan(i - 0.5, i + 0.5, color="#999999", alpha=0.12, zorder=0)
            drawn = True
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            continue
        if families[i] == families[i - 1]:
            ax.axvline(i - 0.5, color="#c0392b", linestyle=":", linewidth=0.7, alpha=0.35, zorder=1)
            drawn = True
            continue
        ax.axvline(i - 0.5, color="#c0392b", linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)
        ax.text(
            i - 0.5, 0.99, f"→ {labels[i]}", transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=6.5, color="#c0392b", alpha=0.9,
        )
        drawn = True
    return drawn
