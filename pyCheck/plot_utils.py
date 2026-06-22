"""plot_utils.py — Shared helpers for day-summary behavior plots.

Small utilities used by both ``day_presentation_plots`` and
``day_behavior_metrics`` (kept here to avoid a circular import between them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _flat(data: Dict[str, np.ndarray], key: str, dtype=float) -> np.ndarray:
    value = np.asarray(data[key]).ravel()
    if dtype is object:
        return value.astype(object)
    return value.astype(dtype)


def _title_suffix(exclude_recs: List[int]) -> str:
    if not exclude_recs:
        return ""
    rec_text = ", ".join(f"rec{rec:03d}" for rec in sorted(exclude_recs))
    plural = "s" if len(exclude_recs) > 1 else ""
    return f" (excluded rec{plural}: {rec_text})"


def _save_figure(fig: plt.Figure, out_path: Path, label: str) -> None:
    print(f"Generating figure: {label}")
    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    plt.close(fig)
