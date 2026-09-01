"""cross_day_co_structured.py — cross-day learning trends for structured CO trials.

Complements `cross_day_plots.py` (which is derived from `AllTrials.mat` and
covers all task types) with a companion figure focused specifically on the
Rust executor's `schedule_phase`-tagged center-out (CO) schedule: how
structured-vs-random success rate, reach speed, path efficiency, and
per-direction success rate evolve across days. Output is a standalone
`cross_day_co_structured.png` plus a JSON/CSV, so the headline cross-day
figure stays uncluttered — the same split used for `cross_day_kinematics.py`.

Per-day aggregation reuses `co_structured.py`'s attempt collection
(`AttemptRecord`, `_collect_rec`, `_tag_pairs`) directly against each day's
raw `joystick.mat`/`recNNN.ev.mat`, so this works independently of whether
`co_structured.py`'s own per-day figures have been generated for that day.

Ring support: when multiple CO ring distances are present (e.g. a near ring
and a far ring), metrics are split per ring.  Backward-compatible: single-ring
data produces the same output as before, just with one ring label.

Metrics per day (medians/rates across the selected recs' peripheral attempts):
  * Structured vs random success rate.
  * Median reach duration and path efficiency (successful peripheral only).
  * Center→peripheral pair completion rate.
  * Direction-resolved success rate, speed, and path efficiency
    (day × ring × angle), for the heatmap panels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .co_structured import (
        AttemptRecord, STRUCTURED_PHASES, _collect_rec, _tag_pairs,
        assign_ring as _assign_ring, detect_rings as _detect_rings,
        ring_labels_sorted as _ring_label_sorted,
    )
    from .cross_day_co_structured_plots import plot_cross_day_co_structured
    from .cross_day_plots import _default_out_dir, _parse_day_rec_selection, _summarize_modes
    from .joystick_validation import load_joystick_dataset
except ImportError:  # pragma: no cover - script-mode fallback
    from co_structured import (
        AttemptRecord, STRUCTURED_PHASES, _collect_rec, _tag_pairs,
        assign_ring as _assign_ring, detect_rings as _detect_rings,
        ring_labels_sorted as _ring_label_sorted,
    )
    from cross_day_co_structured_plots import plot_cross_day_co_structured
    from cross_day_plots import _default_out_dir, _parse_day_rec_selection, _summarize_modes
    from joystick_validation import load_joystick_dataset

# Scalar metric columns (the direction heatmap lives only in the JSON).
_FIELDS = [
    "day", "selected_recs", "control_mode", "position_velocity_blend", "control_mode_mixed",
    "n_structured_attempts", "n_random_attempts",
    "n_peripheral_success", "structured_success_rate", "random_success_rate",
    "median_duration_s", "median_path_efficiency",
    "n_pairs", "n_complete_pairs", "pair_completion_rate",
]


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _success_rate(records: Sequence[AttemptRecord]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([r.outcome == "success" for r in records]))


# ---------------------------------------------------------------------------
# Per-direction, per-ring metrics
# ---------------------------------------------------------------------------

def _direction_ring_metrics(
    peripheral: Sequence[AttemptRecord], rings: List[float],
) -> Dict[str, Any]:
    """Compute direction × ring metrics: success rate, median speed, median path efficiency."""
    ring_labels = _ring_label_sorted(rings)
    angles = sorted({
        int(round(r.target_angle_deg)) % 360
        for r in peripheral if np.isfinite(r.target_angle_deg)
    })

    # Per ring: {ring_label: {angle_str: value}}
    dir_success: Dict[str, Dict[str, float]] = {rl: {} for rl in ring_labels}
    dir_speed: Dict[str, Dict[str, float]] = {rl: {} for rl in ring_labels}
    dir_path_eff: Dict[str, Dict[str, float]] = {rl: {} for rl in ring_labels}
    dir_n: Dict[str, Dict[str, int]] = {rl: {} for rl in ring_labels}

    for angle in angles:
        for rl in ring_labels:
            group = [
                r for r in peripheral
                if (np.isfinite(r.target_angle_deg)
                    and int(round(r.target_angle_deg)) % 360 == angle
                    and _assign_ring(r.reach_distance, rings) == rl)
            ]
            if not group:
                continue
            dir_success[rl][str(angle)] = _success_rate(group)
            dir_n[rl][str(angle)] = len(group)
            successful = [r for r in group if r.outcome == "success"]
            dir_speed[rl][str(angle)] = _median([r.duration_s for r in successful])
            dir_path_eff[rl][str(angle)] = _median([r.path_efficiency for r in successful])

    return {
        "ring_labels": ring_labels,
        "ring_distances": [round(r, 4) for r in rings],
        "dir_success": dir_success,
        "dir_speed": dir_speed,
        "dir_path_eff": dir_path_eff,
        "dir_n": dir_n,
    }


# ---------------------------------------------------------------------------
# Per-ring scalar metrics
# ---------------------------------------------------------------------------

def _per_ring_scalars(
    peripheral: Sequence[AttemptRecord], rings: List[float],
) -> Dict[str, Dict[str, Any]]:
    """Per-ring success rate, median speed, median path efficiency."""
    ring_labels = _ring_label_sorted(rings)
    result: Dict[str, Dict[str, Any]] = {}
    for rl in ring_labels:
        group = [r for r in peripheral if _assign_ring(r.reach_distance, rings) == rl]
        successful = [r for r in group if r.outcome == "success"]
        result[rl] = {
            "n_attempts": len(group),
            "success_rate": _success_rate(group),
            "median_duration_s": _median([r.duration_s for r in successful]),
            "median_path_efficiency": _median([r.path_efficiency for r in successful]),
        }
    return result


# ---------------------------------------------------------------------------
# Day-level aggregation
# ---------------------------------------------------------------------------

def build_day_co_structured(repo_root: Path, day: str, recs: Sequence[int]) -> Dict[str, Any]:
    records: List[AttemptRecord] = []
    next_pair_id = 0
    control_modes: set[str] = set()
    blend_values: set[float] = set()
    for rec in recs:
        rec_str = f"{int(rec):03d}"
        try:
            dataset = load_joystick_dataset(repo_root, day, rec_str)
        except Exception as exc:  # noqa: BLE001 - a bad rec must not kill the day
            print(f"  cross_day_co_structured: could not load {day}/{rec_str} ({exc}); skipping")
            continue
        control_modes.add(str(dataset.control_mode))
        blend_values.add(round(float(dataset.position_velocity_blend), 4))
        rec_records = _collect_rec(dataset, day, int(rec))
        next_pair_id = _tag_pairs(rec_records, next_pair_id)
        records.extend(rec_records)

    structured = [r for r in records if r.schedule_phase in STRUCTURED_PHASES]
    random_inserts = [r for r in records if r.schedule_phase == "random"]
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    peripheral_success = [
        r for r in peripheral if r.outcome == "success" and np.isfinite(r.target_angle_deg)
    ]
    pairs = {r.pair_id for r in records if r.pair_id >= 0}
    complete_pairs = {r.pair_id for r in records if r.pair_id >= 0 and r.pair_complete}

    # Ring detection and per-ring metrics
    rings = _detect_rings(peripheral)
    dir_metrics = _direction_ring_metrics(peripheral, rings)
    ring_scalars = _per_ring_scalars(peripheral, rings)

    # Backward-compatible flat dir_success/dir_n (collapsed across rings)
    flat_dir_success: Dict[str, float] = {}
    flat_dir_n: Dict[str, int] = {}
    angles = sorted({
        int(round(r.target_angle_deg)) % 360
        for r in peripheral if np.isfinite(r.target_angle_deg)
    })
    for angle in angles:
        group = [
            r for r in peripheral
            if np.isfinite(r.target_angle_deg) and int(round(r.target_angle_deg)) % 360 == angle
        ]
        flat_dir_success[str(angle)] = _success_rate(group)
        flat_dir_n[str(angle)] = len(group)

    control_mode_label, blend_repr, mode_mixed = _summarize_modes(control_modes, blend_values)
    return {
        "day": day,
        "selected_recs": [int(r) for r in recs],
        "control_mode": control_mode_label,
        "position_velocity_blend": blend_repr,
        "control_mode_mixed": int(mode_mixed),
        "n_structured_attempts": len(structured),
        "n_random_attempts": len(random_inserts),
        "n_peripheral_success": len(peripheral_success),
        "structured_success_rate": _success_rate(structured),
        "random_success_rate": _success_rate(random_inserts),
        "median_duration_s": _median([r.duration_s for r in peripheral_success]),
        "median_path_efficiency": _median([r.path_efficiency for r in peripheral_success]),
        "n_pairs": len(pairs),
        "n_complete_pairs": len(complete_pairs),
        "pair_completion_rate": (len(complete_pairs) / len(pairs)) if pairs else float("nan"),
        # Backward-compatible flat dicts (collapsed across rings)
        "dir_success": flat_dir_success,
        "dir_n": flat_dir_n,
        # Ring-aware metrics
        "ring_labels": dir_metrics["ring_labels"],
        "ring_distances": dir_metrics["ring_distances"],
        "ring_scalars": ring_scalars,
        "ring_dir_success": dir_metrics["dir_success"],
        "ring_dir_speed": dir_metrics["dir_speed"],
        "ring_dir_path_eff": dir_metrics["dir_path_eff"],
        "ring_dir_n": dir_metrics["dir_n"],
    }


def _row_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: row.get(k) for k in _FIELDS}
    out["selected_recs"] = ",".join(f"{int(r):03d}" for r in row["selected_recs"])
    return out


def _write_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        with open(out_path, "r", newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                merged[str(raw.get("day", "")).strip()] = dict(raw)
    for row in rows:
        merged[str(row["day"])] = _row_for_csv(row)
    ordered = [merged[d] for d in sorted(merged) if d]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in _FIELDS})
    return ordered


def generate_cross_day_co_structured(
    repo_root: str | Path,
    selections: Sequence[str | Sequence[str]],
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir is not None else _default_out_dir(repo_root_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    parsed = [_parse_day_rec_selection(spec, repo_root_path) for spec in selections]
    rows = [
        build_day_co_structured(repo_root_path, day=day, recs=recs)
        for day, recs in parsed
    ]
    rows.sort(key=lambda r: r["day"])

    figure_path = out_dir_path / "cross_day_co_structured.png"
    json_path = out_dir_path / "cross_day_co_structured_metrics.json"
    csv_path = out_dir_path / "cross_day_co_structured.csv"

    plot_cross_day_co_structured(rows, figure_path)
    csv_rows = _write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"days": rows, "figure_path": str(figure_path)}, fh, indent=2)

    print(f"Saved figure: {figure_path}")
    print(f"Saved co_structured cross-day JSON: {json_path}")
    print(f"Saved co_structured cross-day CSV: {csv_path}")
    return {
        "figure_path": str(figure_path),
        "metrics_json_path": str(json_path),
        "metrics_csv_path": str(csv_path),
        "days": rows,
        "csv_days": csv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-day structured center-out (CO) learning trends from schedule_phase-tagged attempts"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--day-recs", action="append", nargs="+", required=True,
        help="Repeatable day selection like 260706:1-3,5 or 260707 for all recs on that day.",
    )
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    generate_cross_day_co_structured(
        repo_root=args.repo_root,
        selections=args.day_recs,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
