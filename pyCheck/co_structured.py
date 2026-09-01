"""co_structured.py — Structured center-out (CO) trial outputs for one day.

The Rust joystick executor tags every attempt in ``behav_result["attempts"]``
with ``schedule_phase``: ``"center"`` / ``"peripheral"`` (center-out mode),
``"sequence"`` (sequence mode) or ``"random"`` (Random mode, an interleave
insert, or a fallback pick). This module separates structured trials from
random inserts and writes, per day:

- ``<day>_co_structured.png`` — a compass grid sized to the day's direction
  count (8 -> 3x3, 16 -> 5x5), one panel per ``(ring, angle)`` laid out by
  direction, showing successful reach trajectories and their average; the
  merged inner block overlays every direction's average.
- ``<day>_co_structured_metrics.png`` — success-rate bars (per direction,
  structured vs random, pair completion), plus median reach duration and path
  efficiency by direction, grouped into one block per ring.
- ``<day>_co_structured_attempts.csv`` — one row per attempt that carries a
  ``schedule_phase`` (rec, trial, attempt, phase, outcome, target identity,
  ring, timing, pair tagging, and the row's slice into the NPZ).
- ``<day>_co_structured_paths.npz`` — pickle-free ragged path storage:
  concatenated ``task_time_s`` / ``cursor_x`` / ``cursor_y`` / ``joystick_x``
  / ``joystick_y`` with ``row_start`` / ``row_len`` indexed by the CSV's
  ``path_row`` column (``sl = slice(row_start[k], row_start[k] + row_len[k])``).

Cursor caveat: paths come from ``joystick_validation``'s reconstruction, which
anchors itself to the cursor the task logs on its own events. Without those
anchors the stateful control modes drift far enough that reaches miss their
targets entirely — see ``pyjoystick.cursor.apply_cursor_anchors``.

Pairing caveat: a shared ``pair_id`` is assigned to attempts *adjacent in the
rec's attempt sequence*, not within a Thalamus trial (the center hold and the
reach it gates are usually two attempts in two trials). Adjacency is
guaranteed only for data recorded
with the pair-boundary executor build (>= 2026-07-07); in older recordings a
random insert could land inside a pair, which simply leaves those attempts
unpaired here.

Days with no ``schedule_phase`` data (pure-Python task, or recordings that
predate the schedule feature) are skipped without writing any files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pyCheck.cross_day_plots import _compute_path_efficiency
from pyCheck.joystick_validation import (
    JoystickDataset,
    get_attempt_segment,
    load_joystick_dataset,
    select_attempt_event,
)

STRUCTURED_PHASES = ("center", "peripheral", "sequence")

# Peripheral targets get a hue-by-angle palette (co_structured_directions), so
# only the non-directional categories need a fixed color.
RANDOM_COLOR = "#8a8a8a"


@dataclass
class AttemptRecord:
    day: str
    rec: int
    trial_index: int
    attempt_index: int
    schedule_phase: str
    outcome: str
    failure_reason: str
    target_index: int
    target_name: str
    target_x: float
    target_y: float
    target_radius: float
    t_target_on: float
    t_end: float
    duration_s: float
    pair_id: int = -1
    pair_complete: bool = False
    path_row: int = -1
    target_angle_deg: float = np.nan
    t_hold_complete: float = np.nan
    reach_distance: float = np.nan
    ring_label: str = ""
    path_efficiency: float = np.nan
    path: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)


def _rec_numbers(day_dir: Path) -> List[int]:
    """Recording numbers present as rec subdirectories (001 -> 1)."""
    recs = []
    for p in sorted(day_dir.iterdir()):
        if p.is_dir() and p.name.isdigit():
            recs.append(int(p.name))
    return recs


def _rec_has_inputs(day_dir: Path, rec: int) -> bool:
    rec_dir = day_dir / f"{rec:03d}"
    return (
        (rec_dir / f"rec{rec:03d}.ev.mat").exists()
        and (rec_dir / f"rec{rec:03d}.bag" / "mat" / "joystick.mat").exists()
    )


def _target_name(dataset: JoystickDataset, trial_index: int, target_index: int) -> str:
    if target_index < 0 or trial_index >= len(dataset.task_configs):
        return ""
    targets = dataset.task_configs[trial_index].get("targets", [])
    if not isinstance(targets, list) or target_index >= len(targets):
        return ""
    target = targets[target_index]
    return str(target.get("name", "")) if isinstance(target, dict) else ""


def _target_angle_deg(target_x: float, target_y: float) -> float:
    """Angle of the target from the task center (0.5, 0.5), degrees normalized to [0, 360)."""
    if not (np.isfinite(target_x) and np.isfinite(target_y)):
        return float(np.nan)
    return float(np.degrees(np.arctan2(target_y - 0.5, target_x - 0.5)) % 360.0)


def detect_rings(peripheral: Sequence["AttemptRecord"], tolerance: float = 0.03) -> List[float]:
    """Cluster peripheral reach distances into distinct rings.

    Returns sorted list of ring center distances (in norm coords).
    Single-ring data returns a list with one element.
    """
    distances = sorted({
        round(r.reach_distance, 3)
        for r in peripheral if np.isfinite(r.reach_distance)
    })
    if not distances:
        return []
    rings: List[float] = [distances[0]]
    for d in distances[1:]:
        if d - rings[-1] > tolerance:
            rings.append(d)
        else:
            # Update ring center to mean of cluster
            rings[-1] = (rings[-1] + d) / 2.0
    return rings


def assign_ring(reach_distance: float, rings: List[float]) -> Optional[str]:
    """Assign a record to the nearest detected ring, returning a label."""
    if not rings or not np.isfinite(reach_distance):
        return None
    idx = int(np.argmin([abs(reach_distance - r) for r in rings]))
    if len(rings) == 1:
        return "ring"
    return f"{'near' if idx == 0 else 'far' if idx == len(rings) - 1 else f'ring{idx}'}"


def ring_labels_sorted(rings: List[float]) -> List[str]:
    """Ring labels in ascending-distance order, matching assign_ring output."""
    if len(rings) <= 1:
        return ["ring"]
    labels = []
    for i in range(len(rings)):
        if i == 0:
            labels.append("near")
        elif i == len(rings) - 1:
            labels.append("far")
        else:
            labels.append(f"ring{i}")
    return labels


def tag_rings(records: List["AttemptRecord"], tolerance: float = 0.03) -> List[float]:
    """Detect the day's rings over every peripheral record and label each one.

    Rings must be detected across the whole day, not per rec: a rec that only
    sampled one ring would otherwise collapse both into a single cluster.
    """
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    rings = detect_rings(peripheral, tolerance)
    for record in peripheral:
        record.ring_label = assign_ring(record.reach_distance, rings) or ""
    return rings


def _attempt_times(attempt: Dict[str, Any]) -> tuple:
    target_on = select_attempt_event(attempt, "target_on", "first")
    end_name = "success" if str(attempt.get("outcome", "")).lower() == "success" else "fail"
    end_event = select_attempt_event(attempt, end_name, "last")
    hold_complete = select_attempt_event(attempt, "hold_complete", "first")
    t_on = float(target_on["time_perf_counter"]) if target_on is not None else np.nan
    t_end = float(end_event["time_perf_counter"]) if end_event is not None else np.nan
    t_hold_complete = float(hold_complete["time_perf_counter"]) if hold_complete is not None else np.nan
    return t_on, t_end, t_hold_complete


def _collect_rec(dataset: JoystickDataset, day: str, rec: int) -> List[AttemptRecord]:
    records: List[AttemptRecord] = []
    for trial_index, behav_result in enumerate(dataset.behav_results):
        attempts = behav_result.get("attempts") or []
        if not attempts and isinstance(behav_result.get("final_attempt"), dict):
            attempts = [behav_result["final_attempt"]]
        trial_records: List[AttemptRecord] = []
        for attempt in attempts:
            if not isinstance(attempt, dict) or "schedule_phase" not in attempt:
                continue
            phase = str(attempt.get("schedule_phase") or "")
            if not phase:
                continue
            target_position = attempt.get("target_position") or {}
            raw_idx = attempt.get("target_index")
            target_index = int(raw_idx) if raw_idx is not None else -1
            t_on, t_end, t_hold_complete = _attempt_times(attempt)
            target_x = float(target_position.get("x_norm", np.nan))
            target_y = float(target_position.get("y_norm", np.nan))
            record = AttemptRecord(
                day=day,
                rec=rec,
                trial_index=trial_index,
                attempt_index=int(attempt.get("attempt_index", -1)),
                schedule_phase=phase,
                outcome=str(attempt.get("outcome") or ""),
                failure_reason=str(attempt.get("failure_reason") or ""),
                target_index=target_index,
                target_name=_target_name(dataset, trial_index, target_index),
                target_x=target_x,
                target_y=target_y,
                target_radius=float(attempt.get("target_radius_ratio") or np.nan),
                t_target_on=t_on,
                t_end=t_end,
                t_hold_complete=t_hold_complete,
                duration_s=float(attempt.get("duration_s") or np.nan),
            )
            if phase == "peripheral":
                record.target_angle_deg = _target_angle_deg(target_x, target_y)
                record.reach_distance = float(np.hypot(target_x - 0.5, target_y - 0.5))
            try:
                # Transport only: no padding, cut at acquisition. Keeps the
                # drawn path on the same interval as path_efficiency below.
                segment = get_attempt_segment(
                    dataset, trial_index, attempt,
                    pre_s=0.0, post_s=0.0, end_event="hold_complete",
                )
                record.path = {
                    "task_time_s": np.asarray(segment.task_time_s, dtype=float),
                    "cursor_x": np.asarray(segment.cursor_x, dtype=float),
                    "cursor_y": np.asarray(segment.cursor_y, dtype=float),
                    "joystick_x": np.asarray(segment.joystick_x, dtype=float),
                    "joystick_y": np.asarray(segment.joystick_y, dtype=float),
                }
            except (ValueError, KeyError):
                record.path = None
            if (
                phase == "peripheral"
                and record.outcome == "success"
                and np.isfinite(t_on)
                and np.isfinite(t_hold_complete)
            ):
                record.path_efficiency = _compute_path_efficiency(
                    dataset, t_on, t_hold_complete, target_x, target_y
                )
            trial_records.append(record)
        records.extend(trial_records)
    return records


def _tag_pairs(records: List[AttemptRecord], next_pair_id: int) -> int:
    """Assign pair_id to consecutive (center, peripheral) attempts within a rec.

    The center hold and the peripheral reach it gates are two attempts, and the
    executor may place them in *different* Thalamus trials (one attempt per
    trial is the common case), so adjacency has to be judged over the rec's
    attempt sequence rather than within a trial.
    """
    ordered = sorted(records, key=lambda r: (r.rec, r.trial_index, r.attempt_index))
    for a, b in zip(ordered, ordered[1:]):
        if a.rec != b.rec:
            continue
        if a.schedule_phase == "center" and b.schedule_phase == "peripheral":
            a.pair_id = b.pair_id = next_pair_id
            a.pair_complete = b.pair_complete = (
                a.outcome == "success" and b.outcome == "success"
            )
            next_pair_id += 1
    return next_pair_id


def _write_csv(records: List[AttemptRecord], out_path: Path) -> None:
    columns = [
        "day", "rec", "trial_index", "attempt_index", "schedule_phase",
        "outcome", "failure_reason", "target_index", "target_name",
        "target_x", "target_y", "target_radius", "target_angle_deg",
        "t_target_on", "t_end", "t_hold_complete", "duration_s", "reach_distance",
        "ring_label", "path_efficiency", "pair_id", "pair_complete", "path_row",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for record in records:
            writer.writerow([
                record.day, record.rec, record.trial_index, record.attempt_index,
                record.schedule_phase, record.outcome, record.failure_reason,
                record.target_index, record.target_name,
                record.target_x, record.target_y, record.target_radius, record.target_angle_deg,
                record.t_target_on, record.t_end, record.t_hold_complete, record.duration_s,
                record.reach_distance, record.ring_label, record.path_efficiency,
                record.pair_id, int(record.pair_complete), record.path_row,
            ])
    print(f"Saved attempt table: {out_path}")


def _write_npz(records: List[AttemptRecord], out_path: Path) -> None:
    """Concatenated ragged paths; records gain path_row indexing row_start/row_len."""
    keys = ("task_time_s", "cursor_x", "cursor_y", "joystick_x", "joystick_y")
    chunks: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    row_start: List[int] = []
    row_len: List[int] = []
    cursor = 0
    row = 0
    for record in records:
        if record.path is None:
            continue
        n = len(record.path["task_time_s"])
        record.path_row = row
        row_start.append(cursor)
        row_len.append(n)
        for key in keys:
            chunks[key].append(record.path[key])
        cursor += n
        row += 1
    arrays = {
        key: (np.concatenate(chunks[key]) if chunks[key] else np.empty(0, dtype=float))
        for key in keys
    }
    np.savez_compressed(
        out_path,
        row_start=np.asarray(row_start, dtype=np.int64),
        row_len=np.asarray(row_len, dtype=np.int64),
        **arrays,
    )
    print(f"Saved reach paths: {out_path} ({row} paths, {cursor} samples)")


def generate_co_structured_outputs(
    repo_root: str | Path,
    day: str,
    out_dir: str | Path | None = None,
    exclude_recs: Optional[List[int]] = None,
) -> Dict[str, object]:
    repo_root_path = Path(repo_root).resolve()
    day_dir = repo_root_path / day
    exclude_recs = exclude_recs or []
    out_dir_path = (
        Path(out_dir).resolve() if out_dir is not None
        else repo_root_path / "claude" / "figures" / day / "beh"
    )

    records: List[AttemptRecord] = []
    recs_used: List[int] = []
    recs_skipped: List[int] = []
    next_pair_id = 0
    ref_task_config: Optional[Dict[str, Any]] = None
    for rec in _rec_numbers(day_dir):
        if rec in exclude_recs:
            continue
        if not _rec_has_inputs(day_dir, rec):
            print(f"co_structured: rec{rec:03d} missing ev.mat/joystick.mat, skipping")
            recs_skipped.append(rec)
            continue
        try:
            dataset = load_joystick_dataset(repo_root_path, day, f"{rec:03d}")
        except Exception as exc:  # noqa: BLE001 - a bad rec must not kill the day
            print(f"co_structured: rec{rec:03d} failed to load ({exc}), skipping")
            recs_skipped.append(rec)
            continue
        if ref_task_config is None and dataset.task_configs:
            ref_task_config = dataset.task_configs[0]
        rec_records = _collect_rec(dataset, day, rec)
        next_pair_id = _tag_pairs(rec_records, next_pair_id)
        records.extend(rec_records)
        recs_used.append(rec)

    if not records:
        print("co_structured: no schedule_phase data found for this day, skipping")
        return {"skipped": "no schedule_phase data"}

    rings = tag_rings(records)
    if len(rings) > 1:
        print(f"co_structured: {len(rings)} rings detected at distances "
              f"{', '.join(f'{r:.3f}' for r in rings)}")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    _write_npz(records, out_dir_path / f"{day}_co_structured_paths.npz")
    _write_csv(records, out_dir_path / f"{day}_co_structured_attempts.csv")

    # Local import: co_structured_plots imports AttemptRecord/palette constants
    # back from this module, so a top-level import here would be circular.
    from pyCheck.co_structured_plots import plot_co_structured_metrics
    from pyCheck.co_structured_traj_plots import plot_co_structured_trajectories

    metrics = plot_co_structured_metrics(
        records, out_dir_path / f"{day}_co_structured_metrics.png", day, exclude_recs
    )
    metrics["trajectories"] = plot_co_structured_trajectories(
        records, out_dir_path / f"{day}_co_structured.png", day, exclude_recs,
        task_config=ref_task_config,
    )
    metrics["rings"] = rings
    metrics["recs_used"] = recs_used
    metrics["recs_skipped"] = recs_skipped
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured center-out reach paths + data from schedule_phase tags"
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--day")
    parser.add_argument("--day-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--exclude-recs", nargs="*", type=int, default=[])
    args = parser.parse_args()

    from pyCheck.day_presentation_plots import _resolve_day_paths

    repo_root, day = _resolve_day_paths(args)
    generate_co_structured_outputs(
        repo_root=repo_root,
        day=day,
        out_dir=args.out_dir,
        exclude_recs=args.exclude_recs,
    )


if __name__ == "__main__":
    main()
