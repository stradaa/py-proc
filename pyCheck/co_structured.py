"""co_structured.py — Structured center-out (CO) trial outputs for one day.

The Rust joystick executor tags every attempt in ``behav_result["attempts"]``
with ``schedule_phase``: ``"center"`` / ``"peripheral"`` (center-out mode),
``"sequence"`` (sequence mode) or ``"random"`` (Random mode, an interleave
insert, or a fallback pick). This module separates structured trials from
random inserts and writes, per day:

- ``<day>_co_structured.png`` — reach paths for structured peripheral /
  center / random-insert attempts plus success-rate bars.
- ``<day>_co_structured_attempts.csv`` — one row per attempt that carries a
  ``schedule_phase`` (rec, trial, attempt, phase, outcome, target identity,
  timing, pair tagging, and the row's slice into the NPZ).
- ``<day>_co_structured_paths.npz`` — pickle-free ragged path storage:
  concatenated ``task_time_s`` / ``cursor_x`` / ``cursor_y`` / ``joystick_x``
  / ``joystick_y`` with ``row_start`` / ``row_len`` indexed by the CSV's
  ``path_row`` column (``sl = slice(row_start[k], row_start[k] + row_len[k])``).

Pairing caveat: a shared ``pair_id`` is assigned to *adjacent*
(center, peripheral) attempts. Adjacency is guaranteed only for data recorded
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
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyCheck.joystick_validation import (
    JoystickDataset,
    get_attempt_segment,
    load_joystick_dataset,
    select_attempt_event,
)
from pyCheck.plot_utils import _save_figure, _title_suffix

STRUCTURED_PHASES = ("center", "peripheral", "sequence")

# Vega category10 minus the grays reserved for random inserts.
TARGET_PALETTE = [
    "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#6b4c9a",
]
RANDOM_COLOR = "#8a8a8a"
CENTER_COLOR = "#4c6a92"


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


def _attempt_times(attempt: Dict[str, Any]) -> tuple:
    target_on = select_attempt_event(attempt, "target_on", "first")
    end_name = "success" if str(attempt.get("outcome", "")).lower() == "success" else "fail"
    end_event = select_attempt_event(attempt, end_name, "last")
    t_on = float(target_on["time_perf_counter"]) if target_on is not None else np.nan
    t_end = float(end_event["time_perf_counter"]) if end_event is not None else np.nan
    return t_on, t_end


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
            t_on, t_end = _attempt_times(attempt)
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
                target_x=float(target_position.get("x_norm", np.nan)),
                target_y=float(target_position.get("y_norm", np.nan)),
                target_radius=float(attempt.get("target_radius_ratio") or np.nan),
                t_target_on=t_on,
                t_end=t_end,
                duration_s=float(attempt.get("duration_s") or np.nan),
            )
            try:
                segment = get_attempt_segment(dataset, trial_index, attempt)
                record.path = {
                    "task_time_s": np.asarray(segment.task_time_s, dtype=float),
                    "cursor_x": np.asarray(segment.cursor_x, dtype=float),
                    "cursor_y": np.asarray(segment.cursor_y, dtype=float),
                    "joystick_x": np.asarray(segment.joystick_x, dtype=float),
                    "joystick_y": np.asarray(segment.joystick_y, dtype=float),
                }
            except (ValueError, KeyError):
                record.path = None
            trial_records.append(record)
        records.extend(trial_records)
    return records


def _tag_pairs(records: List[AttemptRecord], next_pair_id: int) -> int:
    """Assign pair_id to adjacent (center, peripheral) attempts per trial."""
    by_trial: Dict[tuple, List[AttemptRecord]] = {}
    for record in records:
        by_trial.setdefault((record.rec, record.trial_index), []).append(record)
    for trial_records in by_trial.values():
        for a, b in zip(trial_records, trial_records[1:]):
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
        "target_x", "target_y", "target_radius",
        "t_target_on", "t_end", "duration_s",
        "pair_id", "pair_complete", "path_row",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for record in records:
            writer.writerow([
                record.day, record.rec, record.trial_index, record.attempt_index,
                record.schedule_phase, record.outcome, record.failure_reason,
                record.target_index, record.target_name,
                record.target_x, record.target_y, record.target_radius,
                record.t_target_on, record.t_end, record.duration_s,
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


def _draw_paths(ax, records: List[AttemptRecord], color_of) -> None:
    for record in records:
        if record.path is None:
            continue
        success = record.outcome == "success"
        ax.plot(
            record.path["cursor_x"], record.path["cursor_y"],
            color=color_of(record),
            linewidth=1.2 if success else 0.7,
            alpha=0.75 if success else 0.30,
            linestyle="-" if success else "--",
        )


def _setup_reach_axis(ax, title: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.15)


def _plot_figure(
    records: List[AttemptRecord],
    out_path: Path,
    day: str,
    exclude_recs: List[int],
) -> Dict[str, object]:
    peripheral = [r for r in records if r.schedule_phase == "peripheral"]
    center = [r for r in records if r.schedule_phase == "center"]
    sequence = [r for r in records if r.schedule_phase == "sequence"]
    structured = peripheral + center + sequence
    random_inserts = [r for r in records if r.schedule_phase == "random"]
    pairs = {r.pair_id for r in records if r.pair_id >= 0}
    complete_pairs = {r.pair_id for r in records if r.pair_id >= 0 and r.pair_complete}

    target_names = sorted({r.target_name or f"target {r.target_index}" for r in peripheral})
    target_color = {
        name: TARGET_PALETTE[i % len(TARGET_PALETTE)]
        for i, name in enumerate(target_names)
    }

    def peripheral_color(record: AttemptRecord) -> str:
        return target_color[record.target_name or f"target {record.target_index}"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    _draw_paths(ax_a, peripheral, peripheral_color)
    _draw_targets(ax_a, peripheral)
    if center:
        ax_a.scatter(
            [center[0].target_x], [center[0].target_y],
            marker="+", color="#222222", s=70, zorder=5,
        )
    _setup_reach_axis(ax_a, f"Structured peripheral reaches (n={len(peripheral)})")
    for name in target_names:
        ax_a.plot([], [], color=target_color[name], label=name)
    if target_names:
        ax_a.legend(loc="upper right", fontsize=7)

    _draw_paths(ax_b, center, lambda r: CENTER_COLOR)
    _draw_targets(ax_b, center)
    _setup_reach_axis(ax_b, f"Center (return) reaches (n={len(center)})")

    _draw_paths(ax_c, random_inserts, lambda r: RANDOM_COLOR)
    _draw_targets(ax_c, random_inserts, edge_color="#999999")
    _setup_reach_axis(ax_c, f"Random-insert reaches (n={len(random_inserts)})")

    labels: List[str] = []
    values: List[float] = []
    counts: List[int] = []
    colors: List[str] = []
    for name in target_names:
        group = [r for r in peripheral if (r.target_name or f"target {r.target_index}") == name]
        labels.append(name)
        values.append(_success_rate(group))
        counts.append(len(group))
        colors.append(target_color[name])
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
    ax_d.bar(x, values, color=colors, alpha=0.85)
    for xi, value, count in zip(x, values, counts):
        if np.isfinite(value):
            ax_d.text(xi, value + 0.02, f"{value:.0%}\nn={count}",
                      ha="center", va="bottom", fontsize=7)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_d.set_ylim(0.0, 1.15)
    ax_d.set_ylabel("success rate")
    ax_d.set_title("Performance: per target / structured vs random / pairs", fontsize=10)
    ax_d.grid(True, axis="y", alpha=0.15)

    fig.suptitle(f"{day} structured center-out summary{_title_suffix(exclude_recs)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, "co_structured")

    per_target = {
        name: {
            "n": counts[i],
            "success_rate": values[i] if np.isfinite(values[i]) else None,
        }
        for i, name in enumerate(target_names)
    }
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
        rec_records = _collect_rec(dataset, day, rec)
        next_pair_id = _tag_pairs(rec_records, next_pair_id)
        records.extend(rec_records)
        recs_used.append(rec)

    if not records:
        print("co_structured: no schedule_phase data found for this day, skipping")
        return {"skipped": "no schedule_phase data"}

    out_dir_path.mkdir(parents=True, exist_ok=True)
    _write_npz(records, out_dir_path / f"{day}_co_structured_paths.npz")
    _write_csv(records, out_dir_path / f"{day}_co_structured_attempts.csv")
    metrics = _plot_figure(
        records, out_dir_path / f"{day}_co_structured.png", day, exclude_recs
    )
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
