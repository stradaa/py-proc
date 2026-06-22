"""Display-alignment latency computed straight from the thalamus behave file.

The legacy py_proc chain (procDisplay_new.m → proc_display.py) derives the
display latency on the *recorder* clock: it thresholds `recNNN.display.dat` and
subtracts task event times that were re-projected onto the recorder clock via
the fiducial alignment (`w_alignment.mat`). When any link in that chain drifts
— and on the Eevee AlexRig days since 260609 it does, by ~2 s — `disStartOn`
becomes meaningless (median ~1900 ms, most trials NaN) instead of the ~50–80 ms
true display latency.

This module sidesteps that entirely and follows the rule from the pyReplay
README: **`record.time` (ns) is the only clock, shared across every node, and
the task events' `time_perf_counter` are already on it** (single-machine rig,
`perf * 1e9` is directly a behave-clock nanosecond). So display latency is just:

    latency = (first photodiode rising edge at/after target_on) - target_on

with both times read from the same behave file — no recorder clock, no fiducial
fit, no `.dat` sidecars. The photodiode is the `Analog in` node's `Dev1/ai1`
span; per-sample timestamps are reconstructed exactly as
`pyReplay.loader` / `procThalamus_indie.py` do (end at `record.time`, step back
by `sample_intervals`).

The result is a `{"Rec": ..., "disStartOn": ...}` dict — the latency in ms,
relative to the target/start onset — which is exactly what
`day_presentation_plots.plot_display_alignment` expects (it reads only those two
keys), so the existing figure is reused unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from thalamus.record_reader2 import RecordReader

# AlexRig NIDAQ "Analog in" photodiode span (see procThalamus_indie.py / pyReplay).
PHOTODIODE_SPAN = "Dev1/ai1"

# Defaults for matching a target onset to its photodiode edge.
_DEFAULT_MAX_SEARCH_S = 0.4   # widest plausible display latency (s)
_DEFAULT_EDGE_TOL_S = 0.005   # allow an edge a hair before the logged perf time


def _sample_times_ns(t_last: int, n: int, interval_ns: int) -> np.ndarray:
    """Per-sample timestamps for an analog packet.

    `analog.time` is the timestamp of the last sample; earlier samples step back
    by `interval_ns` (matches DataFrameBuilder / pyReplay.loader semantics).
    """
    if n <= 1 or interval_ns <= 0:
        return np.full(n, int(t_last), dtype=np.int64)
    return int(t_last) + (np.arange(n) - (n - 1)) * int(interval_ns)


def _trial_target_on_ns(doc: dict) -> Optional[int]:
    """Earliest `target_on` (the start/onset stimulus) for a trial summary doc.

    Returns the behave-clock ns of the first attempt's first target onset, or
    None if the trial logged no target_on (e.g. an idle/timed-out trial).
    """
    br = doc.get("behav_result")
    if not isinstance(br, dict):
        return None
    attempts = br.get("attempts")
    if not attempts:
        fa = br.get("final_attempt")
        attempts = [fa] if isinstance(fa, dict) else []
    perfs: List[float] = []
    for att in attempts:
        if not isinstance(att, dict):
            continue
        for e in att.get("events") or []:
            if (isinstance(e, dict) and e.get("name") == "target_on"
                    and e.get("time_perf_counter") is not None):
                try:
                    perfs.append(float(e["time_perf_counter"]))
                except (TypeError, ValueError):
                    continue
    if not perfs:
        return None
    return int(min(perfs) * 1e9)


def _trial_outcome(doc: dict) -> str:
    br = doc.get("behav_result")
    if isinstance(br, dict):
        out = str(br.get("final_outcome", "")).lower().strip()
        if out:
            return out
    return "unknown"


def compute_rec_latencies(
    behave_path: str | Path,
    *,
    max_search_s: float = _DEFAULT_MAX_SEARCH_S,
    edge_tol_s: float = _DEFAULT_EDGE_TOL_S,
    task_types: Optional[Sequence[str]] = None,
    progress=None,
) -> Dict[str, np.ndarray]:
    """Per-trial display latency (ms) for one behave capture file.

    One linear `decode_video=False` pass collects the photodiode channel and
    every trial's target onset, then matches each onset to the first photodiode
    rising edge at/after it. Returns arrays keyed by:
      target_on_ns, edge_ns, latency_ms (NaN if unmatched), outcome (object).
    """
    behave_path = str(behave_path)
    want = ({str(t).strip().lower() for t in task_types}
            if task_types else None)

    photo_t: List[np.ndarray] = []
    photo_v: List[np.ndarray] = []
    target_on_ns: List[int] = []
    outcomes: List[str] = []

    with RecordReader(behave_path, decode_video=False) as reader:
        for record in reader:
            body = record.WhichOneof("body")
            if body == "analog" and record.node == "Analog in":
                analog = record.analog
                intervals = list(analog.sample_intervals)
                interval = int(intervals[0]) if intervals else 0
                for span in analog.spans:
                    if span.name != PHOTODIODE_SPAN:
                        continue
                    vals = np.asarray(analog.data[span.begin:span.end], dtype=float)
                    if len(vals):
                        photo_t.append(_sample_times_ns(analog.time, len(vals), interval))
                        photo_v.append(vals)
            elif body == "text":
                txt = record.text.text
                if txt.startswith("BehavState="):
                    continue
                try:
                    doc = json.loads(txt)
                except ValueError:
                    continue
                if not isinstance(doc, dict) or "task_config" not in doc:
                    continue
                if want is not None:
                    tt = str(doc.get("task_config", {}).get("task_type", "")).strip().lower()
                    if tt not in want:
                        continue
                t_on = _trial_target_on_ns(doc)
                if t_on is not None:
                    target_on_ns.append(t_on)
                    outcomes.append(_trial_outcome(doc))

    edge_ns = _rising_edges_ns(photo_t, photo_v)
    target_arr = np.asarray(target_on_ns, dtype=np.int64)
    latency_ms = np.full(len(target_arr), np.nan)
    matched_edge = np.full(len(target_arr), -1, dtype=np.int64)

    if len(edge_ns) and len(target_arr):
        win = int(max_search_s * 1e9)
        tol = int(edge_tol_s * 1e9)
        for i, t_on in enumerate(target_arr):
            idx = int(np.searchsorted(edge_ns, t_on - tol))
            if idx < len(edge_ns) and edge_ns[idx] <= t_on + win:
                matched_edge[i] = edge_ns[idx]
                latency_ms[i] = (edge_ns[idx] - t_on) / 1e6

    return {
        "target_on_ns": target_arr,
        "edge_ns": matched_edge,
        "latency_ms": latency_ms,
        "outcome": np.asarray(outcomes, dtype=object),
    }


def _rising_edges_ns(
    photo_t: List[np.ndarray], photo_v: List[np.ndarray]
) -> np.ndarray:
    """Threshold the pooled photodiode signal and return rising-edge times (ns).

    Threshold is the midpoint between the 2nd and 98th percentiles — robust to
    the occasional spike while tracking the clear low/high (≈0 V / ≈3 V) levels.
    """
    if not photo_t:
        return np.empty(0, dtype=np.int64)
    t = np.concatenate(photo_t)
    v = np.concatenate(photo_v)
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = v[order]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi - lo < 1e-6:
        return np.empty(0, dtype=np.int64)
    on = v > 0.5 * (lo + hi)
    rising = np.where((~on[:-1]) & (on[1:]))[0] + 1
    return t[rising].astype(np.int64)


def _resolve_behave_path(day_dir: Path, rec: int) -> Optional[Path]:
    """behave.<date>.<rec> for a rec number (mirrors pyReplay.run_replay)."""
    matches = sorted(p for p in day_dir.glob(f"behave.*.{rec}")
                     if p.suffix == f".{rec}")
    return matches[0] if matches else None


def _rec_numbers(day_dir: Path) -> List[int]:
    """Recording numbers present as rec subdirectories (001 → 1)."""
    recs = []
    for p in sorted(day_dir.iterdir()):
        if p.is_dir() and p.name.isdigit():
            recs.append(int(p.name))
    return recs


def day_display_data(
    day_dir: str | Path,
    *,
    recs: Optional[Sequence[int]] = None,
    exclude_recs: Optional[Sequence[int]] = None,
    task_types: Optional[Sequence[str]] = None,
    log=print,
) -> Dict[str, object]:
    """Behave-direct display latencies for a whole day, pooled across recs.

    Returns a dict with flat `Rec` and `disStartOn` arrays (one entry per matched
    trial) — drop-in input for `plot_display_alignment` — plus a `per_rec`
    summary and `n_matched` / `n_trials` counts.
    """
    day_dir = Path(day_dir).resolve()
    exclude = set(int(r) for r in (exclude_recs or []))
    rec_list = [int(r) for r in recs] if recs else _rec_numbers(day_dir)
    rec_list = [r for r in rec_list if r not in exclude]

    rec_out: List[int] = []
    lat_out: List[float] = []
    per_rec: Dict[str, Dict[str, float]] = {}
    n_trials_total = 0

    for rec in rec_list:
        behave_path = _resolve_behave_path(day_dir, rec)
        if behave_path is None:
            log(f"  display_behave: no behave capture for rec{rec:03d}, skipping")
            continue
        log(f"  display_behave: reading {behave_path.name} ...")
        res = compute_rec_latencies(behave_path, task_types=task_types)
        lat = res["latency_ms"]
        valid = lat[np.isfinite(lat)]
        n_trials_total += len(lat)
        rec_out.extend([rec] * len(valid))
        lat_out.extend(valid.tolist())
        per_rec[str(rec)] = {
            "n_trials": float(len(lat)),
            "n_matched": float(len(valid)),
            "median_ms": float(np.median(valid)) if len(valid) else float("nan"),
            "mean_ms": float(np.mean(valid)) if len(valid) else float("nan"),
            "min_ms": float(np.min(valid)) if len(valid) else float("nan"),
            "max_ms": float(np.max(valid)) if len(valid) else float("nan"),
        }
        log(f"    rec{rec:03d}: {len(valid)}/{len(lat)} trials matched, "
            f"median {per_rec[str(rec)]['median_ms']:.1f} ms")

    return {
        "Rec": np.asarray(rec_out, dtype=int),
        "disStartOn": np.asarray(lat_out, dtype=float),
        "per_rec": per_rec,
        "n_matched": int(len(lat_out)),
        "n_trials": int(n_trials_total),
    }
