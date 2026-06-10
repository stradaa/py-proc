"""Extract a replay window from a thalamus behave file.

A single fast pass (decode_video=False) over the behave file collects, for the
chosen [start, stop] window: decoded camera frames (each tagged with its own
record.time), joystick X/Y samples, the NIDAQ analog channels (fiducial /
photodiode / reward), and task event/target marks. Decoding video in this same
pass (decode_video=True) is what keeps the cameras synchronized with the other
streams — every frame is placed by its shared-clock timestamp, never by a
framerate.

Because records are time-ordered, the pass breaks as soon as record.time passes
the window stop, so a late window never reads the whole file.

Data access mirrors py_proc/procThalamus_indie.py (same node names, channel
mapping, and joystick span parsing) so the replay stays consistent with the
processing pipeline.
"""

from __future__ import annotations

import collections
import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from thalamus.record_reader2 import RecordReader

from pyReplay import cursor as cursor_mod
from pyReplay.cameras import frame_from_image
from pyReplay.window import (
    AnalogTrack,
    CameraTrack,
    EventMark,
    ReplayWindow,
    TargetSpec,
)

# AlexRig NIDAQ "Analog in" channel mapping (see procThalamus_indie.py).
ANALOG_CHANNELS = {
    "Dev1/ai0": "fiducial",
    "Dev1/ai1": "photodiode",
    "Dev1/ai2": "reward",
}

# Colors for known behavioral state / event labels (loose match to pyCheck).
_EVENT_COLORS = {
    "success": "#2ca02c",
    "fail": "#d62728",
    "failure": "#d62728",
}


def _sample_times_ns(analog_time: int, n: int, interval_ns: int) -> np.ndarray:
    """Per-sample timestamps for an analog packet.

    analog.time is the timestamp of the last sample; earlier samples are spaced
    `interval_ns` apart going backwards (matches DataFrameBuilder semantics).
    """
    if n <= 1 or interval_ns <= 0:
        return np.full(n, int(analog_time), dtype=np.int64)
    return int(analog_time) + (np.arange(n) - (n - 1)) * int(interval_ns)


def _joystick_samples(record_time_ns: int, analog):
    """(times_ns, x, y) for a Joystick analog record. Mirrors
    procThalamus_indie._extract_joystick_samples."""
    x_vals = y_vals = None
    interval = 0
    for span in analog.spans:
        if span.name == "X":
            x_vals = np.asarray(analog.data[span.begin:span.end], dtype=float)
        elif span.name == "Y":
            y_vals = np.asarray(analog.data[span.begin:span.end], dtype=float)
    if x_vals is None and y_vals is None:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    if x_vals is None:
        x_vals = np.full(len(y_vals), np.nan)
    if y_vals is None:
        y_vals = np.full(len(x_vals), np.nan)

    n = max(len(x_vals), len(y_vals))

    def _norm(arr):
        if len(arr) == n:
            return arr
        if len(arr) == 1:
            return np.repeat(arr, n)
        out = np.full(n, np.nan)
        out[:min(len(arr), n)] = arr[:min(len(arr), n)]
        return out

    x_vals, y_vals = _norm(x_vals), _norm(y_vals)
    intervals = np.asarray(analog.sample_intervals, dtype=np.int64)
    if len(intervals):
        interval = int(intervals[0])
    times = _sample_times_ns(record_time_ns, n, interval)
    return times.astype(np.int64), x_vals, y_vals


def _summary_targets(doc: dict):
    """Precise target-on intervals for a trial, from the per-attempt event stream.

    Each presentation is (t_on_ns, t_off_ns, x, y, radius, outcome). The target
    is shown from its `target_on` event until the attempt's resolving event
    (success / *_fail). Event `time_perf_counter` is on the same monotonic clock
    as record.time, so perf * 1e9 is directly a behave-clock nanosecond — no
    cross-clock fitting, and intertrial gaps fall outside every interval.
    """
    br = doc.get("behav_result")
    if not isinstance(br, dict):
        return []
    attempts = br.get("attempts")
    if not attempts:
        fa = br.get("final_attempt")
        attempts = [fa] if isinstance(fa, dict) else []

    out = []
    for att in attempts:
        if not isinstance(att, dict):
            continue
        evs = [e for e in (att.get("events") or []) if isinstance(e, dict)]
        perfs = [float(e["time_perf_counter"]) for e in evs
                 if e.get("time_perf_counter") is not None]
        if not perfs:
            continue
        t_off = max(perfs)
        outcome = "success" if any(e.get("name") == "success" for e in evs) else "fail"
        pos = att.get("target_position") or {}
        for e in evs:
            if e.get("name") != "target_on" or e.get("time_perf_counter") is None:
                continue
            x = e.get("target_x", pos.get("x_norm"))
            y = e.get("target_y", pos.get("y_norm"))
            r = e.get("target_radius_ratio", att.get("target_radius_ratio"))
            try:
                t_on = float(e["time_perf_counter"])
                x, y, r = float(x), float(y), float(r if r is not None else 0.0)
            except (TypeError, ValueError):
                continue
            out.append((int(t_on * 1e9), int(t_off * 1e9), x, y, r, outcome))
    return out


# Colors for trial outcome (target circle tint).
_OUTCOME_COLOR = {"success": "#4caf50", "fail": "#d62728", "failure": "#d62728"}


def load_window(
    behave_path: str,
    start_s: float,
    stop_s: float,
    *,
    camera_names: Optional[Sequence[str]] = None,
    target_h: int = 360,
    progress: Optional[Callable[[float], None]] = None,
) -> ReplayWindow:
    """Preload the [start_s, stop_s] window of `behave_path` into a ReplayWindow.

    start_s / stop_s are seconds relative to the recording start.
    camera_names=None auto-detects every camera node with frames in the window.

    Frames are decoded straight from the behave file (decode_video=True) so each
    one is paired with its own record.time; no .avi sidecars or framerates.
    """
    cam_frames: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    cam_times: Dict[str, List[int]] = collections.defaultdict(list)

    joy_t: List[np.ndarray] = []
    joy_x: List[np.ndarray] = []
    joy_y: List[np.ndarray] = []

    analog_t: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    analog_v: Dict[str, List[np.ndarray]] = collections.defaultdict(list)

    events: List[EventMark] = []
    target_intervals: List = []   # (t_on_ns, t_off_ns, x, y, r, outcome) per target
    control_cfg: dict = {}        # task_config used for cursor reconstruction

    t0_ns: Optional[int] = None
    start_ns = stop_ns = 0

    with RecordReader(behave_path, decode_video=True) as reader:
        last_prog = -1.0
        for record in reader:
            t = record.time
            if t:
                if t0_ns is None:
                    t0_ns = t
                    start_ns = int(t0_ns + start_s * 1e9)
                    stop_ns = int(t0_ns + stop_s * 1e9)
                if t > stop_ns:
                    break
            if progress is not None and stop_ns:
                frac = (t - start_ns) / max(1, (stop_ns - start_ns))
                if frac - last_prog >= 0.02:
                    progress(min(max(frac, 0.0), 1.0))
                    last_prog = frac

            body = record.WhichOneof("body")

            if body == "image":
                node = record.node
                if t0_ns is not None and start_ns <= t <= stop_ns:
                    if camera_names is None or node in camera_names:
                        frame = frame_from_image(record.image, target_h)
                        if frame is not None:
                            cam_frames[node].append(frame)
                            cam_times[node].append(t)
                continue

            in_window = t0_ns is not None and start_ns <= t <= stop_ns

            # Trial summaries are parsed even outside the window so the first
            # in-window trial's start boundary and the control config are known.
            if body == "text":
                text = record.text.text
                if text.startswith("BehavState="):
                    if in_window:
                        label = text.split("=", 1)[1].strip()
                        color = _EVENT_COLORS.get(label.lower(), "#555555")
                        events.append(EventMark(int(t), label, color))
                    continue
                try:
                    doc = json.loads(text)
                except ValueError:
                    doc = None
                if isinstance(doc, dict) and "task_config" in doc:
                    if not control_cfg:
                        control_cfg = doc["task_config"]
                    target_intervals.extend(_summary_targets(doc))
                continue

            if not in_window or body != "analog":
                continue

            analog = record.analog
            if record.node == "Joystick":
                ts, xs, ys = _joystick_samples(t, analog)
                if len(ts):
                    joy_t.append(ts); joy_x.append(xs); joy_y.append(ys)
            elif record.node == "Analog in":
                intervals = list(analog.sample_intervals)
                interval = int(intervals[0]) if intervals else 0
                for span in analog.spans:
                    label = ANALOG_CHANNELS.get(span.name)
                    if label is None:
                        continue
                    vals = np.asarray(analog.data[span.begin:span.end], dtype=float)
                    if not len(vals):
                        continue
                    analog_t[label].append(_sample_times_ns(analog.time, len(vals), interval))
                    analog_v[label].append(vals)

    if t0_ns is None:
        raise ValueError(f"No timestamped records found in {behave_path}")

    win = ReplayWindow(
        source=behave_path, t0_ns=t0_ns, start_ns=start_ns, stop_ns=stop_ns,
    )
    win.joystick_t_ns = np.concatenate(joy_t) if joy_t else np.empty(0, np.int64)
    win.joystick_x = np.concatenate(joy_x) if joy_x else np.empty(0)
    win.joystick_y = np.concatenate(joy_y) if joy_y else np.empty(0)
    # Joystick records may interleave; keep time-sorted.
    if len(win.joystick_t_ns):
        order = np.argsort(win.joystick_t_ns, kind="stable")
        win.joystick_t_ns = win.joystick_t_ns[order]
        win.joystick_x = win.joystick_x[order]
        win.joystick_y = win.joystick_y[order]

    for label in analog_t:
        ts = np.concatenate(analog_t[label])
        vs = np.concatenate(analog_v[label])
        order = np.argsort(ts, kind="stable")
        win.analog[label] = AnalogTrack(label, ts[order].astype(np.int64), vs[order])

    win.events = sorted(events, key=lambda e: e.t_ns)

    # Precise target intervals from trial events ([target_on, resolution]); keep
    # those overlapping the window. Intertrial gaps fall outside every interval.
    target_intervals.sort(key=lambda iv: iv[0])
    trial_starts = [iv[0] for iv in target_intervals]
    for t_on, t_off, x, y, r, outcome in target_intervals:
        if r > 0 and t_off > start_ns and t_on < stop_ns:
            spec = TargetSpec(t_on, t_off, x, y, r, outcome)
            spec.color = _OUTCOME_COLOR.get(outcome.lower(), "#4caf50")
            win.targets.append(spec)
    win.center_gate_radius = float(control_cfg.get("center_gate_radius_ratio", 0.0) or 0.0)

    # Reconstruct cursor (normalized [0,1]) from joystick for the target overlay.
    win.cursor_x, win.cursor_y = cursor_mod.reconstruct(
        win.joystick_t_ns, win.joystick_x, win.joystick_y, control_cfg, trial_starts,
    )

    nodes = camera_names if camera_names is not None else sorted(cam_frames)
    for node in nodes:
        times = np.asarray(cam_times.get(node, []), dtype=np.int64)
        if not len(times):
            continue
        win.cameras[node] = CameraTrack(node, cam_frames[node], times)

    return win
