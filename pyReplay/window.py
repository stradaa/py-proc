"""Data model for a preloaded replay window.

A ReplayWindow holds everything needed to render the chosen [start, stop]
slice of a behave recording: decoded camera frames, joystick samples, analog
NIDAQ channels, and task event marks. All timestamps are nanoseconds on the
behave file's shared monotonic clock. The convenience accessors take a time
expressed in *seconds relative to the recording start* (t0_ns), matching the
window the user selects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraTrack:
    """Decoded frames for one camera within the window.

    frames are grayscale uint8 arrays (H, W), already downsized to the loader's
    target height. frame_times_ns[i] is the behave-clock timestamp of frames[i].
    """

    name: str
    frames: List[np.ndarray]
    frame_times_ns: np.ndarray

    def index_at(self, t_ns: int) -> int:
        """Index of the most recent frame at or before t_ns (clamped)."""
        if len(self.frame_times_ns) == 0:
            return -1
        idx = int(np.searchsorted(self.frame_times_ns, t_ns, side="right")) - 1
        return min(max(idx, 0), len(self.frames) - 1)

    def frame_at(self, t_ns: int) -> Optional[np.ndarray]:
        idx = self.index_at(t_ns)
        return self.frames[idx] if idx >= 0 else None


@dataclass
class AnalogTrack:
    """A uniformly-or-irregularly sampled analog signal within the window."""

    name: str
    t_ns: np.ndarray
    values: np.ndarray

    def slice(self, lo_ns: int, hi_ns: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t_ns, values) for samples in [lo_ns, hi_ns]."""
        if len(self.t_ns) == 0:
            return self.t_ns, self.values
        lo = int(np.searchsorted(self.t_ns, lo_ns, side="left"))
        hi = int(np.searchsorted(self.t_ns, hi_ns, side="right"))
        return self.t_ns[lo:hi], self.values[lo:hi]


@dataclass
class EventMark:
    """A task event (text record) at a point in time."""

    t_ns: int
    label: str
    color: str = "#888888"


@dataclass
class TargetSpec:
    """Joystick target active over a trial interval (from a summary record).

    The trial-summary text record is stamped at the trial *end* (t_end_ns); the
    target is shown over (t_start_ns, t_end_ns]. Coordinates are normalized
    [0,1] task space, matching the reconstructed cursor.
    """

    t_start_ns: int
    t_end_ns: int
    x: float
    y: float
    radius: float
    outcome: str = ""
    color: str = "#4caf50"


@dataclass
class ReplayWindow:
    """Preloaded, ready-to-render slice of a behave recording."""

    source: str
    t0_ns: int                 # recording start (first record.time)
    start_ns: int              # window start (absolute behave-clock ns)
    stop_ns: int               # window stop  (absolute behave-clock ns)

    cameras: Dict[str, CameraTrack] = field(default_factory=dict)

    joystick_t_ns: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    joystick_x: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    joystick_y: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    # Cursor position in normalized [0,1] task space (aligned to joystick_t_ns).
    cursor_x: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    cursor_y: np.ndarray = field(default_factory=lambda: np.empty(0, float))

    analog: Dict[str, AnalogTrack] = field(default_factory=dict)
    events: List[EventMark] = field(default_factory=list)
    targets: List[TargetSpec] = field(default_factory=list)
    center_gate_radius: float = 0.0   # normalized; 0 = unknown/none

    # ---- time helpers (seconds relative to recording start) ----
    @property
    def start_s(self) -> float:
        return (self.start_ns - self.t0_ns) / 1e9

    @property
    def stop_s(self) -> float:
        return (self.stop_ns - self.t0_ns) / 1e9

    @property
    def duration_s(self) -> float:
        return (self.stop_ns - self.start_ns) / 1e9

    def rel_to_ns(self, rel_s: float) -> int:
        return int(self.t0_ns + rel_s * 1e9)

    def ns_to_rel(self, t_ns: int) -> float:
        return (t_ns - self.t0_ns) / 1e9

    def joystick_index_at(self, t_ns: int) -> int:
        if len(self.joystick_t_ns) == 0:
            return -1
        idx = int(np.searchsorted(self.joystick_t_ns, t_ns, side="right")) - 1
        return min(max(idx, 0), len(self.joystick_t_ns) - 1)

    def active_target(self, t_ns: int) -> Optional[TargetSpec]:
        """Target whose trial interval (t_start, t_end] contains t_ns."""
        for tgt in self.targets:
            if tgt.t_start_ns < t_ns <= tgt.t_end_ns:
                return tgt
        return None

    def summary(self) -> str:
        cam = ", ".join(f"{n}:{len(t.frames)}f" for n, t in self.cameras.items())
        return (
            f"ReplayWindow {self.start_s:.2f}-{self.stop_s:.2f}s "
            f"({self.duration_s:.2f}s)\n"
            f"  cameras: {cam or 'none'}\n"
            f"  joystick: {len(self.joystick_t_ns)} samples\n"
            f"  analog: {', '.join(f'{n}:{len(t.t_ns)}' for n, t in self.analog.items()) or 'none'}\n"
            f"  events: {len(self.events)}, targets: {len(self.targets)}"
        )
