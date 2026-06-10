"""pyReplay - interactive replay of thalamus behave-file time windows.

Reads behavioral data streams (cameras, joystick, analog NIDAQ channels,
task events) directly from a thalamus behave file and replays a chosen time
window as synchronized panels in a standalone PyQt6 window.

All data is read from the behave file via thalamus.RecordReader, including the
camera frames (decode_video=True), so every stream — video included — is
timestamped by the same shared monotonic clock (record.time). No .avi sidecars
or framerate metadata are used.
"""

from pyReplay.window import ReplayWindow, CameraTrack, AnalogTrack, EventMark

__all__ = ["ReplayWindow", "CameraTrack", "AnalogTrack", "EventMark"]
