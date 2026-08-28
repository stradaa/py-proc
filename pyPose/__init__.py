"""pyPose — ArUco-based multi-camera 3D calibration for the AlexRig behavior rig.

Establishes a world coordinate frame anchored at the static 2x2 ArUco board and
registers all three cameras (Cam Left, Cam Top, Cam Rear) into it, with Cam Top
bridging Cam Rear (which sees only the Rail board) into the world frame. Provides a
SLEAP-agnostic triangulation utility and joint-view QC figures. Foundation for the
later SLEAP pose pipeline; SLEAP itself is not wired here.
"""

from __future__ import annotations

from .calibrate import Calibration, CameraExtrinsic, calibrate_recording
from .run_calibration import aruco_calibrate_rec
from .store import load_calibration, save_calibration
from .triangulate import triangulate_from_extrinsics, triangulate_nview

__all__ = [
    "Calibration",
    "CameraExtrinsic",
    "calibrate_recording",
    "aruco_calibrate_rec",
    "load_calibration",
    "save_calibration",
    "triangulate_from_extrinsics",
    "triangulate_nview",
]
