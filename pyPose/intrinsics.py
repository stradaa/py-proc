"""intrinsics.py — per-camera K / distortion from the behave DISTORTION nodes.

Same source and read pattern as ``py_proc/procThalamus_indie.py`` (which writes
``<Cam>_camera_info.mat``): each ``DISTORTION`` node carries a ``Camera Matrix``
(K, 3x3) and ``Distortion Coefficients`` (plumb_bob, 5) and names the camera it is
sourced from via ``Source``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    name: str          # camera node name, e.g. "Cam Top"
    K: np.ndarray      # (3, 3) float64
    dist: np.ndarray   # (5,) float64, plumb_bob


def load_intrinsics(config: dict) -> dict[str, CameraIntrinsics]:
    """Return ``{camera_name: CameraIntrinsics}`` from a loaded behave config."""
    out: dict[str, CameraIntrinsics] = {}
    for node in config["nodes"]:
        if node.get("type") != "DISTORTION":
            continue
        cam = node.get("Source")
        if not cam:
            continue
        K = np.asarray(node["Camera Matrix"], dtype=float).reshape(3, 3)
        dist = np.asarray(node["Distortion Coefficients"], dtype=float).ravel()
        out[cam] = CameraIntrinsics(name=cam, K=K, dist=dist)
    if not out:
        raise ValueError("No DISTORTION nodes with intrinsics found in config")
    return out
