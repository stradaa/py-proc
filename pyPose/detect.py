"""detect.py — full-resolution frame access and per-board ArUco pose.

Frames are pulled at chosen ns timestamps from the per-rec ``image.mp4`` rebuilt by
the extraction pass (1:1 with ``<Cam>_image_raw_compressed.mat`` timestamps, so the
frame<->mat scaling is exactly 1.0 — same as ``pyCheck.joystick_validation``).
Detection uses the cv2.aruco 4.13 API: ``ArucoDetector.detectMarkers`` ->
``Board.matchImagePoints`` -> ``cv2.solvePnP`` (IPPE, planar boards).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class BoardObservation:
    rvec: np.ndarray       # (3, 1)  T_board_cam
    tvec: np.ndarray       # (3, 1)  mm
    n_markers: int
    reproj_px: float
    frame_ns: int
    obj_pts: np.ndarray    # (K, 3) mm, board frame
    img_pts: np.ndarray    # (K, 2) px


def make_detector(dictionary):
    """Build the detector with sub-pixel corner refinement enabled.

    ``DetectorParameters()`` defaults to ``CORNER_REFINE_NONE``, which localises
    corners to whole pixels and biases every marker quad slightly inward — the
    triangulated edge lengths come back systematically short of nominal.
    """
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


class FrameSource:
    """Random-access full-resolution grayscale frames for one camera."""

    def __init__(self, bag_dir: str | Path, cam: str) -> None:
        import cv2

        bag = Path(bag_dir)
        mat = loadmat(str(bag / "mat" / f"{cam}_image_raw_compressed.mat"),
                      simplify_cells=True)
        self.times_ns = np.asarray(mat["topic_time_stamp"], dtype=np.uint64).ravel()
        video = bag / "img" / cam / "image.mp4"
        if not video.exists():
            raise FileNotFoundError(f"Missing rebuilt video: {video}")
        self._cap = cv2.VideoCapture(str(video))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open {video}")
        self.cam = cam

    def frame_at(self, t_ns: int) -> tuple[int, np.ndarray] | None:
        import cv2

        idx = int(np.searchsorted(self.times_ns, np.uint64(t_ns), side="left"))
        idx = min(max(idx, 0), len(self.times_ns) - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return int(self.times_ns[idx]), np.ascontiguousarray(gray)

    def frames_at(self, times_ns) -> list[tuple[int, np.ndarray]]:
        out = []
        for t in np.asarray(times_ns).ravel():
            got = self.frame_at(int(t))
            if got is not None:
                out.append(got)
        return out

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def detect_board_pose(gray, board, detector, K, dist, *, min_markers: int = 2):
    """Detect a board in one frame and solve its pose. Returns None on failure."""
    import cv2

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < max(4, 4 * min_markers):
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj_px = float(np.sqrt(np.mean((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2)))
    return BoardObservation(
        rvec=rvec, tvec=tvec,
        n_markers=len(obj_pts) // 4,
        reproj_px=reproj_px,
        frame_ns=-1,
        obj_pts=obj_pts.reshape(-1, 3),
        img_pts=img_pts.reshape(-1, 2),
    )
