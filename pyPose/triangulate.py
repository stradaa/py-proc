"""triangulate.py — multi-view triangulation utility (SLEAP-agnostic).

Given per-camera 2D points and the projection matrices from a calibration, recover
3D world points (mm). This is the seam a future SLEAP step will call — it takes
plain ``(N, 2)`` arrays and knows nothing about SLEAP.

Projection matrices are ``P = K [R | t]`` (see calibrate.py), so 2D points must be
in pixel coordinates of the ideal (K) camera; use :func:`undistort_points` first to
remove lens distortion while staying in pixels.
"""

from __future__ import annotations

import numpy as np


def undistort_points(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort (N,2) pixel points, returning (N,2) pixels in the ideal K camera."""
    import cv2

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, np.asarray(K), np.asarray(dist), P=np.asarray(K))
    return out.reshape(-1, 2)


def triangulate_pair(P1, P2, pts1, pts2) -> np.ndarray:
    """Triangulate (N,2)+(N,2) from two cameras -> (N,3) world mm."""
    import cv2

    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T
    X = cv2.triangulatePoints(np.asarray(P1), np.asarray(P2), pts1, pts2)
    return (X[:3] / X[3]).T


def triangulate_nview(Ps, pts, mask=None) -> np.ndarray:
    """Linear-DLT triangulation over >=2 views per point.

    ``Ps``: list of C (3,4) matrices. ``pts``: list of C (N,2) arrays (pixels in the
    ideal camera). ``mask``: (N,C) bool of which view sees each point (default: all
    finite). Rows with <2 views become NaN.
    """
    Ps = [np.asarray(P, dtype=np.float64) for P in Ps]
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in pts]
    n = pts[0].shape[0]
    c = len(Ps)
    if mask is None:
        mask = np.stack([np.isfinite(p).all(axis=1) for p in pts], axis=1)
    else:
        mask = np.asarray(mask, dtype=bool)

    out = np.full((n, 3), np.nan)
    for i in range(n):
        rows = []
        for j in range(c):
            if not mask[i, j]:
                continue
            x, y = pts[j][i]
            P = Ps[j]
            rows.append(x * P[2] - P[0])
            rows.append(y * P[2] - P[1])
        if len(rows) < 4:  # need >= 2 views (2 rows each)
            continue
        _, _, vt = np.linalg.svd(np.asarray(rows))
        X = vt[-1]
        out[i] = X[:3] / X[3]
    return out


def reproject(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project (N,3) world points through a (3,4) P -> (N,2) pixels."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xh = np.hstack([X, np.ones((X.shape[0], 1))])
    proj = (np.asarray(P) @ Xh.T).T
    return proj[:, :2] / proj[:, 2:3]


def triangulate_from_extrinsics(calib, cam_points: dict[str, np.ndarray],
                                *, undistort: bool = True) -> np.ndarray:
    """Triangulate matched (N,2) points given per-camera arrays keyed by camera name.

    ``calib`` is a :class:`pyPose.calibrate.Calibration`. Cameras missing from
    ``cam_points`` are simply not used.
    """
    cams = [c for c in cam_points if c in calib.cameras]
    Ps, pts = [], []
    for c in cams:
        ce = calib.cameras[c]
        p = np.asarray(cam_points[c], dtype=np.float64).reshape(-1, 2)
        pts.append(undistort_points(p, ce.K, ce.dist) if undistort else p)
        Ps.append(ce.P)
    return triangulate_nview(Ps, pts)
