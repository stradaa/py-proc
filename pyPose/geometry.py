"""geometry.py — rigid-transform algebra for the ArUco multi-camera rig.

Convention
----------
A pose is a 4x4 homogeneous matrix ``T_A_B`` that maps a point expressed in
frame ``A`` to frame ``B``::

    p_B = T_A_B @ [p_A; 1]

OpenCV ``solvePnP(objPoints_in_board, imgPoints, ...)`` returns ``(rvec, tvec)``
with ``p_cam = R @ p_board + t``, i.e. it *is* ``T_board_cam`` (board -> camera).

All translations are in millimetres (board object points are built in mm), so
camera centres and triangulated points come out in mm.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 pose from a (3,3) rotation and a (3,) / (3,1) translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def T_to_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 pose into (R (3,3), t (3,))."""
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV (rvec, tvec) pose to a 4x4 matrix (T_board_cam)."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return rt_to_T(R, np.asarray(tvec, dtype=float).reshape(3))


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 pose (transpose rotation, re-map translation)."""
    R, t = T_to_rt(T)
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*Ts: np.ndarray) -> np.ndarray:
    """Left-to-right matrix product: ``compose(A, B, C) == A @ B @ C``."""
    out = np.eye(4)
    for T in Ts:
        out = out @ np.asarray(T, dtype=float)
    return out


def camera_center(T_world_cam: np.ndarray) -> np.ndarray:
    """World-frame position (3,) of a camera given its world->camera pose."""
    R, t = T_to_rt(T_world_cam)
    return -R.T @ t


def projection_matrix(K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """3x4 projection ``P = K [R | t]`` from intrinsics and a world->camera pose."""
    return np.asarray(K, dtype=float) @ np.asarray(T_world_cam, dtype=float)[:3, :4]


def geodesic_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angular distance (degrees) between two rotation matrices."""
    rel = Ra.T @ Rb
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def average_poses(
    Ts: list[np.ndarray],
    *,
    reject_deg: float = 5.0,
    reject_trans_mad: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Robustly average a list of 4x4 poses.

    Rotations are averaged with :meth:`scipy...Rotation.mean`; translations use a
    median. A single rejection pass drops poses whose rotation is more than
    ``reject_deg`` from the mean or whose translation is more than
    ``reject_trans_mad`` median-absolute-deviations from the median, then the mean
    is recomputed from the survivors.

    Returns ``(T_mean, stats)`` where ``stats`` reports counts and spread.
    """
    Ts = [np.asarray(T, dtype=float) for T in Ts]
    if not Ts:
        raise ValueError("average_poses: empty pose list")

    Rs = np.stack([T[:3, :3] for T in Ts])
    ts = np.stack([T[:3, 3] for T in Ts])

    def _mean(Rs_, ts_):
        R_mean = Rotation.from_matrix(Rs_).mean().as_matrix()
        t_mean = np.median(ts_, axis=0)
        return R_mean, t_mean

    R_mean, t_mean = _mean(Rs, ts)

    # Rejection pass.
    ang = np.array([geodesic_angle_deg(R_mean, R) for R in Rs])
    trans_dev = np.linalg.norm(ts - t_mean, axis=1)
    mad = np.median(np.abs(trans_dev - np.median(trans_dev))) or 1.0
    keep = (ang <= reject_deg) & (trans_dev <= np.median(trans_dev) + reject_trans_mad * mad)
    if keep.sum() >= 2 and keep.sum() < len(Ts):
        R_mean, t_mean = _mean(Rs[keep], ts[keep])
        ang_keep = ang[keep]
    else:
        keep = np.ones(len(Ts), dtype=bool)
        ang_keep = ang

    stats = {
        "n_input": len(Ts),
        "n_used": int(keep.sum()),
        "n_rejected": int((~keep).sum()),
        "rot_spread_deg": float(np.max(ang_keep)) if len(ang_keep) else 0.0,
        "trans_std_mm": float(np.std(ts[keep], axis=0).max()),
    }
    return rt_to_T(R_mean, t_mean), stats
