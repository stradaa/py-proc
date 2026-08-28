"""qc.py — verification checks for an ArUco rig calibration.

Turns a :class:`pyPose.calibrate.Calibration` into pass/fail QC: per-camera
reprojection grade, camera baselines, world-origin image sanity, pose stability,
and a metric triangulation acceptance test (recover known board geometry in mm).
"""

from __future__ import annotations

import numpy as np

from . import triangulate as T

GOOD_PX = 2.0
ACCEPTABLE_PX = 5.0


def reproj_grade(err_px: float) -> str:
    if not np.isfinite(err_px):
        return "fail"
    if err_px < GOOD_PX:
        return "good"
    if err_px < ACCEPTABLE_PX:
        return "acceptable"
    return "fail"


def baselines_mm(calib) -> dict[str, float]:
    cams = list(calib.cameras)
    out = {}
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            d = float(np.linalg.norm(calib.cameras[a].center_mm - calib.cameras[b].center_mm))
            out[f"{a} <-> {b}"] = d
    return out


def origin_in_image(calib) -> dict[str, dict]:
    """Project the world origin into each camera and report pixel + in-bounds."""
    out = {}
    for cam, ce in calib.cameras.items():
        px = T.reproject(ce.P, np.zeros((1, 3)))[0]
        viz = calib.viz_frames.get(cam)
        if viz is not None:
            h, w = viz.gray.shape[:2]
            inside = bool(0 <= px[0] < w and 0 <= px[1] < h)
        else:
            inside = None
        out[cam] = {"pixel": [float(px[0]), float(px[1])], "in_bounds": inside}
    return out


def _matched_image_points(calib, board: str):
    """Per-camera image points for a board, matched across cameras by object point.

    Points are ordered by the board's canonical (marker, corner) layout so each
    consecutive group of 4 is one marker's quad (keeps ``marker_edge_mm`` valid).
    """
    cams = [c for c, v in calib.viz_frames.items() if board in v.obs]
    per_cam_map = {}
    for c in cams:
        o = calib.viz_frames[c].obs[board]
        keys = [tuple(np.round(p, 1)) for p in o.obj_pts]
        per_cam_map[c] = {k: img for k, img in zip(keys, o.img_pts)}
    if len(cams) < 2:
        return cams, None, None
    canonical = [tuple(np.round(p, 1)) for p in
                 np.asarray(calib.board_objpoints_mm[board], dtype=float).reshape(-1, 3)]
    common = [k for k in canonical if all(k in per_cam_map[c] for c in cams)]
    if not common:
        return cams, None, None
    obj = np.array(common, dtype=float)
    img = {c: np.array([per_cam_map[c][k] for k in common]) for c in cams}
    return cams, obj, img


def reconstruct_board(calib, board: str) -> dict | None:
    """Triangulate a board's marker corners across the cameras that see it.

    For the world board, ``obj`` are world coords so we report a direct 3D residual
    (mm). For any board we report recovered marker edge length vs the nominal size.
    """
    cams, obj, img = _matched_image_points(calib, board)
    if obj is None:
        return {"board": board, "cameras": cams, "note": "need >=2 cameras with shared markers"}

    Ps = [calib.cameras[c].P for c in cams]
    Ks = [calib.cameras[c].K for c in cams]
    dists = [calib.cameras[c].dist for c in cams]
    pts = [T.undistort_points(img[c], Ks[i], dists[i]) for i, c in enumerate(cams)]
    X = T.triangulate_nview(Ps, pts)

    result = {"board": board, "cameras": cams, "n_points": int(np.isfinite(X).all(axis=1).sum())}

    # Recovered marker edge length (points are groups of 4 corners per marker).
    edges = []
    for s in range(0, len(X) - 3, 4):
        quad = X[s:s + 4]
        if np.isfinite(quad).all():
            edges += [np.linalg.norm(quad[k] - quad[(k + 1) % 4]) for k in range(4)]
    if edges:
        result["marker_edge_mm"] = {"mean": float(np.mean(edges)), "std": float(np.std(edges))}

    if calib.world_board == board:  # obj is in world coords -> direct residual
        res = np.linalg.norm(X - obj, axis=1)
        res = res[np.isfinite(res)]
        if len(res):
            result["residual_mm"] = {"mean": float(np.mean(res)), "max": float(np.max(res))}
    return result


def pose_stability(calib) -> dict[str, dict]:
    return {
        cam: {"rot_spread_deg": ce.pose_stats.get("rot_spread_deg"),
              "trans_std_mm": ce.pose_stats.get("trans_std_mm"),
              "n_frames": ce.n_frames}
        for cam, ce in calib.cameras.items()
    }


def run_qc(calib, *, min_frames: int = 10, max_reproj_px: float = ACCEPTABLE_PX) -> dict:
    """Assemble the full QC block and an overall pass/fail with reasons."""
    grades = {c: reproj_grade(ce.reproj_px) for c, ce in calib.cameras.items()}
    qc = {
        "reproj_px": {c: float(ce.reproj_px) for c, ce in calib.cameras.items()},
        "reproj_grade": grades,
        "via": {c: ce.via for c, ce in calib.cameras.items()},
        "baselines_mm": baselines_mm(calib),
        "origin_in_image": origin_in_image(calib),
        "pose_stability": pose_stability(calib),
        "reconstruction": {b: reconstruct_board(calib, b) for b in calib.board_objpoints_mm},
    }
    reasons = []
    for c, ce in calib.cameras.items():
        if grades[c] == "fail":
            reasons.append(f"{c}: reproj {ce.reproj_px:.2f}px")
        if ce.n_frames < min_frames:
            reasons.append(f"{c}: only {ce.n_frames} frames")
    qc["passed"] = len(reasons) == 0
    qc["reasons"] = reasons
    return qc
