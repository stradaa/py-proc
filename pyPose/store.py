"""store.py — persist / reload a Calibration as NPZ + JSON (no MATLAB .mat).

Arrays (projection matrices, poses, intrinsics, board geometry) go to
``calibration.npz``; human-readable metadata and QC go to ``calibration.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import Calibration, CameraExtrinsic


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_calibration(calib: Calibration, out_dir: str | Path) -> Path:
    """Write ``calibration.npz`` + ``calibration.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"frame_times_ns": calib.frame_times_ns}
    for cam, ce in calib.cameras.items():
        s = _slug(cam)
        arrays[f"P_{s}"] = ce.P
        arrays[f"R_{s}"] = ce.T_world_cam[:3, :3]
        arrays[f"t_{s}"] = ce.T_world_cam[:3, 3]
        arrays[f"T_world_cam_{s}"] = ce.T_world_cam
        arrays[f"K_{s}"] = ce.K
        arrays[f"dist_{s}"] = ce.dist
        arrays[f"center_{s}"] = ce.center_mm
    for board, Tw in calib.board_world_pose.items():
        arrays[f"T_world_board_{_slug(board)}"] = Tw
    for board, obj in calib.board_objpoints_mm.items():
        arrays[f"objpoints_{_slug(board)}"] = obj
        arrays[f"ids_{_slug(board)}"] = calib.board_ids[board]

    npz_path = out_dir / "calibration.npz"
    np.savez(npz_path, **arrays)

    meta = {
        "day": calib.day,
        "rec": calib.rec,
        "units": calib.units,
        "world_board": calib.world_board,
        "dictionary": calib.dictionary_name,
        "cameras": {
            cam: {
                "slug": _slug(cam),
                "via": ce.via,
                "reproj_px": ce.reproj_px,
                "reproj_grade": calib.qc.get("reproj_grade", {}).get(cam),
                "n_frames": ce.n_frames,
                "n_markers_median": ce.n_markers_median,
                "center_mm": ce.center_mm,
                "pose_stats": ce.pose_stats,
            }
            for cam, ce in calib.cameras.items()
        },
        "visibility": calib.visibility,
        "board_slugs": {b: _slug(b) for b in calib.board_objpoints_mm},
        "frame_selection": calib.selection,
        "qc": calib.qc,
        "npz": "calibration.npz",
        "axis_convention": "world = Screen 2x2 board centre; +X right, +Y up, +Z out of board plane (right-handed); units mm.",
    }
    json_path = out_dir / "calibration.json"
    with open(json_path, "w") as fh:
        json.dump(_jsonify(meta), fh, indent=2)
    return json_path


def load_calibration(path: str | Path) -> Calibration:
    """Reload a Calibration from its ``calibration.json`` (or containing dir)."""
    path = Path(path)
    json_path = path / "calibration.json" if path.is_dir() else path
    with open(json_path) as fh:
        meta = json.load(fh)
    npz = np.load(json_path.parent / meta["npz"], allow_pickle=False)

    cameras = {}
    for cam, cm in meta["cameras"].items():
        s = cm["slug"]
        cameras[cam] = CameraExtrinsic(
            name=cam,
            T_world_cam=npz[f"T_world_cam_{s}"],
            P=npz[f"P_{s}"],
            K=npz[f"K_{s}"],
            dist=npz[f"dist_{s}"],
            center_mm=npz[f"center_{s}"],
            via=cm["via"],
            reproj_px=cm["reproj_px"],
            n_frames=cm["n_frames"],
            n_markers_median=cm["n_markers_median"],
            pose_stats=cm.get("pose_stats", {}),
        )
    board_slugs = meta["board_slugs"]
    return Calibration(
        day=meta["day"], rec=meta["rec"], units=meta["units"],
        world_board=meta["world_board"], dictionary_name=meta["dictionary"],
        cameras=cameras,
        board_world_pose={b: npz[f"T_world_board_{s}"] for b, s in board_slugs.items()},
        board_objpoints_mm={b: npz[f"objpoints_{s}"] for b, s in board_slugs.items()},
        board_ids={b: npz[f"ids_{s}"] for b, s in board_slugs.items()},
        frame_times_ns=npz["frame_times_ns"],
        selection=meta["frame_selection"],
        visibility=meta["visibility"],
        qc=meta.get("qc", {}),
    )
