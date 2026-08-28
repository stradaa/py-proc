"""viz.py — the "joint view" acceptance figures for a rig calibration.

Two artifacts:
  * ``plot_overlays`` — one row per camera: the representative frame with detected
    marker corners and the reprojected world-origin axes drawn on top.
  * ``plot_scene_3d`` — the 3 cameras (position + orientation), every board plane,
    and the world origin in one 3D scene.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry as G  # noqa: E402
from . import triangulate as T  # noqa: E402

_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # X, Y, Z
_BOARD_COLORS = {"Screen 2x2": "#ff7f0e", "Rail 2x2": "#9467bd", "ROI": "#17becf"}
_BOARD_FALLBACK = ("#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f")

# The rig's real "up" is world +X: the Rail and ROI board normals sit within 3
# deg of it, and it is the axis the cameras' image-up vectors align with (Cam
# Top, the downward-looking one, has its optical axis along -X).  The world
# frame comes from the Screen 2x2 board's own marker layout, so its +Y is not
# gravity.  matplotlib always draws its own +Z vertically, so for the scene to
# read correctly -- Rail/ROI flat, Screen 2x2 upright -- world (X, Y, Z) is
# permuted to plot (Y, Z, X) at render time.  A 3-cycle, so handedness is
# preserved; the stored calibration stays in the world frame.
_DISPLAY_PERM = (1, 2, 0)
_DISPLAY_LABELS = ("world Y (mm)", "world Z (mm)", "world X (mm) — up")


def _disp(pts: np.ndarray) -> np.ndarray:
    """World (N,3) -> display (N,3). Valid for points and direction vectors."""
    return np.asarray(pts, dtype=float).reshape(-1, 3)[:, _DISPLAY_PERM]


def _board_colors(boards) -> dict[str, str]:
    """One distinct colour per board, honouring the known names."""
    spare = list(_BOARD_FALLBACK)
    out = {}
    for b in boards:
        if b in _BOARD_COLORS:
            out[b] = _BOARD_COLORS[b]
        else:
            out[b] = spare.pop(0) if spare else "#7f7f7f"
    return out


def _save(fig, out_path: Path, label: str) -> Path:
    print(f"Generating figure: {label}")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def _transform(T_mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return (T_mat[:3, :3] @ pts.T).T + T_mat[:3, 3]


def _board_world_corners(calib, board: str) -> np.ndarray:
    """Board marker corners expressed in world mm, shape (M,4,3)."""
    obj = np.asarray(calib.board_objpoints_mm[board], dtype=float)  # (M,4,3)
    board_to_world = G.invert_T(calib.board_world_pose[board])
    return _transform(board_to_world, obj.reshape(-1, 3)).reshape(obj.shape)


def plot_overlays(calib, out_path: str | Path, *, axis_len_mm: float = 30.0) -> Path | None:
    """2D overlay grid: detected markers + reprojected world axes per camera."""
    cams = [c for c in calib.cameras if c in calib.viz_frames]
    if not cams:
        return None
    fig, axes = plt.subplots(len(cams), 1, figsize=(7, 5.2 * len(cams)))
    if len(cams) == 1:
        axes = [axes]
    axes_pts = np.array([[0, 0, 0], [axis_len_mm, 0, 0], [0, axis_len_mm, 0], [0, 0, axis_len_mm]], float)
    colors = _board_colors(calib.board_objpoints_mm)
    for ax, cam in zip(axes, cams):
        ce = calib.cameras[cam]
        viz = calib.viz_frames[cam]
        ax.imshow(viz.gray, cmap="gray", origin="upper")
        for board, o in viz.obs.items():
            col = colors.get(board, "#7f7f7f")
            ax.scatter(o.img_pts[:, 0], o.img_pts[:, 1], s=18, edgecolor=col,
                       facecolor="none", linewidths=1.4, label=f"{board} ({o.n_markers}m)")
        proj = T.reproject(ce.P, axes_pts)
        o2d = proj[0]
        for k in range(3):
            ax.plot([o2d[0], proj[k + 1, 0]], [o2d[1], proj[k + 1, 1]],
                    color=_AXIS_COLORS[k], lw=2.5)
        ax.set_title(f"{cam}  |  {ce.via}  |  reproj {ce.reproj_px:.2f}px", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"ArUco overlays — {calib.day}/{calib.rec} (world axes = reprojected origin)",
                 fontsize=11)
    return _save(fig, Path(out_path), "aruco overlays")


def _set_equal_3d(ax, pts: np.ndarray) -> None:
    pts = np.asarray(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    r = float((hi - lo).max()) / 2 or 1.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def plot_scene_3d(calib, out_path: str | Path, *, axis_len_mm: float = 40.0,
                  elev: float = 22.0, azim: float = -60.0) -> Path:
    """3D scene: camera positions/orientations, every board, world origin.

    Drawn in the display frame (see ``_DISPLAY_PERM``) so the rig appears the way
    it sits: world +X (up) on the vertical axis, Rail/ROI flat, Screen upright.
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    allpts = [np.zeros(3)]
    colors = _board_colors(calib.board_objpoints_mm)

    # World origin triad (colours stay tied to the world axis, not the plot axis).
    O = _disp(np.zeros(3))[0]
    for k, col in enumerate(_AXIS_COLORS):
        v = _disp(np.eye(3)[k] * axis_len_mm * 1.5)[0]
        ax.plot([O[0], O[0] + v[0]], [O[1], O[1] + v[1]], [O[2], O[2] + v[2]], color=col, lw=2)
    ax.scatter([O[0]], [O[1]], [O[2]], color="k", s=30)

    # Boards as marker-corner quads.
    for board in calib.board_objpoints_mm:
        corners = _board_world_corners(calib, board)  # (M,4,3)
        col = colors.get(board, "#7f7f7f")
        for quad in corners:
            loop = _disp(np.vstack([quad, quad[0]]))
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=col, lw=1.5)
            allpts.append(_disp(quad))
        ax.plot([], [], [], color=col, label=board)

    # Cameras: centre + orientation axes (camera axes in world = R^T columns).
    for cam, ce in calib.cameras.items():
        C = _disp(ce.center_mm)[0]
        R, _ = G.T_to_rt(ce.T_world_cam)
        allpts.append(C[None, :])
        for k, col in enumerate(_AXIS_COLORS):
            d = _disp(R.T[:, k] * axis_len_mm)[0]
            ax.plot([C[0], C[0] + d[0]], [C[1], C[1] + d[1]], [C[2], C[2] + d[2]], color=col, lw=1.5)
        ax.scatter(*C, s=40, marker="^", color="k")
        ax.text(C[0], C[1], C[2], f"  {cam}", fontsize=9)

    _set_equal_3d(ax, np.vstack(allpts))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(_DISPLAY_LABELS[0]); ax.set_ylabel(_DISPLAY_LABELS[1])
    ax.set_zlabel(_DISPLAY_LABELS[2])
    ax.set_title(f"Joint 3-camera view — {calib.day}/{calib.rec}\n"
                 f"world = {calib.world_board} centre; vertical axis = world +X (rig up)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, Path(out_path), "3d rig scene")
