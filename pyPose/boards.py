"""boards.py — parse the Thalamus ``Aruco Static`` config and build cv2 boards.

The behave ``*.json`` node graph carries an ``ARUCO`` node (``Aruco Static``) that
defines the fiducial boards, and one ``DISTORTION`` node per camera carrying that
camera's intrinsics. This module turns those definitions into dataclasses and into
``cv2.aruco`` board objects whose object points are expressed in **millimetres**.

Gotcha handled here: the ARUCO node's ``Sources`` (and the QC channel prefixes) are
*DISTORTION node names* (``"Distortion Cam Top"``), not camera names. We resolve
them to camera names via each DISTORTION node's own ``Source`` field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The static board that defines the world origin.
WORLD_BOARD = "Screen 2x2"
ARUCO_NODE = "Aruco Static"

# ArUco corner order is TL, TR, BR, BL in the marker's own plane (Z out of board).
_UNIT_CORNERS = np.array(
    [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class BoardDef:
    name: str
    ids: tuple[int, ...]
    marker_size_mm: float
    marker_sep_mm: float
    kind: str  # "grid" | "layout"
    cols: int | None = None
    rows: int | None = None
    # layout only: per-marker centre (mm) in board frame, keyed by marker id
    centers_mm: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ArucoConfig:
    dictionary_name: str
    boards: dict[str, BoardDef]
    source_to_camera: dict[str, str]  # "Distortion Cam Top" -> "Cam Top"
    world_board: str = WORLD_BOARD


def load_behave_config(path: str | Path) -> dict:
    """Read a behave ``*.json`` node-graph config."""
    with open(path) as fh:
        return json.load(fh)


def _find_aruco_node(config: dict, node_name: str = ARUCO_NODE) -> dict:
    for node in config["nodes"]:
        if node.get("type") == "ARUCO" and node.get("name") == node_name:
            if not node.get("Running", False):
                raise ValueError(f"ArUco node {node_name!r} is present but not Running")
            return node
    raise ValueError(f"No running ARUCO node named {node_name!r} in config")


def _source_to_camera(config: dict) -> dict[str, str]:
    """Map each DISTORTION node name -> the camera it is sourced from."""
    mapping: dict[str, str] = {}
    for node in config["nodes"]:
        if node.get("type") == "DISTORTION":
            src = node.get("Source")
            if not src:
                raise ValueError(f"DISTORTION node {node.get('name')!r} has no Source")
            mapping[node["name"]] = src
    return mapping


def _parse_board(raw: dict, mm: float = 1000.0) -> BoardDef:
    """Turn one raw board dict into a BoardDef (converting metres -> mm)."""
    name = raw["Name"]
    ids = tuple(int(i) for i in raw["ids"])
    size_mm = float(raw["Marker Size"]) * mm
    sep_mm = float(raw["Marker Separation"]) * mm
    is_layout = raw.get("Type") == "layout" or "Markers" in raw
    if is_layout:
        centers = {
            int(m["id"]): np.array([m["x"], m["y"], m.get("z", 0.0)], dtype=float) * mm
            for m in raw["Markers"]
        }
        return BoardDef(name, ids, size_mm, sep_mm, "layout", centers_mm=centers)
    return BoardDef(
        name, ids, size_mm, sep_mm, "grid",
        cols=int(raw["Columns"]), rows=int(raw["Rows"]),
    )


def parse_aruco_config(config: dict, *, node_name: str = ARUCO_NODE) -> ArucoConfig:
    """Build an :class:`ArucoConfig` from a loaded behave config dict."""
    node = _find_aruco_node(config, node_name)
    boards = {}
    for raw in node["Boards"]:
        bd = _parse_board(raw)
        boards[bd.name] = bd
    return ArucoConfig(
        dictionary_name=node["Dictionary"],
        boards=boards,
        source_to_camera=_source_to_camera(config),
    )


# --------------------------------------------------------------------------- #
# cv2.aruco board construction (object points in mm)
# --------------------------------------------------------------------------- #

def get_dictionary(name: str):
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _grid_centers_mm(bd: BoardDef) -> dict[int, np.ndarray]:
    """Centres (mm) of a GridBoard's markers, origin at the grid's geometric centre.

    ids fill row-major (matching cv2.aruco.GridBoard). +X right, +Y up.
    """
    pitch = bd.marker_size_mm + bd.marker_sep_mm
    cx = (bd.cols - 1) / 2.0
    cy = (bd.rows - 1) / 2.0
    centers = {}
    for idx, mid in enumerate(bd.ids):
        col = idx % bd.cols
        row = idx // bd.cols
        centers[mid] = np.array([(col - cx) * pitch, (cy - row) * pitch, 0.0])
    return centers


def board_marker_centers_mm(bd: BoardDef) -> dict[int, np.ndarray]:
    """Marker centres (mm) in the board's own frame for grid or layout boards."""
    if bd.kind == "grid":
        return _grid_centers_mm(bd)
    return dict(bd.centers_mm)


def board_object_points(bd: BoardDef) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (objPoints list of (4,3) float32, ids (M,) int32) in mm, board frame.

    Both boards are built explicitly (rather than via cv2.aruco.GridBoard) so the
    world board's origin sits at its geometric centre, matching the "board center"
    convention requested for the 3D world frame.
    """
    centers = board_marker_centers_mm(bd)
    obj = []
    ids = []
    for mid in bd.ids:
        c = centers[mid]
        obj.append((c + _UNIT_CORNERS * bd.marker_size_mm).astype(np.float32))
        ids.append(mid)
    return obj, np.asarray(ids, dtype=np.int32)


def build_board(bd: BoardDef, dictionary):
    """Build a ``cv2.aruco.Board`` (mm object points) for detection + pose."""
    import cv2

    obj, ids = board_object_points(bd)
    return cv2.aruco.Board(obj, dictionary, ids)
