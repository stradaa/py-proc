"""calibrate.py — register the three cameras into the 2x2 board's world frame.

Per recording:
  1. parse config, build boards, load intrinsics;
  2. use the QC stream to pick clean frames, then run offline detection to get each
     (camera, board) pose ``T_board_cam``, aggregated robustly over frames;
  3. compose world (Screen 2x2) extrinsics for every camera, using Cam Top as the
     hub that bridges Cam Rear (rail-only) into the world frame;
  4. emit projection matrices + a :class:`Calibration`.

Bridge algebra (see geometry.py for the T_A_B convention):
  world board = W, a bridged board = B, hub camera = h (sees both), target cam = c::

      T_W_B = compose(invert(T_B_h), T_W_h)          # world -> B, via the hub
      T_W_c = compose(T_B_c, T_W_B)                  # world -> c
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as G
from .boards import ArucoConfig, build_board, board_object_points, get_dictionary, load_behave_config, parse_aruco_config
from .detect import BoardObservation, FrameSource, detect_board_pose, make_detector
from .intrinsics import CameraIntrinsics, load_intrinsics
from .aruco_qc import QCTable, SelectedFrames, read_aruco_qc, select_calib_frames


@dataclass
class AggPose:
    T_board_cam: np.ndarray      # (4, 4)
    reproj_px: float             # median over frames
    n_frames: int
    n_markers_median: int
    stats: dict


@dataclass
class CameraExtrinsic:
    name: str
    T_world_cam: np.ndarray      # (4, 4)
    P: np.ndarray                # (3, 4)
    K: np.ndarray
    dist: np.ndarray
    center_mm: np.ndarray        # (3,)
    via: str
    reproj_px: float
    n_frames: int
    n_markers_median: int
    pose_stats: dict


@dataclass
class VizFrame:
    """A representative frame for one camera plus its per-board detections."""
    frame_ns: int
    gray: np.ndarray                        # (H, W) uint8
    obs: dict[str, BoardObservation]        # board name -> detection in this frame


@dataclass
class Calibration:
    day: str
    rec: str
    units: str
    world_board: str
    dictionary_name: str
    cameras: dict[str, CameraExtrinsic]
    board_world_pose: dict[str, np.ndarray]        # board name -> T_world_board (4,4)
    board_objpoints_mm: dict[str, np.ndarray]      # board name -> (M,4,3)
    board_ids: dict[str, np.ndarray]
    frame_times_ns: np.ndarray
    selection: dict
    visibility: dict[str, list[str]]
    qc: dict = field(default_factory=dict)
    # In-memory only (not serialized): for QC + visualization.
    viz_frames: dict[str, VizFrame] = field(default_factory=dict)


def aggregate_pose(obs: list[BoardObservation]) -> AggPose:
    Ts = [G.rvec_tvec_to_T(o.rvec, o.tvec) for o in obs]
    T_mean, stats = G.average_poses(Ts)
    return AggPose(
        T_board_cam=T_mean,
        reproj_px=float(np.median([o.reproj_px for o in obs])),
        n_frames=len(obs),
        n_markers_median=int(np.median([o.n_markers for o in obs])),
        stats=stats,
    )


def observe_all(
    bag_dir: Path,
    config: ArucoConfig,
    intr: dict[str, CameraIntrinsics],
    pairs: list[tuple[str, str]],
    times_ns: np.ndarray,
    *,
    min_obs: int = 5,
):
    """Detect + aggregate a pose for every (camera, board) pair.

    Returns ``(agg, viz)`` where ``viz[cam]`` is a :class:`VizFrame` (the frame with
    the most detected markers for that camera, kept for QC + overlays).
    """
    dictionary = get_dictionary(config.dictionary_name)
    detector = make_detector(dictionary)
    boards = {name: build_board(bd, dictionary) for name, bd in config.boards.items()}
    cams = sorted({c for c, _ in pairs})

    agg: dict[tuple[str, str], AggPose] = {}
    viz: dict[str, "VizFrame"] = {}
    for cam in cams:
        cam_pairs = [b for (c, b) in pairs if c == cam]
        K, dist = intr[cam].K, intr[cam].dist
        with FrameSource(bag_dir, cam) as src:
            frames = src.frames_at(times_ns)
            per_pair: dict[str, list[BoardObservation]] = {b: [] for b in cam_pairs}
            per_frame: dict[int, dict[str, BoardObservation]] = {}
            grays: dict[int, np.ndarray] = {}
            for frame_ns, gray in frames:
                for board_name in cam_pairs:
                    o = detect_board_pose(gray, boards[board_name], detector, K, dist)
                    if o is not None:
                        o.frame_ns = frame_ns
                        per_pair[board_name].append(o)
                        per_frame.setdefault(frame_ns, {})[board_name] = o
                        grays[frame_ns] = gray
            for board_name, obs in per_pair.items():
                if len(obs) >= min_obs:
                    agg[(cam, board_name)] = aggregate_pose(obs)
            if per_frame:
                best_ns = max(per_frame,
                              key=lambda ns: sum(o.n_markers for o in per_frame[ns].values()))
                viz[cam] = VizFrame(best_ns, grays[best_ns], per_frame[best_ns])
    return agg, viz


def compose_extrinsics(
    agg: dict[tuple[str, str], AggPose],
    world_board: str,
) -> tuple[dict[str, tuple[np.ndarray, str]], dict[str, np.ndarray]]:
    """Return (per-camera (T_world_cam, via), board_world_pose{board->T_world_board})."""
    sees: dict[str, set[str]] = {}
    for cam, board in agg:
        sees.setdefault(cam, set()).add(board)

    direct = [c for c, bs in sees.items() if world_board in bs]
    if not direct:
        raise RuntimeError(f"No camera sees the world board {world_board!r}")

    board_world = {world_board: np.eye(4)}
    # For each non-world board, bridge via a hub camera that sees both.
    for cam, board in list(agg.keys()):
        if board in board_world:
            continue
        hubs = [h for h in direct if board in sees[h]]
        if not hubs:
            continue
        hub = min(hubs, key=lambda h: agg[(h, board)].reproj_px + agg[(h, world_board)].reproj_px)
        T_W_B = G.compose(G.invert_T(agg[(hub, board)].T_board_cam),
                          agg[(hub, world_board)].T_board_cam)
        board_world[board] = T_W_B

    extr: dict[str, tuple[np.ndarray, str]] = {}
    for cam, bs in sees.items():
        if world_board in bs:
            extr[cam] = (agg[(cam, world_board)].T_board_cam, f"direct:{world_board}")
            continue
        # Pick the bridge board by pose quality, not by iteration order.  `bs` is
        # a set of board *names*, and Python randomises string hashing per
        # process, so iterating it directly returns a different board from one
        # run to the next -- the same rec would calibrate differently each time.
        # Score a candidate by this camera's pose error on the board plus the
        # hub's; lowest wins, ties break on the board name.
        options = []
        for board in sorted(bs):
            if board not in board_world:
                continue
            hubs = [h for h in direct if board in sees[h]]
            if not hubs:
                continue
            hub = min(hubs, key=lambda h: agg[(h, board)].reproj_px)
            options.append((agg[(cam, board)].reproj_px + agg[(hub, board)].reproj_px, board, hub))
        if not options:
            continue
        _, board, hub = min(options, key=lambda o: (o[0], o[1]))
        T_world_cam = G.compose(agg[(cam, board)].T_board_cam, board_world[board])
        extr[cam] = (T_world_cam, f"bridge:{board} via {hub}")
    return extr, board_world


def calibrate_recording(
    day_dir: str | Path,
    rec: str,
    *,
    behave_json: str | Path | None = None,
    max_frames: int = 40,
    max_reproj_px: float = 3.0,
    max_qc_duration_s: float | None = 120.0,
) -> Calibration:
    """Compute the per-recording ArUco 3-camera calibration."""
    day_dir = Path(day_dir)
    day = day_dir.name
    rec = str(rec).zfill(3)
    bag_dir = day_dir / rec / f"rec{rec}.bag"

    behave_path, json_path = _resolve_behave(day_dir, rec, behave_json)
    config = parse_aruco_config(load_behave_config(json_path))
    intr = load_intrinsics(load_behave_config(json_path))

    qc = read_aruco_qc(behave_path, config, max_duration_s=max_qc_duration_s)
    pairs = qc.pairs
    sel = select_calib_frames(qc, config, required_pairs=pairs,
                              max_reproj_px=max_reproj_px, max_frames=max_frames)

    agg, viz = observe_all(bag_dir, config, intr, pairs, sel.times_ns)
    if not agg:
        raise RuntimeError(f"No board poses detected for {day}/{rec}")
    extr, board_world = compose_extrinsics(agg, config.world_board)

    obj_by_board = {}
    ids_by_board = {}
    for name, bd in config.boards.items():
        obj, ids = board_object_points(bd)
        obj_by_board[name] = np.stack(obj)
        ids_by_board[name] = ids

    cameras: dict[str, CameraExtrinsic] = {}
    for cam, (T_world_cam, via) in extr.items():
        # reproj for the pose the extrinsic came from
        board_used = config.world_board if via.startswith("direct") else via.split(":")[1].split(" via")[0]
        a = agg[(cam, board_used)]
        K, dist = intr[cam].K, intr[cam].dist
        cameras[cam] = CameraExtrinsic(
            name=cam,
            T_world_cam=T_world_cam,
            P=G.projection_matrix(K, T_world_cam),
            K=K, dist=dist,
            center_mm=G.camera_center(T_world_cam),
            via=via,
            reproj_px=a.reproj_px,
            n_frames=a.n_frames,
            n_markers_median=a.n_markers_median,
            pose_stats=a.stats,
        )

    visibility = {c: sorted(b for (cc, b) in agg if cc == c) for c in cameras}
    return Calibration(
        day=day, rec=rec, units="mm",
        world_board=config.world_board,
        dictionary_name=config.dictionary_name,
        cameras=cameras,
        board_world_pose=board_world,
        board_objpoints_mm=obj_by_board,
        board_ids=ids_by_board,
        frame_times_ns=sel.times_ns,
        selection={"n_candidates": sel.n_candidates, "relaxed": sel.relaxed, **sel.criteria},
        visibility=visibility,
        viz_frames=viz,
    )


def _resolve_behave(day_dir: Path, rec: str, behave_json):
    """Find the behave capture file + its json for a rec (e.g. behave.YYYYMMDD.N)."""
    rec_int = int(rec)
    day = day_dir.name
    stem = f"behave.20{day}.{rec_int}"
    capture = day_dir / stem
    json_path = Path(behave_json) if behave_json else day_dir / f"{stem}.json"
    if not capture.exists():
        raise FileNotFoundError(f"Behave capture not found: {capture}")
    if not json_path.exists():
        raise FileNotFoundError(f"Behave config json not found: {json_path}")
    return capture, json_path
