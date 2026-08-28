# pyPose — ArUco multi-camera 3D calibration (AlexRig)

Builds a stable 3D **world coordinate frame** for the Eevee behavior rig, anchored at
the static 2×2 ArUco board, and registers all three cameras (Cam Left, Cam Top, Cam
Rear) into that single frame. This is the geometric foundation for a later SLEAP pose
pipeline: once the 3D space is validated here, SLEAP 2D ROI predictions can be
triangulated into world XYZ. **SLEAP itself is not wired yet** — `triangulate.py` is
the ready seam.

## The rig, as recorded by Thalamus

The behave `*.json` node graph defines an `Aruco Static` node (dictionary
`DICT_4X4_50`, 27 mm markers) with two boards, plus one `DISTORTION` node per camera
carrying that camera's intrinsics (K + plumb_bob).

| Board | Marker IDs | Seen by | Role |
|---|---|---|---|
| **Screen 2x2** | 4, 5, 6, 7 | Cam Left, Cam Top | **World origin** (frame centre) |
| **Rail** | 8, 9 (31.2 mm apart) | Cam Top, Cam Rear | Bridge board |

**Cam Top is the hub** — it sees *both* boards, so it bridges Cam Rear (which sees
only the Rail) into the 2×2 world frame:

```
T_world_rail = invert(T_rail_top) @ T_world_top        # world -> rail, via the hub
T_world_rear = T_rail_rear @ T_world_rail              # world -> rear
```

### Important: what Thalamus persists

The `Aruco Static` node is stored as a **60 Hz analog stream of QC metrics only** —
`jitter_mm`, `n_markers`, `px_per_mm`, `reproj_px` per (camera, board) pair. **No pose
and no marker corners are saved.** So pyPose computes poses *offline*
(`cv2.aruco.detectMarkers` → `Board.matchImagePoints` → `cv2.solvePnP`, IPPE); the
persisted QC stream is used only to cheaply pick the cleanest calibration frames
(all boards fully detected, low reprojection error, before the animal occludes
anything). `Aruco Wand` (a 3-marker wand) is typically not running / not persisted and
is ignored.

## Conventions

- **Units: millimetres.** Board object points are built in mm, so poses, camera
  centres, and triangulated points are all mm.
- **World frame** = Screen 2×2 board **centre**; +X right, +Y up, +Z out of the board
  plane (right-handed).
- A pose `T_A_B` is 4×4 with `p_B = T_A_B @ [p_A; 1]`. `solvePnP` returns
  `T_board_cam`; a camera's projection is `P = K [R | t]` with `(R,t) = T_world_cam`.
- **Cadence: per-recording** — each recording is calibrated from its own frames.

## Usage

```bash
# One recording (writes NPZ+JSON + figures under <REC>.bag/aruco/)
./.venv/bin/python -m pyPose.run_calibration --day-dir <DAY_DIR> --rec 001

# All recordings in a day
./.venv/bin/python -m pyPose.run_calibration --day-dir <DAY_DIR>

# Per-day drift monitor (are the cameras actually stable across recs?)
./.venv/bin/python -m pyPose.drift --day-dir <DAY_DIR>
```

Or as a pipeline step (`py_proc/run_day_pipeline.py`): the `aruco_calib` step runs per
rec after extraction; `--skip-aruco` disables it, `--no-aruco-plots` skips figures.

Downstream (future SLEAP), reload a calibration and triangulate matched 2D points:

```python
from pyPose import load_calibration, triangulate_from_extrinsics
calib = load_calibration(".../rec001.bag/aruco")
xyz = triangulate_from_extrinsics(calib, {"Cam Left": pts_l, "Cam Top": pts_t})  # (N,3) mm
```

## Outputs (`<REC>.bag/aruco/`, no `.mat`)

- **`calibration.npz`** — per camera `P_<slug>` (3×4), `R_/t_/T_world_cam_/K_/dist_/center_`;
  `T_world_board_<slug>` (world→board); `objpoints_<slug>` / `ids_<slug>` (mm); `frame_times_ns`.
- **`calibration.json`** — units, boards, dictionary, per-camera `via`/reproj/frame counts,
  the full QC block, frame-selection criteria, axis convention.
- **`qc_overlays.png`** — per-camera frame with detected markers + reprojected world axes.
- **`scene_3d.png`** — the joint 3-camera view: camera positions/orientations, both
  board planes, world origin.

## QC / acceptance

- Per-camera reprojection grade: `< 2 px` good, `< 5 px` acceptable, else fail.
- Metric self-consistency: triangulate a board's corners and recover the nominal
  27 mm marker edge and (for the world board) a small direct 3D residual.
- Origin sanity, inter-camera baselines, per-frame pose spread.
- `drift.py` compares extrinsics across a day's recs.

## Module map

| Module | Role |
|---|---|
| `boards.py` | parse config, build cv2 boards in mm (world recentred to its centre) |
| `intrinsics.py` | per-camera K/dist from `DISTORTION` nodes (resolves `Distortion Cam X` → `Cam X`) |
| `aruco_qc.py` | read the QC stream (repo's own `RecordReader`) + frame selection |
| `detect.py` | full-res frame access + per-board `solvePnP` pose |
| `geometry.py` | rigid-transform algebra + robust pose averaging |
| `calibrate.py` | rig registration (hub bridge) → `Calibration` |
| `triangulate.py` | SLEAP-agnostic N-view triangulation |
| `qc.py` | verification checks |
| `store.py` | NPZ + JSON I/O |
| `viz.py` | overlay grid + 3D scene |
| `drift.py` | per-day drift monitor |
| `run_calibration.py` | CLI + per-rec entry |

## Validated (260720, 5 recs)

All recs PASS, sub-pixel reprojection (Left ~0.7 px, Top ~0.8 px, Rear ~0.4 px via the
bridge); world-board 3D residual ~0.4 mm; recovered marker edge ~27 mm. Cam Left/Top
(direct on the 2×2) held to ~1–2 mm / <0.15° across the day; Cam Rear (bridged) drifted
~6–7 mm mid-day when the Rail/Rear geometry physically shifted — caught by `drift.py`,
and the reason per-recording cadence is used for the rear camera.

## Notes / limitations

- Reads the behave stream with the repo's own `procThalamus_indie.RecordReader`; the
  thalamus `record_reader2` blocks on the multi-GB file.
- Requires the extraction step's rebuilt `image.mp4` + `<Cam>_image_raw_compressed.mat`.
- Cam Rear's large distortion (`k3 ≈ 7.6`) makes its undistortion the most sensitive;
  watch its reprojection.
- Deps (already in `.venv`): `opencv-contrib-python` (cv2.aruco 4.13 — `estimatePoseBoard`
  is gone; we use `matchImagePoints` + `solvePnP`), numpy 2.x, scipy, matplotlib.
