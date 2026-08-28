"""run_calibration.py — CLI + per-rec entry for the ArUco 3-camera calibration.

Standalone::

    ./.venv/bin/python -m pyPose.run_calibration --day-dir <DAY> [--rec 001] [--no-plots]

Or from the pipeline via :func:`aruco_calibrate_rec`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibrate import calibrate_recording
from .qc import run_qc
from .store import save_calibration
from . import viz


def aruco_calibrate_rec(
    day_dir: str | Path,
    rec: str,
    *,
    make_plots: bool = True,
    max_frames: int = 40,
    max_reproj_px: float = 3.0,
) -> dict:
    """Calibrate one recording, run QC, and write NPZ+JSON (+ optional plots).

    Outputs land in ``<REC>.bag/aruco/`` (NPZ + JSON, no .mat). Returns the QC dict.
    """
    day_dir = Path(day_dir)
    rec = str(rec).zfill(3)
    calib = calibrate_recording(day_dir, rec, max_frames=max_frames, max_reproj_px=max_reproj_px)
    calib.qc = run_qc(calib)

    out_dir = day_dir / rec / f"rec{rec}.bag" / "aruco"
    save_calibration(calib, out_dir)
    if make_plots:
        viz.plot_overlays(calib, out_dir / "qc_overlays.png")
        viz.plot_scene_3d(calib, out_dir / "scene_3d.png")

    print(f"\n[aruco] {calib.day}/{rec}: QC {'PASS' if calib.qc['passed'] else 'FAIL'}")
    for cam, ce in calib.cameras.items():
        print(f"  {cam:10s} via {ce.via:24s} reproj {ce.reproj_px:5.2f}px "
              f"({calib.qc['reproj_grade'][cam]}) n={ce.n_frames}")
    if calib.qc["reasons"]:
        print("  reasons:", "; ".join(calib.qc["reasons"]))
    return calib.qc


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ArUco 3-camera 3D calibration for one day/rec")
    p.add_argument("--day-dir", required=True, help="Path to the day directory (YYMMDD)")
    p.add_argument("--rec", default=None, help="Rec to process (e.g. 001); default all recs")
    p.add_argument("--no-plots", action="store_true", help="Skip QC/3D figures")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--max-reproj-px", type=float, default=3.0)
    args = p.parse_args(argv)

    day_dir = Path(args.day_dir).expanduser().resolve()
    if args.rec is not None:
        recs = [str(args.rec).zfill(3)]
    else:
        recs = sorted(d.name for d in day_dir.iterdir() if d.is_dir() and d.name.isdigit())
    for rec in recs:
        try:
            aruco_calibrate_rec(day_dir, rec, make_plots=not args.no_plots,
                                max_frames=args.max_frames, max_reproj_px=args.max_reproj_px)
        except Exception as exc:  # noqa: BLE001 - report and continue across recs
            print(f"[aruco] {day_dir.name}/{rec}: FAILED — {exc}")


if __name__ == "__main__":
    main()
