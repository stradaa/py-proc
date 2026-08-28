"""drift.py — per-day drift monitor across recordings.

Every rec is calibrated on its own, so a rec-to-rec difference is not an error —
but it does say how much the physical rig moved during the day. This module
answers two separate questions:

  * *between recs* — where does each ArUco marker sit in the world frame from one
    rec to the next, and how far does each camera move?
  * *within a rec* — how tightly do the ~40 calibration frames of a single rec
    agree with each other? That is the noise floor any between-rec number has to
    be read against.

Both land in one figure plus a printed summary.

CLI::

    ./.venv/bin/python -m pyPose.drift --day-dir <DAY> [--no-plot]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from . import geometry as G  # noqa: E402
from .store import load_calibration  # noqa: E402
from .viz import _BOARD_COLORS, _BOARD_FALLBACK, _save  # noqa: E402

_CAM_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def load_day(day_dir: str | Path):
    """Load every rec's calibration under a day dir. Returns ``[(rec, Calibration)]``."""
    day_dir = Path(day_dir)
    out = []
    for rec_dir in sorted(d for d in day_dir.iterdir() if d.is_dir() and d.name.isdigit()):
        rec = rec_dir.name
        aruco = rec_dir / f"rec{rec}.bag" / "aruco"
        if (aruco / "calibration.json").exists():
            out.append((rec, load_calibration(aruco)))
    return out


# --------------------------------------------------------------------------- #
# Between-rec: where the markers and cameras sit
# --------------------------------------------------------------------------- #

def marker_world_centers(cal) -> dict[tuple[str, int], np.ndarray]:
    """Marker centres in world mm, keyed by ``(board, marker_id)``.

    ``board_world_pose`` maps world -> board, so it is inverted to place the
    board's own object points into the world frame.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for board, obj in cal.board_objpoints_mm.items():
        board_to_world = G.invert_T(cal.board_world_pose[board])
        obj = np.asarray(obj, dtype=float).reshape(-1, 4, 3)
        ids = np.asarray(cal.board_ids[board]).ravel()
        for mid, quad in zip(ids, obj):
            centre = quad.mean(axis=0)
            out[(board, int(mid))] = board_to_world[:3, :3] @ centre + board_to_world[:3, 3]
    return out


def marker_drift(cals: list) -> dict:
    """Per-marker world position across recs, plus its displacement from the median.

    The world board's own markers are fixed at the origin by construction, so
    their drift is identically zero — they are kept as the reference floor.
    """
    per_rec = [marker_world_centers(c) for c in cals]
    keys = sorted(set.intersection(*(set(d) for d in per_rec)))
    out = {}
    for key in keys:
        P = np.stack([d[key] for d in per_rec])           # (n_recs, 3)
        med = np.median(P, axis=0)
        disp = np.linalg.norm(P - med, axis=1)            # (n_recs,)
        out[key] = {
            "positions_mm": P,
            "disp_mm": disp,
            "max_disp_mm": float(disp.max()),
            "span_mm": float(np.linalg.norm(P.max(0) - P.min(0))),
            "is_world_board": key[0] == cals[0].world_board,
        }
    return out


def board_orientation_drift(cals: list) -> dict:
    """Per-board angular deviation (deg) from the day-mean board orientation."""
    out = {}
    for board in cals[0].board_world_pose:
        Rs = [G.invert_T(c.board_world_pose[board])[:3, :3] for c in cals]
        Rmean = Rotation.from_matrix(Rs).mean().as_matrix()
        angs = np.array([G.geodesic_angle_deg(Rmean, R) for R in Rs])
        out[board] = {"angles_deg": angs, "max_deg": float(angs.max())}
    return out


def camera_drift(cals: list) -> dict:
    """Per-camera centre displacement from the day median, and orientation spread."""
    out = {}
    for cam in cals[0].cameras:
        C = np.stack([c.cameras[cam].center_mm for c in cals])
        med = np.median(C, axis=0)
        disp = np.linalg.norm(C - med, axis=1)
        Rs = [c.cameras[cam].T_world_cam[:3, :3] for c in cals]
        Rmean = Rotation.from_matrix(Rs).mean().as_matrix()
        angs = np.array([G.geodesic_angle_deg(Rmean, R) for R in Rs])
        out[cam] = {
            "centers_mm": C,
            "disp_mm": disp,
            "max_disp_mm": float(disp.max()),
            "center_span_mm": (C.max(0) - C.min(0)).tolist(),
            "center_std_mm": float(np.linalg.norm(C.std(0))),
            "orient_deg": angs,
            "orient_span_deg": float(angs.max()),
            "via": cals[0].cameras[cam].via,
        }
    return out


# --------------------------------------------------------------------------- #
# Within-rec: how well the frames of one rec agree
# --------------------------------------------------------------------------- #

def within_rec_consistency(cals: list) -> dict:
    """Per-camera within-rec pose spread, straight from each rec's pose_stats.

    ``trans_std_mm`` and ``rot_spread_deg`` come from the robust average over that
    rec's calibration frames, so they are the per-rec noise floor.
    """
    out = {}
    for cam in cals[0].cameras:
        stats = [c.cameras[cam].pose_stats for c in cals]
        out[cam] = {
            "trans_std_mm": np.array([s.get("trans_std_mm", np.nan) for s in stats]),
            "rot_spread_deg": np.array([s.get("rot_spread_deg", np.nan) for s in stats]),
            "n_used": np.array([s.get("n_used", 0) for s in stats]),
            "n_input": np.array([s.get("n_input", 0) for s in stats]),
            "reproj_px": np.array([c.cameras[cam].reproj_px for c in cals]),
        }
    return out


def day_drift(cals: list) -> dict:
    """Full drift summary: markers, boards, cameras, baselines, within-rec floor."""
    cams = list(cals[0].cameras)
    baselines = {}
    for a, b in itertools.combinations(cams, 2):
        d = np.array([float(np.linalg.norm(c.cameras[a].center_mm - c.cameras[b].center_mm))
                      for c in cals])
        baselines[f"{a} <-> {b}"] = {"values_mm": d, "mean": float(d.mean()),
                                     "span": float(d.max() - d.min())}
    return {
        "markers": marker_drift(cals),
        "boards": board_orientation_drift(cals),
        "cameras": camera_drift(cals),
        "baselines_mm": baselines,
        "within_rec": within_rec_consistency(cals),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def _board_color_map(boards) -> dict[str, str]:
    spare = list(_BOARD_FALLBACK)
    return {b: _BOARD_COLORS.get(b, spare.pop(0) if spare else "#7f7f7f") for b in boards}


def plot_day_drift(recs: list[str], summary: dict, world_board: str,
                   out_path: str | Path, day: str = "") -> Path:
    """2x2 drift figure: between-rec marker/camera motion over within-rec spread."""
    x = np.arange(len(recs))
    boards = list(summary["boards"])
    bcol = _board_color_map(boards)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) per-marker displacement from the day median.
    ax = axes[0, 0]
    seen = set()
    for (board, mid), m in sorted(summary["markers"].items()):
        label = board if board not in seen else None
        seen.add(board)
        ax.plot(x, m["disp_mm"], marker="o", ms=3, lw=1.2, color=bcol[board],
                alpha=0.85, label=label)
    ax.set_title(f"Per-marker drift between recs\n(distance from that marker's day median)",
                 fontsize=10)
    ax.set_ylabel("displacement (mm)")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.94, f"{world_board} markers are 0 by construction",
            transform=ax.transAxes, fontsize=7.5, style="italic", va="top")

    # (0,1) camera centre displacement + board orientation drift.
    ax = axes[0, 1]
    for k, (cam, c) in enumerate(sorted(summary["cameras"].items())):
        ax.plot(x, c["disp_mm"], marker="o", ms=3, lw=1.2,
                color=_CAM_COLORS[k % len(_CAM_COLORS)], label=cam)
    ax.set_title("Camera centre drift between recs\n(distance from that camera's day median)",
                 fontsize=10)
    ax.set_ylabel("displacement (mm)")
    ax.legend(fontsize=8)

    # (1,0) within-rec translation spread.
    ax = axes[1, 0]
    for k, (cam, w) in enumerate(sorted(summary["within_rec"].items())):
        ax.plot(x, w["trans_std_mm"], marker="s", ms=3, lw=1.2,
                color=_CAM_COLORS[k % len(_CAM_COLORS)], label=cam)
    ax.set_title("Within-rec consistency — translation\n(spread across that rec's calibration frames)",
                 fontsize=10)
    ax.set_ylabel("pose translation std (mm)")
    ax.legend(fontsize=8)

    # (1,1) within-rec rotation spread.
    ax = axes[1, 1]
    for k, (cam, w) in enumerate(sorted(summary["within_rec"].items())):
        ax.plot(x, w["rot_spread_deg"], marker="s", ms=3, lw=1.2,
                color=_CAM_COLORS[k % len(_CAM_COLORS)], label=cam)
    ax.set_title("Within-rec consistency — rotation\n(max frame deviation from that rec's mean pose)",
                 fontsize=10)
    ax.set_ylabel("pose rotation spread (deg)")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels([f"rec{r}" for r in recs], rotation=45, fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"ArUco calibration drift — {day} ({len(recs)} recs), world = {world_board} centre\n"
                 "top row: between recs   |   bottom row: within each rec",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, Path(out_path), "aruco day drift")


def print_report(day_dir: str | Path, make_plot: bool = True) -> dict:
    day_dir = Path(day_dir)
    loaded = load_day(day_dir)
    if len(loaded) < 2:
        print(f"Need >=2 calibrated recs under {day_dir}; found {len(loaded)}")
        return {}
    recs = [rec for rec, _ in loaded]
    cals = [c for _, c in loaded]
    summary = day_drift(cals)
    world_board = cals[0].world_board

    print(f"\nDrift across {len(cals)} recs ({', '.join(recs)}) — world = {world_board} centre")

    print("\nBETWEEN RECS — markers (world mm):")
    by_board: dict[str, list[float]] = {}
    for (board, mid), m in sorted(summary["markers"].items()):
        by_board.setdefault(board, []).append(m["max_disp_mm"])
    for board, vals in by_board.items():
        tag = "  (world board — 0 by construction)" if board == world_board else ""
        print(f"  {board:11s} max drift {max(vals):6.2f} mm   median {np.median(vals):6.2f} mm"
              f"   orient {summary['boards'][board]['max_deg']:5.2f} deg{tag}")

    print("\nBETWEEN RECS — cameras:")
    print(f"  {'camera':10s} {'via':28s} {'max drift':>10s} {'std':>7s} {'orient°':>8s}")
    for cam, c in sorted(summary["cameras"].items()):
        print(f"  {cam:10s} {c['via']:28s} {c['max_disp_mm']:9.2f}mm {c['center_std_mm']:7.2f} "
              f"{c['orient_span_deg']:8.2f}")
    print("  baselines (mm): " + "   ".join(
        f"{k}: {v['mean']:.1f} (span {v['span']:.1f})" for k, v in summary["baselines_mm"].items()))

    print("\nWITHIN REC — noise floor any number above must be read against:")
    print(f"  {'camera':10s} {'trans std (mm)':>22s} {'rot spread (deg)':>22s} {'frames used':>14s}")
    for cam, w in sorted(summary["within_rec"].items()):
        print(f"  {cam:10s} {np.nanmedian(w['trans_std_mm']):9.2f} "
              f"(max {np.nanmax(w['trans_std_mm']):5.2f}) "
              f"{np.nanmedian(w['rot_spread_deg']):9.2f} (max {np.nanmax(w['rot_spread_deg']):5.2f}) "
              f"{int(w['n_used'].sum()):6d}/{int(w['n_input'].sum()):<6d}")

    if make_plot:
        out_dir = day_dir / "aruco"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_day_drift(recs, summary, world_board, out_dir / "day_drift.png", day=day_dir.name)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Per-day ArUco calibration drift monitor")
    p.add_argument("--day-dir", required=True)
    p.add_argument("--no-plot", action="store_true", help="Skip the drift figure")
    args = p.parse_args(argv)
    print_report(Path(args.day_dir).expanduser().resolve(), make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
