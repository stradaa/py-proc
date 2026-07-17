"""Rapid testbed for choosing per-camera brightness/contrast/gamma values.

Grabs a few frames per camera from the first seconds of a behave file (so it
decodes in ~2 s, not the whole recording) and renders a comparison grid PNG:
one row per camera, one column per candidate correction. Open the PNG, pick the
correction that looks best for each camera, then set those values in
pyReplay/adjust.py's DEFAULTS and re-run to confirm.

Usage:
    python -m pyReplay.tune_contrast BEHAVE_PATH [--out PNG] [--n 3] [--scan 4]
"""

from __future__ import annotations

import argparse
import collections
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure

from thalamus.record_reader2 import RecordReader

from pyReplay.adjust import CameraAdjust
from pyReplay.cameras import frame_from_image

# Same clock-glitch guard as loader.load_window: some behave files interleave
# spurious-node records whose timestamps are ~1e17 ns (years off) and must not be
# trusted as real behave-clock times. See pyReplay/loader.py.
_MAX_REC_NS = 24 * 3600 * 1_000_000_000  # 24 h in ns


def grab_sample_frames(
    behave_path: str,
    n_per_cam: int = 3,
    max_scan_s: float = 4.0,
    camera_names: Optional[Sequence[str]] = None,
    target_h: int = 360,
) -> Dict[str, List[np.ndarray]]:
    """Collect up to `n_per_cam` frames per camera from the file's first seconds.

    Reads from t0 with decode_video=True, stopping once every camera has its
    quota or `max_scan_s` of behave-clock time has elapsed. Returns
    {camera_name: [frame, ...]} with grayscale uint8 frames downsized to target_h.
    """
    frames: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    t0_ns: Optional[int] = None
    stop_ns = 0
    with RecordReader(behave_path, decode_video=True) as reader:
        for record in reader:
            t = record.time
            if t:
                if t0_ns is None:
                    t0_ns = t
                    stop_ns = int(t0_ns + max_scan_s * 1e9)
                elif abs(t - t0_ns) > _MAX_REC_NS:
                    continue  # clock-glitch timestamp; skip
                if t > stop_ns:
                    break
            if record.WhichOneof("body") != "image":
                continue
            node = record.node
            if camera_names is not None and node not in camera_names:
                continue
            if len(frames[node]) >= n_per_cam:
                continue
            frame = frame_from_image(record.image, target_h)
            if frame is not None:
                frames[node].append(frame)
            # Early exit once every seen camera has its quota (and we've seen a few).
            if camera_names is not None and all(
                len(frames[c]) >= n_per_cam for c in camera_names
            ):
                break
    return dict(frames)


def default_candidates() -> List[Tuple[str, CameraAdjust]]:
    """Starter set of candidate corrections shown alongside the raw frame."""
    return [
        ("gamma 0.75", CameraAdjust(gamma=0.75)),   # note: <1 = darken (for reference)
        ("gamma 1.5", CameraAdjust(gamma=1.5)),
        ("gamma 2.0", CameraAdjust(gamma=2.0)),
        ("gain 1.6 +10", CameraAdjust(gain=1.6, bias=10)),
        ("gain 2.0 +20", CameraAdjust(gain=2.0, bias=20)),
        ("g1.8 gain1.4", CameraAdjust(gain=1.4, gamma=1.8)),
    ]


def comparison_grid(
    frames: Dict[str, List[np.ndarray]],
    candidates: Sequence[Tuple[str, CameraAdjust]],
    out_png: str,
    frame_index: int = 0,
) -> str:
    """Render one row per camera, columns = [raw] + candidates, to out_png."""
    cams = list(frames)
    ncol = 1 + len(candidates)
    nrow = max(len(cams), 1)
    fig = Figure(figsize=(2.6 * ncol, 2.6 * nrow), facecolor="white")
    for r, cam in enumerate(cams):
        cam_frames = frames[cam]
        if not cam_frames:
            continue
        base = cam_frames[min(frame_index, len(cam_frames) - 1)]
        cells = [("raw", base)] + [(lbl, adj.apply(base)) for lbl, adj in candidates]
        for c, (lbl, img) in enumerate(cells):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=9)
            if c == 0:
                ax.set_ylabel(cam, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.02,
                        wspace=0.05, hspace=0.08)
    fig.savefig(out_png, dpi=110)
    return out_png


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("behave_path")
    ap.add_argument("--out", default="contrast_grid.png")
    ap.add_argument("--n", type=int, default=3, help="frames grabbed per camera")
    ap.add_argument("--scan", type=float, default=4.0, help="max seconds to scan")
    ap.add_argument("--frame", type=int, default=0, help="which grabbed frame to show")
    args = ap.parse_args(argv)

    frames = grab_sample_frames(args.behave_path, n_per_cam=args.n, max_scan_s=args.scan)
    if not frames:
        raise SystemExit("No camera frames found in the scanned span.")
    for cam, fl in frames.items():
        h, w = fl[0].shape
        print(f"{cam}: {len(fl)} frame(s), {w}x{h}, mean={np.mean(fl[0]):.1f}")
    out = comparison_grid(frames, default_candidates(), args.out, frame_index=args.frame)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
