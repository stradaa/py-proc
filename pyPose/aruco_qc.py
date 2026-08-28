"""aruco_qc.py — parse the persisted ``Aruco Static`` QC stream and pick frames.

Thalamus persists the ``Aruco Static`` node as an *analog* record — 16 channels of
QC metrics only (``jitter_mm``, ``n_markers``, ``px_per_mm``, ``reproj_px``) for
each (camera, board) pair, emitted at whatever rate the node is configured for
(3 Hz on the 260824 rig). No pose or corners are stored, so we
cannot recover extrinsics from it; but it is a cheap, per-frame signal for picking
the cleanest calibration frames (all needed boards fully detected, low reprojection
error, before the animal occludes anything) without decoding any video.

We read it with the repo's own ``RecordReader`` (``procThalamus_indie``), which
iterates this 9 GB stream in ~20 s; the thalamus ``record_reader2`` blocks on it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boards import ARUCO_NODE, ArucoConfig

_METRICS = ("jitter_mm", "n_markers", "px_per_mm", "reproj_px")


def _load_record_reader():
    """Import the RecordReader class from py_proc/procThalamus_indie.py."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "py_proc" / "procThalamus_indie.py"
    spec = importlib.util.spec_from_file_location("_proc_thalamus_indie", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RecordReader


@dataclass
class QCTable:
    """Per-(camera, board) QC metric time series on the shared ns clock."""

    times_ns: np.ndarray                                   # (N,)
    data: dict[tuple[str, str], dict[str, np.ndarray]]     # (cam,board)->metric->(N,)
    source_to_camera: dict[str, str] = field(default_factory=dict)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return list(self.data.keys())


@dataclass
class SelectedFrames:
    times_ns: np.ndarray            # (K,) chosen QC sample times
    n_candidates: int               # fully-clean samples available
    relaxed: bool                   # True if we fell back to best-effort ranking
    criteria: dict


def _parse_span_name(name: str, sources: list[str], boards: list[str]):
    """Split ``"<source>_<board>_<metric>"`` handling spaces in names."""
    for metric in _METRICS:
        suffix = "_" + metric
        if not name.endswith(suffix):
            continue
        rest = name[: -len(suffix)]
        for src in sources:
            if rest.startswith(src + "_"):
                board = rest[len(src) + 1:]
                if board in boards:
                    return src, board, metric
        return None
    return None


def read_aruco_qc(
    behave_path: str | Path,
    config: ArucoConfig,
    *,
    node_name: str = ARUCO_NODE,
    max_duration_s: float | None = 120.0,
) -> QCTable:
    """Scan the behave stream and collect the ``Aruco Static`` QC records.

    ``max_duration_s`` stops the scan that many seconds after the first ArUco
    record (calibration uses early, pre-occlusion frames); pass ``None`` for the
    whole recording.
    """
    RecordReader = _load_record_reader()
    sources = list(config.source_to_camera.keys())
    boards = list(config.boards.keys())
    s2c = config.source_to_camera

    times: list[int] = []
    cols: dict[tuple[str, str, str], list[float]] = {}
    layout: list[tuple[int, tuple[str, str, str]]] | None = None
    t_first: int | None = None

    with RecordReader(str(behave_path), False, False) as reader:
        for record in reader:
            if record.node != node_name or record.WhichOneof("body") != "analog":
                continue
            analog = record.analog
            if layout is None:
                layout = []
                for span in analog.spans:
                    parsed = _parse_span_name(span.name, sources, boards)
                    if parsed is not None:
                        cam = s2c[parsed[0]]
                        layout.append((span.begin, (cam, parsed[1], parsed[2])))
                        cols.setdefault((cam, parsed[1], parsed[2]), [])
            times.append(record.time)
            for begin, key in layout:
                cols[key].append(analog.data[begin])
            if t_first is None:
                t_first = record.time
            elif max_duration_s is not None and (record.time - t_first) > max_duration_s * 1e9:
                break

    times_ns = np.asarray(times, dtype=np.uint64)
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (cam, board, metric), vals in cols.items():
        data.setdefault((cam, board), {})[metric] = np.asarray(vals, dtype=float)
    return QCTable(times_ns=times_ns, data=data, source_to_camera=s2c)


def select_calib_frames(
    qc: QCTable,
    config: ArucoConfig,
    *,
    required_pairs: list[tuple[str, str]] | None = None,
    max_reproj_px: float = 3.0,
    max_frames: int = 40,
    min_frames: int = 10,
    min_full_frac: float = 0.05,
) -> SelectedFrames:
    """Pick QC sample times where every required (camera, board) pair is clean.

    A pair is "clean" when it sees all its markers and its reprojection error is
    finite and ``<= max_reproj_px``. If fewer than ``min_frames`` fully-clean
    samples exist, fall back to ranking all samples by a detection score.

    Pairs that are never fully detected (below ``min_full_frac`` of samples) are
    dropped first: a camera that simply cannot see a board — a normal rig
    geometry outcome — would otherwise make the all-pairs intersection empty and
    silently force the relaxed fallback on every recording.
    """
    pairs = [p for p in (required_pairs if required_pairs is not None else qc.pairs)
             if p in qc.data]
    n = len(qc.times_ns)
    full = {p: len(config.boards[p[1]].ids) for p in pairs}

    full_frac = {p: float(np.mean(qc.data[p]["n_markers"] >= full[p])) for p in pairs}
    dropped = [p for p in pairs if full_frac[p] < min_full_frac]
    kept = [p for p in pairs if full_frac[p] >= min_full_frac]
    if kept:  # never drop every pair, however bad the recording
        pairs = kept
    else:
        dropped = []

    clean = np.ones(n, dtype=bool)
    score = np.zeros(n, dtype=float)
    for p in pairs:
        m = qc.data[p]
        nm = m["n_markers"]
        rp = m["reproj_px"]
        ok = (nm >= full[p]) & np.isfinite(rp) & (rp > 0) & (rp <= max_reproj_px)
        clean &= ok
        score += nm - np.where(np.isfinite(rp), rp, 100.0)

    cand = np.flatnonzero(clean)
    relaxed = False
    if len(cand) >= min_frames:
        chosen = cand
    else:
        relaxed = True
        chosen = np.argsort(score)[::-1][: max(max_frames, min_frames)]
        chosen = np.sort(chosen)

    if len(chosen) > max_frames:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_frames).astype(int)]

    return SelectedFrames(
        times_ns=qc.times_ns[chosen],
        n_candidates=int(len(cand)),
        relaxed=relaxed,
        criteria={
            "required_pairs": [list(p) for p in pairs],
            "dropped_pairs": [list(p) for p in dropped],
            "full_frac": {f"{c}|{b}": round(v, 4) for (c, b), v in full_frac.items()},
            "max_reproj_px": max_reproj_px,
            "full_markers": {f"{c}|{b}": full[p] for p in pairs for (c, b) in [p]},
            "n_qc_samples": n,
        },
    )
