"""simple_touch_metrics.py — Day-summary panels for the simple_touch task.

The joystick panels in ``day_behavior_metrics`` all read ``Joystick*`` fields and
produce empty axes for touch trials, so simple_touch gets its own set here.

Two things the task does *not* give us, and where they come from instead:

* **Touch position.**  ``behav_result`` carries only ``presented_target_ids`` and
  ``selected_target_id`` (identical for the Python ``simple_touch_task`` and the
  Rust ``simple_touch_rust`` port), so the endpoint is read back off the hand
  trace in ``recNNN.scaledhnd.dat`` around ``StartAq``.
* **Failure mode.**  There is no ``failure_reason`` for this task.  The task drops
  ``selected_target_id`` entirely when the subject never acquires the target, so a
  no-start fail and a released-too-early fail separate on that key alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from py_proc.helpers import loadroshnd
from pyCheck.plot_utils import _flat, _save_figure, _title_suffix

# Touch is sampled at ~44 Hz during contact, so a +/-25 ms window around StartAq
# spans at least one report without running into the release.  Checked on 261028
# rec001: the median error is flat across window width (3.73 deg at +/-10 ms,
# 3.75 at +/-25, 3.85 at +/-100) and agrees with first-touch-after-StartAq
# (3.66), so the long tail is the animal, not a time-base offset.  A window
# placed *before* StartAq does diverge (4.87 deg, max 18.6) -- the hand is still
# arriving -- so keep this centred on StartAq, never leading it.
_TOUCH_WINDOW_MS = 25
# proc_hand writes this where the screen reported no contact.
_NOTOUCH = -100.0


def _rec_prefix(repo_root: Path, day: str, rec: int) -> str:
    return str(repo_root / day / f"{rec:03d}" / f"rec{rec:03d}")


def _absolute_ms(data: Dict[str, np.ndarray], field: str) -> np.ndarray:
    """``field`` as an absolute sample index into the rec's .dat files.

    ``save_trials`` stores StartOn as the absolute timestamp and every other
    event relative to it (save_trials.py:8-9), so AllTrials columns must be
    re-based before they can index a .dat.  Events.mat is already absolute and
    carries no StartOn, in which case the column is returned unchanged.
    """
    values = _flat(data, field, float)
    if "StartOn" not in data:
        return values
    return values + _flat(data, "StartOn", float)


def touch_endpoints(repo_root: Path, day: str, data: Dict[str, np.ndarray]) -> np.ndarray:
    """(n_trials, 2) median touch position in degrees at StartAq, NaN where the
    screen reported no contact or the hand trace is missing."""
    start_aq = _absolute_ms(data, "StartAq")
    rec = _flat(data, "Rec", int)
    out = np.full((len(start_aq), 2), np.nan, dtype=float)

    for r in sorted(np.unique(rec).tolist()):
        rows = np.flatnonzero(rec == r)
        pref = _rec_prefix(repo_root, day, r)
        # loadroshnd indexes Events[ref][itrial], so a bare dict of the one
        # column it needs is enough.
        events = {"StartAq": start_aq[rows]}
        for i, row in enumerate(rows):
            if not np.isfinite(start_aq[row]):
                continue
            hand = loadroshnd(pref, events, i, "StartAq", [-_TOUCH_WINDOW_MS, _TOUCH_WINDOW_MS])
            if hand.shape[1] == 0:
                continue
            touching = (hand[0] != _NOTOUCH) & (hand[1] != _NOTOUCH)
            touching &= np.isfinite(hand[0]) & np.isfinite(hand[1])
            if not np.any(touching):
                continue
            out[row] = np.median(hand[:, touching], axis=1)
    return out


def radial_error(data: Dict[str, np.ndarray], endpoints: np.ndarray) -> np.ndarray:
    """Distance in degrees from each touch endpoint to its target centre."""
    target = np.asarray(data["TargetLocation"], dtype=float)
    if target.ndim == 1:
        target = target.reshape(-1, 2)
    return np.hypot(endpoints[:, 0] - target[:, 0], endpoints[:, 1] - target[:, 1])


def acceptance_window_deg(repo_root: Path, day: str, recs: List[int]) -> Optional[float]:
    """The targN_window_size used on this day, or None if the recs disagree or
    none could be read. Drawn on the scatter so error reads against the actual
    acceptance criterion rather than an arbitrary radius."""
    windows: set[float] = set()
    for rec in recs:
        ev_path = repo_root / day / f"{rec:03d}" / f"rec{rec:03d}.ev.mat"
        try:
            ev = loadmat(ev_path, simplify_cells=True)
        except Exception:  # noqa: BLE001 - provenance must never break a panel
            continue
        used = ev.get("used_values", [])
        used = list(used) if isinstance(used, (list, np.ndarray)) else [used]
        for entry in used[:1]:
            try:
                doc = json.loads(str(entry))
            except ValueError:
                continue
            for key, value in doc.items():
                if key.startswith("targ") and key.endswith("_window_size"):
                    windows.add(round(float(value), 4))
    if len(windows) == 1:
        return windows.pop()
    return None


def touch_failure_modes(repo_root: Path, day: str, data: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-row failure label: "" for successes, else "never acquired" (the task
    omits selected_target_id on a no-start fail) or "released too early"."""
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    labels = np.full(len(rec), "", dtype=object)

    for r in sorted(np.unique(rec).tolist()):
        rows = np.flatnonzero(rec == r)
        ev_path = repo_root / day / f"{r:03d}" / f"rec{r:03d}.ev.mat"
        try:
            ev = loadmat(ev_path, simplify_cells=True)
        except Exception as exc:  # noqa: BLE001
            print(f"simple_touch: could not read {ev_path} ({exc}); failure modes unlabelled")
            continue
        behav = ev.get("behav_results", [])
        behav = list(behav) if isinstance(behav, (list, np.ndarray)) else [behav]
        for k, row in enumerate(rows[: len(behav)]):
            if success[row] >= 0.5:
                continue
            br = behav[k]
            if isinstance(br, str):
                try:
                    br = json.loads(br)
                except ValueError:
                    br = {}
            if not isinstance(br, dict):
                br = {}
            selected = br.get("selected_target_id", None)
            labels[row] = "never acquired" if selected is None else "released too early"
    return labels


def plot_simple_touch_performance(
    data: Dict[str, np.ndarray],
    out_path: Path,
    repo_root: Path,
    day: str,
    exclude_recs: Optional[List[int]] = None,
) -> Dict[str, object]:
    exclude_recs = exclude_recs or []
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    recs = sorted(np.unique(rec).tolist())

    labels = touch_failure_modes(repo_root, day, data)
    modes = ["never acquired", "released too early"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    counts = [int(np.sum(rec == r)) for r in recs]
    rates = [float(np.nanmean(success[rec == r])) for r in recs]
    x = np.arange(len(recs))
    ax.bar(x, counts, color="#9aa7b4", label="trials")
    ax.set_xticks(x)
    ax.set_xticklabels([f"rec{r:03d}" for r in recs], rotation=45, ha="right")
    ax.set_ylabel("trials")
    ax.set_xlabel("recording")
    twin = ax.twinx()
    twin.plot(x, np.array(rates) * 100.0, "o-", color="#d1495b", label="success %")
    twin.set_ylim(0, 105)
    twin.set_ylabel("success rate (%)")
    ax.set_title(f"{day} simple_touch by recording{_title_suffix(exclude_recs)}")

    ax = axes[1]
    bottoms = np.zeros(len(recs))
    colors = {"never acquired": "#e8a33d", "released too early": "#4c6ef5"}
    for mode in modes:
        vals = np.array([int(np.sum((rec == r) & (labels == mode))) for r in recs], dtype=float)
        ax.bar(x, vals, bottom=bottoms, label=mode, color=colors[mode])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"rec{r:03d}" for r in recs], rotation=45, ha="right")
    ax.set_ylabel("failed trials")
    ax.set_xlabel("recording")
    ax.set_title("failure modes")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save_figure(fig, out_path, "simple_touch performance")

    return {
        "n_trials": int(len(rec)),
        "overall_success_rate": float(np.nanmean(success)) if len(success) else float("nan"),
        "per_rec": {
            f"rec{r:03d}": {
                "trial_count": int(np.sum(rec == r)),
                "success_rate": float(np.nanmean(success[rec == r])),
            }
            for r in recs
        },
        "failure_modes": {mode: int(np.sum(labels == mode)) for mode in modes},
    }


def frame_is_consistent(
    err: np.ndarray, success: np.ndarray, window: Optional[float]
) -> bool:
    """Do the touch trace and TargetLocation live in the same frame?

    The touch transform is built from the Qt widget that reports canvas_info
    (task_context.py:636-662).  On Rust task recs that widget is the 650x508
    control window rather than the stimulus surface, so the mapped touch lands
    tens of degrees away from the target while the animal was in fact hitting
    it.  A high success rate alongside a median error outside the acceptance
    window is exactly that disagreement, and it must not be reported as the
    animal's spatial accuracy.
    """
    ok = np.isfinite(err)
    if window is None or not np.any(ok):
        return True
    accepted = np.nanmean(success[ok]) if np.any(ok) else float("nan")
    return not (float(np.median(err[ok])) > window and accepted > 0.5)


def _mismatch_metrics(
    err: np.ndarray, ok: np.ndarray, success: np.ndarray, window: Optional[float]
) -> Dict[str, object]:
    """What we can honestly say when the two frames disagree: counts, and why
    the spatial numbers are withheld."""
    return {
        "n_trials_with_touch": int(np.sum(ok)),
        "n_trials": int(len(err)),
        "acceptance_window_deg": window,
        "frame_mismatch": (
            "touch trace and TargetLocation disagree: median error "
            f"{float(np.median(err[ok])):.1f} deg exceeds the {window} deg window "
            f"while {100 * float(np.nanmean(success[ok])):.0f}% of these trials were "
            "accepted. The touch transform is built from the Qt widget reporting "
            "canvas_info (task_context.py:636-662), which on Rust task recs is the "
            "control window, not the stimulus surface. Spatial accuracy is not "
            "reportable for these recs; success rate and trial counts are."
        ),
    }


def plot_simple_touch_accuracy(
    data: Dict[str, np.ndarray],
    out_path: Path,
    repo_root: Path,
    day: str,
    exclude_recs: Optional[List[int]] = None,
) -> Dict[str, object]:
    exclude_recs = exclude_recs or []
    rec = _flat(data, "Rec", int)
    success = _flat(data, "Success", float)
    endpoints = touch_endpoints(repo_root, day, data)
    err = radial_error(data, endpoints)
    window = acceptance_window_deg(repo_root, day, sorted(np.unique(rec).tolist()))

    target = np.asarray(data["TargetLocation"], dtype=float).reshape(-1, 2)
    ok = np.isfinite(err)

    if not frame_is_consistent(err, success, window):
        # Draw the disagreement instead of a plausible-looking error cloud.
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        ax.scatter(endpoints[ok, 0], endpoints[ok, 1], s=14, alpha=0.5,
                   color="#d1495b", label="mapped touch")
        ax.scatter(target[:, 0], target[:, 1], s=60, marker="+",
                   color="#2b8a3e", label="target")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("horizontal (deg)")
        ax.set_ylabel("vertical (deg)")
        ax.set_title(f"{day} touch frame mismatch - accuracy not reportable")
        ax.legend(frameon=False, fontsize=8)
        ax.text(0.5, -0.16,
                "Touch maps tens of degrees off target while these trials were "
                "accepted.\ncanvas_info comes from the Qt control window, not the "
                "stimulus surface, on Rust recs.",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color="#d1495b")
        fig.tight_layout()
        _save_figure(fig, out_path, "simple_touch frame mismatch")
        return _mismatch_metrics(err, ok, success, window)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    if np.any(ok):
        ax.hist(err[ok], bins=24, color="#4c6ef5", edgecolor="white")
        ax.axvline(float(np.median(err[ok])), color="#d1495b", lw=2,
                   label=f"median {np.median(err[ok]):.2f}°")
    if window is not None:
        ax.axvline(window, color="#2b8a3e", ls="--", lw=2, label=f"window {window:.1f}°")
    ax.set_xlabel("radial error (deg)")
    ax.set_ylabel("trials")
    ax.set_title(f"{day} touch accuracy (n={int(np.sum(ok))}){_title_suffix(exclude_recs)}")
    ax.legend(frameon=False, fontsize=8)

    # Endpoints in the target frame: every target collapsed onto one origin, so
    # a systematic bias shows up as an off-centre cloud.
    ax = axes[1]
    dx = endpoints[:, 0] - target[:, 0]
    dy = endpoints[:, 1] - target[:, 1]
    hit = ok & (success >= 0.5)
    miss = ok & (success < 0.5)
    ax.scatter(dx[hit], dy[hit], s=14, alpha=0.5, color="#4c6ef5", label="success")
    if np.any(miss):
        ax.scatter(dx[miss], dy[miss], s=22, alpha=0.8, color="#d1495b", marker="x", label="fail")
    if window is not None:
        ax.add_patch(plt.Circle((0, 0), window, fill=False, color="#2b8a3e", ls="--", lw=2))
    ax.axhline(0, color="#aaa", lw=0.8)
    ax.axvline(0, color="#aaa", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("horizontal error (deg)")
    ax.set_ylabel("vertical error (deg)")
    ax.set_title("touch endpoint relative to target centre")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save_figure(fig, out_path, "simple_touch accuracy")

    valid = err[ok]
    return {
        "n_trials_with_touch": int(np.sum(ok)),
        "n_trials": int(len(err)),
        "acceptance_window_deg": window,
        "median_radial_error_deg": float(np.median(valid)) if valid.size else float("nan"),
        "p25_radial_error_deg": float(np.percentile(valid, 25)) if valid.size else float("nan"),
        "p75_radial_error_deg": float(np.percentile(valid, 75)) if valid.size else float("nan"),
        "mean_bias_deg": {
            "x": float(np.nanmean(dx[ok])) if valid.size else float("nan"),
            "y": float(np.nanmean(dy[ok])) if valid.size else float("nan"),
        },
        "per_rec_median_error_deg": {
            f"rec{r:03d}": float(np.median(err[ok & (rec == r)]))
            if np.any(ok & (rec == r)) else float("nan")
            for r in sorted(np.unique(rec).tolist())
        },
    }
