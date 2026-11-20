"""Checks for the simple_touch day-summary metrics and the proc_reach fix.

Run from the py-proc repo root:   python tests/test_simple_touch_metrics.py

No pytest dependency: plain asserts, prints PASS lines, non-zero exit on failure.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from py_proc.proc_reach import proc_reach
from pyCheck.simple_touch_metrics import (
    acceptance_window_deg,
    frame_is_consistent,
    radial_error,
    touch_endpoints,
    touch_failure_modes,
)

NOTOUCH = -100.0


def _write_scaledhnd(path: Path, n_samples: int, touches):
    """touches: {sample_index: (h, v)}; everything else is the no-touch sentinel."""
    data = np.full((2, n_samples), NOTOUCH, dtype=np.float32)
    for idx, (h, v) in touches.items():
        data[0, idx] = h
        data[1, idx] = v
    # column-major interleave, matching MATLAB fwrite of a 2xN array
    data.T.astype(np.float32).ravel().tofile(path)


def _rec_dir(root: Path, day: str, rec: int) -> Path:
    d = root / day / f"{rec:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_touch_endpoint_and_radial_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 1)
        # Trial 1 touches at (3, 4) -> 5 deg from a target at the origin.
        # Trial 2 has no contact anywhere in its window -> NaN, not a false 0.
        _write_scaledhnd(rec_dir / "rec001.scaledhnd.dat", 2000,
                         {i: (3.0, 4.0) for i in range(490, 511)})
        data = {
            "Rec": np.array([1, 1]),
            "StartAq": np.array([500.0, 1500.0]),
            "TargetLocation": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "Success": np.array([1.0, 0.0]),
        }
        pts = touch_endpoints(root, "261028", data)
        assert np.allclose(pts[0], [3.0, 4.0]), pts[0]
        assert np.all(np.isnan(pts[1])), pts[1]

        err = radial_error(data, pts)
        assert abs(err[0] - 5.0) < 1e-6, err[0]
        assert np.isnan(err[1]), err[1]
    print("PASS touch endpoint read from scaledhnd, no-contact stays NaN")


def test_touch_endpoint_rebases_alltrials_times():
    """AllTrials stores StartOn absolute and every other event relative to it
    (save_trials.py:8-9).  Feeding those relative times straight to loadroshnd
    reads the wrong slice of the .dat, which silently inflates the error rather
    than failing, so pin both time bases to the same answer."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 1)
        _write_scaledhnd(rec_dir / "rec001.scaledhnd.dat", 20000,
                         {i: (3.0, 4.0) for i in range(9330, 9351)})
        target = np.array([[0.0, 0.0]])

        # Events.mat style: StartAq already absolute, no StartOn column.
        absolute = {
            "Rec": np.array([1]),
            "StartAq": np.array([9340.0]),
            "TargetLocation": target,
            "Success": np.array([1.0]),
        }
        # AllTrials style: the same instant, split across StartOn + StartAq.
        relative = {
            "Rec": np.array([1]),
            "StartOn": np.array([9000.0]),
            "StartAq": np.array([340.0]),
            "TargetLocation": target,
            "Success": np.array([1.0]),
        }
        pts_abs = touch_endpoints(root, "261028", absolute)
        pts_rel = touch_endpoints(root, "261028", relative)
        assert np.allclose(pts_abs[0], [3.0, 4.0]), pts_abs[0]
        assert np.allclose(pts_rel[0], pts_abs[0]), (pts_rel[0], pts_abs[0])
    print("PASS AllTrials relative times are re-based onto StartOn")


def test_touch_endpoint_offset_target():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 2)
        _write_scaledhnd(rec_dir / "rec002.scaledhnd.dat", 1200,
                         {i: (9.0, 1.0) for i in range(295, 306)})
        data = {
            "Rec": np.array([2]),
            "StartAq": np.array([300.0]),
            "TargetLocation": np.array([[10.0, 0.0]]),
            "Success": np.array([1.0]),
        }
        err = radial_error(data, touch_endpoints(root, "261028", data))
        # (9,1) vs (10,0) -> sqrt(2)
        assert abs(err[0] - np.sqrt(2.0)) < 1e-6, err[0]
    print("PASS radial error measured against the trial's own target centre")


def test_failure_modes_split_on_selected_target_id():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 1)
        behav = [
            {"presented_target_ids": [3], "selected_target_id": 3},   # success
            {"presented_target_ids": [3]},                            # never acquired
            {"presented_target_ids": [3], "selected_target_id": 3},   # released early
        ]
        savemat(rec_dir / "rec001.ev.mat",
                {"behav_results": np.array([json.dumps(b) for b in behav], dtype=object)})
        data = {"Rec": np.array([1, 1, 1]), "Success": np.array([1.0, 0.0, 0.0])}
        labels = touch_failure_modes(root, "261028", data)
        assert labels.tolist() == ["", "never acquired", "released too early"], labels
    print("PASS failure modes split on the missing selected_target_id key")


def test_acceptance_window_from_used_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 1)
        used = {"targ0_window_size": 8.0, "targ1_window_size": 8.0, "hold_timeout": 0.2}
        savemat(rec_dir / "rec001.ev.mat",
                {"used_values": np.array([json.dumps(used)], dtype=object)})
        assert acceptance_window_deg(root, "261028", [1]) == 8.0
        # A rec that cannot be read must not invent a window.
        assert acceptance_window_deg(root, "261028", [9]) is None
    print("PASS acceptance window read from used_values")


def test_frame_mismatch_is_detected_not_reported():
    """A Rust-rec touch frame maps tens of degrees off target while the trials
    were accepted.  That combination is a frame disagreement, not the animal's
    accuracy, and must not be reported as a spatial-error number."""
    success = np.ones(50)
    # Good day: error well inside the 8 deg window.
    good = np.full(50, 3.4)
    assert frame_is_consistent(good, success, 8.0)
    # Rust day: 66 deg median with ~99% of trials accepted.
    bad = np.full(50, 66.8)
    assert not frame_is_consistent(bad, success, 8.0)
    # Large error with a LOW success rate is the animal missing, which is real
    # data and must still be reported.
    assert frame_is_consistent(bad, np.zeros(50), 8.0)
    # No acceptance window read -> no basis to judge, so do not suppress.
    assert frame_is_consistent(bad, success, None)
    print("PASS frame mismatch detected without suppressing genuine misses")


def test_proc_reach_simple_touch_runs_without_go():
    """The simple_touch branch keys off StartAq. Go is always NaN for this task,
    so a Go guard ahead of the branch silently skipped every touch trial."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec_dir = _rec_dir(root, "261028", 1)
        # Hand parked, then a fast leftward excursion: diff < -20 marks the reach
        # start, diff > 20 the return.
        n = 9000
        trace = np.full(n, 0.0, dtype=np.float32)
        trace[1000:1100] = -50.0   # reach inside trial 1's StartAq window
        trace[5000:5100] = -50.0   # and inside trial 2's
        data = np.vstack([trace, np.zeros(n, dtype=np.float32)])
        data.T.astype(np.float32).ravel().tofile(rec_dir / "rec001.scaledhnd.dat")

        # Two trials: loadmat(simplify_cells=True) collapses a 1-element array
        # to a bare scalar, which is a fixture artifact, not the behaviour under
        # test (real recs have many trials).
        events = {
            "Reach": np.array([1.0, 1.0]),
            "Go": np.array([np.nan, np.nan]),
            "StartAq": np.array([3000.0, 7000.0]),
            "TargAq": np.array([np.nan, np.nan]),
            "End": np.array([3200.0, 7200.0]),
            "PyTaskType": np.array(["simple_touch_rust", "simple_touch_task"], dtype=object),
            "ReachStart": np.array([np.nan, np.nan]),
            "ReachStop": np.array([np.nan, np.nan]),
            "ReachSize": np.array([np.nan, np.nan]),
        }
        savemat(rec_dir / "rec001.Events.mat", {"Events": events})

        proc_reach("261028", "001", str(root))

        out = loadmat(rec_dir / "rec001.Events.mat", simplify_cells=True)["Events"]
        assert np.isfinite(out["ReachStart"]).all(), out["ReachStart"]
        assert np.isfinite(out["ReachStop"]).all(), out["ReachStop"]
    print("PASS proc_reach handles simple_touch with Go=NaN")


if __name__ == "__main__":
    test_touch_endpoint_and_radial_error()
    test_touch_endpoint_rebases_alltrials_times()
    test_touch_endpoint_offset_target()
    test_failure_modes_split_on_selected_target_id()
    test_acceptance_window_from_used_values()
    test_frame_mismatch_is_detected_not_reported()
    test_proc_reach_simple_touch_runs_without_go()
    print("\nALL TESTS PASSED")
