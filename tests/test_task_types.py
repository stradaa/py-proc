"""Checks for Rust-delegate task aliases and screen-touch fail exclusion.

Run from the py-proc repo root:   python tests/test_task_types.py

No pytest dependency: plain asserts, prints PASS lines, non-zero exit on failure.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyCheck.cross_day_plots import _is_requested_task
from pyCheck.day_presentation_plots import _touch_fail_flags
from py_proc.proc_events import _TASK_CODES
from pyjoystick.tasks import canonical_task_type, is_touch_input_fail


def _behav(outcome, reason=None):
    attempt = {"outcome": outcome, "failure_reason": reason, "events": []}
    return {"final_outcome": outcome, "final_attempt": attempt, "attempts": [attempt]}


def test_aliases():
    assert canonical_task_type("simple_touch_rust") == "simple_touch_task"
    assert canonical_task_type("joystick_intro_rust") == "joystick_intro"
    assert canonical_task_type("simple_touch_task") == "simple_touch_task"
    assert canonical_task_type(None) == ""
    # simple_touch_rust must get the reach flag, or proc_reach skips it.
    assert _TASK_CODES[canonical_task_type("simple_touch_rust")][1] is True
    print("PASS Rust task types alias to their Python task")


def test_task_filter():
    assert _is_requested_task("joystick_intro_rust", {"joystick_intro"})
    assert _is_requested_task("joystick_intro", {"joystick_intro_rust"})
    assert not _is_requested_task("simple_touch_rust", {"joystick_intro"})
    assert _is_requested_task("anything", set())
    print("PASS task filter matches Rust and Python variants")


def test_touch_fail_detection():
    assert is_touch_input_fail(_behav("fail", "touch_input"))
    assert not is_touch_input_fail(_behav("fail", "timeout_after_movement"))
    assert not is_touch_input_fail(_behav("success"))
    assert not is_touch_input_fail({})
    # final_attempt missing: falls back to the last attempt
    assert is_touch_input_fail({"attempts": [{"failure_reason": "touch_input"}]})
    print("PASS touch-input fails detected from behav_result")


def test_touch_fail_flags_column_and_fallback():
    data = {"Rec": np.array([1, 1, 1, 2, 2]),
            "JoystickTouchFail": np.array([0.0, 1.0, np.nan, 1.0, 0.0])}
    flags = _touch_fail_flags(Path("/nonexistent"), "000000", data)
    assert flags.tolist() == [False, True, False, True, False]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recs = {1: [_behav("success"), _behav("fail", "touch_input"), _behav("fail", "touch_input")],
                2: [_behav("fail", "timeout_without_movement")]}
        for rec, behav in recs.items():
            rec_dir = root / "261102" / f"{rec:03d}"
            rec_dir.mkdir(parents=True)
            savemat(rec_dir / f"rec{rec:03d}.ev.mat",
                    {"behav_results": np.array([json.dumps(b) for b in behav], dtype=object)})
        legacy = {"Rec": np.array([1, 1, 1, 2])}
        flags = _touch_fail_flags(root, "261102", legacy)
        assert flags.tolist() == [False, True, True, False], flags
    print("PASS touch-fail flags from AllTrials column and ev.mat fallback")


if __name__ == "__main__":
    test_aliases()
    test_task_filter()
    test_touch_fail_detection()
    test_touch_fail_flags_column_and_fallback()
    print("\nALL TESTS PASSED")
