"""Parity + endpoint checks for the shared joystick->cursor reconstruction.

Run from the py-proc repo root:   python tests/test_cursor_reconstruct.py

No pytest dependency: plain asserts, prints PASS lines, non-zero exit on failure.
Verifies (1) the pyCheck and pyReplay wrappers produce identical trajectories
for every control_mode, and (2) blend reduces to direct at b=0 and cumulative
at b=1.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyCheck.joystick_validation import reconstruct_cursor as check_reconstruct
from pyReplay.cursor import reconstruct as replay_reconstruct
from pyjoystick.cursor import reconstruct_cursor as core_reconstruct


def _synthetic():
    fps = 240.0
    t_s = np.arange(0.0, 2.0, 1.0 / fps)
    # Start centered, then hold a diagonal deflection; add a small wobble.
    jx = np.where(t_s < 0.2, 0.0, 0.6) + 0.02 * np.sin(2 * np.pi * 3 * t_s)
    jy = np.where(t_s < 0.2, 0.0, -0.4) + 0.02 * np.cos(2 * np.pi * 2 * t_s)
    jx[0] = 0.0
    jy[0] = 0.0
    return t_s, jx, jy


BASE = {
    "direct_range": 0.45,
    "cumulative_speed": 0.70,
    "zero_drift_mode": True,
    "zero_drift_buffer": 0.03,
    "reset_cursor_each_trial": True,
}


def _cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def test_wrapper_parity():
    """pyCheck and pyReplay wrappers must return identical trajectories."""
    t_s, jx, jy = _synthetic()
    t_ns = (t_s * 1e9).astype(np.int64)
    trial_starts_s = [0.0, 1.0]
    trial_starts_ns = [int(s * 1e9) for s in trial_starts_s]
    for mode, extra in [
        ("direct", {}),
        ("cumulative", {}),
        ("blend", {"position_velocity_blend": 0.0}),
        ("blend", {"position_velocity_blend": 0.35}),
        ("blend", {"position_velocity_blend": 1.0}),
    ]:
        cfg = _cfg(control_mode=mode, **extra)
        cx_c, cy_c = check_reconstruct(jx, jy, t_s, cfg, trial_starts_s)
        cx_r, cy_r = replay_reconstruct(t_ns, jx, jy, cfg, trial_starts_ns)
        label = f"{mode}{extra.get('position_velocity_blend', '')}"
        assert np.allclose(cx_c, cx_r, atol=1e-9), f"x parity {label}"
        assert np.allclose(cy_c, cy_r, atol=1e-9), f"y parity {label}"
    print("PASS wrapper parity (pyCheck == pyReplay) for all modes")


def test_blend_endpoints():
    """blend b=0 -> direct; blend b=1 -> cumulative (same trial resets)."""
    t_s, jx, jy = _synthetic()
    starts = [0.0, 1.0]

    cx_d, cy_d = core_reconstruct(t_s, jx, jy, _cfg(control_mode="direct"), starts)
    cx_b0, cy_b0 = core_reconstruct(
        t_s, jx, jy, _cfg(control_mode="blend", position_velocity_blend=0.0), starts
    )
    # direct is stateless; blend b=0 re-centers at trial starts (index 0 and the
    # t=1.0 boundary), so compare away from those reset samples.
    reset_idx = {0, int(np.searchsorted(t_s, 1.0))}
    keep = np.array([i for i in range(len(t_s)) if i not in reset_idx])
    assert np.allclose(cx_d[keep], cx_b0[keep], atol=1e-9), "blend b=0 == direct"
    assert np.allclose(cy_d[keep], cy_b0[keep], atol=1e-9), "blend b=0 == direct"
    print("PASS blend b=0 reduces to direct")

    cx_c, cy_c = core_reconstruct(t_s, jx, jy, _cfg(control_mode="cumulative"), starts)
    cx_b1, cy_b1 = core_reconstruct(
        t_s, jx, jy, _cfg(control_mode="blend", position_velocity_blend=1.0), starts
    )
    assert np.allclose(cx_c, cx_b1, atol=1e-12), "blend b=1 == cumulative"
    assert np.allclose(cy_c, cy_b1, atol=1e-12), "blend b=1 == cumulative"
    print("PASS blend b=1 reduces to cumulative")


def test_blend_monotone_memory():
    """Sanity: after releasing the stick to center, higher blend retains more
    displacement (position memory grows with blend)."""
    fps = 240.0
    t_s = np.arange(0.0, 3.0, 1.0 / fps)
    jx = np.where(t_s < 1.0, 0.6, 0.0)  # hold 1s then release to center
    jy = np.zeros_like(t_s)
    finals = []
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cx, _ = core_reconstruct(
            t_s, jx, jy, _cfg(control_mode="blend", position_velocity_blend=b), []
        )
        finals.append(cx[-1])
    diffs = np.diff(finals)
    assert np.all(diffs >= -1e-9), f"expected non-decreasing hold, got {finals}"
    print(f"PASS blend position-memory monotone in b: {[round(f,3) for f in finals]}")


if __name__ == "__main__":
    test_wrapper_parity()
    test_blend_endpoints()
    test_blend_monotone_memory()
    print("\nALL TESTS PASSED")
