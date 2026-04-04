"""
proc_hand.py — Port of procHand.m.

Loads serialhnd.mat from bag/mat/, applies clock alignment, interpolates to
1 ms grid using 'previous' method (hold-last value), applies touch transform,
saves recNNN.hnd.dat (raw interpolated, float32 2×N) and
recNNN.scaledhnd.dat (degrees, float32 2×N, notouch=-100).
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

from .helpers import rec_paths, get_fs_rec, get_touch_transform, load_w_alignment


def proc_hand(day, rec, monkeydir, out_suffix=''):
    """
    Port of procHand(day, rec).
    """
    _, rec_dir, rec_pref, bag_dir = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix
    print(f'\nproc_hand: day={day} rec={rec}')

    serialhnd_file = os.path.join(bag_dir, 'mat', 'serialhnd.mat')
    if not os.path.exists(serialhnd_file):
        print(f'  serialhnd.mat not found at {serialhnd_file}, skipping')
        return

    # ---- Load hand data -----------------------------------------------------
    hnd_mat = loadmat(serialhnd_file, simplify_cells=True)
    hnd_msgs = hnd_mat.get('serialhnd', hnd_mat)

    if not hnd_msgs or not (isinstance(hnd_msgs, dict) and hnd_msgs):
        print('  No hand data present, skipping')
        return

    t_hand_header = (np.asarray(hnd_msgs['header_stamp_sec'], dtype=float).ravel()
                     + np.asarray(hnd_msgs['header_stamp_nanosec'], dtype=float).ravel() * 1e-9)
    hand = np.vstack([
        np.asarray(hnd_msgs['x'], dtype=float).ravel(),
        np.asarray(hnd_msgs['y'], dtype=float).ravel()
    ])  # 2 × n_hand_samples

    # ---- Recording length / time grid ---------------------------------------
    Fs_rec = get_fs_rec(rec_pref)
    fiducial = np.asarray(
        loadmat(f'{rec_pref}.fiducial.mat', simplify_cells=True)['fiducial'],
        dtype=float).ravel()
    N_rec = len(fiducial)
    t_rec = np.arange(N_rec) / Fs_rec
    # 1 ms grid matching MATLAB's t_rec(1):0.001:t_rec(end)
    n_ds = int(np.floor((t_rec[-1] - t_rec[0]) / 1e-3)) + 1
    t_rec_ds = t_rec[0] + np.arange(n_ds) * 1e-3

    # ---- Clock alignment ----------------------------------------------------
    w_drift_ros = load_w_alignment(py_pref)
    t_hand_rec = w_drift_ros[0] + w_drift_ros[1] * t_hand_header

    # ---- Handle NaN at first sample (fill with second) ----------------------
    if np.isnan(hand[0, 0]):
        if hand.shape[1] > 1:
            hand[:, 0] = hand[:, 1]

    # ---- Interpolate to 1 ms grid (previous-value hold) --------------------
    hand_rec_interp = np.full((2, len(t_rec_ds)), np.nan)
    for ch in range(2):
        valid = np.isfinite(hand[ch]) & np.isfinite(t_hand_rec)
        if np.sum(valid) < 2:
            continue
        f = interp1d(t_hand_rec[valid], hand[ch, valid],
                     kind='previous', bounds_error=False,
                     fill_value=(hand[ch, valid][0], hand[ch, valid][-1]))
        hand_rec_interp[ch] = f(t_rec_ds)

    # ---- Save .hnd.dat (raw interpolated, float32 column-major) -------------
    hnd_file = f'{py_pref}.hnd.dat'
    hand_rec_interp.T.astype(np.float32).flatten().tofile(hnd_file)
    print(f'  Saved {hnd_file}')

    # ---- Reload .hnd.dat (consistent with MATLAB procHand) ------------------
    hnd = np.fromfile(hnd_file, dtype=np.float32).reshape(-1, 2).T  # 2 × N

    # ---- Touch transform (raw → mm) -----------------------------------------
    try:
        M, dpi = get_touch_transform(bag_dir)
    except Exception as e:
        print(f'  WARNING: could not get touch transform: {e}')
        return

    # notouch_val = [0, 0] for serial touch screen
    notouch_val = np.array([0.0, 0.0])
    notouch_times = np.where(
        (hnd[0] == notouch_val[0]) | (hnd[1] == notouch_val[1])
    )[0]

    # Apply homogeneous transform: M @ [x; y; 1]
    ones = np.ones((1, hnd.shape[1]))
    mapped_mm = M @ np.vstack([hnd, ones])  # 3 × N
    mapped_mm = mapped_mm[:2, :]  # 2 × N

    # ---- Convert mm → degrees (eye-to-screen distance = 400 mm) ------------
    deg_per_mm = np.degrees(np.arctan(1.0 / 400.0))
    mapped_deg = mapped_mm * deg_per_mm

    # ---- Mark no-touch samples with -100 ------------------------------------
    mapped_deg[:, notouch_times] = -100.0

    # ---- Save .scaledhnd.dat (float32 column-major) -------------------------
    scaled_file = f'{py_pref}.scaledhnd.dat'
    mapped_deg.T.astype(np.float32).flatten().tofile(scaled_file)
    print(f'  Saved {scaled_file}')
