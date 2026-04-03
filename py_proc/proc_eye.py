"""
proc_eye.py — Port of procEye.m.

Loads oculomatic eye data, applies quadrant-gain transform, pchip-interpolates
to 1 ms grid, multitaper low-pass filters at 50 Hz, saves recNNN.lp.seye.dat
(float32, 2×N interleaved column-major).
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
from scipy.signal.windows import dpss

from .helpers import rec_paths, get_fs_rec, get_gaze_transform, load_w_alignment


def proc_eye(day, rec, monkeydir, out_suffix=''):
    """
    Port of procEye(day, rec).

    Saves recNNN.lp.seye.dat as float32 (2×N column-major, i.e. interleaved
    H/V pairs): [H0, V0, H1, V1, ...].
    """
    _, rec_dir, rec_pref, bag_dir = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix
    print(f'\nproc_eye: day={day} rec={rec}')

    # ---- Sampling rate / recording length -----------------------------------
    Fs_rec = get_fs_rec(rec_pref)
    fiducial = np.asarray(
        loadmat(f'{rec_pref}.fiducial.mat', simplify_cells=True)['fiducial'],
        dtype=float).ravel()
    N_rec = len(fiducial)
    t_rec = np.arange(N_rec) / Fs_rec  # seconds, starts at 0

    # ---- Clock alignment ----------------------------------------------------
    w_drift_ros = load_w_alignment(py_pref)

    # ---- Load eye data from bag/mat/ ----------------------------------------
    eye_mat = loadmat(os.path.join(bag_dir, 'mat', 'oculomatic_eye.mat'),
                      simplify_cells=True)
    eye_msgs = eye_mat.get('oculomatic_eye', eye_mat)

    if not eye_msgs or not (isinstance(eye_msgs, dict) and eye_msgs):
        print('  No eye data present, skipping')
        return

    t_eye_header = (np.asarray(eye_msgs['header_stamp_sec'], dtype=float).ravel()
                    + np.asarray(eye_msgs['header_stamp_nanosec'], dtype=float).ravel() * 1e-9)

    eye_in = np.vstack([
        np.asarray(eye_msgs['x'], dtype=float).ravel(),
        np.asarray(eye_msgs['y'], dtype=float).ravel()
    ])  # 2 × n_eye_samples

    # Mask invalid (oculomatic outputs 1e6 when pupil not visible)
    eye_in[eye_in == 1e6] = np.nan

    # ---- Quadrant gain transform → mm ----------------------------------------
    quadrant_gains = get_gaze_transform(bag_dir)  # 2 × 4
    eye_mm = _apply_quadrant_gains(eye_in, quadrant_gains)

    # ---- Fill leading NaNs per channel (for stable pchip interpolation) ------
    for ch in range(2):
        first_valid = np.where(~np.isnan(eye_mm[ch]))[0]
        if len(first_valid) > 0 and first_valid[0] > 0:
            eye_mm[ch, :first_valid[0]] = eye_mm[ch, first_valid[0]]

    # ---- Convert mm → degrees (eye-to-screen distance = 400 mm) -------------
    deg_per_mm = np.degrees(np.arctan(1.0 / 400.0))
    eye_deg = deg_per_mm * eye_mm

    # ---- Clock-align eye timestamps to recorder timebase --------------------
    t_eye_rec = w_drift_ros[0] + w_drift_ros[1] * t_eye_header

    # ---- Pchip interpolation onto 1 ms grid ----------------------------------
    Fs_ds = 1000.0
    # 1 ms grid matching MATLAB's t_rec(1):0.001:t_rec(end)
    n_ds = int(np.floor((t_rec[-1] - t_rec[0]) * Fs_ds)) + 1
    t_rec_ds = t_rec[0] + np.arange(n_ds) / Fs_ds

    eye_deg_rec = np.full((2, len(t_rec_ds)), np.nan)
    for ch in range(2):
        valid = ~np.isnan(eye_deg[ch])
        if np.sum(valid) < 2:
            continue
        pchip = PchipInterpolator(t_eye_rec[valid], eye_deg[ch, valid], extrapolate=False)
        eye_deg_rec[ch] = pchip(t_rec_ds)

    # ---- Fill NaN gaps before FFT-based filter --------------------------------
    # MATLAB's conv-based mtfilter propagates NaN only locally; Python's FFT
    # propagates NaN to the entire output.  Fill remaining NaN values with
    # linear interpolation over gaps so the FFT has a clean input.  The NaN
    # positions are restored after filtering.
    eye_deg_rec_filled = eye_deg_rec.copy()
    for ch in range(2):
        y = eye_deg_rec_filled[ch]
        nan_mask = np.isnan(y)
        if nan_mask.all():
            y[:] = 0.0
        elif nan_mask.any():
            idx = np.arange(len(y))
            y[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], y[~nan_mask])
        eye_deg_rec_filled[ch] = y

    # ---- Multitaper low-pass filter at 50 Hz ---------------------------------
    yA = _mtfilter(eye_deg_rec_filled, T=0.05, W=50.0, Fs=Fs_ds)

    # Patch first floor(T*Fs) = 50 samples: hold value at sample 50
    patch_n = int(np.floor(0.05 * Fs_ds))
    for ch in range(2):
        yA[ch, :patch_n] = yA[ch, patch_n]

    # Re-center to original median (robust to outliers, matches procEye.m fix)
    for ch in range(2):
        yA[ch] += np.nanmedian(eye_deg_rec[ch]) - np.nanmedian(yA[ch])

    # ---- Save recNNN.lp.seye.dat as float32 (2×N column-major) --------------
    # Column-major: [H_0, V_0, H_1, V_1, ...] = yA.T.flatten()
    out_file = f'{py_pref}.lp.seye.dat'
    yA.T.astype(np.float32).flatten().tofile(out_file)
    print(f'  Saved {out_file}  ({yA.shape[1]} samples at {int(Fs_ds)} Hz)')


# ---------------------------------------------------------------------------
# Quadrant gain application (port of load_and_scale_task_controller_gaze_positions)
# ---------------------------------------------------------------------------

def _apply_quadrant_gains(eye_in, quadrant_gains):
    """
    Apply per-quadrant gain to eye_in (2 × N).
    quadrant_gains is 2 × 4 (mm/input_unit).
    Quadrant assignment:
      Q0: x>=0, y>=0   Q1: x<0, y>=0   Q2: x<0, y<0   Q3: x>=0, y<0
    """
    eye_mm = eye_in.copy()
    iq = [
        (eye_in[0] >= 0) & (eye_in[1] >= 0),
        (eye_in[0] < 0)  & (eye_in[1] >= 0),
        (eye_in[0] < 0)  & (eye_in[1] < 0),
        (eye_in[0] >= 0) & (eye_in[1] < 0),
    ]
    for q in range(4):
        idx = iq[q]
        eye_mm[0, idx] = eye_in[0, idx] * quadrant_gains[0, q]
        eye_mm[1, idx] = eye_in[1, idx] * quadrant_gains[1, q]
    return eye_mm


# ---------------------------------------------------------------------------
# Multitaper LP filter (port of mtfilter.m)
# ---------------------------------------------------------------------------

def _mtfilter(data, T=0.05, W=50.0, Fs=1000.0):
    """
    Convolution-based multitaper low-pass filter.

    Port of mtfilter.m / mtfilt.m / dp_proj.m (Pesaran toolbox).

    The filter kernel is built from DPSS tapers of length n = floor(T*Fs)
    via the projection matrix, giving a kernel of length 2n.  The kernel is
    applied via convolution and the output is trimmed to match the input.
    This matches MATLAB's conv-based implementation (not spectral on the full
    signal) and propagates NaN only locally.

    Parameters
    ----------
    data : 2 × N array
    T    : taper half-duration (s), e.g. 0.05
    W    : frequency cutoff (Hz), e.g. 50
    Fs   : sampling rate (Hz)

    Returns
    -------
    y : 2 × N filtered array
    """
    n = int(np.floor(T * Fs))   # taper / half-kernel length = 50
    NW = T * W                  # time-bandwidth product = 2.5
    K = max(1, int(2 * NW) - 1) # number of tapers = 4

    # DPSS tapers: scipy returns K × n; transpose to n × K (MATLAB convention)
    tapers_kn = dpss(n, NW, K)   # K × n
    vt = tapers_kn.T              # n × K

    # Projection matrix (n × n), matches MATLAB: pr = pr_op * pr_op'
    pr = vt @ vt.T

    # Build filter kernel of length 2n via MATLAB's mtfilt.m loop:
    #   for t = 0:N-1: X(t+1:t+N) += pr(:, N-t)'   (1-indexed)
    # In 0-indexed Python: filt[t:t+n] += pr[:, n-1-t]
    filt = np.zeros(2 * n)
    for t in range(n):
        filt[t:t+n] += pr[:, n - 1 - t]
    filt /= n   # MATLAB: X = X./N

    # Apply via convolution and trim to input length.
    # MATLAB (flag=0): Y(ii,:) = tmp(N/2 : szX(2)+N/2-1)  [1-indexed]
    # where N = length(filt) = 2n = 100.
    # 0-indexed start = N/2 - 1 = n - 1 = 49.
    N_filt = len(filt)   # = 2n = 100
    start = N_filt // 2 - 1   # = 49  (matches MATLAB's 1-indexed N/2 = 50)
    M = data.shape[1]

    y = np.zeros_like(data, dtype=float)
    for ch in range(data.shape[0]):
        tmp = np.convolve(data[ch].astype(float), filt)
        y[ch] = tmp[start : start + M]

    return y
