"""
helpers.py — Shared utility functions for the AlexRig processing pipeline.

Port of MATLAB helpers in /vol/brains/raid/analyze/proc/PyTaskCtrl/v7/ and
/vol/brains/raid/analyze/utils/, /vol/brains/raid/analyze/get/.
"""

import os
import re
import json
import yaml
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import fftconvolve
from scipy.stats import norm as scipy_norm

# ---------------------------------------------------------------------------
# Directory / file helpers
# ---------------------------------------------------------------------------

def get_recs(day_dir):
    """Return sorted list of 3-digit rec directory names (e.g. ['001', '002'])."""
    entries = []
    for name in os.listdir(day_dir):
        if (os.path.isdir(os.path.join(day_dir, name))
                and re.fullmatch(r'\d{3}', name)):
            entries.append(name)
    return sorted(entries)


def rec_paths(day, rec, monkeydir):
    """Return (day_dir, rec_dir, rec_pref, bag_dir) for a given day/rec."""
    day_dir = os.path.join(monkeydir, day)
    rec_dir = os.path.join(day_dir, rec)
    rec_pref = os.path.join(rec_dir, f'rec{rec}')
    bag_dir = f'{rec_pref}.bag'
    return day_dir, rec_dir, rec_pref, bag_dir


def load_experiment(rec_pref):
    """Load experiment struct from recNNN.experiment.mat."""
    exp_file = f'{rec_pref}.experiment.mat'
    data = loadmat(exp_file, simplify_cells=True)
    return data.get('experiment', {})


def get_fs_rec(rec_pref):
    """Get recording sampling rate from experiment.mat."""
    experiment = load_experiment(rec_pref)
    try:
        acq = experiment['hardware']['acquisition']
        if isinstance(acq, list):
            return float(acq[0]['samplingrate'])
        return float(acq['samplingrate'])
    except (KeyError, TypeError, IndexError):
        return 30000.0


def load_mat_bag(bag_dir, filename):
    """Load a .mat file from bag/mat/."""
    return loadmat(os.path.join(bag_dir, 'mat', filename), simplify_cells=True)


def load_w_alignment(rec_pref):
    """Load w_drift_ros from .w_alignment.mat → 1-D array [offset, slope]."""
    data = loadmat(f'{rec_pref}.w_alignment.mat', simplify_cells=True)
    return np.asarray(data['w_drift_ros'], dtype=float).ravel()


# ---------------------------------------------------------------------------
# Clock alignment
# ---------------------------------------------------------------------------

def timestamp_alignment_distance(a, b, ioffset=0, do_robust=False):
    """
    Port of timestamp_alignment_distance.m.

    Iterative nearest-neighbour linear regression between timestamp trains a and b.
    Returns (distance, w, res) where w = [offset, slope] such that b ≈ w[0] + w[1]*a.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if ioffset > 0:
        a = a[ioffset:]
    elif ioffset < 0:
        b = b[-ioffset:]

    first_diff = b[0] - a[0]
    b = b - first_diff

    min_max = min(a.max(), b.max())
    b = b[b < min_max]
    a = a[a < min_max]

    w = np.array([0.0, 1.0])
    difftol = 1e-9

    for _ in range(20):
        n = len(a)
        X = np.column_stack([np.ones(n), a])
        corrected = X @ w

        matched_times = np.array([b[np.argmin(np.abs(corrected[i] - b))]
                                   for i in range(n)])

        w_prev = w.copy()
        w, _, _, _ = np.linalg.lstsq(X, matched_times, rcond=None)
        res = X @ w - matched_times

        if np.linalg.norm(w - w_prev) < difftol:
            break

    corrected = X @ w
    w[0] += first_diff
    distance = float(np.max(np.abs(corrected - matched_times)))
    return distance, w, res


# ---------------------------------------------------------------------------
# Transform helpers (gaze / touch)
# ---------------------------------------------------------------------------

def _parse_first_trial_config(bag_dir):
    """Parse the first entry of trial_summary_task_config.yaml → dict."""
    yaml_file = os.path.join(bag_dir, 'mat', 'trial_summary_task_config.yaml')
    with open(yaml_file) as fh:
        raw = yaml.safe_load(fh)

    entries = raw if isinstance(raw, list) else [raw]
    first = entries[0]
    if isinstance(first, str):
        try:
            first = json.loads(first)
        except Exception:
            first = yaml.safe_load(first)
    return first


def get_gaze_transform(bag_dir):
    """
    Port of get_task_controller_gaze_transform_from_bag.m.
    Returns quadrant_gains: 2×4 float array (mm/input_unit).
    """
    cfg = _parse_first_trial_config(bag_dir)

    gaze_tforms = np.array(cfg['gaze_tforms'], dtype=float)
    dpi = float(cfg['dpi'])

    # gaze_tforms is (4, 9) — each row is a flattened 3×3 homogeneous matrix.
    # Columns 0 and 4 are the diagonal gains of the 2×2 scaling part.
    quadrant_gains = gaze_tforms[:, [0, 4]].T  # shape 2×4
    quadrant_gains = quadrant_gains / dpi * 25.4
    return quadrant_gains


def get_touch_transform(bag_dir):
    """
    Port of get_task_controller_touch_transform_from_bag.m.
    Returns (M, dpi) where M is 3×3 homogeneous transform (pixels/raw → mm).
    """
    cfg = _parse_first_trial_config(bag_dir)

    M = np.array(cfg['touch_tform'], dtype=float).reshape(3, 3, order='F')
    dpi = float(cfg['dpi'])

    if 'local_to_global_translation_x' in cfg:
        ltg = np.array([cfg['local_to_global_translation_x'],
                        cfg['local_to_global_translation_y']], dtype=float)
    else:
        ltg = np.array([-1920.0, 0.0])

    M[:2, 2] += ltg

    center = np.array([cfg['frame_width'] / 2.0, cfg['frame_height'] / 2.0])
    M[:2, 2] -= center

    # y-flip (y-axis convention: up is positive)
    yflipper = np.eye(3)
    yflipper[1, 1] = -1.0
    M = yflipper @ M

    M = (1.0 / dpi * 25.4) * M
    return M, dpi


# ---------------------------------------------------------------------------
# Used-values struct
# ---------------------------------------------------------------------------

def get_used_trial_values_struct(used_values_entry):
    """
    Port of get_used_trial_values_struct.m.
    Returns (used_values_dict, target_values_list).
    target_values_list: list of dicts, one per target, with fields from targ0_*.
    """
    if isinstance(used_values_entry, dict):
        uv = used_values_entry
    elif isinstance(used_values_entry, str):
        try:
            uv = json.loads(used_values_entry)
        except Exception:
            uv = yaml.safe_load(used_values_entry) or {}
    else:
        uv = {}

    if not uv:
        return {}, []

    # Find how many targets there are (targ0_name, targ1_name, ...)
    targ_name_keys = [k for k in uv if re.match(r'targ\d+_name', k)]
    target_indices = sorted(set(
        int(re.search(r'targ(\d+)_', k).group(1)) for k in targ_name_keys
    ))

    if not target_indices:
        return uv, []

    # Fields come from targ0_*
    targ0_keys = [k for k in uv if re.match(r'targ0_', k)]
    target_fields = [re.sub(r'^targ0_', '', k) for k in targ0_keys]
    target_fields.append('mean_luminance')

    target_values = []
    for itarg in target_indices:
        tv = {}
        for field in target_fields:
            fname = f'targ{itarg}_{field}'
            if fname in uv:
                tv[field] = uv[fname]
            elif 'mean_luminance' in uv:
                tv[field] = uv['mean_luminance']
            else:
                tv[field] = float('nan')
        target_values.append(tv)

    return uv, target_values


# ---------------------------------------------------------------------------
# Binary .dat slice loaders
# ---------------------------------------------------------------------------

def _load_dat_slice(dat_file, start_samp, end_samp):
    """
    Load a [start_samp, end_samp) slice from a 2-channel interleaved float32 binary.
    The MATLAB fwrite(fid, data_2xN, 'float') layout is column-major:
      [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...].
    Returns a 2×N float32 array.
    """
    if not os.path.exists(dat_file):
        return np.full((2, 0), np.nan, dtype=np.float32)

    file_size = os.path.getsize(dat_file)
    n_total_samples = file_size // (2 * 4)  # 2 channels × 4 bytes

    s0 = max(0, min(int(start_samp), n_total_samples))
    s1 = max(0, min(int(end_samp), n_total_samples))
    n_read = s1 - s0
    if n_read <= 0:
        return np.full((2, 0), np.nan, dtype=np.float32)

    with open(dat_file, 'rb') as fh:
        fh.seek(s0 * 2 * 4)
        raw = np.frombuffer(fh.read(n_read * 2 * 4), dtype=np.float32)

    return raw.reshape(-1, 2).T  # 2 × n_read


def _event_ref_sample(Events, itrial, ref_event):
    """Return start sample (ms = sample index at 1000 Hz) for ref_event."""
    val = Events[ref_event]
    if hasattr(val, '__len__'):
        val = val[itrial]
    if np.isnan(float(val)):
        return np.nan
    return float(val)


def loadlpeye(rec_pref, Events, itrial, ref_event, window_ms):
    """
    Load a trial slice from .lp.seye.dat.

    Args:
        rec_pref : path prefix (directory/recNNN)
        Events   : dict of Events fields (timestamps in ms)
        itrial   : 0-based trial index
        ref_event: string field name in Events (e.g. 'Go')
        window_ms: [start_offset, end_offset] in ms relative to ref_event

    Returns:
        2×N float32 array of eye position (degrees).
    """
    ref_ms = _event_ref_sample(Events, itrial, ref_event)
    if np.isnan(ref_ms):
        return np.full((2, 1), np.nan, dtype=np.float32)

    s0 = round(ref_ms + window_ms[0])
    s1 = round(ref_ms + window_ms[1])
    return _load_dat_slice(f'{rec_pref}.lp.seye.dat', s0, s1)


def loadroshnd(rec_pref, Events, itrial, ref_event, window_ms):
    """
    Load a trial slice from .scaledhnd.dat.
    Same interface as loadlpeye.
    """
    ref_ms = _event_ref_sample(Events, itrial, ref_event)
    if np.isnan(ref_ms):
        return np.full((2, 1), np.nan, dtype=np.float32)

    s0 = round(ref_ms + window_ms[0])
    s1 = round(ref_ms + window_ms[1])
    return _load_dat_slice(f'{rec_pref}.scaledhnd.dat', s0, s1)


# ---------------------------------------------------------------------------
# Event matching (port of event_match.m)
# ---------------------------------------------------------------------------

def _compute_conv_corr(self_conv, match_conv, max_shift):
    """
    Port of compute_conv_corr sub-function in event_match.m.
    Returns (conv_corr, conv_dev).
    """
    mx_ind = int(np.argmax(match_conv))
    shift = round(mx_ind - len(match_conv) / 2.0)
    if abs(shift) > max_shift:
        shift = int(round(np.sign(shift) * max_shift))

    if shift > 0:
        sc_tmp = np.concatenate([np.zeros(shift), self_conv])[:len(match_conv)]
        ind = np.arange(shift, len(match_conv) - shift)
        if len(ind) < 2:
            return np.nan, float(np.sum(np.abs(match_conv - sc_tmp)))
        cc = np.corrcoef(match_conv[ind], sc_tmp[ind])
        conv_corr = float(cc[0, 1])
        conv_dev = float(np.sum(np.abs(match_conv - sc_tmp)))
    else:
        shift = abs(shift)
        mc_tmp = np.concatenate([np.zeros(shift), match_conv])[:len(self_conv)]
        ind = np.arange(shift, len(self_conv) - shift)
        if len(ind) < 2:
            return np.nan, float(np.sum(np.abs(mc_tmp - self_conv)))
        cc = np.corrcoef(self_conv[ind], mc_tmp[ind])
        conv_corr = float(cc[0, 1])
        conv_dev = float(np.sum(np.abs(mc_tmp - self_conv)))

    return conv_corr, conv_dev


def event_match(ref_event_times, match_event_times,
                Fs_rec, median_latency, min_event_interval,
                max_latency=0.8, reverse_order=False):
    """
    Port of event_match.m.

    Match two event trains using cross-correlation with a Gaussian kernel.
    When there are more match events than ref events, iteratively removes the
    least-matched display event.

    Returns:
        matched_ref_events   : 1-D array (NaN where no match)
        matched_match_events : 1-D array (NaN where no match)
        conv_corr            : scalar correlation quality metric
    """
    ref_event_times = np.asarray(ref_event_times, dtype=float).ravel()
    match_event_times = np.asarray(match_event_times, dtype=float).ravel()

    max_shift = median_latency * Fs_rec * 2

    # Build time axis
    dt = 1.0 / Fs_rec
    t0 = ref_event_times[0] - max_latency
    t1 = ref_event_times[-1] + max_latency
    filter_time = np.arange(t0, t1 + dt, dt)

    def _place_events(times, ft):
        arr = np.zeros(len(ft))
        for t in times:
            idx = int(np.argmin(np.abs(ft - t)))
            arr[idx] = 1.0
        return arr

    filter_init = _place_events(ref_event_times, filter_time)
    candidate = _place_events(match_event_times, filter_time)

    mu = np.mean(filter_time)
    latency_kernel = scipy_norm.pdf(filter_time, loc=mu, scale=median_latency * 2)
    fast_kernel = scipy_norm.pdf(filter_time, loc=mu, scale=min_event_interval)
    filter_kernel = (latency_kernel / latency_kernel.max()
                     + 0.1 * fast_kernel / fast_kernel.max())
    filt = fftconvolve(filter_init, filter_kernel, mode='same')
    filt = filt / filt.max()

    self_conv = fftconvolve(filter_init, filt[::-1], mode='same')
    match_conv = fftconvolve(candidate, filt[::-1], mode='same')

    n_events = max(len(match_event_times), len(ref_event_times))
    matched_ref = np.full(n_events, np.nan)
    matched_match = np.full(n_events, np.nan)

    if len(match_event_times) == len(ref_event_times):
        matched_ref = ref_event_times.copy()
        matched_match = match_event_times.copy()
        conv_corr, _ = _compute_conv_corr(self_conv, match_conv, np.inf)

    elif len(match_event_times) < len(ref_event_times):
        # Recurse with roles swapped
        matched_match, matched_ref, conv_corr = event_match(
            match_event_times, ref_event_times,
            Fs_rec, median_latency, min_event_interval,
            max_latency, reverse_order=True)

    else:
        # More match events than ref events — remove extras.
        # MATLAB initialises matched_match_events = match_event_times here (line 62),
        # so surviving display times remain at their original positions after trimming.
        matched_match[:] = match_event_times
        unmatched = np.zeros(n_events, dtype=bool)
        orig_n = list(range(n_events))
        cand = candidate.copy()
        met = list(match_event_times.copy())

        while len(met) > len(ref_event_times):
            match_dev = np.zeros(len(met))
            for i in range(len(met)):
                remove_idx = int(np.argmin(np.abs(met[i] - filter_time)))
                cand_tmp = cand.copy()
                cand_tmp[remove_idx] = 0.0
                conv_tmp = fftconvolve(cand_tmp, filt[::-1], mode='same')
                _, dev = _compute_conv_corr(self_conv, conv_tmp, max_shift)
                match_dev[i] = dev

            min_n = int(np.argmin(match_dev))
            remove_idx = int(np.argmin(np.abs(met[min_n] - filter_time)))
            cand[remove_idx] = 0.0
            match_conv = fftconvolve(cand, filt[::-1], mode='same')
            unmatched[orig_n[min_n]] = True
            orig_n.pop(min_n)
            met.pop(min_n)

        matched_ref[~unmatched] = ref_event_times
        conv_corr, _ = _compute_conv_corr(self_conv, match_conv, np.inf)

    return matched_ref, matched_match, conv_corr
