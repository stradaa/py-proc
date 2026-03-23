"""
proc_display.py — Port of procDisplay_new.m.

Reads the (corrected) display signal, thresholds it, finds transition
times, then matches those transitions to task-controller event times using
event_match() per trial.  Saves recNNN.displayEvents.mat.
"""

import os
import json
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import medfilt

from .helpers import (rec_paths, get_fs_rec, event_match)
from .pre_proc import load_ev_mat


# Display states that trigger a screen update (match across all task types)
_DISPLAY_STATES_BY_TASK = {
    'gaze_anchoring':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success',
         'start_on', 'targs_on', 'go', 'targs_acq', 'targs2_on', 'success'},
    'gaze_anchoring_fast':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'doublestep_saccade':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'doublestep_saccade_fast':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'doublestep_saccade_and_touch':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'doublestep_saccade_and_touch_fast':
        {'start_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'doublestep_saccade_and_touch_sequence':
        {'start_on', 'start_gaze_on', 'targs_on', 'go', 'targs_acq', 'targ2_on', 'success'},
    'delayed_reach':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_reach_gif':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_reach_stim_task':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_read_stim_task':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_reach_and_saccade':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_saccade':
        {'start_on', 'targs_on', 'go', 'success'},
    'delayed_saccade_touch':
        {'start_on', 'targs_on', 'go', 'success'},
    'simple_touch_task':
        {'start_on', 'success'},
    'simple_touch_task_feedback':
        {'start_on', 'success'},
    'simple_touch_and_look_task':
        {'start_on', 'success'},
    'simple_saccade_touch_task':
        {'start_on', 'success'},
    'luminance_reward_selection':
        {'start_on', 'go', 'success'},
    'calibrate_eye_reach':
        {'start_on', 'targs_on', 'go', 'success'},
    'calibrate_eye_saccade':
        {'start_on', 'targs_on', 'go', 'success'},
}


def proc_display(day, rec, monkeydir, out_suffix=''):
    """
    Port of procDisplay_new(day, rec).

    Reads recNNN.display_corrected.dat (or .display.dat), thresholds,
    matches display transitions to task events, saves recNNN.displayEvents.mat.
    """
    _, rec_dir, rec_pref, bag_dir = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix

    # ---- Load sampling rate --------------------------------------------------
    Fs_rec = get_fs_rec(rec_pref)
    time_multiplier = 1.0

    # ---- Load display signal -------------------------------------------------
    corr_file = f'{py_pref}.display_corrected.dat'
    raw_file = f'{rec_pref}.display.dat'
    if os.path.exists(corr_file):
        display_file = corr_file
        print(f'  Using corrected display: {corr_file}')
    else:
        display_file = raw_file

    display_pre = np.fromfile(display_file, dtype=np.float32)
    display = medfilt(display_pre.astype(np.float64), kernel_size=99)
    display_time = np.arange(1, len(display) + 1) / Fs_rec  # seconds

    # ---- Threshold -----------------------------------------------------------
    _, edges = np.histogram(display, bins=10)
    midpt = np.mean(edges)
    display_on = display > midpt

    display_change = np.concatenate([[0], np.abs(np.diff(display_on.astype(np.int8)))])

    if midpt > 2:
        # Fallback: quantile-based change detection on downsampled signal
        print(f'  WARNING: midpt={midpt:.2f} > 2 V on rec {rec}, using quantile method')
        ds_step = max(1, Fs_rec // 500)
        display_ds = display[::ds_step]
        display_diff_ds = np.abs(np.diff(display_ds))
        thresh_q = np.quantile(display_diff_ds, 0.997)
        display_changed_ds = np.concatenate([[0], display_diff_ds > thresh_q])
        display_changed = np.zeros(len(display))
        display_changed[::ds_step] = display_changed_ds[:len(display_changed[::ds_step])]

        # Refine: remove too-close transitions
        changed_ind = np.where(display_changed == 1)[0]
        refined = list(changed_ind)
        diffs = np.diff(changed_ind)
        for k in range(len(diffs) - 1):
            if diffs[k] < 0.02 * Fs_rec:
                for idx in (changed_ind[k], changed_ind[k + 1]):
                    if idx in refined:
                        refined.remove(idx)

        display_change = np.zeros(len(display))
        display_change[refined] = 1.0

    # ---- Load pre-proc ev data -----------------------------------------------
    ev, state_ts_rec, i_complete_starts, trial_config, used_values, behav_results = \
        load_ev_mat(py_pref)

    t_starton = state_ts_rec[i_complete_starts]  # seconds
    t_starton_ms = t_starton * 1e3

    # Display transition times (seconds)
    t_display_changes = np.where(display_change)[0] / Fs_rec

    # Rough estimate of median display latency.
    # Mirrors MATLAB procDisplay_new.m lines 122-127:
    #   [~,imin] = min(abs(bsxfun(@minus, t_display_changes, t_starton')), [], 2)
    #   display_latencies = t_display_changes(imin) - t_starton   % (1 x N_trials) vector
    #   median_display_latency = median(display_latencies(find(display_latencies)>0))
    # For each trial, find nearest display transition; compute display - trial_start.
    # MATLAB's find(x)>0 trick = all nonzero elements (find indices are 1-based, always > 0).
    if len(t_display_changes) > 0 and len(t_starton) > 0:
        imin = np.argmin(
            np.abs(t_display_changes[None, :] - t_starton[:, None]), axis=1)  # (N_trials,)
        nearest_display = t_display_changes[imin]  # (N_trials,)
        display_latencies = nearest_display - t_starton  # (N_trials,)
        nonzero_lat = display_latencies[display_latencies != 0]
        median_display_latency = float(np.median(nonzero_lat)) if len(nonzero_lat) > 0 else 0.03
    else:
        median_display_latency = 0.03  # 30 ms default

    median_display_latency_ms = median_display_latency * 1000
    # NO < 10ms guard — MATLAB procDisplay_new.m has none
    print(f'  Median display latency: {median_display_latency_ms:.1f} ms')

    # ---- Match per trial -----------------------------------------------------
    numtrials = len(ev)
    displayEvents = []
    displayLatencyAcrossStates = []
    conv_corr_arr = np.zeros(numtrials)
    missing_states = np.zeros(numtrials)
    max_latency = 0.4  # seconds
    min_event_interval = 0.01  # seconds

    for iTr in range(numtrials):
        tc = trial_config[iTr] if iTr < len(trial_config) else ''
        if isinstance(tc, str):
            try:
                tc_struct = json.loads(tc)
            except Exception:
                try:
                    import yaml as _yaml
                    tc_struct = _yaml.safe_load(tc) or {}
                except Exception:
                    tc_struct = {}
        else:
            tc_struct = tc if isinstance(tc, dict) else {}

        trial_type = tc_struct.get('task_type', '')
        display_states = _DISPLAY_STATES_BY_TASK.get(trial_type, set())

        # Get task-controller event times for this trial (seconds)
        trial_events = ev[iTr]['events']
        trial_event_times_s = ev[iTr]['eventTimes'] * time_multiplier  # seconds

        # Filter to display states
        mask = [e.lower().strip() in display_states for e in trial_events]
        tmpEv = [e.lower().strip() for e, m in zip(trial_events, mask) if m]
        tmpEvTimes = trial_event_times_s[mask]

        if len(tmpEvTimes) == 0:
            missing_states[iTr] = np.nan
            displayEvents.append(_empty_display_event(ev[iTr]['trialNumber']))
            continue

        # Display changes in the trial window (strictly after first task event)
        t0 = tmpEvTimes[0]
        t1 = tmpEvTimes[-1] + max_latency
        dispEv_time = t_display_changes[(t_display_changes > t0) &
                                        (t_display_changes < t1)]

        if len(dispEv_time) == 0:
            missing_states[iTr] = np.nan
            displayEvents.append(_empty_display_event(ev[iTr]['trialNumber']))
            continue

        if len(dispEv_time) >= 10 * len(tmpEvTimes):
            print(f'  Too many display events for trial {iTr + 1}')
            displayEvents.append(_empty_display_event(ev[iTr]['trialNumber']))
            continue

        matched_tmp, matched_disp, cc = event_match(
            tmpEvTimes, dispEv_time,
            Fs_rec, median_display_latency, min_event_interval, max_latency)

        conv_corr_arr[iTr] = cc if not np.isnan(cc) else 0.0
        missing_states[iTr] = int(np.sum(np.isnan(matched_disp)))

        # Intersect with original event list
        event_ind = []
        for t in matched_tmp:
            if not np.isnan(t):
                diffs = np.abs(tmpEvTimes - t)
                event_ind.append(int(np.argmin(diffs)))
        devents = [tmpEv[i] for i in event_ind]

        # Remove NaN entries from matched
        nan_mask = np.isnan(matched_tmp)
        matched_disp_clean = matched_disp[~nan_mask]
        matched_tmp_clean = matched_tmp[~nan_mask]

        latency_ms = (matched_disp_clean - matched_tmp_clean) * 1000.0

        displayEvents.append({
            'trialNumber': ev[iTr]['trialNumber'],
            'events': devents,
            'eventTimes': matched_disp_clean,
            'eventTimes_in_ms': matched_disp_clean * 1000.0,
            'latency_in_ms': latency_ms,
        })
        displayLatencyAcrossStates.extend(latency_ms.tolist())

    # ---- Summary -------------------------------------------------------------
    valid_cc = conv_corr_arr[conv_corr_arr > 0]
    if len(valid_cc) > 0:
        print(f'  Conv correlation: mean={np.mean(valid_cc):.3f}  '
              f'min={np.min(valid_cc):.3f}  max={np.max(valid_cc):.3f}')

    n_no_display = int(np.sum(np.isnan(missing_states)))
    n_with_missing = int(np.sum((missing_states > 0) & ~np.isnan(missing_states)))
    print(f'  Trials with no display events: {n_no_display} / {numtrials}')
    print(f'  Trials with missing display states: {n_with_missing} / {numtrials}')
    if displayLatencyAcrossStates:
        arr = np.array(displayLatencyAcrossStates)
        print(f'  Display latency: min={np.nanmin(arr):.1f} ms  max={np.nanmax(arr):.1f} ms')

    # ---- Save ----------------------------------------------------------------
    lat_arr = np.array(displayLatencyAcrossStates) if displayLatencyAcrossStates else np.array([])
    savemat(f'{py_pref}.displayEvents.mat', {
        'displayEvents': _display_events_to_mat(displayEvents),
        'displayLatencyAcrossStates': lat_arr,
    })
    print(f'  Saved {py_pref}.displayEvents.mat')


def _empty_display_event(trial_number):
    return {
        'trialNumber': trial_number,
        'events': [],
        'eventTimes': np.array([]),
        'eventTimes_in_ms': np.array([]),
        'latency_in_ms': np.array([]),
    }


def _display_events_to_mat(de_list):
    """Convert list of display-event dicts to object array for savemat."""
    n = len(de_list)
    arr = np.empty(n, dtype=object)
    for i, de in enumerate(de_list):
        arr[i] = de
    return arr
