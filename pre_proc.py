"""
pre_proc.py — Port of preProcPyTask.m.

Reads bag-dir MAT/YAML files, aligns timestamps, identifies valid trials,
and saves recNNN.ev.mat.
"""

import os
import re
import json
import yaml
import numpy as np
from scipy.io import loadmat, savemat

from .helpers import rec_paths, load_w_alignment


def pre_proc(day, rec, monkeydir, out_suffix=''):
    """
    Port of preProcPyTask(day, rec).

    Reads from bag/mat/:
      trial_summary.mat, state.mat, deliver_reward.mat,
      state_state.yaml, trial_summary_task_config.yaml,
      trial_summary_used_values.yaml, trial_summary_task_result.yaml,
      trial_summary_behav_result.yaml

    Aligns timestamps via w_drift_ros, identifies valid trials, saves
    recNNN.ev.mat with:
      ev, state_ts_rec, i_complete_starts, trial_config, used_values,
      behav_results
    """
    day_dir, rec_dir, rec_pref, bag_dir = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix

    # ---- Load MAT files from bag/mat/ ----------------------------------------
    def _load_bag(name):
        return loadmat(os.path.join(bag_dir, 'mat', name), simplify_cells=True)

    trial_summary_mat = _load_bag('trial_summary.mat')
    state_mat = _load_bag('state.mat')
    reward_mat = _load_bag('deliver_reward.mat')

    # trial_summary may be nested
    ts = trial_summary_mat.get('trial_summary', trial_summary_mat)
    st = state_mat.get('state', state_mat)
    rd = reward_mat.get('deliver_reward', reward_mat)

    # ---- Load YAML files -------------------------------------------------------
    def _load_yaml(fname):
        fpath = os.path.join(bag_dir, 'mat', fname)
        with open(fpath) as fh:
            return yaml.safe_load(fh)

    state_names_raw = _load_yaml('state_state.yaml')
    trial_config_raw = _load_yaml('trial_summary_task_config.yaml')
    used_values_raw = _load_yaml('trial_summary_used_values.yaml')
    trial_results_raw = _load_yaml('trial_summary_task_result.yaml')
    behav_results_raw = _load_yaml('trial_summary_behav_result.yaml')

    # Normalise to lists
    state_names = state_names_raw if isinstance(state_names_raw, list) else [state_names_raw]
    trial_config = trial_config_raw if isinstance(trial_config_raw, list) else [trial_config_raw]
    used_values = used_values_raw if isinstance(used_values_raw, list) else [used_values_raw]
    trial_results = trial_results_raw if isinstance(trial_results_raw, list) else [trial_results_raw]
    behav_results = behav_results_raw if isinstance(behav_results_raw, list) else [behav_results_raw]

    # ---- Timestamps ------------------------------------------------------------
    def _ts(d, sec_field, ns_field):
        sec = np.asarray(d[sec_field], dtype=np.float64)
        nsec = np.asarray(d[ns_field], dtype=np.float64)
        return sec + nsec * 1e-9

    state_ts = _ts(st, 'header_stamp_sec', 'header_stamp_nanosec')
    trial_summary_ts = _ts(ts, 'header_stamp_sec', 'header_stamp_nanosec')

    # ---- Clock alignment -------------------------------------------------------
    w_drift_ros = load_w_alignment(py_pref)
    state_ts_rec = w_drift_ros[0] + w_drift_ros[1] * state_ts
    trial_summary_ts_rec = w_drift_ros[0] + w_drift_ros[1] * trial_summary_ts

    # ---- Find intertrial and start-on indices ----------------------------------
    start_codes = {'start_on', 'START_ON', 'start_touch_on'}
    end_codes = {'success', 'fail', 'SUCCESS', 'FAIL'}
    intertrial_codes = {'intertrial', 'INTERTRIAL'}

    iintertrial = [i for i, s in enumerate(state_names) if s in intertrial_codes]
    istarton = [i for i, s in enumerate(state_names) if s in start_codes]

    if len(iintertrial) != len(trial_results):
        print(f'  WARNING: unexpected number of trial results in {day} rec {rec}. '
              f'Attempting to match trials.')

    intertrial_times = state_ts_rec[iintertrial]

    # Trim trial_summary entries that fall outside intertrial window
    if len(intertrial_times) > 0 and len(trial_summary_ts_rec) > 0:
        if len(intertrial_times) < len(trial_summary_ts_rec):
            if trial_summary_ts_rec[0] < intertrial_times[0]:
                trial_summary_ts_rec = trial_summary_ts_rec[1:]
                trial_summary_ts = trial_summary_ts[1:]
                trial_config = trial_config[1:]
                trial_results = trial_results[1:]
                used_values = used_values[1:]
                behav_results = behav_results[1:]
            if len(trial_summary_ts_rec) > 0 and len(intertrial_times) > 0:
                if trial_summary_ts_rec[-1] > intertrial_times[-1]:
                    trial_summary_ts_rec = trial_summary_ts_rec[:-1]
                    trial_summary_ts = trial_summary_ts[:-1]
                    trial_config = trial_config[:-1]
                    trial_results = trial_results[:-1]
                    used_values = used_values[:-1]
                    behav_results = behav_results[:-1]

    # Build trial_times: 2 × n_intertrial_gaps matrix
    n_iti = len(iintertrial)
    trial_times = []
    for n in range(n_iti - 1):
        trial_times.append((intertrial_times[n], intertrial_times[n + 1]))
    # Extend with dummy pairs if needed
    while len(trial_times) < len(trial_summary_ts_rec):
        if intertrial_times.size > 0:
            trial_times.append((intertrial_times[-1], intertrial_times[-1]))
        else:
            break

    # Match each trial_summary timestamp to an intertrial interval
    matched_trials = np.zeros(len(trial_summary_ts_rec), dtype=int)  # 1-indexed like MATLAB
    for n, ts_r in enumerate(trial_summary_ts_rec):
        for m, (t_start, t_end) in enumerate(trial_times):
            if t_start < ts_r < t_end:
                if not np.any(matched_trials == (m + 1)):
                    matched_trials[n] = m + 1  # 1-based
                    break

    # valid_itis: which intertrial intervals had a matched trial summary
    valid_itis = np.zeros(len(iintertrial), dtype=bool)
    for m in matched_trials:
        if m > 0:
            valid_itis[m - 1] = True

    # ---- Identify complete trials (start + end code, valid ITI) ---------------
    i_complete_starts = []
    i_complete_ends = []
    trial_time_pairs = []
    valid_trial_mask = np.zeros(len(trial_results), dtype=bool)

    for n in range(n_iti - 1):
        trial_ind = list(range(iintertrial[n], iintertrial[n + 1] + 1))
        trial_states = [state_names[i] for i in trial_ind]

        n_start = sum(1 for s in trial_states if s in start_codes)
        n_end = sum(1 for s in trial_states if s in end_codes)
        has_valid_iti = valid_itis[n]

        if n_start == 1 and n_end == 1 and has_valid_iti:
            t_start_local = next(i for i, s in enumerate(trial_states) if s in start_codes)
            t_end_local = next(i for i, s in enumerate(trial_states) if s in end_codes)
            i_complete_starts.append(trial_ind[t_start_local])
            i_complete_ends.append(trial_ind[t_end_local])
            trial_time_pairs.append((intertrial_times[n], intertrial_times[n + 1]))
            # Mark the trial_summary that matched this intertrial interval
            for k, m in enumerate(matched_trials):
                if m == (n + 1):
                    valid_trial_mask[k] = True
                    break

    i_complete_starts = np.array(i_complete_starts, dtype=int)
    i_complete_ends = np.array(i_complete_ends, dtype=int)

    # Filter trial data to valid trials only
    trial_config = [tc for tc, v in zip(trial_config, valid_trial_mask) if v]
    trial_results = [tr for tr, v in zip(trial_results, valid_trial_mask) if v]
    used_values = [uv for uv, v in zip(used_values, valid_trial_mask) if v]
    behav_results = [br for br, v in zip(behav_results, valid_trial_mask) if v]
    trial_summary_ts = trial_summary_ts[valid_trial_mask]
    trial_summary_ts_rec_filt = trial_summary_ts_rec[valid_trial_mask]

    ntrials = len(trial_results)

    # ---- Sanity checks --------------------------------------------------------
    assert len(i_complete_starts) == len(i_complete_ends), \
        'state screening resulted in unequal trial starts and ends'
    assert len(i_complete_starts) == len(trial_config), \
        f'number of trial starts ({len(i_complete_starts)}) != trial configs ({len(trial_config)})'

    # ---- Build ev struct array ------------------------------------------------
    trial_start_times = state_ts_rec[i_complete_starts]
    trial_end_times = state_ts_rec[i_complete_ends]
    ts_rec_multiplier = 1.0

    myevs = []
    for iTr in range(ntrials):
        # Parse task_type to check for null
        tc = trial_config[iTr]
        if isinstance(tc, str):
            try:
                tc_struct = json.loads(tc)
            except Exception:
                tc_struct = yaml.safe_load(tc) or {}
        else:
            tc_struct = tc if isinstance(tc, dict) else {}

        task_type = tc_struct.get('task_type', '')

        if task_type == 'null':
            # Null task: find pulse_start within a short window
            if iTr >= ntrials - 2:
                current_indices = list(range(i_complete_starts[iTr], i_complete_ends[iTr] + 1))
            else:
                tmp_end = min(i_complete_starts[iTr] + 11, len(state_names))
                tmp_indices = list(range(i_complete_starts[iTr], tmp_end))
                tmp_states = [state_names[i] for i in tmp_indices]

                pulse_ind = next(
                    (k for k, s in enumerate(tmp_states) if s == 'pulse_start'),
                    None)
                if pulse_ind is None:
                    current_indices = list(range(i_complete_starts[iTr], i_complete_ends[iTr] + 1))
                else:
                    # Find last start_on before pulse
                    new_start_idx = tmp_indices[0]
                    for k in range(pulse_ind, -1, -1):
                        if tmp_states[k] == 'start_on':
                            new_start_idx = tmp_indices[k]
                            break
                    # Find success
                    success_idx = None
                    for k, s in enumerate(tmp_states):
                        if s == 'success':
                            success_idx = tmp_indices[k]
                            break
                    if success_idx is None:
                        success_idx = i_complete_ends[iTr]
                    current_indices = list(range(new_start_idx, success_idx + 1))
        else:
            current_indices = list(range(i_complete_starts[iTr], i_complete_ends[iTr] + 1))

        values = [state_names[i] for i in current_indices]
        # Fix targs2_on → targ2_on
        values = ['targ2_on' if s == 'targs2_on' else s for s in values]

        event_times = state_ts_rec[current_indices] * ts_rec_multiplier

        myevs.append({
            'trialNumber': iTr + 1,  # 1-based
            'events': values,
            'eventTimes': event_times,
            'trialresult': trial_results[iTr],
        })

    ev = myevs
    state_ts_rec_out = state_ts_rec * ts_rec_multiplier

    # ---- Save -----------------------------------------------------------------
    out_file = f'{py_pref}.ev.mat'
    savemat(out_file, {
        'ev': _ev_list_to_struct(ev),
        'state_ts_rec': state_ts_rec_out,
        'i_complete_starts': i_complete_starts + 1,  # 1-based for MATLAB compat
        'trial_config': _cell_to_mat(trial_config),
        'used_values': _cell_to_mat(used_values),
        'behav_results': _cell_to_mat(behav_results),
    })
    print(f'  pre_proc: saved {out_file} ({ntrials} trials)')

    return ev, state_ts_rec_out, i_complete_starts, trial_config, used_values, behav_results


# ---------------------------------------------------------------------------
# MATLAB savemat helpers
# ---------------------------------------------------------------------------

def _ev_list_to_struct(ev_list):
    """Convert list of dicts to object array for savemat (MATLAB struct array)."""
    import numpy as np
    # savemat handles a list of dicts as a struct array when passed as an
    # object array with dtype=object. Simpler: just save the key fields.
    # Use numpy object array of dicts.
    n = len(ev_list)
    arr = np.empty(n, dtype=object)
    for i, ev in enumerate(ev_list):
        arr[i] = ev
    return arr


def _cell_to_mat(lst):
    """Convert list of strings/dicts to MATLAB-compatible cell array."""
    n = len(lst)
    arr = np.empty(n, dtype=object)
    for i, item in enumerate(lst):
        if isinstance(item, dict):
            arr[i] = json.dumps(item)
        elif item is None:
            arr[i] = ''
        else:
            arr[i] = str(item) if not isinstance(item, str) else item
    return arr


# ---------------------------------------------------------------------------
# Load ev.mat back (for use by proc_events etc.)
# ---------------------------------------------------------------------------

def load_ev_mat(rec_pref):
    """
    Load recNNN.ev.mat and return a Python-native list of ev dicts.
    Also returns (state_ts_rec, i_complete_starts, trial_config, used_values,
    behav_results) as Python objects.
    """
    data = loadmat(f'{rec_pref}.ev.mat', simplify_cells=True)

    # ev: MATLAB struct array → list of dicts
    ev_raw = data.get('ev', [])
    ev = []
    if hasattr(ev_raw, '__len__') and len(ev_raw) > 0:
        if isinstance(ev_raw[0], dict):
            for e in ev_raw:
                events = e.get('events', [])
                if not isinstance(events, list):
                    if isinstance(events, np.ndarray):
                        events = events.tolist()
                    else:
                        events = [events]
                ev.append({
                    'trialNumber': int(e.get('trialNumber', 0)),
                    'events': events,
                    'eventTimes': np.asarray(e.get('eventTimes', []), dtype=float).ravel(),
                    'trialresult': e.get('trialresult', ''),
                })

    state_ts_rec = np.asarray(data.get('state_ts_rec', []), dtype=float).ravel()
    i_complete_starts = np.asarray(data.get('i_complete_starts', []), dtype=int).ravel() - 1  # 0-based

    def _load_cell(key):
        raw = data.get(key, [])
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            return list(raw)
        elif isinstance(raw, list):
            return raw
        return [raw]

    trial_config = _load_cell('trial_config')
    used_values = _load_cell('used_values')
    behav_results = _load_cell('behav_results')

    return ev, state_ts_rec, i_complete_starts, trial_config, used_values, behav_results
