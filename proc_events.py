"""
proc_events.py — Port of procEvents.m (v7).

Main event-extraction pass.  Calls pre_proc and (optionally) proc_display
internally, then populates all Events fields and saves recNNN.Events.mat.

When use_display=False this mimics procEvents_noDisplay.m: task-controller
timestamps are used for all fields; the dis* fields remain NaN.
"""

import os
import json
import yaml
import numpy as np
from scipy.io import loadmat, savemat

from .helpers import (rec_paths, get_fs_rec,
                      timestamp_alignment_distance, get_used_trial_values_struct)
from .pre_proc import pre_proc, load_ev_mat


def proc_events(day, rec, monkeydir, use_display=True, out_suffix=''):
    """
    Port of procEvents(day, rec) / procEvents_noDisplay(day, rec).

    Parameters
    ----------
    day         : str  e.g. '260220'
    rec         : str  e.g. '001'
    use_display : bool  If False, skip proc_display (noDisplay pass)
    out_suffix  : str  Appended to rec_pref for all Python pipeline output files
    """
    _, rec_dir, rec_pref, bag_dir = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix
    print(f'\nproc_events: day={day} rec={rec} use_display={use_display}')

    # ---- Sampling rate -------------------------------------------------------
    Fs_rec = get_fs_rec(rec_pref)
    Fs_rec_ds = 1000.0  # downsampled 1 ms grid

    # ---- Clock alignment (AlexRig single-clock shortcut) --------------------
    _compute_and_save_alignment(rec_pref, py_pref, Fs_rec, bag_dir)

    # ---- Pre-process (state → ev.mat) ----------------------------------------
    pre_proc(day, rec, monkeydir, out_suffix)

    ev, state_ts_rec, i_complete_starts, trial_config, used_values, behav_results = \
        load_ev_mat(py_pref)

    # ---- Display sensor events -----------------------------------------------
    if use_display:
        _task_type = _get_first_task_type(trial_config)
        if _task_type == 'null':
            print('  Null task — skipping display sensor')
            displayEvents = None
            median_displayLatency = 0.0
        else:
            from .proc_display import proc_display
            proc_display(day, rec, monkeydir, out_suffix)
            disp_data = loadmat(f'{py_pref}.displayEvents.mat', simplify_cells=True)
            displayEvents = _parse_display_events(disp_data)
            lat = np.asarray(disp_data.get('displayLatencyAcrossStates', []), dtype=float).ravel()
            median_displayLatency = float(np.nanmedian(lat)) if len(lat) > 0 else 60.0
            if median_displayLatency < 10 or np.isnan(median_displayLatency):
                median_displayLatency = 30.0
    else:
        displayEvents = None
        median_displayLatency = 30.0

    # ---- Populate Events struct -----------------------------------------------
    ntrials = len(ev)
    Events = _init_events(ntrials)

    for itrial in range(ntrials):
        Events['Trial'][itrial] = itrial + 1  # 1-based

        # ---- Parse trial config ----------------------------------------------
        tc = trial_config[itrial] if itrial < len(trial_config) else ''
        tc_struct = _parse_config(tc)
        trial_type = tc_struct.get('task_type', '')

        if trial_type == 'null':
            Events['PyTaskType'].append(trial_type)
            Events['UsedValues'].append({})
            Events['TargetConfigs'].append([])
            Events['TargetValues'].append([])
            target_values = []
            test_targets = []
        else:
            test_targets = tc_struct.get('targets', [])
            uv_entry = used_values[itrial] if itrial < len(used_values) else ''
            used_values_struct, target_values = get_used_trial_values_struct(uv_entry)
            Events['PyTaskType'].append(trial_type)
            Events['UsedValues'].append(used_values_struct)
            Events['TargetConfigs'].append(_rename_target_configs(test_targets))
            Events['TargetValues'].append(target_values)

            is_choice = tc_struct.get('is_choice', False)
            if is_choice:
                if isinstance(test_targets, list):
                    Events['Choice'][itrial] = sum(
                        1 for t in test_targets
                        if not (t.get('is_fixation', False) if isinstance(t, dict) else False))
                else:
                    Events['Choice'][itrial] = 1
            else:
                Events['Choice'][itrial] = 1 if trial_type != 'null' else 0

        # ---- TaskCode + Reach/Saccade flags ----------------------------------
        _set_task_code(Events, itrial, trial_type)

        # ---- Event times (ms) ------------------------------------------------
        trial_events = ev[itrial]['events']
        trial_event_times_ms = ev[itrial]['eventTimes'] * 1e3  # s → ms

        # ---- Display event times for this trial ------------------------------
        if displayEvents is not None and itrial < len(displayEvents):
            de = displayEvents[itrial]
            if de and de.get('trialNumber') == ev[itrial]['trialNumber']:
                trial_displayEvents = de.get('events', [])
                trial_displayEventTimes_ms = np.asarray(
                    de.get('eventTimes_in_ms', []), dtype=float)
            else:
                trial_displayEvents = []
                trial_displayEventTimes_ms = np.array([])
        else:
            trial_displayEvents = []
            trial_displayEventTimes_ms = np.array([])

        def _task_ev_time(name_set):
            """First occurrence of any name in name_set in trial_events (ms)."""
            for i, e in enumerate(trial_events):
                if e.lower().strip() in name_set:
                    return float(trial_event_times_ms[i])
            return np.nan

        def _disp_ev_time(name_set):
            """First display event matching any name in name_set (ms)."""
            if len(trial_displayEvents) == 0:
                return np.nan
            for i, e in enumerate(trial_displayEvents):
                if e.lower().strip() in name_set:
                    if i < len(trial_displayEventTimes_ms):
                        v = float(trial_displayEventTimes_ms[i])
                        return v if not np.isnan(v) else np.nan
            return np.nan

        # ---- StartTrial / StartOn --------------------------------------------
        start_names = {'start_on', 'start_touch_on'}
        StartTrial = _task_ev_time(start_names)
        Events['StartTrial'][itrial] = StartTrial

        # StartOn: prefer display time, fall back to task controller
        dis_StartOn = _disp_ev_time(start_names)
        if not np.isnan(dis_StartOn):
            Events['StartOn'][itrial] = dis_StartOn
        else:
            Events['StartOn'][itrial] = StartTrial
        Events['disStartOn'][itrial] = dis_StartOn

        # ---- start_gaze_on (doublestep_saccade_and_touch_sequence) -----------
        if any(e.lower().strip() == 'start_gaze_on' for e in trial_events):
            Events['StartGazeOn'][itrial] = _task_ev_time({'start_gaze_on'})
            Events['disStartGazeOn'][itrial] = _disp_ev_time({'start_gaze_on'})

        # ---- StartAq ---------------------------------------------------------
        StartAq = _task_ev_time({'start_acq'})
        Events['StartAq'][itrial] = StartAq

        # If StartAq < StartOn, fall back to task-controller StartOn
        if not np.isnan(StartAq) and not np.isnan(Events['StartOn'][itrial]):
            if StartAq < Events['StartOn'][itrial]:
                Events['StartOn'][itrial] = StartTrial

        # ---- TargsOn ---------------------------------------------------------
        if _is_simple_task(trial_type) and not any(
                e.lower().strip() in {'targs_on'} for e in trial_events):
            Events['TargsOn'][itrial] = StartTrial
            Events['disTargsOn'][itrial] = np.nan
        elif any(e.lower().strip() == 'targs_on' for e in trial_events):
            Events['TargsOn'][itrial] = _task_ev_time({'targs_on'})
            Events['disTargsOn'][itrial] = _disp_ev_time({'targs_on'})

        # ---- Go --------------------------------------------------------------
        if any(e.lower().strip() == 'go' for e in trial_events):
            Events['Go'][itrial] = _task_ev_time({'go'})
            Events['disGo'][itrial] = _disp_ev_time({'go'})

        # ---- TargAq ----------------------------------------------------------
        if any(e.lower().strip() == 'targs_acq' for e in trial_events):
            Events['TargAq'][itrial] = _task_ev_time({'targs_acq'})

        # ---- Targ2On ---------------------------------------------------------
        if any(e.lower().strip() == 'targ2_on' for e in trial_events):
            Events['Targ2On'][itrial] = _task_ev_time({'targ2_on'})
            Events['disTarg2On'][itrial] = _disp_ev_time({'targ2_on'})

        # ---- Targ2Aq ---------------------------------------------------------
        if any(e.lower().strip() == 'targ2_acq' for e in trial_events):
            Events['Targ2Aq'][itrial] = _task_ev_time({'targ2_acq'})

        # ---- Pulse_start / Pulse_end -----------------------------------------
        if any(e.lower().strip() == 'pulse_start' for e in trial_events):
            Events['Pulse_start'][itrial] = _task_ev_time({'pulse_start'})
        if any(e.lower().strip() == 'pulse_end' for e in trial_events):
            Events['Pulse_end'][itrial] = _task_ev_time({'pulse_end'})

        # ---- Optogenetics params from config ---------------------------------
        if 'stim_start' in tc_struct:
            Events['StimStart'][itrial] = float(tc_struct['stim_start'])
        if 'intan_cfg' in tc_struct:
            Events['Intan_Cfg'][itrial] = float(tc_struct.get('intan_cfg', 0))
            Events['Pulse_count'][itrial] = float(tc_struct.get('pulse_count', 0))
            Events['Pulse_freq'][itrial] = float(tc_struct.get('pulse_frequency', 0))
            Events['Pulse_width'][itrial] = float(tc_struct.get('pulse_width', 0))

        # ---- Success / Fail / End --------------------------------------------
        if any(e.lower().strip() == 'success' for e in trial_events):
            Events['Success'][itrial] = 1
            Events['End'][itrial] = _task_ev_time({'success'})
        elif any(e.lower().strip() == 'fail' for e in trial_events):
            Events['Success'][itrial] = 0
            Events['End'][itrial] = _task_ev_time({'fail'})

        # ---- Behav results: targets, choices, locations ----------------------
        br = behav_results[itrial] if itrial < len(behav_results) else None
        if trial_type != 'null' and br:
            _populate_behav_results(Events, itrial, trial_type, br,
                                    target_values, tc_struct)

        # ---- TargSeq ---------------------------------------------------------
        if trial_type != 'null' and target_values:
            Events['TargSeq'].append(
                _build_targ_seq(Events, itrial, trial_events, trial_type,
                                test_targets, target_values))
        else:
            Events['TargSeq'].append([])

    # ---- Finalise and save ---------------------------------------------------
    Events['Trial'] = np.arange(1, ntrials + 1)

    print(f'  # trials: {ntrials}')
    savemat(f'{py_pref}.Events.mat', {'Events': _events_to_mat(Events)})
    print(f'  Saved {py_pref}.Events.mat')


# ---------------------------------------------------------------------------
# Clock alignment
# ---------------------------------------------------------------------------

def _compute_and_save_alignment(rec_pref, py_pref, Fs_rec, bag_dir):
    """
    Compute and save .w_alignment.mat.

    AlexRig single-clock shortcut: when ros ≈ local, simhead_weights = [0, 1].
    Then timestamp_alignment_distance(ev_times_header_est, ev_times_rec).
    """
    out_file = f'{py_pref}.w_alignment.mat'

    # Load fiducial data
    fid_data = loadmat(os.path.join(bag_dir, 'mat', 'fiducial.mat'),
                       simplify_cells=True)
    fid_cmd_data = loadmat(os.path.join(bag_dir, 'mat', 'fiducial_pulse.mat'),
                           simplify_cells=True)
    fiducial_rec = np.asarray(
        loadmat(f'{rec_pref}.fiducial.mat', simplify_cells=True)['fiducial'],
        dtype=float).ravel()

    if isinstance(fid_data.get('fiducial', fid_data), dict):
        fd = fid_data.get('fiducial', fid_data)
    else:
        fd = fid_data
    if isinstance(fid_cmd_data.get('fiducial_pulse', fid_cmd_data), dict):
        fcd = fid_cmd_data.get('fiducial_pulse', fid_cmd_data)
    else:
        fcd = fid_cmd_data

    first_fid_ts_ns = int(np.asarray(fd['topic_time_stamp']).ravel()[0])
    first_fid_ts = first_fid_ts_ns / 1e9

    if first_fid_ts < 100:
        first_fid_ts_ns = int(np.asarray(fd['time_ref_sec']).ravel()[0])
        first_fid_ts = float(first_fid_ts_ns)

    # ev_times_ros: from fiducial_pulse stamp
    try:
        ev_times_ros = (np.asarray(fcd['stamp_sec'], dtype=float).ravel()
                        + np.asarray(fcd['stamp_nanosec'], dtype=float).ravel() / 1e9
                        - first_fid_ts)
    except KeyError:
        ev_times_ros = (np.asarray(fd['topic_time_stamp'], dtype=float).ravel()
                        - first_fid_ts_ns) / 1e9

    ev_times_local = (np.asarray(fd['time_ref_sec'], dtype=float).ravel()
                      + np.asarray(fd['time_ref_nanosec'], dtype=float).ravel() / 1e9
                      - first_fid_ts)

    if len(ev_times_ros) < 2:
        print(f'  WARNING: too few fiducial events, using identity alignment')
        w_drift_ros = np.array([0.0, 1.0])
        alignment_distance = 0.0
        res = np.zeros(1)
        savemat(out_file, {'alignment_distance': alignment_distance,
                           'w_drift_ros': w_drift_ros, 'res': res})
        return

    # AlexRig single-clock shortcut
    if np.max(np.abs(ev_times_ros - ev_times_local)) < 1e-9:
        simhead_weights = np.array([0.0, 1.0])
    else:
        iquick = (ev_times_ros - ev_times_local) < np.percentile(
            ev_times_ros - ev_times_local, 80)
        try:
            _, simhead_weights, _ = timestamp_alignment_distance(
                ev_times_local[iquick], ev_times_ros[iquick], 0, False)
        except Exception:
            simhead_weights = np.array([0.0, 1.0])

    ev_times_header_est = simhead_weights[0] + simhead_weights[1] * ev_times_local

    # Recorder-side fiducial pulse times.
    # Mirrors MATLAB: find([0, diff(fiducial)>0]) which is 1-indexed, so position 1
    # (the prepended 0) never fires.  In Python, position 0 of the diff fires if
    # fiducial_rec[0] > 0 (recording started mid-pulse — spurious edge at t=0).
    # Conditionally drop it to match MATLAB on all days.
    _edges = np.where(np.diff(np.concatenate([[0], fiducial_rec])) > 0)[0]
    if len(_edges) > 0 and _edges[0] == 0:
        _edges = _edges[1:]
    ev_times_rec = _edges / Fs_rec

    if len(ev_times_header_est) == 0 or len(ev_times_rec) == 0:
        w_drift_ros = np.array([0.0, 1.0])
        alignment_distance = 0.0
        res = np.zeros(1)
    else:
        alignment_distance, w_drift_ros, res = timestamp_alignment_distance(
            ev_times_header_est, ev_times_rec, 0, False)
        w_drift_ros[0] -= w_drift_ros[1] * first_fid_ts

        if alignment_distance > 1e-2:
            # Try dropping leading analog pulses — handles spurious recording-start
            # edges (ev_times_rec[0] ≈ 0) that cause a one-period misalignment.
            n_extra = len(ev_times_rec) - len(ev_times_header_est)
            for n_drop in range(1, min(4, n_extra + 2)):
                dist_d, w_d, res_d = timestamp_alignment_distance(
                    ev_times_header_est, ev_times_rec[n_drop:], 0, False)
                if dist_d < alignment_distance:
                    w_d[0] -= w_d[1] * first_fid_ts
                    alignment_distance = dist_d
                    w_drift_ros = w_d
                    res = res_d
                    print(f'  Alignment improved: dropped first {n_drop} analog '
                          f'pulse(s) (dist={dist_d:.2e})')
                    break

        if alignment_distance > 1e-2 and np.abs(np.median(res)) > 3e-3:
            # Fallback: least-squares
            n_max = min(len(ev_times_header_est), len(ev_times_rec))
            xx = ev_times_header_est[:n_max]
            yy = ev_times_rec[:n_max]
            coeffs = np.polyfit(xx, yy, 1)
            w_ls = np.array([coeffs[1], coeffs[0]])  # [offset, slope]
            res_ls = (w_ls[0] + w_ls[1] * xx) - yy
            dist_ls = float(np.max(np.abs(res_ls)))
            if dist_ls < alignment_distance:
                alignment_distance = dist_ls
                w_drift_ros = w_ls
                res = res_ls
                w_drift_ros[0] -= w_drift_ros[1] * first_fid_ts
                print('  Least-squares alignment succeeded')

    savemat(out_file, {'alignment_distance': float(alignment_distance),
                       'w_drift_ros': w_drift_ros, 'res': res})
    print(f'  Alignment saved: offset={w_drift_ros[0]:.6f}  '
          f'slope={w_drift_ros[1]:.9f}  dist={alignment_distance:.2e}')


# ---------------------------------------------------------------------------
# Helper: parse config entry
# ---------------------------------------------------------------------------

def _parse_config(tc):
    if isinstance(tc, dict):
        return tc
    elif isinstance(tc, str):
        try:
            return json.loads(tc)
        except Exception:
            try:
                return yaml.safe_load(tc) or {}
            except Exception:
                return {}
    return {}


def _get_first_task_type(trial_config):
    if not trial_config:
        return ''
    tc = _parse_config(trial_config[0])
    return tc.get('task_type', '')


def _is_simple_task(trial_type):
    return 'simple' in trial_type


# ---------------------------------------------------------------------------
# Init Events dict
# ---------------------------------------------------------------------------

def _init_events(ntrials):
    def _nan(n, shape2=None):
        if shape2:
            return np.full((n, shape2), np.nan)
        return np.full(n, np.nan)

    def _zeros(n):
        return np.zeros(n)

    n = ntrials
    return {
        'Trial': np.zeros(n, dtype=int),
        'TaskCode': _nan(n),
        'PyTaskType': [],
        'StartTrial': _zeros(n),
        'StartOn': _zeros(n),
        'disStartOn': _nan(n),
        'StartGazeOn': _nan(n),
        'disStartGazeOn': _nan(n),
        'StartAq': _nan(n),
        'TargsOn': _nan(n),
        'disTargsOn': _nan(n),
        'CueOn': _nan(n),
        'CueRt': _zeros(n),
        'CueLt': _zeros(n),
        'Go': _nan(n),
        'disGo': _nan(n),
        'Targ2On': _nan(n),
        'disTarg2On': _nan(n),
        'SaccStart': _nan(n),
        'SaccStop': _nan(n),
        'Sacc2Start': _nan(n),
        'Sacc2Stop': _nan(n),
        'ReachStart': _nan(n),
        'ReachStop': _nan(n),
        'Reach2Start': _nan(n),
        'Reach2Stop': _nan(n),
        'TargAq': _nan(n),
        'Targ2Aq': _nan(n),
        'TargetOff': _nan(n),
        'Choice': _zeros(n),
        'Target': _nan(n),
        'Target1': _nan(n),
        'Target2': _nan(n),
        'ChosenTarget': _nan(n),
        'ChosenTarget2': _nan(n),
        'Reach': _zeros(n),
        'Saccade': _zeros(n),
        'End': _nan(n),
        'Success': _zeros(n),
        'RewardReceived': _zeros(n),
        'TargetConfigs': [],
        'TargetValues': [],
        'UsedValues': [],
        'TargSeq': [],
        'TargetLocation': _nan(n, 2),
        'Target2Location': _nan(n, 2),
        'EyeTargetLocation': _nan(n, 2),
        'HandTargetLocation': _nan(n, 2),
        'EyeTarg2Location': _nan(n, 2),
        'TargetAngle': _nan(n),
        'Target2Angle': _nan(n),
        # LRS
        'targ1': [None] * n,
        'targ2': [None] * n,
        'targ1_lum': _nan(n),
        'targ2_lum': _nan(n),
        'targ1_height': _nan(n),
        'targ2_height': _nan(n),
        'targ1_width': _nan(n),
        'targ2_width': _nan(n),
        'targ1_angle': _nan(n),
        'targ2_angle': _nan(n),
        'targ1_reward': _nan(n),
        'targ2_reward': _nan(n),
        'targ1_location': _nan(n, 2),
        'targ2_location': _nan(n, 2),
        # Stim / opto
        'StimStart': _nan(n),
        'Intan_Cfg': _nan(n),
        'Pulse_count': _nan(n),
        'Pulse_freq': _nan(n),
        'Pulse_width': _nan(n),
        'Pulse_start': _nan(n),
        'Pulse_end': _nan(n),
        # Reach size
        'ReachSize': _nan(n),
    }


# ---------------------------------------------------------------------------
# TaskCode / Reach / Saccade flags
# ---------------------------------------------------------------------------

_TASK_CODES = {
    'simple_touch_task': (1, True, False),
    'simple_saccade_touch_task': (5, True, True),
    'simple_touch_and_look_task': (4, True, True),
    'delayed_reach': (9, True, True),
    'delayed_reach_gif': (9, True, True),
    'suppressed_reach': (45, True, False),
    'distractor_suppression_reach': (44, True, False),
    'delayed_saccade': (11, False, True),
    'delayed_saccade_touch': (111, True, True),
    'context_dependent_delay_reach': (50, False, False),
    'gaze_anchoring': (38, True, True),
    'gaze_anchoring_fast': (381, True, True),
    'doublestep_saccade': (39, False, True),
    'doublestep_saccade_fast': (391, False, True),
    'doublestep_saccade_and_touch': (392, True, True),
    'doublestep_saccade_and_touch_fast': (393, True, True),
    'doublestep_saccade_and_touch_sequence': (394, True, True),
    'delayed_reach_and_saccade': (13, True, True),
    'delayed_read_stim_task': (109, True, False),
    'delayed_reach_stim_task': (109, True, False),
    'null_task': (100, False, False),
    'luminance_reward_selection': (11, False, True),
    'calibrate_eye_reach': (2, True, True),
    'calibrate_eye_saccade': (3, True, True),
}


def _set_task_code(Events, itrial, trial_type):
    code_reach_sac = _TASK_CODES.get(trial_type, (np.nan, False, False))
    Events['TaskCode'][itrial] = code_reach_sac[0]
    Events['Reach'][itrial] = 1 if code_reach_sac[1] else 0
    Events['Saccade'][itrial] = 1 if code_reach_sac[2] else 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TargetConfigs key renaming (MATLAB field names ≤ 31 chars)
# ---------------------------------------------------------------------------

_TARGET_KEY_RENAMES = {
    'auditory_spatial_offset_around_fixation': 'aud_offset_fixation',
    'auditory_spatial_offset': 'aud_offset',
}

def _rename_target_configs(targets):
    """Rename target dict keys that exceed MATLAB's 31-char field name limit."""
    if not targets:
        return targets
    out = []
    for t in targets:
        if isinstance(t, dict) and any(k in _TARGET_KEY_RENAMES for k in t):
            t = {_TARGET_KEY_RENAMES.get(k, k): v for k, v in t.items()}
        out.append(t)
    return out


# Behavioural results parsing
# ---------------------------------------------------------------------------

def _parse_behav(br):
    if isinstance(br, dict):
        return br
    elif isinstance(br, str):
        try:
            return json.loads(br)
        except Exception:
            try:
                return yaml.safe_load(br) or {}
            except Exception:
                return {}
    return {}


def _populate_behav_results(Events, itrial, trial_type, br_raw,
                             target_values, tc_struct):
    br = _parse_behav(br_raw)
    if not br:
        return

    if trial_type == 'luminance_reward_selection':
        _populate_lrs(Events, itrial, br, target_values)
        return

    # ---- Target / Target2 ---------------------------------------------------
    if not _is_simple_task(trial_type):
        pti = br.get('presented_target_ids')
        if pti is not None:
            if isinstance(pti, (list, np.ndarray)):
                Events['Target'][itrial] = int(pti[0]) + 1
            else:
                Events['Target'][itrial] = int(pti) + 1
    else:
        if trial_type == 'simple_touch_and_look_task':
            _tid = br.get('touch_target_id', np.nan)
            _gid = br.get('gaze_target_id', np.nan)
            if isinstance(_tid, (list, np.ndarray)):
                _tid = _tid[0]
            if isinstance(_gid, (list, np.ndarray)):
                _gid = _gid[0]
            Events['Target'][itrial] = _tid
            Events['Target2'][itrial] = _gid
        else:
            pti = br.get('presented_target_ids', np.nan)
            if isinstance(pti, (list, np.ndarray)):
                Events['Target'][itrial] = int(pti[0])
            elif pti is not None:
                Events['Target'][itrial] = int(pti)

    if not _is_simple_task(trial_type):
        pt2 = br.get('presented_targ2_id')
        if pt2 is not None:
            Events['Target2'][itrial] = int(pt2) + 1

    # ---- ChosenTarget / location --------------------------------------------
    sel = br.get('selected_target_id')
    if sel is not None:
        ct = int(sel) + 1
        Events['ChosenTarget'][itrial] = ct
        if ct >= 1 and ct <= len(target_values):
            tv = target_values[ct - 1]
            _set_target_location(Events, itrial, tv)
    elif Events['Success'][itrial] == 1:
        pti = br.get('presented_target_ids')
        if pti is not None:
            ct = (int(pti[0]) + 1) if isinstance(pti, (list, np.ndarray)) else (int(pti) + 1)
            Events['ChosenTarget'][itrial] = ct
            if ct >= 1 and ct <= len(target_values):
                tv = target_values[ct - 1]
                _set_target_location(Events, itrial, tv)

    # ---- ChosenTarget2 / location -------------------------------------------
    if Events['Success'][itrial] == 1:
        pt2 = br.get('presented_targ2_id')
        if pt2 is not None:
            ct2 = int(Events['Target2'][itrial])
            if ct2 >= 1 and ct2 <= len(target_values):
                tv2 = target_values[ct2 - 1]
                rad2 = tv2.get('radius', np.nan)
                ang2 = tv2.get('angle', np.nan)
                Events['Target2Angle'][itrial] = ang2
                Events['Target2Location'][itrial, :] = [
                    rad2 * np.cos(np.radians(ang2)),
                    rad2 * np.sin(np.radians(ang2))
                ]
                Events['ChosenTarget2'][itrial] = ct2 - 2  # MATLAB convention

    # ---- RewardReceived (LRS) -----------------------------------------------
    rwd = br.get('reward_given')
    if rwd is not None:
        Events['RewardReceived'][itrial] = float(rwd)


def _set_target_location(Events, itrial, tv):
    rad = tv.get('radius', np.nan)
    ang = tv.get('angle', np.nan)
    if not (np.isnan(float(rad)) or np.isnan(float(ang))):
        Events['TargetAngle'][itrial] = float(ang)
        Events['TargetLocation'][itrial, :] = [
            float(rad) * np.cos(np.radians(float(ang))),
            float(rad) * np.sin(np.radians(float(ang)))
        ]


def _populate_lrs(Events, itrial, br, target_values):
    """Populate luminance_reward_selection fields."""
    uv = Events['UsedValues'][itrial] if isinstance(Events['UsedValues'], list) else {}
    if not uv or not br:
        return

    Events['targ1'][itrial] = uv.get('targ1_name', np.nan)
    Events['targ2'][itrial] = uv.get('targ2_name', np.nan)
    for ch in ('lum', 'height', 'width', 'angle'):
        Events[f'targ1_{ch}'][itrial] = uv.get(f'targ1_luminance' if ch == 'lum' else f'targ1_{ch}', np.nan)
        Events[f'targ2_{ch}'][itrial] = uv.get(f'targ2_luminance' if ch == 'lum' else f'targ2_{ch}', np.nan)

    rewards = br.get('rewards', [np.nan, np.nan])
    Events['targ1_reward'][itrial] = rewards[0] if len(rewards) > 0 else np.nan
    Events['targ2_reward'][itrial] = rewards[1] if len(rewards) > 1 else np.nan

    for targ in ('targ1', 'targ2'):
        rad = uv.get(f'{targ}_radius', np.nan)
        ang = uv.get(f'{targ}_angle', np.nan)
        Events[f'{targ}_location'][itrial, :] = [
            float(rad) * np.cos(np.radians(float(ang))),
            float(rad) * np.sin(np.radians(float(ang)))
        ]

    ct = br.get('chosen_target', np.nan)
    Events['ChosenTarget'][itrial] = float(ct) if ct is not None else np.nan
    rwd = br.get('reward_given', np.nan)
    Events['RewardReceived'][itrial] = float(rwd) if rwd is not None else np.nan


# ---------------------------------------------------------------------------
# TargSeq builder
# ---------------------------------------------------------------------------

def _extract_targ_colors(target_values, ids):
    colors = []
    for idx in (ids if hasattr(ids, '__iter__') else [ids]):
        i = int(idx) - 1
        if 0 <= i < len(target_values):
            colors.append(target_values[i].get('color', None))
        else:
            colors.append(None)
    return colors


def _build_targ_seq(Events, itrial, trial_events, trial_type, test_targets, target_values):
    tseq = []
    has_targs_on = any(e.lower().strip() == 'targs_on' for e in trial_events)

    # Targets at start_on
    if not has_targs_on:
        start_ids = list(range(1, len(target_values) + 1))
    elif isinstance(test_targets, list) and test_targets:
        start_ids = [i + 1 for i, t in enumerate(test_targets)
                     if isinstance(t, dict) and t.get('is_fixation', False)]
    else:
        start_ids = []

    if start_ids:
        tseq.append({
            'targ_ids': start_ids,
            'event': 'start_on',
            'time_ms': float(Events['StartOn'][itrial]),
            'colors': _extract_targ_colors(target_values, start_ids),
        })

    # Peripheral targets at targs_on
    if has_targs_on and not np.isnan(Events['TargsOn'][itrial]):
        if isinstance(test_targets, list) and test_targets:
            peri_ids = [i + 1 for i, t in enumerate(test_targets)
                        if isinstance(t, dict) and not t.get('is_fixation', False)]
        else:
            peri_ids = list(range(1, len(target_values) + 1))

        targ2_idx = int(Events['Target2'][itrial]) if not np.isnan(Events['Target2'][itrial]) else None
        if targ2_idx is not None:
            peri_ids = [p for p in peri_ids if p != targ2_idx]

        if peri_ids:
            tseq.append({
                'targ_ids': peri_ids,
                'event': 'targs_on',
                'time_ms': float(Events['TargsOn'][itrial]),
                'colors': _extract_targ_colors(target_values, peri_ids),
            })

    # Second peripheral target at targ2_on
    if not np.isnan(Events['Targ2On'][itrial]) and not np.isnan(Events['Target2'][itrial]):
        t2 = int(Events['Target2'][itrial])
        tseq.append({
            'targ_ids': [t2],
            'event': 'targ2_on',
            'time_ms': float(Events['Targ2On'][itrial]),
            'colors': _extract_targ_colors(target_values, [t2]),
        })

    return tseq


# ---------------------------------------------------------------------------
# Display events loader
# ---------------------------------------------------------------------------

def _parse_display_events(disp_data):
    """Convert displayEvents from loadmat to list of dicts.

    squeeze_me=True collapses single-element cell arrays:
      - events becomes a plain str instead of ['start_on']
      - eventTimes_in_ms becomes a float instead of [1767.0]
    Normalise both so downstream code can always iterate over lists/arrays.
    """
    raw = disp_data.get('displayEvents', [])
    if not hasattr(raw, '__len__') or len(raw) == 0:
        return []

    result = []
    for de in raw:
        if isinstance(de, dict):
            # Normalise events: str → [str]
            evs = de.get('events', [])
            if isinstance(evs, str):
                evs = [evs]
            elif isinstance(evs, np.ndarray):
                evs = evs.tolist()
            # Normalise eventTimes_in_ms: scalar → 1-d array
            ets = de.get('eventTimes_in_ms', [])
            if isinstance(ets, (float, int, np.floating, np.integer)):
                ets = np.array([float(ets)])
            else:
                ets = np.asarray(ets, dtype=float).ravel()
            result.append({**de, 'events': evs, 'eventTimes_in_ms': ets})
        else:
            result.append({})
    return result


# ---------------------------------------------------------------------------
# Events → savemat format
# ---------------------------------------------------------------------------

def _events_to_mat(Events):
    """
    Convert Events dict to a MATLAB-compatible struct dict.
    Cell/object fields become numpy object arrays.
    """
    out = {}
    for key, val in Events.items():
        if isinstance(val, list):
            n = len(val)
            arr = np.empty(n, dtype=object)
            for i, v in enumerate(val):
                if v is None:
                    arr[i] = np.nan
                elif isinstance(v, dict):
                    arr[i] = json.dumps(v)
                elif isinstance(v, list):
                    if len(v) == 0:
                        arr[i] = np.array([], dtype=object)
                    else:
                        inner = np.empty(len(v), dtype=object)
                        for j, item in enumerate(v):
                            if isinstance(item, dict):
                                inner[j] = json.dumps(item)
                            else:
                                inner[j] = item
                        arr[i] = inner
                else:
                    arr[i] = v
            out[key] = arr
        else:
            out[key] = val
    return out
