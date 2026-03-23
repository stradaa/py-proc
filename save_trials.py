"""
save_trials.py — Build Trials.mat and AllTrials.mat from per-recording Events.mat files.

Mirrors the MATLAB schema:
  Trials.mat    — successful trials only  (dbdatabasePyTask.m)
  AllTrials.mat — all trials              (dbAlldatabasePyTask.m)

StartOn is saved as the absolute timestamp (ms).
All other event timestamps are relative to StartOn (i.e. t - StartOn).
"""

import os
import json
import numpy as np
from scipy.io import loadmat, savemat

from .helpers import rec_paths, get_recs


def save_trials(day, monkeydir, out_suffix=''):
    """
    Aggregate Events.mat files for all recs in `day` into Trials.mat / AllTrials.mat.

    Saves to:
      monkeydir/DAY/mat/Trials{out_suffix}.mat     — successful trials only
      monkeydir/DAY/mat/AllTrials{out_suffix}.mat  — all trials
    """
    day_dir = os.path.join(monkeydir, day)
    recs = get_recs(day_dir)
    if not recs:
        print(f'save_trials: no recs found for day {day}')
        return

    all_trials = []

    for rec in recs:
        _, rec_dir, rec_pref, _ = rec_paths(day, rec, monkeydir)
        py_pref = rec_pref + out_suffix
        events_file = f'{py_pref}.Events.mat'
        if not os.path.exists(events_file):
            print(f'  save_trials: {events_file} not found, skipping')
            continue

        ev_data = loadmat(events_file, simplify_cells=True)
        Events = ev_data.get('Events', {})
        if not Events:
            continue

        ntrials = _n_trials(Events)
        for itrial in range(ntrials):
            t = _extract_trial(Events, itrial, day, rec)
            all_trials.append(t)

    if not all_trials:
        print(f'save_trials: no valid trials for day {day}')
        return

    mat_dir = os.path.join(day_dir, 'mat')
    os.makedirs(mat_dir, exist_ok=True)

    # AllTrials — every trial
    AllTrials = _concat_trials(all_trials)
    all_file = os.path.join(mat_dir, f'AllTrials{out_suffix}.mat')
    savemat(all_file, {'AllTrials': AllTrials})
    print(f'save_trials: saved {all_file}  ({len(all_trials)} trials)')

    # Trials — successful trials only
    success_trials = [t for t in all_trials if t.get('Success', 0) == 1]
    Trials = _concat_trials(success_trials)
    out_file = os.path.join(mat_dir, f'Trials{out_suffix}.mat')
    savemat(out_file, {'Trials': Trials})
    print(f'save_trials: saved {out_file}  ({len(success_trials)} successful trials)')


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------

_TS_FIELDS = [
    'StartAq', 'TargsOn', 'Go', 'TargAq', 'Targ2On', 'Targ2Aq',
    'SaccStart', 'SaccStop', 'Sacc2Start', 'Sacc2Stop',
    'ReachStart', 'ReachStop', 'Reach2Start', 'Reach2Stop',
    'TargetOff', 'End', 'Pulse_start', 'Pulse_end',
    'disStartOn', 'disTargsOn', 'disGo', 'disTarg2On',
    'StartGazeOn', 'disStartGazeOn',
    'EyeTargetLocation', 'HandTargetLocation', 'EyeTarg2Location',
    'TargetLocation', 'Target2Location',
    'targ1_location', 'targ2_location',
]


def _scalar(arr, i, default=np.nan):
    """Safe scalar extraction from array at index i."""
    try:
        v = arr[i]
        return float(v) if not hasattr(v, '__len__') else v
    except (IndexError, TypeError):
        return default


def _row(arr, i, ncol=2):
    """Safe 1×ncol row extraction."""
    try:
        row = arr[i]
        if hasattr(row, '__len__'):
            return np.asarray(row, dtype=float).ravel()[:ncol]
        return np.full(ncol, np.nan)
    except (IndexError, TypeError):
        return np.full(ncol, np.nan)


def _str(arr, i):
    try:
        v = arr[i]
        return str(v)
    except (IndexError, TypeError):
        return ''


def _extract_trial(Events, itrial, day, rec):
    """Return a flat dict for one trial."""
    so = _scalar(Events.get('StartOn', [np.nan]), itrial, np.nan)

    def _ts(field):
        v = _scalar(Events.get(field, [np.nan]), itrial, np.nan)
        if np.isnan(so) or np.isnan(v):
            return np.nan
        return v - so

    def _ts_row(field, ncol=2):
        row = _row(Events.get(field, np.full((1, ncol), np.nan)), itrial, ncol)
        if np.isnan(so):
            return row
        return row - so

    t = {
        # Identity
        'Trial': int(_scalar(Events.get('Trial', [itrial + 1]), itrial, itrial + 1)),
        'Day': int(day),
        'Rec': int(rec),
        'Fs': 20000.0,  # stub (electrophysiology fs)

        # StartOn — absolute timestamp (ms), matches MATLAB dbdatabasePyTask.m
        'StartOn': float(so) if not np.isnan(so) else np.nan,

        # Relative timestamps
        'StartAq': _ts('StartAq'),
        'End': _ts('End'),
        'TargsOn': _ts('TargsOn'),
        'Go': _ts('Go'),
        'TargAq': _ts('TargAq'),
        'Targ2On': _ts('Targ2On'),
        'Targ2Aq': _ts('Targ2Aq'),
        'TargetOff': _ts('TargetOff'),
        'SaccStart': _ts('SaccStart'),
        'SaccStop': _ts('SaccStop'),
        'Sacc2Start': _ts('Sacc2Start'),
        'Sacc2Stop': _ts('Sacc2Stop'),
        'ReachStart': _ts('ReachStart'),
        'ReachStop': _ts('ReachStop'),
        'Reach2Start': _ts('Reach2Start'),
        'Reach2Stop': _ts('Reach2Stop'),
        'StimStart': _scalar(Events.get('StimStart', [np.nan]), itrial, np.nan),
        'Pulse_start': _ts('Pulse_start'),
        'Pulse_end': _ts('Pulse_end'),
        'disStartOn': _ts('disStartOn'),
        'disTargsOn': _ts('disTargsOn'),
        'disGo': _ts('disGo'),
        'disTarg2On': _ts('disTarg2On'),
        'StartGazeOn': _ts('StartGazeOn'),
        'disStartGazeOn': _ts('disStartGazeOn'),

        # 2-D locations (subtract so from each element)
        'EyeTargetLocation': _ts_row('EyeTargetLocation'),
        'HandTargetLocation': _ts_row('HandTargetLocation'),
        'EyeTarg2Location': _ts_row('EyeTarg2Location'),
        'TargetLocation': _row(Events.get('TargetLocation', np.full((1, 2), np.nan)), itrial),
        'Target2Location': _row(Events.get('Target2Location', np.full((1, 2), np.nan)), itrial),

        # Flags / codes
        'Success': int(_scalar(Events.get('Success', [0]), itrial, 0)),
        'Reach': int(_scalar(Events.get('Reach', [0]), itrial, 0)),
        'Saccade': int(_scalar(Events.get('Saccade', [0]), itrial, 0)),
        'Choice': int(_scalar(Events.get('Choice', [0]), itrial, 0)),
        'TaskCode': float(_scalar(Events.get('TaskCode', [np.nan]), itrial, np.nan)),
        'Target': float(_scalar(Events.get('Target', [np.nan]), itrial, np.nan)),
        'Target1': float(_scalar(Events.get('Target1', [np.nan]), itrial, np.nan)),
        'Target2': float(_scalar(Events.get('Target2', [np.nan]), itrial, np.nan)),
        'ChosenTarget': float(_scalar(Events.get('ChosenTarget', [np.nan]), itrial, np.nan)),
        'ChosenTarget2': float(_scalar(Events.get('ChosenTarget2', [np.nan]), itrial, np.nan)),
        'TargetAngle': float(_scalar(Events.get('TargetAngle', [np.nan]), itrial, np.nan)),
        'Target2Angle': float(_scalar(Events.get('Target2Angle', [np.nan]), itrial, np.nan)),

        # LRS
        'targ1_lum': float(_scalar(Events.get('targ1_lum', [np.nan]), itrial, np.nan)),
        'targ2_lum': float(_scalar(Events.get('targ2_lum', [np.nan]), itrial, np.nan)),
        'targ1_height': float(_scalar(Events.get('targ1_height', [np.nan]), itrial, np.nan)),
        'targ2_height': float(_scalar(Events.get('targ2_height', [np.nan]), itrial, np.nan)),
        'targ1_width': float(_scalar(Events.get('targ1_width', [np.nan]), itrial, np.nan)),
        'targ2_width': float(_scalar(Events.get('targ2_width', [np.nan]), itrial, np.nan)),
        'targ1_angle': float(_scalar(Events.get('targ1_angle', [np.nan]), itrial, np.nan)),
        'targ2_angle': float(_scalar(Events.get('targ2_angle', [np.nan]), itrial, np.nan)),
        'targ1_reward': float(_scalar(Events.get('targ1_reward', [np.nan]), itrial, np.nan)),
        'targ2_reward': float(_scalar(Events.get('targ2_reward', [np.nan]), itrial, np.nan)),
        'targ1_location': _row(Events.get('targ1_location', np.full((1, 2), np.nan)), itrial),
        'targ2_location': _row(Events.get('targ2_location', np.full((1, 2), np.nan)), itrial),

        # Stim / opto
        'Intan_Cfg': float(_scalar(Events.get('Intan_Cfg', [np.nan]), itrial, np.nan)),
        'Pulse_count': float(_scalar(Events.get('Pulse_count', [np.nan]), itrial, np.nan)),
        'Pulse_freq': float(_scalar(Events.get('Pulse_freq', [np.nan]), itrial, np.nan)),
        'Pulse_width': float(_scalar(Events.get('Pulse_width', [np.nan]), itrial, np.nan)),
        'ReachSize': float(_scalar(Events.get('ReachSize', [np.nan]), itrial, np.nan)),
        'RewardReceived': float(_scalar(Events.get('RewardReceived', [0]), itrial, 0)),

        # Electrophysiology stubs
        'MT1': np.nan, 'MT2': np.nan, 'Ch': np.nan, 'Gain': np.nan,
        'Iso': np.nan, 'Depth': np.nan, 'Joystick': np.nan, 'HandCode': np.nan,
        'StartHand': np.nan, 'StartEye': np.nan,
        'RewardDur': np.nan, 'RewardMag': np.nan,

        # PyTask extras (cell fields)
        'PyTaskType': _str(Events.get('PyTaskType', []), itrial),
        'TargetValues': _cell_str(Events.get('TargetValues', []), itrial),
        'TargetConfigs': _cell_str(Events.get('TargetConfigs', []), itrial),
        'UsedValues': _cell_str(Events.get('UsedValues', []), itrial),
        'TargSeq': _cell_str(Events.get('TargSeq', []), itrial),
    }
    return t


def _cell_str(arr, i):
    """Extract cell element as JSON string or empty string."""
    try:
        v = arr[i]
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return json.dumps(v)
        elif isinstance(v, list):
            return json.dumps(v)
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            return ''
        return str(v)
    except (IndexError, TypeError):
        return ''


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def _n_trials(Events):
    """Return number of trials from a field with known length."""
    for key in ('Trial', 'Success', 'Reach'):
        val = Events.get(key, None)
        if val is not None and hasattr(val, '__len__'):
            return len(val)
    return 0


def _concat_trials(trial_list):
    """Convert list of per-trial dicts → dict of numpy arrays."""
    if not trial_list:
        return {}

    keys = trial_list[0].keys()
    out = {}
    for k in keys:
        vals = [t[k] for t in trial_list]
        first = vals[0]
        if isinstance(first, np.ndarray):
            out[k] = np.vstack([v.reshape(1, -1) for v in vals])
        elif isinstance(first, str):
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
            out[k] = arr
        else:
            try:
                out[k] = np.array(vals, dtype=float)
            except (ValueError, TypeError):
                arr = np.empty(len(vals), dtype=object)
                for i, v in enumerate(vals):
                    arr[i] = v
                out[k] = arr
    return out
