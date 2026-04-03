"""
proc_reach.py — Port of procReach.m.

Detects reach start/stop from .scaledhnd.dat derivative thresholding.
Updates and saves recNNN.Events.mat.
"""

import os
import numpy as np
from scipy.io import loadmat, savemat

from .helpers import rec_paths, loadroshnd


def proc_reach(day, rec, monkeydir, out_suffix=''):
    """
    Port of procReach(day, rec).
    """
    _, rec_dir, rec_pref, _ = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix
    print(f'\nproc_reach: day={day} rec={rec}')

    events_file = f'{py_pref}.Events.mat'
    if not os.path.exists(events_file):
        print(f'  Events.mat not found: {events_file}')
        return

    ev_data = loadmat(events_file, simplify_cells=True)
    Events = ev_data['Events']

    reach_trials = np.where(np.asarray(Events['Reach'], dtype=float) == 1)[0]
    if len(reach_trials) == 0:
        print('  No reach trials, skipping')
        return

    print(f'  Processing {len(reach_trials)} reach trials')

    for itrial in reach_trials:
        go_t = float(Events['Go'][itrial])
        if np.isnan(go_t):
            continue

        task_type = _get_task_type(Events, itrial)

        # ---- Blink timeout from used values ----------------------------------
        blink = _get_reach_blink(Events, itrial)

        # ---- simple_touch_task branch ----------------------------------------
        if task_type == 'simple_touch_task':
            H = loadroshnd(py_pref, Events, itrial, 'StartAq', [-3000, 200])
            if H.shape[1] == 0:
                continue
            A = np.diff(H[0])

            T1 = _find_first(A, '<', -20)
            T2 = _find_first(A, '>', 20)

            if T1 is None:
                Events['ReachStart'][itrial] = np.nan
                Events['ReachStop'][itrial] = np.nan
            elif T2 is None:
                # T1 found (1-indexed in MATLAB), convert: offset from StartAq
                Events['ReachStart'][itrial] = float(Events['StartAq'][itrial]) - (T1 + 1)
                Events['ReachStop'][itrial] = np.nan
            else:
                Events['ReachStart'][itrial] = float(Events['StartAq'][itrial]) - (T1 + 1)
                Events['ReachStop'][itrial] = float(Events['StartAq'][itrial]) - (T2 + 1)
            continue

        # ---- All other reach tasks -------------------------------------------
        targaq_t = float(Events['TargAq'][itrial])
        if np.isnan(targaq_t):
            targaqtime = 0.0
        else:
            targaqtime = targaq_t - go_t

        if targaqtime < 0:
            Events['ReachStart'][itrial] = np.nan
            Events['ReachStop'][itrial] = np.nan
            continue

        end_t = float(Events['End'][itrial])
        if np.isnan(end_t):
            endtime = 1000.0
        else:
            endtime = end_t + 500.0 - go_t

        H = loadroshnd(py_pref, Events, itrial, 'Go', [0, endtime])
        if H.shape[1] == 0:
            continue

        # Detect first reach
        T1, T2, d, Trev, Hextend = _detect_reach(H, rec_pref, Events, itrial)

        T1_found = T1 is not None
        T1_val = T1 if T1_found else 0
        T2_val = T2 if T2 is not None else np.nan
        Trev_val = (Trev if Trev is not None else 0) if T1_found else 0

        ReachStart = T1_val - Trev_val + go_t
        ReachStop = (T2_val - Trev_val + go_t) if T1_found and not np.isnan(T2_val) else np.nan

        Events['ReachStart'][itrial] = float(ReachStart) if T1_found else np.nan
        Events['ReachStop'][itrial] = float(ReachStop) if not np.isnan(ReachStop) else np.nan
        Events['ReachSize'][itrial] = float(d) if d is not None else 0.0

        # Hand target location
        if Hextend is not None:
            _set_hand_target_location(Events, itrial, Hextend, T1_val)

        # ---- Second reach detection ------------------------------------------
        if not np.isnan(Events['ReachStop'][itrial]):
            reach_stop_t = float(Events['ReachStop'][itrial])
            end_t2 = (float(Events['End'][itrial]) + 500.0 - reach_stop_t
                      if not np.isnan(Events['End'][itrial]) else 500.0)

            if end_t2 > 500.0:
                min_reach_int = 50.0
                H2 = loadroshnd(py_pref, Events, itrial, 'ReachStop',
                                [min_reach_int, end_t2])
                if H2.shape[1] > 0:
                    T1b, T2b, db, Trevb, Hextend2 = _detect_reach(H2, rec_pref, Events, itrial)

                    if T1b is not None:
                        Trevb = Trevb if Trevb is not None else 0
                        # +1: T1b/T2b are 0-indexed; MATLAB find() is 1-indexed
                        Rs2 = (T1b + 1) + reach_stop_t + min_reach_int
                        Rp2 = ((T2b + 1) + reach_stop_t + min_reach_int
                               if T2b is not None and not np.isnan(T2b) else np.nan)
                        Events['Reach2Start'][itrial] = float(Rs2)
                        Events['Reach2Stop'][itrial] = float(Rp2) if not np.isnan(Rp2) else np.nan

    n_trials = len(Events['ReachStart'])
    print(f'  # trials: {n_trials}')
    savemat(events_file, {'Events': Events})
    print(f'  Saved {events_file}')


# ---------------------------------------------------------------------------
# Reach detection logic
# ---------------------------------------------------------------------------

def _detect_reach(H, rec_pref, Events, itrial):
    """
    Port of the inner reach-detection while loop in procReach.m.

    Returns (T1, T2, d, Trev, Hextend).
    T1, T2 are 0-based indices into Hextend.
    d is the reach distance.
    Trev is the extension length prepended from before Go.
    """
    A = np.diff(H[0])

    if H.shape[1] > 0 and H[0, 0] < -20:
        # Hand starts off-screen: prepend from before Go
        Hrev = loadroshnd(rec_pref, Events, itrial, 'Go', [-300, 0])
        Hrev = Hrev[:, ::-1]
        Arev = np.diff(Hrev[0])
        Trev_idx = _find_first(Arev, '>', 20)
        if Trev_idx is not None and Trev_idx < len(Arev):
            Arevrev = -Arev[:Trev_idx + 1][::-1]
            Hrevrev = Hrev[:, :Trev_idx + 1][:, ::-1]
            Aextend = np.concatenate([Arevrev, A])
            Hextend = np.hstack([Hrevrev, H])
            Trev = Trev_idx if Trev_idx is not None else 0
        else:
            Trev = 0
            Hextend = H
            Aextend = A
    else:
        Trev = 0
        Hextend = H
        Aextend = A

    Trev = Trev if Trev is not None else 0

    T1 = _find_first(Aextend, '<', -20)
    T2 = _find_first(Aextend, '>', 20)
    d = 0.0

    if T1 is not None and T2 is not None:
        try:
            d = np.sqrt((Hextend[0, T1] - Hextend[0, T2 + 2]) ** 2
                        + (Hextend[1, T1] - Hextend[1, T2 + 2]) ** 2)
        except IndexError:
            d = 0.0

        # Skip spurious reaches with distance < 0.5 deg
        if T2 < len(Aextend) - 2:
            while d < 0.5 and T1 is not None:
                Aextend[:T2 + 2] = 0
                T1 = _find_first(Aextend, '<', -20)
                T2 = _find_first(Aextend, '>', 20)
                if T1 is None or T2 is None:
                    d = 0.0
                    break
                try:
                    d = np.sqrt((Hextend[0, T1 - 1] - Hextend[0, T2 + 2]) ** 2
                                + (Hextend[1, T1 - 1] - Hextend[1, T2 + 2]) ** 2)
                except IndexError:
                    d = 0.0
                    break

    if T1 is None:
        return None, None, 0.0, Trev, Hextend
    return T1, T2, d, Trev, Hextend


def _find_first(arr, op, threshold):
    """Return 0-based index of first element satisfying arr[i] op threshold, or None."""
    if op == '<':
        idx = np.where(arr < threshold)[0]
    elif op == '>':
        idx = np.where(arr > threshold)[0]
    else:
        return None
    return int(idx[0]) if len(idx) > 0 else None


def _get_task_type(Events, itrial):
    tt = Events.get('PyTaskType', [])
    if hasattr(tt, '__len__') and len(tt) > itrial:
        return str(tt[itrial])
    return ''


def _get_reach_blink(Events, itrial):
    """Extract blink/hold timeout from UsedValues (ms)."""
    uv = Events.get('UsedValues', [])
    if not hasattr(uv, '__len__') or len(uv) <= itrial:
        return 0.0
    entry = uv[itrial]
    if isinstance(entry, str):
        import json, yaml
        try:
            entry = json.loads(entry)
        except Exception:
            try:
                entry = yaml.safe_load(entry)
            except Exception:
                return 0.0
    if not isinstance(entry, dict):
        return 0.0
    for key in ('blink', 'hand_blink', 'blink_timeout'):
        val = entry.get(key)
        if val is not None:
            return float(val) * 1000.0
    return 0.0


def _set_hand_target_location(Events, itrial, Hextend, T1):
    """Set HandTargetLocation from the position just before/at reach start."""
    if T1 == 0:
        idx = min(1, Hextend.shape[1] - 1)
        Events['HandTargetLocation'][itrial, 0] = float(Hextend[0, idx])
        Events['HandTargetLocation'][itrial, 1] = float(Hextend[1, idx])
    elif T1 < Hextend.shape[1] and Hextend[0, T1] == -100:
        idx = max(0, T1 - 1)
        Events['HandTargetLocation'][itrial, 0] = float(Hextend[0, idx])
        Events['HandTargetLocation'][itrial, 1] = float(Hextend[1, idx])
    elif T1 < Hextend.shape[1]:
        Events['HandTargetLocation'][itrial, 0] = float(Hextend[0, T1])
        Events['HandTargetLocation'][itrial, 1] = float(Hextend[1, T1])
