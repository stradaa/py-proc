"""
proc_saccade.py — Port of procSaccade.m.

Detects saccades using eye velocity (Savitzky-Golay derivative) and
findpeaks logic.  Updates and saves recNNN.Events.mat.
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter, find_peaks

from .helpers import rec_paths, loadlpeye


def proc_saccade(day, rec, monkeydir, out_suffix=''):
    """
    Port of procSaccade(day, rec).

    For each trial with Saccade==1:
      - Loads eye velocity from .lp.seye.dat
      - Detects first saccade (around Go ± window) and second saccade (Targ2On window)
      - Writes SaccStart/Stop, Sacc2Start/Stop, EyeTargetLocation into Events
    """
    _, rec_dir, rec_pref, _ = rec_paths(day, rec, monkeydir)
    py_pref = rec_pref + out_suffix
    print(f'\nproc_saccade: day={day} rec={rec}')

    events_file = f'{py_pref}.Events.mat'
    if not os.path.exists(events_file):
        print(f'  Events.mat not found: {events_file}')
        return

    ev_data = loadmat(events_file, simplify_cells=True)
    Events = ev_data['Events']

    sacc_trials = np.where(np.asarray(Events['Saccade'], dtype=float) == 1)[0]
    if len(sacc_trials) == 0:
        print('  No saccade trials, skipping')
        return

    print(f'  Processing {len(sacc_trials)} saccade trials')

    for itrial in sacc_trials:
        go_t = float(Events['Go'][itrial])
        if np.isnan(go_t):
            continue

        task_type = _get_task_type(Events, itrial)

        # simple_touch_task: null out saccade fields and skip
        if task_type == 'simple_touch_task':
            Events['SaccStart'][itrial] = np.nan
            Events['SaccStop'][itrial] = np.nan
            Events['Sacc2Start'][itrial] = np.nan
            Events['Sacc2Stop'][itrial] = np.nan
            continue

        # ---- First saccade (Go-centered) ------------------------------------
        targaq_t = float(Events['TargAq'][itrial])
        if np.isnan(targaq_t):
            targaqtime = 0.0
        else:
            targaqtime = targaq_t - go_t
        if targaqtime < 0:
            targaqtime = abs(targaqtime) + 110.0

        E = loadlpeye(py_pref, Events, itrial, 'Go', [-70, targaqtime + 100])
        start_ind, end_ind, max_vel = _detect_saccade(E)

        if max_vel < 70:
            start_ind = np.nan
            end_ind = np.nan

        Events['SaccStart'][itrial] = (_add_if_not_nan(start_ind, go_t - 70)
                                       if not np.isnan(start_ind) else np.nan)
        Events['SaccStop'][itrial] = (_add_if_not_nan(end_ind, go_t - 70)
                                      if not np.isnan(end_ind) else np.nan)

        # Eye position at saccade stop
        if not np.isnan(end_ind):
            eye_loc_ind = min(E.shape[1] - 1, max(0, round(end_ind)))
            Events['EyeTargetLocation'][itrial, 0] = float(E[0, eye_loc_ind])
            Events['EyeTargetLocation'][itrial, 1] = float(E[1, eye_loc_ind])
        else:
            Events['EyeTargetLocation'][itrial, :] = np.nan

        # ---- Second saccade (Targ2On-centered) ------------------------------
        targ2aq_t = float(Events['Targ2Aq'][itrial])
        targ2on_t = float(Events['Targ2On'][itrial])

        if not np.isnan(targ2aq_t) and not np.isnan(targ2on_t):
            targ2aqtime = targ2aq_t - targ2on_t
            if targ2aqtime > 0:
                SaccStart_ref = float(Events['SaccStart'][itrial])
                if np.isnan(SaccStart_ref):
                    SaccStart_ref = -50.0

                Sacc2Start = SaccStart_ref
                t_offset = 0.0
                used_t_offset = 0.0
                max_iter = 20

                for _ in range(max_iter):
                    E2 = loadlpeye(py_pref, Events, itrial, 'Targ2On',
                                   [1 + t_offset, targ2aqtime + 100 + t_offset])
                    E2 = np.where(np.isnan(E2), 0.0, E2)

                    s2, e2, mv2 = _detect_saccade(E2)
                    if mv2 < 70:
                        s2 = np.nan
                        e2 = np.nan

                    Sacc2Start = (_add_if_not_nan(s2, targ2on_t + t_offset)
                                  if not np.isnan(s2) else SaccStart_ref + 51.0)
                    if np.isnan(s2):
                        used_t_offset = t_offset
                        break

                    if (Sacc2Start - SaccStart_ref) >= 50:
                        used_t_offset = t_offset
                        break

                    used_t_offset = t_offset
                    t_offset += 10.0

                Events['Sacc2Start'][itrial] = (
                    _add_if_not_nan(s2, targ2on_t + used_t_offset) if not np.isnan(s2) else np.nan)
                Events['Sacc2Stop'][itrial] = (
                    _add_if_not_nan(e2, targ2on_t + used_t_offset) if not np.isnan(e2) else np.nan)

                if not np.isnan(e2) and E2.shape[1] > 0:
                    eye_loc_ind2 = min(E2.shape[1] - 1, max(0, round(e2)))
                    Events['EyeTarg2Location'][itrial, 0] = float(E2[0, eye_loc_ind2])
                    Events['EyeTarg2Location'][itrial, 1] = float(E2[1, eye_loc_ind2])
                else:
                    Events['EyeTarg2Location'][itrial, :] = np.nan

    savemat(events_file, {'Events': Events})
    print(f'  Saved {events_file}')


# ---------------------------------------------------------------------------
# Saccade detection helper
# ---------------------------------------------------------------------------

def _calc_eye_velocity(E):
    """
    Compute eye speed (deg/s) from 2×N eye position array using
    Savitzky-Golay derivative filter (window=51, poly=5, deriv=1, delta=1ms).

    Mirrors calcEyeVelocity_v2.m: after computing velocity, the first and last
    N samples are replaced with the nearest valid interior value, suppressing
    edge artifacts that would otherwise cause spurious saccade detections.
    """
    if E.shape[1] < 52:
        return np.zeros(E.shape[1])

    vel_H = savgol_filter(E[0], window_length=51, polyorder=5, deriv=1, delta=1e-3)
    vel_V = savgol_filter(E[1], window_length=51, polyorder=5, deriv=1, delta=1e-3)
    vel = np.sqrt(vel_H ** 2 + vel_V ** 2)

    # Flatten edge samples to suppress SG filter artifacts (matches MATLAB)
    N = 51
    vel[:N] = vel[N]
    vel[len(vel) - N:] = vel[len(vel) - N - 1]
    return vel


def _detect_saccade(E):
    """
    Find the largest velocity peak and its start/stop indices.

    Returns (start_ind, end_ind, max_vel) — indices are into E (0-based).
    Indices are floats (may be nan).
    """
    if E.shape[1] < 52:
        return np.nan, np.nan, 0.0

    vel = _calc_eye_velocity(E)
    peaks, props = find_peaks(vel)

    if len(peaks) == 0:
        # MATLAB: isempty(pks) → start_ind/end_ind are immediately overwritten
        # with nan by the `if max_vel < 70 || isempty(pks)` block regardless
        # of max_vel.  Return nan here to match that behaviour.
        max_vel = float(np.max(vel))
        return np.nan, np.nan, max_vel

    max_peak_idx = int(np.argmax(vel[peaks]))
    max_vel = float(vel[peaks[max_peak_idx]])
    sacc_loc = int(peaks[max_peak_idx])
    vel_thresh = 0.1

    # Start: go backward from peak until vel < 10% of peak
    vel_before = vel[:sacc_loc + 1][::-1]
    back = np.where(vel_before < max_vel * vel_thresh)[0]
    if len(back) > 0:
        start_ind = float(sacc_loc - back[0])
    else:
        # Fallback: half-width from peak
        widths = _peak_half_width(vel, sacc_loc)
        start_ind = float(sacc_loc - widths // 2)

    # Stop: go forward from peak until vel < 10% of peak.
    # +1 matches MATLAB: vel_after = vel(sacc_loc:end) is 1-indexed so
    # find() returns k where vel(sacc_loc+k-1)<thresh → end_ind = sacc_loc+k = E+1.
    vel_after = vel[sacc_loc:]
    fwd = np.where(vel_after < max_vel * vel_thresh)[0]
    if len(fwd) > 0:
        end_ind = float(sacc_loc + fwd[0] + 1)
    else:
        widths = _peak_half_width(vel, sacc_loc)
        end_ind = float(sacc_loc + widths // 2 + 1)

    return start_ind, end_ind, max_vel


def _peak_half_width(vel, sacc_loc):
    """Rough half-width estimate for a peak at sacc_loc."""
    half_max = vel[sacc_loc] / 2.0
    # search backward
    left = 0
    for i in range(sacc_loc, -1, -1):
        if vel[i] < half_max:
            left = sacc_loc - i
            break
    # search forward
    right = 0
    for i in range(sacc_loc, len(vel)):
        if vel[i] < half_max:
            right = i - sacc_loc
            break
    return max(left + right, 1)


def _get_task_type(Events, itrial):
    tt = Events.get('PyTaskType', [])
    if hasattr(tt, '__len__') and len(tt) > itrial:
        return str(tt[itrial])
    return ''


def _add_if_not_nan(val, offset):
    if np.isnan(val):
        return np.nan
    return float(val) + float(offset)
