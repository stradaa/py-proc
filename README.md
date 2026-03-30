# proc_pipeline — Python Port of AlexRig Behavioral Processing Pipeline

Python port of the MATLAB v7 pipeline (`/vol/brains/raid/analyze/proc/PyTaskCtrl/v7/`).
Produces identical output file formats so existing MATLAB analysis scripts continue to work.

Written by Indie Garwood using [Claude Code](https://claude.ai/claude-code) (Anthropic).
Created: 2026-03-04

Forked 2026-03-24 and modified for joystick behavior support by Alex Estrada.
---

## Pipeline Overview

Orchestrated by `../p260303_procLoop.py`. Run order:

```
PREREQUISITE (must be run first, once per day):
  procThalamus_indie.py                      ← extracts raw signals from behave bag files

Pass 1 (noDisplay):
  proc_events(day, rec, use_display=False)   ← per rec

detect_display_states(day)                   ← per day

Pass 2 (full):
  proc_events(day, rec, use_display=True)    ← per rec
  proc_eye(day, rec)
  proc_saccade(day, rec)
  proc_hand(day, rec)
  proc_reach(day, rec)

save_trials(day)                             ← per day
```

---

## Module Index

| File | MATLAB source | Output |
|---|---|---|
| `helpers.py` | Multiple utils | — (shared library) |
| `pre_proc.py` | `preProcPyTask.m` | `recNNN.ev.mat` |
| `proc_events.py` | `procEvents.m` | `recNNN.Events.mat`, `recNNN.w_alignment.mat` |
| `proc_display.py` | `procDisplay_new.m` | `recNNN.displayEvents.mat` |
| `detect_display_states.py` | `m260216_detect_display_states.m` | `recNNN.display_corrected.dat`, diagnostic PNG |
| `proc_eye.py` | `procEye.m` | `recNNN.lp.seye.dat` |
| `proc_saccade.py` | `procSaccade.m` | updates `recNNN.Events.mat` |
| `proc_hand.py` | `procHand.m` | `recNNN.hnd.dat`, `recNNN.scaledhnd.dat` |
| `proc_reach.py` | `procReach.m` | updates `recNNN.Events.mat` |
| `save_trials.py` | `dbdatabasePyTask.m` + `dbAlldatabasePyTask.m` | `DAY/mat/Trials.mat` (successful only), `DAY/mat/AllTrials.mat` (all trials) |

Current AlexRig extension: `procThalamus_indie.py` also writes `bag/mat/joystick.mat` when a `Joystick` node is present.

---

## Key Technical Decisions

### Clock alignment (AlexRig single-clock shortcut)
AlexRig is a single-machine rig: `remote_time ≈ local_time`. When `max(|ros - local|) < 1e-9`,
`simhead_weights = [0, 1]` is used directly (no regression needed). Stored in `recNNN.w_alignment.mat`
as `w_drift_ros = [offset, slope]`, applied as `t_rec = w[0] + w[1] * t_ros`.

### Binary .dat file format
All `.dat` files are flat `float32`, 2-channel interleaved, column-major (matching MATLAB `fwrite`):
```
[ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
```
Read in Python as: `np.fromfile(f, dtype=np.float32).reshape(-1, 2).T` → shape `(2, N)`.

### Eye processing (proc_eye)
- Source: `oculomatic_eye.mat` (from `bag/mat/`)
- `1e6` sentinel → NaN masking
- Quadrant-gain gaze transform: `get_gaze_transform(bag_dir)` → `quadrant_gains` (2×4 array)
- pchip interpolation to 1 ms grid: `scipy.interpolate.PchipInterpolator`
- NaN gaps filled with `np.interp` before filter (FFT propagates NaN globally; MATLAB conv does not)
- Multitaper LP filter at 50 Hz: ports `mtfilter.m`/`mtfilt.m` — DPSS tapers of length `n=floor(T*Fs)=50`,
  projection matrix builds 100-tap FIR kernel, applied via `np.convolve` and trimmed to match MATLAB's
  `tmp(N/2 : szX(2)+N/2-1)` indexing. MAE vs MATLAB: 0.01–0.02 deg.
- Patch first 50 samples; re-center to original median (robust to outliers)
- Output: `recNNN.lp.seye.dat` (float32, 1000 Hz, degrees)

### Saccade detection (proc_saccade)
- Eye velocity: `savgol_filter(E, window=51, poly=5, deriv=1, delta=1e-3)` → deg/s
- Peak detection: `scipy.signal.find_peaks`
- Start/stop: search back/forward from peak until velocity < 10% of peak
- `max_vel < 70 deg/s` → NaN (not a saccade)
- Second saccade: Targ2On window, 50 ms minimum separation loop

### Hand / reach processing
- `serialhnd.mat` (timestamps + X, Y) from `bag/mat/` — written by `procThalamus_indie.py`
- Clock-aligned, interpolated to 1 ms grid with `interp1d(kind='previous')`
- Touch transform: `get_touch_transform(bag_dir)` → 3×3 homogeneous matrix M
- `notouch_val = [0, 0]` → `-100` in `scaledhnd.dat`
- Reach detection: `diff(H[0]) < -20` (reach start), `> 20` (reach stop)
- Distance filter: skip spurious reaches with `d < 0.5 deg`

### Event matching (display)
`event_match()` in `helpers.py` ports `event_match.m`:
- Cross-correlation with Gaussian kernel (`scipy.signal.fftconvolve`)
- Iteratively removes unmatched display transitions
- `median_latency` estimated from early matched events

### MATLAB → Python mappings
| MATLAB | Python |
|---|---|
| `sgolay` + derivative | `savgol_filter(..., deriv=1, delta=1e-3)` |
| `findpeaks` | `scipy.signal.find_peaks` |
| `pchip` | `scipy.interpolate.PchipInterpolator` |
| `interp1(...,'previous')` | `interp1d(..., kind='previous')` |
| `medfilt1(x, 99)` | `scipy.signal.medfilt(x, 99)` |
| `normpdf` | `scipy.stats.norm.pdf` |
| `conv(...,'same')` | `scipy.signal.fftconvolve(..., mode='same')` |
| `fitlm` | `np.polyfit(x, y, 1)` |
| `ReadYaml` / `jsondecode` | `yaml.safe_load` / `json.loads` |
| `loadmat` / `save` | `scipy.io.loadmat(simplify_cells=True)` / `scipy.io.savemat` |
| `mtfilter([0.05 50], 1000)` | `dpss(n=50, NW=2.5, K=4)` → projection-matrix FIR kernel → `np.convolve` |

---

## What Was Removed vs MATLAB

- All `notbag` fallback paths (AlexRig uses only the `bag/` directory)
- Commented-out dead code blocks in `procSaccade`, `procReach`, `procDisplay_new` (~150–300 lines each)
- `notbag` parallel loading paths in helpers

---

## Validation Results

All 10 recording days validated against MATLAB AllTrials output (2026-03-11).
Comparison metric: fraction of timestamp events with |Python − MATLAB| > 5 ms.

| Day | Matched trials | Events compared | > 5 ms | % | Notes |
|---|---|---|---|---|---|
| 260212 | 1125 | 8240 | 14 | 0.2% | |
| 260213 | 849 | 6922 | 6 | 0.1% | |
| 260216 | 990 | 9213 | 25 | 0.3% | |
| 260217 | 1362 | 7088 | 16 | 0.2% | |
| 260218 | 1389 | 8277 | 14 | 0.2% | |
| 260219 | 956 | 8182 | 29 | 0.4% | |
| 260220 | 1680 | 10710 | 43 | 0.4% | |
| 260226 | 2255 | 15122 | 11 | 0.1% | |
| 260302 | 1006 | 10504 | 22 | 0.2% | |
| **260303** | **1269** | **15709** | **10** | **0.1%** | **New day — not used in port development** |
| **TOTAL** | **11881** | **99967** | **190** | **0.2%** | |

Days 260212–260302 were used during pipeline development. **Day 260303 was processed blind** (not
used in any debugging or tuning) and achieved 0.1% — confirming the pipeline generalises correctly.

All remaining errors are SaccStart/SaccStop outliers (2–6 per rec) from peak disambiguation in
multi-saccade trials. These reflect genuine ambiguity between two valid velocity peaks, not a
pipeline bug.

### Field-level summary (typical values across all days)

| Field | MAE | Notes |
|---|---|---|
| `StartOn`, `Go`, `TargAq`, `End` | 0.01–0.33 ms | Clock precision residual only |
| `SaccStart` / `SaccStop` | median 0.7–1.1 ms | 2–6 outliers/rec from peak disambiguation |
| `ReachStart` / `ReachStop` | MAE 1.3–1.5 ms, max ~3 ms | Within sampling limits; see note below |
| Eye trace (`.lp.seye.dat`) | 0.01–0.02 deg | Over a 150 deg range |
| `scaledhnd.dat` touch positions | < 0.01 deg | Matches MATLAB exactly after transform fix |

### ReachStart timing note

The 1.3–1.5 ms mean error in ReachStart / ReachStop is **within the fundamental precision of the data**
and is not a bug. Sources:

1. **Clock alignment residual (~0.28 ms)**: Python's `proc_events` computes slightly different
   `w_drift_ros` than MATLAB's `procEvents` (independent regression on the same pulse train). This
   shifts serial touch timestamps by ~0.28 ms in recording time.
2. **1 ms grid quantization**: The `scaledhnd.dat` is at 1 ms resolution.  When a touch-lift event
   (reach start trigger) straddles a 1 ms boundary due to the 0.28 ms shift, the detected index
   moves by 1 ms.  This accounts for the remaining ~1 ms.
3. **Fundamental touch sampling limit**: The serial touchscreen reports events at ~44 Hz during
   contact (22 ms interval).  This means the actual liftoff moment has up to ±11 ms uncertainty
   from sensor latency alone.  The 2 ms Python–MATLAB discrepancy is 5–10× smaller than this.

---

## Bug Fixes Applied (proc_pipeline — development history)

### `proc_saccade.py` — SaccStop end_ind off-by-one (2026-03-05)

**Root cause**: MATLAB's `vel_after = vel(sacc_loc:end)` is 1-indexed, so
`find(vel_after < thresh, 1, 'first')` returns k where `vel(sacc_loc+k-1) < thresh`, giving
`end_ind = sacc_loc + k = E + 1` (one sample past the threshold crossing E).
Python's `np.where(vel_after < thresh)[0]` is 0-indexed, giving `end_ind = E` — one sample earlier.

**Fix**: Added `+1` to `end_ind` in `_detect_saccade()` (both normal path and fallback).

**Effect**: SaccStop median error reduced from +1.7–2.1 ms to +0.7–1.1 ms, now matching SaccStart
(both reflect only the ~0.3 ms Go clock alignment residual).

### `helpers.py` — touch transform matrix column-major bug (2026-03-05)

**Root cause**: `get_touch_transform()` used `np.array(v).reshape(3, 3)` (Python row-major) to parse
`touch_tform`, but MATLAB's `reshape(v, 3, 3)` is column-major.  The result was the **transpose** of the
correct transform matrix: the translation was in the last row instead of the last column, producing
coordinates around −132 deg (wrong) instead of ±15 deg (on-screen).

**Fix**: Changed to `reshape(3, 3, order='F')` (Fortran/column-major order).

**Evidence**: For raw touch (2290, 2981), buggy Python gave H=−130.6 deg; fixed Python gives
H=−0.18 deg, exactly matching MATLAB's output.

### `proc_hand.py` / `proc_eye.py` — grid endpoint off-by-one (2026-03-05)

**Root cause**: `np.arange(t_rec[0], t_rec[-1] + step, step)` added one extra sample beyond
MATLAB's `t_rec(1):step:t_rec(end)` because the fractional millisecond in `t_rec[-1]` rounded
up to include one more grid point.

**Fix**: Replaced with `n_ds = floor((t_rec[-1] - t_rec[0]) / step) + 1; t_rec[0] + arange(n_ds)*step`,
which exactly replicates MATLAB's colon operator count.  Output lengths now match MATLAB exactly.

### `proc_display.py` — median_display_latency guard removed (2026-03-09)

**Root cause**: Python had a `< 10 ms` guard on `median_display_latency` that fell back to 30 ms.
For some recs (e.g. 260217 rec007) the per-trial median is legitimately negative (~−85 ms, because
the nearest display transition precedes trial start). The guard fired → fallback to 30 ms → wrong
Gaussian kernel in `event_match` → large StartOn errors (up to 2000 ms) for many trials.

**Fix**: Removed the guard entirely. MATLAB's `procDisplay_new.m` has no such guard.

**Effect**: 260217 rec007 dropped from 25.2% to 0.2% errors.

### `proc_reach.py` — reach distance index correction (2026-03-10)

**Root cause**: MATLAB uses 1-indexed `T1−1` and `T2+1` for the distance computation; Python's
0-indexed equivalents are `T1` and `T2+2` (initial d) and `T1−1` and `T2+2` (while-loop d).

**Fix**: Corrected all four index expressions in `_detect_reach()`.

**Effect**: Affects only the spurious-reach filter (d < 0.5 deg), not timing. No measurable change
in error counts on any validated day.

### `proc_eye.py` — re-centering mean → median (2026-03-11)

**Root cause**: The post-filter re-centering step used `nanmean`. A single outlier sample in the
raw eye data can corrupt the mean, shifting the entire filtered trace by millions of degrees
(observed on 260302 rec002: H channel ≈ 11,364,006 deg constant).

**Fix**: Changed to `nanmedian`, matching the same fix applied to `procEye.m`.

**Effect**: 260302 rec002 saccade detection restored. No change on other days.

---

## Joystick Support (Current State)

### What is already done

- The joystick task now emits only sparse canonical `BehavState` values for trial structure: `intertrial`, `start_on`, `success`, `fail`.
- Rich within-trial joystick task events remain in `behav_result['attempts'][...]['events']`, including `target_on`, `first_joystick_movement`, `target_entry`, `target_exit`, `hold_start`, `hold_break`, `hold_complete`, `reward_triggered`, `bonus_reward_triggered`, `ignored_idle_timeout`, `free_play_start`, and `free_play_end_requested`.
- The photodiode / display square is toggled only on `start_on`, so display matching should be used only to refine the true visual target-onset time.
- `procThalamus_indie.py` now extracts the recorded `Joystick` analog node into `bag/mat/joystick.mat` as timestamped `x`/`y` samples, preserving every sample instead of collapsing by timestamp.
- `proc_events.py` now writes joystick-related fields into `Events.mat` / `AllTrials.mat` / `Trials.mat`:
  `JoystickTargetOn`, `JoystickFirstMovement`, `JoystickTargetEntry`,
  `JoystickTargetEntryFinal`, `JoystickTargetExit`, `JoystickTargetExitFinal`,
  `JoystickHoldStart`, `JoystickHoldStartFinal`, `JoystickHoldBreak`,
  `JoystickHoldBreakFinal`, `JoystickHoldComplete`, `JoystickReward`,
  and `JoystickAttemptCount`.
- A separate validation / visualization folder now exists at `pyCheck/`.
  It reconstructs cursor position from `joystick.mat` using the real task logic in
  `thalamus/task_controller/joystick_intro.py`, loads `AllTrials.mat`, and generates
  timeseries / trajectory plots for inspection.

### Current joystick contract

- Primary continuous joystick source: recorded `Joystick` node from the behave bag.
- Fallback / debugging source: task-side `behav_result['joystick_samples']` if needed, but this is not the preferred analysis source.
- Saved intermediate file: `bag/mat/joystick.mat` with `header_stamp_sec`, `header_stamp_nanosec`, `x`, `y`.
- These timestamps are still in task / ROS-side time and should be aligned later using `w_drift_ros`, just like other task-side modalities.

### Validation status on day `260324`

- `260324` has now been processed both with and without video.
- Video-enabled extraction completes successfully. On Windows systems without symlink privilege,
  `procThalamus_indie.py` now falls back from symlink to normal file copy for the camera MP4s.
- `joystick.mat` for `260324` contains `26030` samples, with monotonic timestamps and plausible
  X/Y range.
- `pyCheck` currently produces:
  - joystick / cursor timeseries plots
  - reconstructed cursor trajectory plots
  - an alignment summary histogram
  - a constant-shift sweep figure for debugging timestamp offsets
- The **current plotting code is good enough for qualitative inspection**.
  The timeseries and trajectory plots now use raw task attempt events from `behav_result`
  as the primary annotation source, so repeated events such as `target_entry`, `target_exit`,
  `target_entry_2`, `target_exit_2`, `target_entry_3`, etc. are visible at the correct
  within-trial spacing.
- `proc_events.py` joystick event promotion has now been fixed:
  processed `Joystick*` fields are generated from absolute task timestamps in
  `behav_result['final_attempt']['events'][...]['time_perf_counter']`, aligned into recorder time
  through `w_drift_ros`, instead of being reconstructed backwards from `End`.
- On the explicit failure case `260324`, trial 4, all processed `Joystick*` fields now round-trip
  back to the raw `behav_result` event times with `0.0 ms` error in `pyCheck`.
- The remaining joystick timing concern is now **not** the processed `Joystick*` fields.
  It is the relation between task `StartOn` and photodiode-confirmed `disStartOn`.

### Current photodiode / target-onset status

- `StartOn` and `disStartOn` are now intentionally kept separate.
  `StartOn` remains the original task-controller timestamp from the `BehavState=start_on` event.
  `disStartOn` stores the photodiode/display-matched timestamp from `proc_display.py`.
- This is intentional: downstream analysis can choose whether to use task-side onset or
  display-confirmed onset.

#### Current unresolved concern: `disStartOn` appears far too late

For `260324`, rec `001`, the currently matched display onset is much later than the task-side
`StartOn`. Example first five trials:

- trial 1: `disStartOn - StartOn = 941 ms`
- trial 2: `1227 ms`
- trial 3: `1519 ms`
- trial 4: `1428 ms`
- trial 5: `1680 ms`

Across all 30 trials on this rec:

- median `disStartOn - StartOn`: `1219 ms`
- mean: `1206 ms`
- range: `641–1680 ms`

This is implausibly large if the photodiode square really turns on in the same rendered frame as
the target. A normal monitor / photodiode lag should be on the order of tens of milliseconds, not
~1 second.

#### What has been ruled out so far

- The joystick `Joystick*` fields are no longer the source of the problem; they match raw
  `behav_result` timing correctly.
- The `disStartOn - StartOn` offset does **not** show a meaningful linear drift across trials.
  A least-squares fit versus trial number gives approximately:
  `slope = -1.67 ms / trial`, `R^2 = 0.0028`.
- The offset also does **not** correlate meaningfully with absolute recording time or trial duration.
- Around each task `StartOn`, the raw thresholded display trace currently shows only one detectable
  transition within `±2 s`, and it is the late one currently being matched.

**Interpretation**:

- The present `disStartOn` values should **not** yet be trusted as the true visual onset.
- The remaining problem is likely in photodiode/display detection or display-event matching,
  not in joystick event promotion.

### Recommended field policy

- `StartOn`: original task-controller `start_on` event.
- `disStartOn`: separate photodiode / display-confirmed onset from `proc_display.py`.
- `Success` / `Fail`: task-controller timed.
- `target_entry`, `hold_start`, `hold_complete`: initially task-emitted from `behav_result`.
- Continuous joystick trajectory: derived from `joystick.mat` after clock alignment.

### Resume checklist for a future session

When resuming joystick work, inspect these in order:

1. `proc/PyTaskCtrl/py_proc/procThalamus_indie.py`
   Confirm `joystick.mat` is being written and sample counts look correct.
   Also confirm the Windows video-link fallback still works (`copyfile` fallback when symlink fails).
2. `proc/PyTaskCtrl/py_proc/pre_proc.py`
   Confirm sparse joystick task `BehavState` values produce valid trials.
3. `proc/PyTaskCtrl/py_proc/proc_events.py`
   Confirm the fixed joystick event-promotion path is still intact:
   `Joystick*` fields should be generated from absolute `time_perf_counter` values and aligned via
   `w_drift_ros`, not reconstructed backwards from `End`.
4. `pyCheck/joystick_validation.py`
   Use this as the current ground-truth inspection tool.
   The plots look okay now because they use raw `behav_result` attempt events for annotation.
5. One real processed recording with video, currently `260324`
   Compare:
   - photodiode / `start_on`
   - visible cursor movement in video
   - raw `behav_result['final_attempt']['events']`
   - processed `Joystick*` fields in `AllTrials.mat`
6. Keep the `260324`, trial 4 example in mind
   This is still the best concrete trial to compare across:
   task `StartOn`, `disStartOn`, raw `behav_result` joystick events, `joystick.mat`, and video.
7. Inspect the raw display signal around trial onset
   Determine whether the matched edge is really the photodiode square turning on, or whether the
   current threshold / matching logic is locking onto the wrong transition.
8. Do **not** treat `disStartOn` as trustworthy visual onset until the large `~1.2 s` offset is explained
   For now, retain both task `StartOn` and photodiode `disStartOn` and compare them explicitly.

---

## Known Issues / TODO

- **Saccade outliers (2–6/rec)**: A few trials per rec have SaccStart/SaccStop errors > 5 ms.
  These are cases where Python and MATLAB find different velocity peaks from the same ambiguous
  profile (e.g., post-saccadic oscillations or adjacent saccades). Not a bug — both algorithms are
  genuinely ambiguous; they just pick differently.
- **`MONKEYDIR` portability**: All pipeline functions take `monkeydir` as an explicit parameter.
  Only the runner script (`p260303_procLoop.py`) has it hardcoded.
- **Photodiode / display onset mismatch on joystick day `260324`**:
  `disStartOn` is now being populated for `joystick_intro`, but the matched photodiode onset is
  implausibly late relative to task `StartOn` (median ~1219 ms, range 641–1680 ms on rec `001`).
  This does not show a simple linear drift across trials, so the likely remaining issue is in
  display signal interpretation or event matching rather than joystick event timing.
