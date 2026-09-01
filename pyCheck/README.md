# pyCheck

Validation and visualization helpers for joystick days.

What it does:
- Loads `joystick.mat`, `AllTrials.mat`, `recNNN.ev.mat`, and `recNNN.w_alignment.mat`
- Reconstructs cursor position from joystick samples using the real `joystick_intro.py` direction-influence and control-mode logic
- Compares processed trial timestamps against `behav_result` event times
- Compares reconstructed cursor positions against task-logged `cursor_x` / `cursor_y` at `target_entry`, `target_exit`, and `hold_start`
- Generates trial trajectory plots and a summary figure
- Renders replay MP4s that show the reconstructed cursor path and target over selected trials
- Generates day-level presentation plots from `AllTrials.mat`

Tools in this folder:
- `run_joystick_validation.py`
  Runs the existing validation workflow for one day and one rec, writes alignment summaries, and can render replay videos.
- `day_presentation_plots.py`
  Generates presentation-style day summaries directly from `AllTrials.mat`, such as:
  `*_overview_by_rec.png`, `*_performance_over_time.png`, `*_display_alignment.png`, `*_target_performance.png`,
  `*_success_failure_timing.png`, and `*_summary_metrics.json`.
- `day_behavior_metrics.py`
  Additional day-summary behavior plots wired into `generate_day_presentation_plots`, all derived from fields
  already in `AllTrials.mat`: `*_kinematic_intervals.png` (reaction / movement / hold-duration histograms),
  `*_failure_modes.png` (why failed trials failed, per rec), `*_overshoot_attempts.png` (target re-entry rate and
  attempts per trial), and `*_engagement.png` (inter-trial interval, cumulative trials vs rewards, trial rate per rec).
- `co_structured.py` / `co_structured_traj_plots.py` / `co_structured_plots.py`
  Structured center-out (CO) outputs for days recorded with the Rust executor's target schedule: separates
  `schedule_phase`-tagged attempts (`center` / `peripheral` / `sequence` vs `random` inserts), writes
  `*_co_structured.png` (per-direction compass trajectory grid), `*_co_structured_metrics.png` (performance /
  speed / path-efficiency bars by direction and ring), `*_co_structured_attempts.csv` (one row per attempt),
  and `*_co_structured_paths.npz` (full cursor/joystick paths). Runs automatically inside
  `generate_day_presentation_plots` (skipped when a day has no `schedule_phase` data) or standalone via
  `python -m pyCheck.co_structured` with the same `--repo-root/--day/--day-dir/--out-dir/--exclude-recs` args.
- `plot_utils.py`
  Small shared helpers (`_flat`, `_save_figure`, `_title_suffix`) used by both day-plot modules.
- `cross_day_plots.py`
  Generates cross-day summaries for selected day/recording combinations, including:
  median successful trial duration by day, hold-break rate, first-entry success rate,
  median path efficiency, and successful trial count.

Optional GUI:
- `proc_gui/run_gui.py`
  Launches a PyQt6 front end for selecting day/rec, generating validation or day-summary plots, and browsing the resulting PNGs.

Usage:

```powershell
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324
```

Outputs are written by default to:

```text
/vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/claude/figures/<DAY>/beh
```

Typical files:
- `alignment_summary.png`
- `validation_summary.json`
- `trial_XXX_trajectory.png`
- `trial_XXX_timeseries.png`

Important path note:
- `run_joystick_validation.py` assumes `--repo-root` points at the day-data root that contains folders like `260331/`, not necessarily this code repository.
- For AlexRig days stored under `/vol/cortex/.../Bowser_Behavior_AlexRig/`, pass:
  `--repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig`
- You can also pass `--day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260331`.
- If that figures root is read-only, also pass a writable `--out-dir`.

Replay examples:

```powershell
# Continuous range
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 10-20

# Discrete trials
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4 7 12 18

# Custom output path and frame rate
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4-5 --render-out pyCheck\output\260324\trial_replay_004_005.mp4 --fps 20

Further optional statements and examples:
  --out-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/python/alex/beh_joystick_analysis \
  --exclude-recs 3

# Updated 
> python pyCheck/run_joystick_validation.py --day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260402 --rec 001

> python pyCheck/day_presentation_plots.py  --day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260331 \
  --task-types joystick_intro
```

Linux / shared-storage example:

```bash
./.venv/bin/python pyCheck/run_joystick_validation.py \
  --repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig \
  --day 260331 \
  --rec 005 \
  --skip-report \
  --render-trials 1 2 3 4 5 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --render-out /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/claude/figures/260331/beh/trial_replay_260331_005.mp4
```

Equivalent direct day-directory usage:

```bash
./.venv/bin/python pyCheck/run_joystick_validation.py \
  --day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260331 \
  --rec 005 \
  --skip-report \
  --render-trials 1 2 3 4 5
```

Presentation plot example:

```bash
./.venv/bin/python pyCheck/day_presentation_plots.py \
  --repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig \
  --day 260331 \
  --exclude-recs 3
```

Equivalent direct day-directory usage:

```bash
./.venv/bin/python pyCheck/day_presentation_plots.py \
  --day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260331 \
  --exclude-recs 3
```

Task-filtered usage:

```bash
./.venv/bin/python pyCheck/day_presentation_plots.py \
  --day-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260331 \
  --task-types joystick_intro
```

By default this writes into:
- `/vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/claude/figures/260331/beh`

Inside that folder it writes:
- `260331_overview_by_rec.png`
- `260331_performance_over_time.png`
- `260331_display_alignment.png`
- `260331_target_performance.png`
- `260331_success_failure_timing.png` — success-vs-failure timing histograms (first movement, target entry, hold complete, trial end)
- `260331_kinematic_intervals.png` — reaction time, movement time, and hold duration histograms
- `260331_failure_modes.png` — failed-trial breakdown per rec (no movement / never acquired target / broke hold)
- `260331_overshoot_attempts.png` — target re-entry rate per rec and attempts-per-trial distribution
- `260331_engagement.png` — inter-trial interval, cumulative trials vs rewards, and trial rate per rec
- `260331_summary_metrics.json`
- `260331_co_structured.png` / `_co_structured_metrics.png` / `_co_structured_attempts.csv` /
  `_co_structured_paths.npz` — only for days whose attempts carry `schedule_phase` (Rust executor target
  schedule); see "Structured center-out outputs" below

Notes:
- `--out-dir` is optional; if provided, it overrides the default `claude/figures/<day>/beh` path.
- `--task-types` filters trials by exact `AllTrials.PyTaskType` value, so mixed-task recordings can be summarized without excluding entire recs.
- `run_joystick_validation.py` now defaults both report figures and replay videos into the same `claude/figures/<day>/beh` folder used by `day_presentation_plots.py`.
- `display_alignment.png` is computed **directly from the behave file** (`pyCheck/display_behave.py`): the display latency is the first photodiode (`Analog in` / `Dev1/ai1`) rising edge at/after each trial's task `target_on`, both read on the shared `record.time` clock (`target_on.time_perf_counter * 1e9` is already a behave-clock ns — see `pyReplay/README.md`). This replaces the legacy recorder-clock `AllTrials.disStartOn`, which drifted by ~2 s on AlexRig days since 260609. If the behave captures are missing, `generate_day_presentation_plots` falls back to `AllTrials.disStartOn`.
- `--render-trials` accepts ranges like `10-20`, discrete integers like `4 7 12`, or comma-separated tokens.
- Output is a single concatenated MP4 replaying the selected trials in order.
- The replay is based on reconstructed cursor position from `joystick.mat`, not the camera video.
- `day_presentation_plots.py` currently uses `Target` ID for target-wise summaries; some days may not have reliable `TargetAngle` or `TargetLocation` fields.
- If you want to exclude problematic blocks from the day-level summary, pass them via `--exclude-recs`.

## Day behavior metrics

`day_behavior_metrics.py` adds four derived figures (auto-generated by `generate_day_presentation_plots`) that
turn the raw `AllTrials.mat` joystick timestamps into motor-learning readouts. All timestamps are milliseconds
relative to per-trial `StartOn`; `StartOn` itself is an absolute, per-rec monotonic clock used for inter-trial
intervals. Trials with `End <= 0` are degenerate (aborted/placeholder rows, e.g. a calibration rec) and are
excluded from every interval and engagement metric.

**`*_kinematic_intervals.png`** — three overlaid success/failure histograms, x-clipped at the 99th percentile
with outliers piled into the last bin:
- **Reaction time** = `JoystickFirstMovement − JoystickTargetOn`. Time from target appearing to movement onset —
  the classic learning/engagement readout. Note: in direct-control mode first movement fires almost immediately
  after target on, so RT is near-zero and movement time is the more meaningful speed metric.
- **Movement time** = `JoystickTargetEntry − JoystickFirstMovement`. Time to transport the cursor to the target;
  shortens as the reach is learned.
- **Hold duration** = `JoystickHoldComplete − JoystickHoldStart`. Time spent holding to complete the trial.

**`*_failure_modes.png`** — stacked bar per rec categorizing each failed trial (`Success == 0`) by the funnel
stage it reached, so you can see *why* trials fail rather than just the failure rate:
- **No movement** — never produced a first movement (no `JoystickFirstMovement`). Usually disengagement.
- **Never acquired target** — moved but never entered the target (no `JoystickTargetEntry`). Reach accuracy/range.
- **Broke hold** — entered the target but never completed the hold (no `JoystickHoldComplete`). Hold stability.

**`*_overshoot_attempts.png`** — two panels:
- **Overshoot / corrective re-entry rate** per rec — fraction of acquired trials where the cursor exited and
  re-entered (`JoystickTargetEntryFinal > JoystickTargetEntry`). A proxy for endpoint precision; typically drops
  with learning.
- **Attempts per trial** — distribution of `JoystickAttemptCount` with the successful count overlaid, showing
  effort/persistence and how success rate varies with number of attempts.

**`*_engagement.png`** — three panels tracking motivation and fatigue across the session:
- **Inter-trial interval** — histogram of within-rec `diff(StartOn)` (x-clipped at the 95th percentile to drop
  long pauses/breaks), with the median marked.
- **Cumulative trials vs rewards** — diverging lines show reward-rate drift over the day.
- **Trial rate by recording** — trials/min per rec from the `StartOn` span, flagging blocks where the subject
  slowed down.

## Structured center-out outputs

`co_structured.py` separates structured trials from random inserts using the per-attempt `schedule_phase`
field the Rust joystick executor writes into `behav_result["attempts"]` (`center` / `peripheral` in
center-out mode, `sequence` in sequence mode, `random` for Random mode, interleave inserts, and fallback
picks). It reads attempts straight from `recNNN.ev.mat` `behav_results` — no pipeline or `AllTrials.mat`
schema change — and reconstructs each attempt's cursor path from `joystick.mat`.

Outputs (same `claude/figures/<day>/beh` folder):
- **`<day>_co_structured.png`** — a compass grid sized to the number of directions the day ran: 8 directions
  give a 3x3, 16 give a 5x5, 24 give a 7x7 (grid size `G` is the smallest odd number whose `4(G-1)` perimeter
  cells fit every direction). Each perimeter panel sits at its own direction (the 90° target's panel is
  top-middle, 0° is middle-right), and shows every successful reach to that target (thin, translucent) plus
  one bold average trajectory (per-trial cursor paths resampled onto a common normalized-time grid and
  averaged). Panels are keyed by `(ring, angle)`, so a two-ring day gets one panel per ring per direction and
  never averages a near reach together with a far one; hue carries direction and the outer ring is darker.
  Reaches are drawn `target_on` → `hold_complete` — transport only. The return leg is a separate
  `center`-phase attempt and isn't drawn here. The merged inner block overlays every direction's average.
  Targets are drawn as **ellipses**: `target_radius_ratio` scales by the task region's *min* dimension, so a
  target that is a circle on screen is an ellipse in the normalized coordinates this figure plots in (see
  `fitts_throughput.pixel_scale`). The dotted ellipse at center is `center_gate_radius_ratio`.
- **`<day>_co_structured_metrics.png`** — three bar-chart panels, directions ordered by angle and grouped into
  one contiguous block per ring (labelled `near` / `far` when a day ran more than one):
  (1) success rate per peripheral target / structured vs random overall / center→peripheral pair completion,
  (2) median reach duration by direction (successful attempts only), and (3) median path efficiency by
  direction (successful attempts only; actual path length / straight-line distance from `target_on` to
  `hold_complete`, with a dashed reference line at 1.0 = perfectly straight — values can occasionally dip
  below 1.0 when a reach undershoots into the near edge of a wide target, since the ratio is measured against
  the nominal target center rather than the actual stop point).
- **`<day>_co_structured_attempts.csv`** — one row per `schedule_phase` attempt:
  `day, rec, trial_index, attempt_index, schedule_phase, outcome, failure_reason, target_index, target_name,
  target_x, target_y, target_radius, target_angle_deg, t_target_on, t_end, t_hold_complete, duration_s,
  reach_distance, ring_label, path_efficiency, pair_id, pair_complete, path_row`.
  `target_angle_deg` is the target's true angle from center (0.5, 0.5) in degrees [0, 360), derived
  geometrically (never from `target_name`, which is an arbitrary experimenter string) — populated only for
  `schedule_phase == "peripheral"` (NaN otherwise). `reach_distance` is the target's normalized distance from
  center and `ring_label` its detected ring (`ring` for a single-ring day, else `near` / `ring<i>` / `far`),
  both peripheral-only. `path_efficiency` is populated only for successful peripheral attempts. `pair_id >= 0`
  groups an adjacent center→peripheral pair; `path_row` indexes into the NPZ (−1 = no path).
- **`<day>_co_structured_paths.npz`** — pickle-free ragged path storage: concatenated `task_time_s`,
  `cursor_x`, `cursor_y`, `joystick_x`, `joystick_y` plus `row_start` / `row_len`. Reconstruct row `k` with
  `sl = slice(row_start[k], row_start[k] + row_len[k])`, e.g.:

  ```python
  import numpy as np, csv
  npz = np.load("260707_co_structured_paths.npz")
  with open("260707_co_structured_attempts.csv") as fh:
      rows = list(csv.DictReader(fh))
  k = int(rows[0]["path_row"])
  sl = slice(npz["row_start"][k], npz["row_start"][k] + npz["row_len"][k])
  x, y, t = npz["cursor_x"][sl], npz["cursor_y"][sl], npz["task_time_s"][sl]
  ```

Notes:
- Pair adjacency (`pair_id`) is guaranteed only for data recorded with the pair-boundary executor build
  (≥ 2026-07-07); in older recordings a random insert could land inside a pair, leaving those attempts unpaired.
  Pairing runs over the rec's attempt sequence, not within a Thalamus trial — the center hold and the reach it
  gates are commonly two attempts in two different trials.
- The `cursor_x` / `cursor_y` in the NPZ are reconstructed from `joystick.mat`, then **anchored** to the
  cursor the task itself logs on its `target_entry` / `hold_start` / `target_exit` events. Under the stateful
  control modes (`cumulative` / `blend`) the offline integration otherwise drifts without bound — on 260901
  it reached 0.38 in normalized units, four times a target radius, and only 43% of successful peripheral
  reaches ended inside their target. With anchoring that is 98%. `load_joystick_dataset` prints the anchor
  coverage per rec and warns when a stateful rec has no anchors at all.
- The schedule advances per *presented* trial, not per success — filter on `outcome == "success"` (or
  `pair_complete`) for completed reaches.
- Attempts without a `schedule_phase` key (pure-Python task, pre-schedule days) are ignored; if the whole day
  has none, the module prints a note and writes nothing.

Cross-day plot example:

```bash
./.venv/bin/python pyCheck/cross_day_plots.py \
  --repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig \
  --day-recs 260330:3-5 \
  --day-recs 260331:1,2,4,5 \
  --day-recs 260402:1-4 \
  --day-recs 260406:3-5 \
  --day-recs 260407:2-5,7 \
  --day-recs 260408:1-7 \
  --day-recs 260409:1-6 \
  --day-recs 260410:1-5, 7-10 \
  --day-recs 260415:1,4,5 \
  --day-recs 260416:1-7 \
  --task-types joystick_intro
```

Cross-day selection notes:
- `--day-recs` is repeatable and lets you choose which recordings belong to each day.
- Use `260410` with no colon to include all recs found for that day.
- Use comma-separated and range syntax like `1,2,5-7`.
- Outputs are written by default to `claude/figures/cross_day_beh`.
- Cross-day outputs now include `cross_day_metrics.png`, `cross_day_summary_metrics.json`, and `cross_day_summary_metrics.csv`.
- A companion **`cross_day_kinematics.png`** (plus `cross_day_kinematics_metrics.json` / `.csv`) is written alongside the headline figure by `cross_day_kinematics.py`. It adds finer-grained motor-learning metrics — movement-time decomposition (reaction / transport / acquisition), reaction-time variability, initial reach-direction error, submovement count, SPARC smoothness, trajectory & endpoint variability, engagement (attempts/trial, idle rate, trials/min), and a direction-resolved success-rate heatmap. All are computed on the task/behave clock (event `time_perf_counter` + reconstructed cursor), like the existing cross-day metrics. Run it standalone with the same `--day-recs` / `--task-types` args as `cross_day_plots.py`, or get it automatically from `generate_cross_day_plots` (the GUI's cross-day button).
  - Note: in direct control mode `first_joystick_movement` fires almost immediately after `target_on`, so the "reaction" component is near-zero and the **transport time (move→entry) is the meaningful reach-speed metric**.
- A second companion **`cross_day_co_structured.png`** (plus `cross_day_co_structured_metrics.json` / `.csv`) is written alongside the headline figure by `cross_day_co_structured.py`, on days that carry `schedule_phase`-tagged attempts (see "Structured center-out outputs" above). It tracks structured-CO learning specifically: structured-vs-random success rate, median reach duration and path efficiency by day (successful peripheral reaches only), center→peripheral pair completion rate, attempt counts by day, and a direction-resolved success-rate heatmap (day x angle) that shows whether directional bias shrinks over days. Aggregation reuses `co_structured.py`'s own attempt collection directly against each day's raw data, so it doesn't depend on that day's per-day `co_structured` figures having been generated first. Run it standalone with the same `--day-recs` args as `cross_day_plots.py`, or get it automatically from `generate_cross_day_plots`. Skips cleanly (prints a note, no figure) on days/selections with no `schedule_phase` data.
- The duration metric is computed on successful trials as `hold_complete - target_on`.
- Hold-break rate is computed among trials that reached first target entry.
- First-entry success means the first target entry reached hold completion before any exit or hold break.
- Path efficiency is computed on successful trials as cursor path length from `target_on` to `hold_complete`, divided by straight-line distance from cursor position at `target_on` to the target center.
- The cross-day CSV is cumulative: rerunning with new days adds or updates rows by day instead of discarding the older saved summary rows.
