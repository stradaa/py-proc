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
  `*_overview_by_rec.png`, `*_performance_over_time.png`, `*_target_performance.png`, and `*_summary_metrics.json`.

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
- `260331_summary_metrics.json`

Notes:
- `--out-dir` is optional; if provided, it overrides the default `claude/figures/<day>/beh` path.
- `--task-types` filters trials by exact `AllTrials.PyTaskType` value, so mixed-task recordings can be summarized without excluding entire recs.
- `run_joystick_validation.py` now defaults both report figures and replay videos into the same `claude/figures/<day>/beh` folder used by `day_presentation_plots.py`.
- `display_alignment.png` summarizes `AllTrials.disStartOn`, which is already stored relative to `StartOn` in milliseconds.
- `--render-trials` accepts ranges like `10-20`, discrete integers like `4 7 12`, or comma-separated tokens.
- Output is a single concatenated MP4 replaying the selected trials in order.
- The replay is based on reconstructed cursor position from `joystick.mat`, not the camera video.
- `day_presentation_plots.py` currently uses `Target` ID for target-wise summaries; some days may not have reliable `TargetAngle` or `TargetLocation` fields.
- If you want to exclude problematic blocks from the day-level summary, pass them via `--exclude-recs`.
