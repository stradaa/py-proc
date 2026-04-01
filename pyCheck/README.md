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

Usage:

```powershell
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324
```

Outputs are written by default to:

```text
pyCheck/output/<DAY>/
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
- If that root is read-only, also pass a writable `--out-dir`.

Replay examples:

```powershell
# Continuous range
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 10-20

# Discrete trials
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4 7 12 18

# Custom output path and frame rate
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4-5 --render-out pyCheck\output\260324\trial_replay_004_005.mp4 --fps 20
```

Linux / shared-storage example:

```bash
./.venv/bin/python pyCheck/run_joystick_validation.py \
  --repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig \
  --day 260331 \
  --rec 005 \
  --skip-report \
  --render-trials 1 2 3 4 5 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --render-out /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/python/alex/beh_joystick_analysis/260331_rec005_first20_successful_reaches.mp4
```

Presentation plot example:

```bash
./.venv/bin/python pyCheck/day_presentation_plots.py \
  --repo-root /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig \
  --day 260331 \
  --out-dir /vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/python/alex/beh_joystick_analysis \
  --exclude-recs 3
```

This writes:
- `260331_overview_by_rec.png`
- `260331_performance_over_time.png`
- `260331_target_performance.png`
- `260331_summary_metrics.json`

Notes:
- `--render-trials` accepts ranges like `10-20`, discrete integers like `4 7 12`, or comma-separated tokens.
- Output is a single concatenated MP4 replaying the selected trials in order.
- The replay is based on reconstructed cursor position from `joystick.mat`, not the camera video.
- `day_presentation_plots.py` currently uses `Target` ID for target-wise summaries; some days may not have reliable `TargetAngle` or `TargetLocation` fields.
- If you want to exclude problematic blocks from the day-level summary, pass them via `--exclude-recs`.
