# pyCheck

Validation and visualization helpers for joystick days.

What it does:
- Loads `joystick.mat`, `AllTrials.mat`, `recNNN.ev.mat`, and `recNNN.w_alignment.mat`
- Reconstructs cursor position from joystick samples using the real `joystick_intro.py` direction-influence and control-mode logic
- Compares processed trial timestamps against `behav_result` event times
- Compares reconstructed cursor positions against task-logged `cursor_x` / `cursor_y` at `target_entry`, `target_exit`, and `hold_start`
- Generates trial trajectory plots and a summary figure
- Renders replay MP4s that show the reconstructed cursor path and target over selected trials

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

Replay examples:

```powershell
# Continuous range
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 10-20

# Discrete trials
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4 7 12 18

# Custom output path and frame rate
.venv\Scripts\python.exe pyCheck\run_joystick_validation.py --day 260324 --skip-report --render-trials 4-5 --render-out pyCheck\output\260324\trial_replay_004_005.mp4 --fps 20
```

Notes:
- `--render-trials` accepts ranges like `10-20`, discrete integers like `4 7 12`, or comma-separated tokens.
- Output is a single concatenated MP4 replaying the selected trials in order.
- The replay is based on reconstructed cursor position from `joystick.mat`, not the camera video.
