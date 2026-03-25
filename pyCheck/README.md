# pyCheck

Validation and visualization helpers for joystick days.

What it does:
- Loads `joystick.mat`, `AllTrials.mat`, `recNNN.ev.mat`, and `recNNN.w_alignment.mat`
- Reconstructs cursor position from joystick samples using the real `joystick_intro.py` direction-influence and control-mode logic
- Compares processed trial timestamps against `behav_result` event times
- Compares reconstructed cursor positions against task-logged `cursor_x` / `cursor_y` at `target_entry`, `target_exit`, and `hold_start`
- Generates trial trajectory plots and a summary figure

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
