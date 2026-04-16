# proc_gui

Optional PyQt6 desktop GUI for the `pyCheck` workflows.

What it supports:
- choose a day directory and recording
- run the full `run_day_pipeline.py` workflow from a Processing tab
- generate day presentation plots
- build a cross-day list of selected days and recordings, then generate cross-day summary plots
- generate joystick validation summary plots
- generate a selected trial trajectory or timeseries plot
- browse generated PNGs inside the chosen output directory

Launch:

```bash
./.venv/bin/python proc_gui/run_gui.py
```

Notes:
- GUI code lives only in `proc_gui/`.
- The GUI reuses the existing `pyCheck` functions directly; it does not replace the CLI scripts.
- The cross-day GUI workflow keeps `cross_day_summary_metrics.csv` cumulative so you can keep adding new days as data accrues.
- Install `PyQt6` in the active environment before launching the GUI.
