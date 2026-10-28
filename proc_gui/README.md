# proc_gui

Optional PyQt6 desktop GUI for the `pyCheck` workflows.

What it supports:
- choose a day directory and recording
- run the full `run_day_pipeline.py` workflow from a Processing tab
- generate day presentation plots
- build a cross-day list of selected days and recordings, then generate cross-day summary plots
- generate joystick validation summary plots
- generate a selected trial trajectory or timeseries plot
- browse generated PNGs inside the chosen output directory, with wheel zoom and drag panning
- minimise any panel out of the way and have the arrangement restored next launch

Launch:

```bash
./.venv/bin/python proc_gui/run_gui.py
```

Notes:
- GUI code lives only in `proc_gui/`.
- The GUI reuses the existing `pyCheck` functions directly; it does not replace the CLI scripts.
- The cross-day GUI workflow keeps `cross_day_summary_metrics.csv` cumulative so you can keep adding new days as data accrues.
- Install `PyQt6` in the active environment before launching the GUI.

Layout:
- Every panel (Monkey Repo, Day Notes, Controls and its Inputs/Actions/Output Log rows,
  Figures) has a `▾` marker in its title bar. Clicking it — or double-clicking the title
  bar — minimises the panel to a narrow labelled strip; clicking the strip's `▸` gives the
  panel its previous size back, taking the space from its roomiest neighbour.
- The `View` menu lists every panel with `Ctrl+Shift+<n>` toggles, plus
  `Maximise Figure Preview` (`Ctrl+Shift+F`) which minimises everything at once and
  restores it on the second press, `Show All Panels` (`Ctrl+Shift+0`) and `Reset Layout`.
- Window geometry, splitter positions, which panels are minimised and the active tab are
  saved on close and restored on the next launch (`QSettings`, `PesaranLab/ProcGui`).
  `Reset Layout` forgets them.

Figure preview:
- Wheel zooms around the cursor, drag pans, double-click fits.
- Toolbar: `−` / `+` / `Fit` / `1:1` with a zoom readout and the figure's pixel size.
- Shortcuts: `Ctrl+0` fit, `Ctrl+1` native size, `Ctrl++` / `Ctrl+-` zoom.
