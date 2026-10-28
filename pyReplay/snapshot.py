"""Render the same one second of a recording in every layout, side by side.

Use this to choose a layout before committing to a long export: it loads one
window, then renders a short clip (and a PNG still) of each arrangement in
pyReplay/layouts.py from that single decode, plus an optional contact sheet of
all the stills.

Examples
--------
  python -m pyReplay.snapshot --file /cdz/.../260924/behave.20260924.1 --at 120
  python -m pyReplay.snapshot --day-dir /cdz/.../260924 --rec 1 --at 120 \\
      --seconds 2 --out-dir ~/layout_previews --dpi 160
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional, Sequence

_PRE_ROLL_S = 2.5   # decoded before the snapshot instant so trails/traces fill


def render_snapshots(
    win,
    out_dir: str,
    at_s: float,
    *,
    seconds: float = 1.0,
    layouts: Optional[Sequence[str]] = None,
    scroll_s: float = 2.0,
    fps: int = 30,
    dpi: int = 120,
    figsize=(16.0, 9.0),
    contact_sheet: bool = True,
    log=print,
) -> List[str]:
    """Render `seconds` of `win` starting at `at_s` in each layout. Returns paths."""
    from pyReplay.export import export_mp4, export_still
    from pyReplay.layouts import LAYOUTS

    names = list(layouts) if layouts else list(LAYOUTS)
    out = pathlib.Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for name in names:
        png = out / f"layout_{name}.png"
        mp4 = out / f"layout_{name}.mp4"
        log(f"  {name}: still -> {png.name}")
        export_still(win, str(png), at_s, layout=name, scroll_s=scroll_s,
                     figsize=figsize, dpi=dpi)
        log(f"  {name}: {seconds:g}s clip -> {mp4.name}")
        export_mp4(win, str(mp4), layout=name, scroll_s=scroll_s, fps=fps,
                   figsize=figsize, dpi=dpi, start_s=at_s, duration_s=seconds)
        written += [str(png), str(mp4)]

    if contact_sheet and len(names) > 1:
        sheet = _contact_sheet([out / f"layout_{n}.png" for n in names], names,
                               out / "layout_contact_sheet.png")
        log(f"  contact sheet -> {pathlib.Path(sheet).name}")
        written.append(sheet)
    return written


def _contact_sheet(pngs, names, out_path) -> str:
    """Tile the layout stills into one labelled comparison image."""
    import matplotlib.image as mpimg
    from matplotlib.figure import Figure

    n = len(pngs)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig = Figure(figsize=(9 * ncol, 5.3 * nrow), facecolor="white")
    gs = fig.add_gridspec(nrow, ncol, hspace=0.10, wspace=0.03,
                          left=0.01, right=0.99, top=0.95, bottom=0.01)
    for i, (png, name) in enumerate(zip(pngs, names)):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        ax.imshow(mpimg.imread(str(png)))
        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_axis_off()
    fig.savefig(str(out_path), dpi=110, facecolor="white")
    return str(out_path)


def main(argv=None) -> int:
    from pyReplay.run_replay import _resolve_behave, _float_or_end  # noqa: F401

    p = argparse.ArgumentParser(
        prog="pyReplay.snapshot", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="Path to a behave capture file")
    p.add_argument("--day-dir", help="Day directory; used with --rec")
    p.add_argument("--rec", default="1", help="Recording number (default: 1)")
    p.add_argument("--at", type=float, required=True, metavar="SEC",
                   help="Instant to snapshot, seconds from recording start")
    p.add_argument("--seconds", type=float, default=1.0, metavar="SEC",
                   help="Clip length per layout (default: 1.0)")
    p.add_argument("--out-dir", default="layout_previews",
                   help="Directory for the stills/clips (default: ./layout_previews)")
    p.add_argument("--layouts", nargs="*", default=None, metavar="NAME",
                   help="Subset of layouts to render (default: all)")
    p.add_argument("--cameras", nargs="*", default=None, metavar="NAME")
    p.add_argument("--scroll", type=float, default=2.0, metavar="SEC")
    p.add_argument("--target-h", type=int, default=480, metavar="PX",
                   help="Decoded camera frame height (default: 480)")
    p.add_argument("--max-fps", type=float, default=0.0, metavar="FPS",
                   help="Cap retained camera fps; 0 = keep every frame (default)")
    p.add_argument("--dpi", type=int, default=120,
                   help="120 = 1920x1080, 160 = 2560x1440 (default: 120)")
    args = p.parse_args(argv)

    from pyReplay.loader import load_window

    behave = _resolve_behave(args)
    start = max(args.at - _PRE_ROLL_S, 0.0)
    stop = args.at + args.seconds + 0.5
    print(f"Loading {behave}  [{start:.2f}s - {stop:.2f}s] ...", flush=True)
    win = load_window(behave, start, stop, camera_names=args.cameras,
                      target_h=args.target_h, max_fps=args.max_fps or None,
                      progress=lambda f: print(f"\r  {100 * f:5.1f}%", end="", flush=True))
    print("\n" + win.summary(), flush=True)

    print(f"Rendering {args.seconds:g}s per layout at {args.dpi} dpi:")
    written = render_snapshots(win, args.out_dir, args.at, seconds=args.seconds,
                               layouts=args.layouts, scroll_s=args.scroll,
                               dpi=args.dpi)
    print(f"\n{len(written)} files in {pathlib.Path(args.out_dir).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
