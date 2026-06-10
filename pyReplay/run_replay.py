"""CLI entry point for the pyReplay interactive player.

Examples:
  python -m pyReplay --file /cdz/.../260608/behave.20260608.1 --start 30 --stop 60
  python -m pyReplay --day-dir /cdz/.../260608 --rec 1 --start 30 --stop 60
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _resolve_behave(args: argparse.Namespace) -> str:
    if args.file:
        return str(pathlib.Path(args.file).resolve())
    if not args.day_dir:
        raise SystemExit("Provide --file or --day-dir (+ --rec).")
    day_dir = pathlib.Path(args.day_dir).resolve()
    matches = sorted(p for p in day_dir.glob(f"behave.*.{args.rec}")
                     if p.suffix == f".{args.rec}")
    if not matches:
        raise SystemExit(f"No behave.*.{args.rec} capture file found in {day_dir}")
    return str(matches[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pyReplay", description=__doc__)
    parser.add_argument("--file", help="Path to a behave capture file")
    parser.add_argument("--day-dir", help="Day directory (used with --rec)")
    parser.add_argument("--rec", default="1", help="Recording number (with --day-dir)")
    parser.add_argument("--start", type=float, required=True,
                        help="Window start, seconds from recording start")
    parser.add_argument("--stop", type=float, required=True,
                        help="Window stop, seconds from recording start")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Camera node names (default: auto-detect)")
    parser.add_argument("--scroll", type=float, default=2.0,
                        help="Trace scroll half-window (s)")
    parser.add_argument("--target-h", type=int, default=360,
                        help="Decoded camera frame height (px)")
    args = parser.parse_args(argv)

    behave = _resolve_behave(args)

    # Imported lazily so --help works without Qt / a display.
    from PyQt6.QtWidgets import QApplication
    from pyReplay.loader import load_window
    from pyReplay.player import ReplayPlayer

    print(f"Loading {behave}  [{args.start}-{args.stop}s] ...", flush=True)
    win = load_window(behave, args.start, args.stop,
                      camera_names=args.cameras, target_h=args.target_h,
                      progress=lambda f: print(f"\r  {100 * f:5.1f}%", end="", flush=True))
    print("\n" + win.summary(), flush=True)

    app = QApplication.instance() or QApplication(sys.argv)
    player = ReplayPlayer(win, scroll_s=args.scroll)
    player.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
