"""CLI entry point for the pyReplay interactive player.

Specify the time window with --start / --stop (seconds from recording start).
Use 'end' for --stop to read until the end of the file.

Examples
--------
  # 30-second window starting at second 0 (file path form):
  python -m pyReplay --file /cdz/.../260609/behave.20260609.1 --start 0 --stop 30

  # Same window using day-dir + rec number:
  python -m pyReplay --day-dir /cdz/.../260609 --rec 1 --start 0 --stop 30

  # From second 60 to the end of the file:
  python -m pyReplay --file /cdz/.../260609/behave.20260609.1 --start 60 --stop end

  # From the start to the end of the file:
  python -m pyReplay --file /cdz/.../260609/behave.20260609.1 --start 0 --stop end

  # Pick specific cameras and a taller display:
  python -m pyReplay --file .../behave.20260609.1 --start 10 --stop 40 \\
      --cameras "Cam Top" "Cam Left" --target-h 480
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_LARGE_S = 999_999.0  # sentinel for "end of file" — no recording is this long


def _float_or_end(s: str) -> float:
    """Accept a number like '30.5' or the literal string 'end'."""
    if s.strip().lower() == "end":
        return _LARGE_S
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number or 'end', got '{s}'"
        )


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
    parser = argparse.ArgumentParser(
        prog="pyReplay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",
                        help="Path to a behave capture file "
                             "(e.g. /cdz/.../260609/behave.20260609.1)")
    parser.add_argument("--day-dir",
                        help="Day directory; used together with --rec "
                             "(e.g. /cdz/.../260609)")
    parser.add_argument("--rec", default="1",
                        help="Recording number when using --day-dir (default: 1)")
    parser.add_argument("--start", type=float, required=True,
                        metavar="SEC",
                        help="Window start, seconds from recording start (e.g. 0, 30.5)")
    parser.add_argument("--stop", type=_float_or_end, required=True,
                        metavar="SEC|end",
                        help="Window stop, seconds from recording start, "
                             "or 'end' to read until end of file")
    parser.add_argument("--cameras", nargs="*", default=None,
                        metavar="NAME",
                        help="Camera node names to include "
                             "(default: auto-detect all cameras). "
                             "Example: --cameras \"Cam Top\" \"Cam Left\"")
    parser.add_argument("--scroll", type=float, default=2.0,
                        metavar="SEC",
                        help="Joystick trace scroll half-window in seconds (default: 2.0)")
    parser.add_argument("--target-h", type=int, default=360,
                        metavar="PX",
                        help="Decoded camera frame height in pixels (default: 360)")
    args = parser.parse_args(argv)

    behave = _resolve_behave(args)
    stop_label = "end" if args.stop >= _LARGE_S else f"{args.stop}s"

    # Imported lazily so --help works without Qt / a display.
    from PyQt6.QtWidgets import QApplication
    from pyReplay.loader import load_window
    from pyReplay.player import ReplayPlayer

    print(f"Loading {behave}  [{args.start}s – {stop_label}] ...", flush=True)
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
