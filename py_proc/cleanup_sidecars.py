"""Delete the legacy day-level constant-fps sidecar .avi files.

The extract step rebuilds each camera as a per-rec, timestamp-driven
``rec{N}.bag/img/{cam}/image.mp4`` (1:1 with the ``*_image_raw_compressed.mat``
timestamps — the correct video). The old day-level sidecars named
``behave.<day>.<idx>.<cam>.avi`` are constant-fps, unreliable, and redundant with
the mp4s. This is a cleanup step to remove them once the mp4s are in place; the
video is always reconstructable from the behave files, so this deletes no data.

Safety: dry-run by default; ``delete_sidecar_avi`` only removes a sidecar when the
matching per-rec ``image.mp4`` exists (unless ``require_mp4=False``), so it never
removes the only copy of a camera's video for a rec that wasn't rebuilt.

CLI:
    python -m py_proc.cleanup_sidecars --day-dir DAY [--rec 001] [--apply]
    python -m py_proc.cleanup_sidecars --day-dir DAY --apply --no-require-mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def _mp4_for(day_dir: Path, idx: str, cam: str) -> Path:
    """Per-rec rebuilt video that replaces the ``<idx>`` / ``<cam>`` sidecar."""
    rec = idx.zfill(3)
    return day_dir / rec / f"rec{rec}.bag" / "img" / cam / "image.mp4"


def find_sidecar_avi(day_dir, rec: Optional[str] = None) -> List[Path]:
    """Day-level ``behave.<day>.<idx>.<cam>.avi`` sidecars, optionally one rec.

    ``rec`` may be given as ``'001'`` or ``'1'`` (matched on the numeric ``idx``).
    """
    day_dir = Path(day_dir)
    rec_int = int(rec) if rec is not None else None
    out: List[Path] = []
    for p in sorted(day_dir.glob("behave.*.avi")):
        parts = p.name.split(".")  # ['behave', day, idx, 'Cam X', 'avi']
        if len(parts) != 5 or parts[0] != "behave" or parts[-1] != "avi":
            continue
        idx = parts[2]
        if rec_int is not None:
            try:
                if int(idx) != rec_int:
                    continue
            except ValueError:
                continue
        out.append(p)
    return out


def delete_sidecar_avi(
    day_dir,
    rec: Optional[str] = None,
    dry_run: bool = True,
    require_mp4: bool = True,
) -> dict:
    """Delete day-level sidecar .avi files (dry-run by default).

    Returns a summary dict: ``candidates``, ``deleted``, ``skipped_no_mp4``,
    ``bytes_deleted``, ``dry_run``. With ``require_mp4`` (default) a sidecar is
    skipped when its per-rec ``image.mp4`` is missing.
    """
    day_dir = Path(day_dir)
    candidates = find_sidecar_avi(day_dir, rec)
    deleted: List[str] = []
    skipped: List[str] = []
    freed = 0
    for p in candidates:
        parts = p.name.split(".")
        idx, cam = parts[2], parts[3]
        if require_mp4 and not _mp4_for(day_dir, idx, cam).exists():
            skipped.append(str(p))
            continue
        size = p.stat().st_size
        if not dry_run:
            p.unlink()
        deleted.append(str(p))
        freed += size
    return {
        "day_dir": str(day_dir),
        "candidates": [str(p) for p in candidates],
        "deleted": deleted,
        "skipped_no_mp4": skipped,
        "bytes_deleted": freed,
        "dry_run": dry_run,
    }


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Delete legacy day-level sidecar .avi files")
    ap.add_argument("--day-dir", required=True)
    ap.add_argument("--rec", default=None, help="Only this rec (e.g. 001); default all")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (default is a dry-run preview)")
    ap.add_argument("--no-require-mp4", action="store_true",
                    help="Delete even if the replacement image.mp4 is missing")
    args = ap.parse_args(argv)

    res = delete_sidecar_avi(
        args.day_dir, rec=args.rec, dry_run=not args.apply,
        require_mp4=not args.no_require_mp4,
    )
    verb = "Deleted" if args.apply else "Would delete"
    print(f"{verb} {len(res['deleted'])} sidecar .avi ({_fmt_gb(res['bytes_deleted'])}) "
          f"in {res['day_dir']}")
    for f in res["deleted"]:
        print(f"  {verb.lower()}: {f}")
    if res["skipped_no_mp4"]:
        print(f"Skipped {len(res['skipped_no_mp4'])} (no matching image.mp4 — "
              f"use --no-require-mp4 to force):")
        for f in res["skipped_no_mp4"]:
            print(f"  skipped: {f}")
    if not args.apply and res["deleted"]:
        print("\nDry-run only. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
