"""cross_day_cache.py — Load each rec once per run, and each day once per project.

The cross-day figures are built by four independent walkers (summary,
kinematics, co_structured, throughput).  Each used to take ``(repo_root,
selections)`` and re-open every rec for itself, so a selection cost four full
passes over the raw data.  Loading one rec takes ~0.5 s, over half of it
``json.loads`` on ``behav_results``, so that redundancy dominated the runtime.

Two caches fix that, at different timescales:

``DatasetCache``
    Within one run, holds ``(day, rec) -> JoystickDataset`` so the four builders
    share a single load.  Scoped to one day at a time by the caller, so peak
    memory stays at one day's recs rather than the whole selection.

``DayRowCache``
    Between runs, stores each day's four finished row dicts in ``out_dir``
    keyed by a fingerprint of the inputs that produced them.  Adding one day to
    a long selection then costs one day of work instead of all of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

CACHE_FILENAME = "cross_day_row_cache.json"
# Bump when a builder's output shape or semantics change, so stale rows from an
# older version are never reused.
CACHE_VERSION = 1


class DatasetCache:
    """``(day, rec) -> JoystickDataset`` for the span the caller keeps it open."""

    def __init__(self, loader: Optional[Callable[..., Any]] = None) -> None:
        if loader is None:
            from pyCheck.joystick_validation import load_joystick_dataset
            loader = load_joystick_dataset
        self._loader = loader
        self._cache: Dict[tuple, Any] = {}
        self.loads = 0
        self.hits = 0

    def __call__(self, repo_root, day: str, rec: str = "001"):
        key = (str(repo_root), str(day), str(rec))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.loads += 1
        dataset = self._loader(repo_root, day, rec)
        self._cache[key] = dataset
        return dataset

    def clear(self) -> None:
        self._cache.clear()


def _source_paths(repo_root: Path, day: str, recs: Sequence[int]) -> list[Path]:
    """Every file a day's builders read, so a change to any of them invalidates."""
    paths = [repo_root / day / "mat" / "AllTrials.mat"]
    for rec in recs:
        rec_str = f"{int(rec):03d}"
        rec_dir = repo_root / day / rec_str
        paths.append(rec_dir / f"rec{rec_str}.ev.mat")
        paths.append(rec_dir / f"rec{rec_str}.w_alignment.mat")
        paths.append(rec_dir / f"rec{rec_str}.bag" / "mat" / "joystick.mat")
    return paths


def fingerprint(
    repo_root: Path,
    day: str,
    recs: Sequence[int],
    task_types: Optional[Sequence[str]] = None,
) -> str:
    """Identity of a day's inputs: the recs, the task filter, and the size+mtime
    of every file read. A missing file contributes a distinct marker, so a day
    that gains a rec's data does not silently reuse the old row."""
    parts = [
        f"v{CACHE_VERSION}",
        f"recs={[int(r) for r in recs]}",
        f"tasks={sorted({str(t).strip().lower() for t in (task_types or []) if str(t).strip()})}",
    ]
    for path in _source_paths(repo_root, day, recs):
        try:
            st = path.stat()
            parts.append(f"{path.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{path.name}:missing")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


class DayRowCache:
    """Finished day rows from previous runs, keyed by input fingerprint."""

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / CACHE_FILENAME
        self._data: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict) and loaded.get("version") == CACHE_VERSION:
                self._data = loaded.get("days", {})
        except (OSError, ValueError):
            # An unreadable or stale-format cache is simply no cache.
            self._data = {}

    def get(self, day: str, fp: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(day)
        if entry is not None and entry.get("fingerprint") == fp:
            self.hits += 1
            return entry.get("rows")
        self.misses += 1
        return None

    def put(self, day: str, fp: str, rows: Dict[str, Any]) -> None:
        self._data[day] = {"fingerprint": fp, "rows": rows}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"version": CACHE_VERSION, "days": self._data}, fh)
        except OSError as exc:  # noqa: BLE001 - a cache that cannot be written is not fatal
            print(f"cross_day cache: could not write {self.path} ({exc})")
