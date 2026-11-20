"""Checks for the cross-day dataset/row caches.

Run from the py-proc repo root:   python tests/test_cross_day_cache.py

The row cache persists finished day rows between runs, so the thing that must
never happen is serving a stale row after its inputs changed.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyCheck.cross_day_cache import (
    CACHE_VERSION,
    DatasetCache,
    DayRowCache,
    fingerprint,
)


def _make_day(root: Path, day: str, recs) -> None:
    """Minimal tree of the files fingerprint() stats."""
    (root / day / "mat").mkdir(parents=True, exist_ok=True)
    (root / day / "mat" / "AllTrials.mat").write_bytes(b"all")
    for rec in recs:
        r = f"{rec:03d}"
        d = root / day / r
        (d / f"rec{r}.bag" / "mat").mkdir(parents=True, exist_ok=True)
        (d / f"rec{r}.ev.mat").write_bytes(b"ev")
        (d / f"rec{r}.w_alignment.mat").write_bytes(b"w")
        (d / f"rec{r}.bag" / "mat" / "joystick.mat").write_bytes(b"joy")


def test_dataset_cache_loads_each_rec_once():
    calls = []

    def loader(repo_root, day, rec):
        calls.append((day, rec))
        return {"day": day, "rec": rec}

    cache = DatasetCache(loader=loader)
    # Four builders asking for the same two recs is one load each.
    for _ in range(4):
        cache("/root", "261028", "001")
        cache("/root", "261028", "002")
    assert len(calls) == 2, calls
    assert cache.loads == 2 and cache.hits == 6, (cache.loads, cache.hits)

    # Clearing between days releases the datasets, so the next day reloads.
    cache.clear()
    cache("/root", "261028", "001")
    assert len(calls) == 3, calls
    print("PASS dataset cache loads each rec once, clears between days")


def test_fingerprint_tracks_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_day(root, "261028", [1, 2])
        base = fingerprint(root, "261028", [1, 2], [])

        assert fingerprint(root, "261028", [1, 2], []) == base, "must be stable"
        assert fingerprint(root, "261028", [1], []) != base, "rec selection matters"
        assert fingerprint(root, "261028", [1, 2], ["joystick_intro"]) != base, \
            "task filter changes the result, so it must change the key"

        # A rewritten source file invalidates, even at the same size.
        ev = root / "261028" / "001" / "rec001.ev.mat"
        st = ev.stat()
        os.utime(ev, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert fingerprint(root, "261028", [1, 2], []) != base, "mtime must invalidate"

        # So does one appearing where there was none.
        _make_day(root, "261029", [1])
        missing = fingerprint(root, "261029", [1, 2], [])
        _make_day(root, "261029", [1, 2])
        assert fingerprint(root, "261029", [1, 2], []) != missing, \
            "a rec whose data arrived later must not reuse the missing-file row"
    print("PASS fingerprint tracks recs, task filter and every source file")


def test_row_cache_round_trip_and_invalidation():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rows = {"summary": {"day": "261028", "n": 1}, "kinematics": None}

        c1 = DayRowCache(out)
        assert c1.get("261028", "fp1") is None
        c1.put("261028", "fp1", rows)
        c1.save()

        c2 = DayRowCache(out)
        assert c2.get("261028", "fp1") == rows, "must survive a round trip"
        assert c2.get("261028", "fp2") is None, "a changed fingerprint must miss"
        assert c2.get("261029", "fp1") is None, "another day must miss"

        # A cache written by a different code version is ignored, not trusted.
        path = out / "cross_day_row_cache.json"
        path.write_text(path.read_text().replace(
            f'"version": {CACHE_VERSION}', f'"version": {CACHE_VERSION + 1}'))
        assert DayRowCache(out).get("261028", "fp1") is None
    print("PASS row cache round-trips, and misses on changed inputs or version")


def test_corrupt_cache_is_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "cross_day_row_cache.json").write_text("{not json")
        assert DayRowCache(out).get("261028", "fp1") is None
    print("PASS a corrupt cache file degrades to no cache")


if __name__ == "__main__":
    test_dataset_cache_loads_each_rec_once()
    test_fingerprint_tracks_inputs()
    test_row_cache_round_trip_and_invalidation()
    test_corrupt_cache_is_not_fatal()
    print("\nALL TESTS PASSED")
