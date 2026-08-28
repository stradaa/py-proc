from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import traceback

from scipy.io import loadmat

# Skip video extraction
# python run_day_pipeline.py --day-dir "/vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260323" --skip-video

# Skip redoing video if camera outputs already exist
# python run_day_pipeline.py --day-dir "/vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260323" --skip-existing-video

# Only do extraction + no-display Events pass + save_trials
# python run_day_pipeline.py --day-dir "/vol/cortex/cd4/pesaranlab/Bowser_Behavior_AlexRig/260323" --no-display

# Process only rec005 for that day
# python run_day_pipeline.py --day-dir "dir" --rec 005 --skip-video

# Full path also works
# python run_day_pipeline.py --day-dir "dir" --rec 005

def _bootstrap_local_package(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "py_proc",
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to bootstrap local package from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["py_proc"] = module
    spec.loader.exec_module(module)
    return module


def _events_trial_count(events_file: Path) -> int:
    if not events_file.exists():
        return 0
    data = loadmat(events_file, simplify_cells=True)
    events = data.get("Events", {})
    if not events:
        return 0
    trial = events.get("Trial", [])
    try:
        return len(trial)
    except TypeError:
        return 0


def _safe(failures: list[str], what: str, fn, *fn_args, **fn_kwargs) -> bool:
    """Run one processing stage, recording failures instead of aborting the day.

    Which nodes were plugged in that day should not decide whether the rest of
    the pipeline runs, so a stage that raises is reported and skipped.
    """
    try:
        fn(*fn_args, **fn_kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {what}: FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures.append(f"{what}: {type(exc).__name__}: {exc}")
        return False


_ALL_STEPS = frozenset({"extract", "events_nodisplay", "detect_display", "full_proc", "save_trials", "aruco_calib"})


def _run_aruco_calib(repo_root: Path, day_dir: Path, recs: list[str], *, make_plots: bool) -> None:
    """Run the pyPose ArUco calibration per rec (independent of trial count).

    pyPose lives at the actual repo root (parent of this py_proc package dir), so we
    put that on sys.path before importing. Failures are reported per rec and never
    abort the rest of the pipeline.
    """
    repo_top = repo_root.parent
    if str(repo_top) not in sys.path:
        sys.path.insert(0, str(repo_top))
    try:
        from pyPose.run_calibration import aruco_calibrate_rec
    except Exception as exc:  # noqa: BLE001
        print(f"[aruco] skipped — could not import pyPose: {exc}")
        return
    for rec in recs:
        try:
            aruco_calibrate_rec(day_dir, rec, make_plots=make_plots)
        except Exception as exc:  # noqa: BLE001
            print(f"[aruco] {day_dir.name}/{rec}: FAILED — {exc}")

    # Day-level drift monitor. Needs >=2 calibrated recs, so it is skipped
    # silently on a single-rec run; it reads whatever is already on disk rather
    # than only the recs processed just now.
    try:
        from pyPose.drift import print_report
        print_report(day_dir, make_plot=make_plots)
    except Exception as exc:  # noqa: BLE001
        print(f"[aruco] drift report FAILED — {exc}")


def run_day_pipeline(
    day_dir: str | Path,
    skip_video: bool = False,
    skip_existing_video: bool = False,
    no_display: bool = False,
    rec: str | None = None,
    steps: set[str] | None = None,
    skip_aruco: bool = False,
    aruco_plots: bool = True,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent
    day_dir = Path(day_dir).expanduser().resolve()
    monkeydir = day_dir.parent
    day = day_dir.name

    if not day_dir.exists():
        raise FileNotFoundError(f"Day directory not found: {day_dir}")

    if steps is None:
        run_steps = set(_ALL_STEPS)
        if no_display:
            run_steps -= {"detect_display", "full_proc"}
        if skip_aruco:
            run_steps.discard("aruco_calib")
    else:
        run_steps = set(steps)

    extract_cmd = [sys.executable, str(repo_root / "procThalamus_indie.py"), "-d", str(day_dir)]
    if skip_video:
        extract_cmd.append("--skip-video")
    if skip_existing_video:
        extract_cmd.append("--skip-existing-video")

    if "extract" in run_steps:
        print(f"\n=== Step 1: Extract raw day data from {day_dir} ===")
        subprocess.run(extract_cmd, check=True, cwd=repo_root)

    _bootstrap_local_package(repo_root)

    from py_proc.detect_display_states import detect_display_states
    from py_proc.helpers import get_recs
    from py_proc.proc_events import proc_events
    from py_proc.proc_eye import proc_eye
    from py_proc.proc_hand import proc_hand
    from py_proc.proc_reach import proc_reach
    from py_proc.proc_saccade import proc_saccade
    from py_proc.save_trials import save_trials

    recs: list[str] = []
    if run_steps - {"extract"}:
        recs = get_recs(str(day_dir))
        if not recs:
            raise RuntimeError(f"No rec folders found under {day_dir}")
        if rec is not None:
            rec = str(rec).zfill(3)
            if rec not in recs:
                raise RuntimeError(f"Requested rec {rec} not found under {day_dir}. Available: {', '.join(recs)}")
            recs = [rec]

    failures: list[str] = []

    if "events_nodisplay" in run_steps:
        print(f"\n=== Step 2: No-display Events pass for {len(recs)} rec(s) ===")
        for rec in recs:
            _safe(failures, f"rec{rec} proc_events(no-display)",
                  proc_events, day, rec, str(monkeydir), use_display=False)

    if "detect_display" in run_steps:
        print(f"\n=== Step 3: Detect display states ===")
        _safe(failures, "detect_display_states", detect_display_states, day, str(monkeydir))

    if "full_proc" in run_steps:
        print(f"\n=== Step 4: Full per-rec processing ===")
        for rec in recs:
            if not _safe(failures, f"rec{rec} proc_events",
                         proc_events, day, rec, str(monkeydir), use_display=True):
                continue
            n_trials = _events_trial_count(day_dir / rec / f"rec{rec}.Events.mat")
            if n_trials == 0:
                print(f"rec{rec}: 0 trials after proc_events, skipping eye/saccade/hand/reach")
                continue
            # Each modality is independent: a missing node (eye tracker,
            # touchscreen) must not stop the others from being processed.
            for name, fn in (("proc_eye", proc_eye), ("proc_saccade", proc_saccade),
                             ("proc_hand", proc_hand), ("proc_reach", proc_reach)):
                _safe(failures, f"rec{rec} {name}", fn, day, rec, str(monkeydir))

    if "aruco_calib" in run_steps:
        print(f"\n=== Step: ArUco 3-camera calibration ===")
        _run_aruco_calib(repo_root, day_dir, recs, make_plots=aruco_plots)

    if "save_trials" in run_steps:
        print(f"\n=== Step 5: Aggregate trials ===")
        _safe(failures, "save_trials", save_trials, day, str(monkeydir))

    if failures:
        print(f"\nPipeline finished with {len(failures)} failed stage(s):")
        for item in failures:
            print(f"  - {item}")
    else:
        print("\nPipeline complete.")
    print(f"Day directory: {day_dir}")
    if recs:
        print(f"Recs processed: {', '.join(recs)}")
        print(f"Events path example: {day_dir / recs[0] / f'rec{recs[0]}.Events.mat'}")
    print(f"AllTrials path: {day_dir / 'mat' / 'AllTrials.mat'}")
    print(f"Trials path: {day_dir / 'mat' / 'Trials.mat'}")
    return {
        "day_dir": str(day_dir),
        "recs_processed": recs,
        "events_example": str(day_dir / recs[0] / f"rec{recs[0]}.Events.mat") if recs else "",
        "all_trials_path": str(day_dir / "mat" / "AllTrials.mat"),
        "trials_path": str(day_dir / "mat" / "Trials.mat"),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full py-proc pipeline for one day directory"
    )
    parser.add_argument(
        "--day-dir",
        required=True,
        help="Full path or relative path to the day directory containing behave.* files",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Pass --skip-video through to procThalamus_indie.py",
    )
    parser.add_argument(
        "--skip-existing-video",
        action="store_true",
        help="Pass --skip-existing-video through to procThalamus_indie.py",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip display correction and only run the no-display Events pass",
    )
    parser.add_argument(
        "--rec",
        default=None,
        help="Optional single rec to process (e.g. 005). By default all recs in the day are processed.",
    )
    parser.add_argument(
        "--skip-aruco",
        action="store_true",
        help="Skip the ArUco 3-camera calibration step",
    )
    parser.add_argument(
        "--no-aruco-plots",
        action="store_true",
        help="Run ArUco calibration but skip its QC/3D figures",
    )
    args = parser.parse_args()

    run_day_pipeline(
        day_dir=args.day_dir,
        skip_video=args.skip_video,
        skip_existing_video=args.skip_existing_video,
        no_display=args.no_display,
        rec=args.rec,
        skip_aruco=args.skip_aruco,
        aruco_plots=not args.no_aruco_plots,
    )


if __name__ == "__main__":
    main()
