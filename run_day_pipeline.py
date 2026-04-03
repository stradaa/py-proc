from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

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


def run_day_pipeline(
    day_dir: str | Path,
    skip_video: bool = False,
    skip_existing_video: bool = False,
    no_display: bool = False,
    rec: str | None = None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent
    day_dir = Path(day_dir).expanduser().resolve()
    monkeydir = day_dir.parent
    day = day_dir.name

    if not day_dir.exists():
        raise FileNotFoundError(f"Day directory not found: {day_dir}")

    extract_cmd = [sys.executable, str(repo_root / "procThalamus_indie.py"), "-d", str(day_dir)]
    if skip_video:
        extract_cmd.append("--skip-video")
    if skip_existing_video:
        extract_cmd.append("--skip-existing-video")

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

    recs = get_recs(str(day_dir))
    if not recs:
        raise RuntimeError(f"No rec folders were created under {day_dir}")
    if rec is not None:
        rec = str(rec).zfill(3)
        if rec not in recs:
            raise RuntimeError(f"Requested rec {rec} not found under {day_dir}. Available recs: {', '.join(recs)}")
        recs = [rec]

    print(f"\n=== Step 2: No-display Events pass for {len(recs)} rec(s) ===")
    for rec in recs:
        proc_events(day, rec, str(monkeydir), use_display=False)

    if not no_display:
        print(f"\n=== Step 3: Detect display states ===")
        detect_display_states(day, str(monkeydir))

        print(f"\n=== Step 4: Full per-rec processing ===")
        for rec in recs:
            proc_events(day, rec, str(monkeydir), use_display=True)
            n_trials = _events_trial_count(day_dir / rec / f"rec{rec}.Events.mat")
            if n_trials == 0:
                print(f"rec{rec}: 0 trials after proc_events, skipping eye/saccade/hand/reach")
                continue
            proc_eye(day, rec, str(monkeydir))
            proc_saccade(day, rec, str(monkeydir))
            proc_hand(day, rec, str(monkeydir))
            proc_reach(day, rec, str(monkeydir))

    print(f"\n=== Step 5: Aggregate trials ===")
    save_trials(day, str(monkeydir))

    print("\nPipeline complete.")
    print(f"Day directory: {day_dir}")
    print(f"Recs processed: {', '.join(recs)}")
    print(f"Events path example: {day_dir / recs[0] / f'rec{recs[0]}.Events.mat'}")
    print(f"AllTrials path: {day_dir / 'mat' / 'AllTrials.mat'}")
    print(f"Trials path: {day_dir / 'mat' / 'Trials.mat'}")
    return {
        "day_dir": str(day_dir),
        "recs_processed": recs,
        "events_example": str(day_dir / recs[0] / f"rec{recs[0]}.Events.mat"),
        "all_trials_path": str(day_dir / "mat" / "AllTrials.mat"),
        "trials_path": str(day_dir / "mat" / "Trials.mat"),
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
    args = parser.parse_args()

    run_day_pipeline(
        day_dir=args.day_dir,
        skip_video=args.skip_video,
        skip_existing_video=args.skip_existing_video,
        no_display=args.no_display,
        rec=args.rec,
    )


if __name__ == "__main__":
    main()
