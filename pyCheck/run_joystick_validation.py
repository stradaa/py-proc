from pathlib import Path
import argparse
import json

from joystick_validation import (
    build_validation_report,
    parse_trial_tokens,
    render_trial_replay_video,
)


def _resolve_day_paths(args: argparse.Namespace) -> tuple[Path, str]:
    if args.day_dir is not None:
        day_dir = Path(args.day_dir).resolve()
        return day_dir.parent, day_dir.name

    if args.repo_root is None or args.day is None:
        raise ValueError("Provide either --day-dir or both --repo-root and --day.")

    return Path(args.repo_root).resolve(), str(args.day)


def _default_out_dir(repo_root: Path, day: str) -> Path:
    return repo_root / "claude" / "figures" / day / "beh"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate joystick alignment and render replay videos")
    parser.add_argument("--repo-root")
    parser.add_argument("--day")
    parser.add_argument("--day-dir")
    parser.add_argument("--rec", default="001")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sample-trials", nargs="*", default=None)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--render-trials", nargs="*", default=None)
    parser.add_argument("--render-out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier; 1.0 = real-time")
    args = parser.parse_args()

    repo_root, day = _resolve_day_paths(args)
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else _default_out_dir(repo_root, day)
    sample_trials = parse_trial_tokens(args.sample_trials)
    render_trials = parse_trial_tokens(args.render_trials)

    print(f"Using output directory: {out_dir}")

    if not args.skip_report:
        summary = build_validation_report(
            repo_root=repo_root,
            day=day,
            rec=args.rec,
            out_dir=out_dir,
            sample_trials=sample_trials,
        )
        print(json.dumps(summary, indent=2))

    if render_trials:
        render_out = (
            Path(args.render_out).resolve()
            if args.render_out is not None
            else out_dir / f"trial_replay_{day}_{args.rec}.mp4"
        )
        out_path = render_trial_replay_video(
            repo_root=repo_root,
            day=day,
            rec=args.rec,
            trial_numbers=render_trials,
            out_path=render_out,
            fps=args.fps,
            playback_speed=args.speed,
        )
        print(f"Saved replay video to {out_path}")


if __name__ == "__main__":
    main()
