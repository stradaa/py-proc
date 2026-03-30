from pathlib import Path
import argparse
import json

from joystick_validation import (
    build_validation_report,
    parse_trial_tokens,
    render_trial_replay_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate joystick alignment and render replay videos")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--day", required=True)
    parser.add_argument("--rec", default="001")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sample-trials", nargs="*", default=None)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--render-trials", nargs="*", default=None)
    parser.add_argument("--render-out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier; 1.0 = real-time")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sample_trials = parse_trial_tokens(args.sample_trials)
    render_trials = parse_trial_tokens(args.render_trials)

    if not args.skip_report:
        summary = build_validation_report(
            repo_root=repo_root,
            day=args.day,
            rec=args.rec,
            out_dir=args.out_dir,
            sample_trials=sample_trials,
        )
        print(json.dumps(summary, indent=2))

    if render_trials:
        render_out = (
            Path(args.render_out)
            if args.render_out is not None
            else repo_root / "pyCheck" / "output" / args.day / f"trial_replay_{args.day}_{args.rec}.mp4"
        )
        out_path = render_trial_replay_video(
            repo_root=repo_root,
            day=args.day,
            rec=args.rec,
            trial_numbers=render_trials,
            out_path=render_out,
            fps=args.fps,
            playback_speed=args.speed,
        )
        print(f"Saved replay video to {out_path}")


if __name__ == "__main__":
    main()
