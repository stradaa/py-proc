from pathlib import Path
import argparse
import json
import sys

from joystick_validation import build_validation_report

# python pyCheck/run_joystick_validation.py --day 260324 --sample-trials 1 2 3 4 5

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Validate joystick.mat alignment against AllTrials/Trials")
#     parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
#     parser.add_argument("--day", required=True)
#     parser.add_argument("--rec", default="001")
#     parser.add_argument("--out-dir", default=None)
#     parser.add_argument("--sample-trials", nargs="*", type=int, default=None)
#     args = parser.parse_args()

#     summary = build_validation_report(
#         repo_root=args.repo_root,
#         day=args.day,
#         rec=args.rec,
#         out_dir=args.out_dir,
#         sample_trials=args.sample_trials,
#     )
#     print(json.dumps(summary, indent=2)) 


# if __name__ == "__main__":
#     main()

summary = build_validation_report(
    repo_root=Path(""),
    day="260324",
    rec="001",
    out_dir=None,
    sample_trials=[1, 2, 3, 4, 5],
)

print(summary)
