from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from procThalamus_indie import RecordReader, get_capture_stem, get_rec_number, is_capturefile

# ./.venv/bin/python check_processability.py --day-dir <dir> --max-records 50000

def _scan_capture(path: Path, max_records: Optional[int]) -> Dict[str, Any]:
    counts = {
        "text_records": 0,
        "behavstate_records": 0,
        "task_json_records": 0,
        "other_json_records": 0,
    }
    nodes = set()
    samples: List[Dict[str, str]] = []

    with RecordReader(path, False, False) as reader:
        for i, record in enumerate(reader):
            body = record.WhichOneof("body")
            node = record.node
            nodes.add(node)

            if body == "text":
                counts["text_records"] += 1
                text = record.text.text
                if text.startswith("BehavState="):
                    counts["behavstate_records"] += 1
                    if len(samples) < 5:
                        samples.append({"kind": "BehavState", "sample": text[:200]})
                else:
                    try:
                        doc = json.loads(text)
                        if isinstance(doc, dict) and "task_config" in doc:
                            counts["task_json_records"] += 1
                            if len(samples) < 5:
                                samples.append({"kind": "task_json", "sample": str(sorted(doc.keys()))[:200]})
                        else:
                            counts["other_json_records"] += 1
                            if len(samples) < 5:
                                samples.append({"kind": "json", "sample": text[:200]})
                    except Exception:
                        if len(samples) < 5:
                            samples.append({"kind": "text", "sample": text[:200]})

            if max_records is not None and i + 1 >= max_records:
                break

    processable = counts["behavstate_records"] > 0 and counts["task_json_records"] > 0
    notes: List[str] = []
    if counts["text_records"] == 0:
        notes.append("no text stream found")
    elif counts["behavstate_records"] == 0:
        notes.append("text exists but no BehavState messages")
    elif counts["task_json_records"] == 0:
        notes.append("text exists but no task summary JSON")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "processable": processable,
        "counts": counts,
        "nodes": sorted(nodes),
        "samples": samples,
        "notes": notes,
    }


def _collect_recordings(day_dir: Path) -> Dict[str, Dict[str, Path]]:
    per_stem: Dict[str, Dict[str, Path]] = {}
    for path in sorted(day_dir.iterdir()):
        if not path.name.startswith("behave") or not path.is_file():
            continue
        if path.suffix in {".json", ".log", ".avi", ".mp4"}:
            continue
        if not is_capturefile(path):
            continue

        stem = get_capture_stem(path)
        entry = per_stem.setdefault(stem, {})
        if path.suffix == ".novideo":
            entry["novideo"] = path
        else:
            entry["full"] = path
    return per_stem


def _load_status(day_dir: Path, stem: str) -> str:
    json_path = day_dir / f"{stem}.json"
    if not json_path.exists():
        return ""
    try:
        with open(json_path, encoding="utf-8") as fh:
            return str(json.load(fh).get("status", ""))
    except Exception:
        return ""


def _sidecar_log_bytes(day_dir: Path, stem: str) -> Optional[int]:
    log_path = day_dir / f"{stem}.log"
    if not log_path.exists():
        return None
    return log_path.stat().st_size


def _print_human_summary(day_dir: Path, reports: List[Dict[str, Any]], skip_video: bool) -> None:
    print(f"Day: {day_dir}")
    print(f"Mode checked: {'skip-video selection' if skip_video else 'full-capture selection'}")
    print()
    for report in reports:
        print(f"Rec {report['rec']:>3}  stem={report['stem']}")
        if report["status"]:
            print(f"  Status: {report['status']}")
        if report["log_bytes"] is None:
            print("  Sidecar .log: missing")
        else:
            print(f"  Sidecar .log: {report['log_bytes']} bytes")

        for label in ("selected", "full", "novideo"):
            scan = report.get(label)
            if not scan:
                continue
            marker = " [pipeline input]" if label == "selected" else ""
            print(f"  {label}{marker}: {scan['path']}")
            print(
                "    processable={p} text={t} behavstate={b} task_json={j}".format(
                    p=scan["processable"],
                    t=scan["counts"]["text_records"],
                    b=scan["counts"]["behavstate_records"],
                    j=scan["counts"]["task_json_records"],
                )
            )
            if scan["notes"]:
                print(f"    notes: {'; '.join(scan['notes'])}")
            if scan["samples"]:
                preview = " | ".join(f"{s['kind']}: {s['sample']}" for s in scan["samples"][:2])
                print(f"    samples: {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether behave.* recordings contain the embedded trial text needed by the pipeline."
    )
    parser.add_argument("--day-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Mimic run_day_pipeline/procThalamus_indie selection when .novideo files exist.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on records scanned per file for a faster but non-exhaustive check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human summary.",
    )
    args = parser.parse_args()

    day_dir = args.day_dir.expanduser().resolve()
    recordings = _collect_recordings(day_dir)
    reports: List[Dict[str, Any]] = []

    for stem in sorted(recordings, key=lambda s: get_rec_number(recordings[s].get("novideo", recordings[s]["full"]))):
        files = recordings[stem]
        full_path = files.get("full")
        novideo_path = files.get("novideo")
        selected_path = novideo_path if args.skip_video and novideo_path is not None else full_path or novideo_path
        assert selected_path is not None

        report: Dict[str, Any] = {
            "stem": stem,
            "rec": f"{get_rec_number(selected_path):03}",
            "status": _load_status(day_dir, stem),
            "log_bytes": _sidecar_log_bytes(day_dir, stem),
        }
        report["selected"] = _scan_capture(selected_path, args.max_records)
        if full_path is not None and full_path != selected_path:
            report["full"] = _scan_capture(full_path, args.max_records)
        if novideo_path is not None and novideo_path != selected_path:
            report["novideo"] = _scan_capture(novideo_path, args.max_records)
        reports.append(report)

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        _print_human_summary(day_dir, reports, skip_video=args.skip_video)


if __name__ == "__main__":
    main()
