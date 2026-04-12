#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVENT_H = "best_test_(highest_valid_acc)"
EVENT_L = "best_test_(lowest_valid_loss)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect OGBN task-pool best_test(H/L) ACC metrics.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="", help="CSV output path. Default: <manifest>_best_test_summary.csv")
    p.add_argument("--include-incomplete", action="store_true")
    return p.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def get_flag(command: list[str], flag: str, default: str = "") -> str:
    for i in range(len(command) - 1):
        if command[i] == flag:
            return command[i + 1]
    return default


def load_latest_events(metrics_path: Path) -> dict[str, dict[str, Any]]:
    picked: dict[str, dict[str, Any]] = {}
    if not metrics_path.exists():
        return picked
    with metrics_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            event = row.get("event")
            if event in {EVENT_H, EVENT_L}:
                picked[event] = row
    return picked


def default_output_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}_best_test_summary.csv")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(manifest_path)
    tasks = read_manifest(manifest_path)

    rows_out: list[dict[str, Any]] = []
    complete_count = 0
    for task in tasks:
        command = task.get("command", [])
        workdir = Path(task.get("workdir", ".")).expanduser()
        if not workdir.is_absolute():
            workdir = (manifest_path.parent / workdir).resolve()

        task_name = get_flag(command, "--task_name")
        conv_name = get_flag(command, "--conv_name")
        model_dir = get_flag(command, "--model_dir")
        seed = get_flag(command, "--seed")
        cuda = get_flag(command, "--cuda")

        metrics_path = workdir / model_dir / conv_name / "metrics" / f"{task_name}.jsonl"
        events = load_latest_events(metrics_path)
        row_h = events.get(EVENT_H, {})
        row_l = events.get(EVENT_L, {})
        has_h = bool(row_h)
        has_l = bool(row_l)
        complete = has_h and has_l
        if complete:
            complete_count += 1
        if not complete and not args.include_incomplete:
            continue

        rows_out.append(
            {
                "unit_name": task.get("unit_name", ""),
                "task_name": task_name,
                "conv_name": conv_name,
                "seed": seed,
                "cuda": cuda,
                "metrics_path": str(metrics_path),
                "has_best_test_h": int(has_h),
                "has_best_test_l": int(has_l),
                "best_test_h_acc": row_h.get("acc", ""),
                "best_test_h_timestamp": row_h.get("timestamp", ""),
                "best_test_l_acc": row_l.get("acc", ""),
                "best_test_l_timestamp": row_l.get("timestamp", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit_name",
        "task_name",
        "conv_name",
        "seed",
        "cuda",
        "metrics_path",
        "has_best_test_h",
        "has_best_test_l",
        "best_test_h_acc",
        "best_test_h_timestamp",
        "best_test_l_acc",
        "best_test_l_timestamp",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[collect] manifest={manifest_path}")
    print(f"[collect] total_tasks={len(tasks)} complete_best_test_hl={complete_count}")
    print(f"[collect] rows_written={len(rows_out)}")
    print(f"[collect] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
