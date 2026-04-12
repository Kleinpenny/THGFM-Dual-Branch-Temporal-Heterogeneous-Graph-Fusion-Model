#!/usr/bin/env python3
"""
Run a JSONL task manifest with subprocess workers (no systemd, no sudo).

GPU assignment matches run_systemd_task_pool: pick the GPU with
the fewest running jobs (tie-break lower index). With max_per_gpu=1 and max_running=len(gpus),
each GPU runs at most one job — workloads spread evenly across GPUs.

Manifest format: same as run_systemd_task_pool.py (unit_name, workdir, log_file, setenv, command).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    unit_name: str
    log_file: Path
    command: list[str]
    workdir: Path
    setenv: dict[str, str]


@dataclass(frozen=True)
class Running:
    popen: subprocess.Popen
    task: Task
    gpu: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-GPU subprocess pool for JSONL task manifests.")
    p.add_argument("--manifest", required=True, help="Task manifest JSONL.")
    p.add_argument("--project-dir", default="", help="Default workdir parent if manifest paths are relative.")
    p.add_argument("--gpu-ids", nargs="+", default=["0", "1", "2", "3"], help="Physical GPU indices.")
    p.add_argument("--max-running", type=int, default=0, help="Max concurrent jobs (0 = len(gpu_ids)).")
    p.add_argument("--max-per-gpu", type=int, default=1, help="Max concurrent jobs per GPU.")
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--cuda-arg", default="--cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def infer_default_project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def read_manifest(path: Path, default_project_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            missing = [k for k in ("unit_name", "log_file", "command") if k not in item]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {missing}")
            command = item["command"]
            if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
                raise ValueError(f"{path}:{line_no}: command must be string list")

            workdir = Path(item.get("workdir", str(default_project_dir)))
            if not workdir.is_absolute():
                workdir = default_project_dir / workdir
            log_file = Path(item["log_file"])
            if not log_file.is_absolute():
                log_file = workdir / log_file
            setenv = item.get("setenv", {})
            if not isinstance(setenv, dict):
                raise ValueError(f"{path}:{line_no}: setenv must be object")
            tasks.append(
                Task(
                    unit_name=str(item["unit_name"]),
                    log_file=log_file,
                    command=command,
                    workdir=workdir,
                    setenv={str(k): str(v) for k, v in setenv.items()},
                )
            )
    if not tasks:
        raise ValueError(f"{path}: no tasks found")
    return tasks


def set_cuda_arg(command: list[str], cuda_arg: str, gpu_id: str) -> list[str]:
    updated = list(command)
    for i in range(len(updated) - 1):
        if updated[i] == cuda_arg:
            updated[i + 1] = str(gpu_id)
            return updated
    raise ValueError(f"command missing required CUDA arg {cuda_arg!r}: {shlex.join(command)}")


def next_available_gpu(gpu_ids: list[str], running_per_gpu: dict[str, int], max_per_gpu: int) -> str | None:
    cand = [g for g in gpu_ids if running_per_gpu.get(g, 0) < max_per_gpu]
    if not cand:
        return None
    return min(cand, key=lambda g: (running_per_gpu.get(g, 0), int(g) if g.isdigit() else g))


def launch_one(task: Task, gpu: str, cuda_arg: str) -> subprocess.Popen:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = set_cuda_arg(list(task.command), cuda_arg, gpu)
    env = os.environ.copy()
    env.update(task.setenv)
    log_f = task.log_file.open("a", encoding="utf-8", buffering=1)
    print(f"[launch] {task.unit_name} gpu={gpu}")
    print(f"         log: {task.log_file}")
    print(f"         cmd: {shlex.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=str(task.workdir),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    default_project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else infer_default_project_dir()
    gpu_ids = [str(x) for x in args.gpu_ids]
    max_running = int(args.max_running) if args.max_running > 0 else len(gpu_ids)
    if args.max_per_gpu <= 0 or args.poll_seconds <= 0:
        raise SystemExit("--max-per-gpu and --poll-seconds must be positive")
    if not gpu_ids:
        raise SystemExit("--gpu-ids requires at least one id")

    tasks = read_manifest(manifest_path, default_project_dir)
    running_per_gpu = {g: 0 for g in gpu_ids}
    pending = list(tasks)
    running: dict[str, Running] = {}
    finished_ok: list[str] = []
    finished_fail: list[str] = []

    print(f"[pool] manifest={manifest_path}")
    print(f"[pool] total_tasks={len(tasks)} max_running={max_running}")
    print(f"[pool] gpu_ids={','.join(gpu_ids)} max_per_gpu={args.max_per_gpu}")

    if args.dry_run:
        for t in tasks:
            print(f"[dry-run] {t.unit_name}: {shlex.join(t.command)}")
        return 0

    try:
        while pending or running:
            while pending and len(running) < max_running:
                gpu = next_available_gpu(gpu_ids, running_per_gpu, args.max_per_gpu)
                if gpu is None:
                    break
                task = pending.pop(0)
                popen = launch_one(task, gpu, args.cuda_arg)
                running[task.unit_name] = Running(popen=popen, task=task, gpu=gpu)
                running_per_gpu[gpu] += 1

            finished_now: list[str] = []
            for unit_name, item in list(running.items()):
                code = item.popen.poll()
                if code is None:
                    continue
                running_per_gpu[item.gpu] -= 1
                if code == 0:
                    finished_ok.append(unit_name)
                    print(f"[done] {unit_name} gpu={item.gpu} exit=0")
                else:
                    finished_fail.append(unit_name)
                    print(f"[fail] {unit_name} gpu={item.gpu} exit={code}")
                finished_now.append(unit_name)
            for name in finished_now:
                running.pop(name, None)

            if running or pending:
                slots = " ".join(f"{g}:{running_per_gpu[g]}/{args.max_per_gpu}" for g in gpu_ids)
                print(
                    f"[pool] running={len(running)} pending={len(pending)} "
                    f"ok={len(finished_ok)} fail={len(finished_fail)} | gpu_slots {slots}"
                )
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[pool] interrupted; terminating children...")
        for _name, item in list(running.items()):
            if item.popen.poll() is None:
                item.popen.terminate()
        time.sleep(2)
        for _name, item in list(running.items()):
            if item.popen.poll() is None:
                item.popen.kill()
        return 130

    print(f"[summary] finished={len(finished_ok) + len(finished_fail)}/{len(tasks)} ok={len(finished_ok)} fail={len(finished_fail)}")
    if finished_fail:
        for name in finished_fail:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
