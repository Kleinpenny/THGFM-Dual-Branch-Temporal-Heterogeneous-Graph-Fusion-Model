#!/usr/bin/env python3
"""
Run a JSONL task manifest (same schema as run_systemd_task_pool.py) using subprocess,
with optional multi-GPU placement (--gpu-ids / --max-per-gpu).

Use this when systemd-run is unavailable or you want a foreground pool without sudo.
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
    popen: subprocess.Popen[bytes]
    task: Task
    gpu: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subprocess task pool for HTAG manifest JSONL.")
    p.add_argument("--manifest", required=True, help="Task manifest JSONL.")
    p.add_argument("--project-dir", default="", help="Default workdir if manifest omits it.")
    p.add_argument("--max-running", type=int, default=4)
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--gpu-ids", nargs="+", default=["0"])
    p.add_argument(
        "--max-per-gpu",
        type=int,
        default=0,
        help="Max concurrent jobs per GPU (required >0; typically match --max-running for a single GPU).",
    )
    p.add_argument("--cuda-arg", default="--cuda")
    p.add_argument(
        "--inherit-env",
        nargs="*",
        default=["WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"],
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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
    names = [t.unit_name for t in tasks]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicated unit_name")
    return tasks


def set_cuda_arg(command: list[str], cuda_arg: str, gpu_id: str) -> list[str]:
    updated = list(command)
    for i in range(len(updated) - 1):
        if updated[i] == cuda_arg:
            updated[i + 1] = str(gpu_id)
            return updated
    raise ValueError(f"command missing {cuda_arg!r}: {shlex.join(command)}")


def next_available_gpu(gpu_ids: list[str], running_per_gpu: dict[str, int], max_per_gpu: int) -> str | None:
    cand = [g for g in gpu_ids if running_per_gpu.get(g, 0) < max_per_gpu]
    if not cand:
        return None
    return min(cand, key=lambda g: (running_per_gpu.get(g, 0), int(g) if g.isdigit() else g))


def launch_one(
    task: Task,
    gpu: str,
    cuda_arg: str,
    inherited: dict[str, str],
) -> subprocess.Popen[bytes]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = set_cuda_arg(task.command, cuda_arg, gpu)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(inherited)
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
    default_project_dir = (
        Path(args.project_dir).expanduser().resolve()
        if args.project_dir
        else Path(__file__).resolve().parents[2]
    )

    tasks = read_manifest(manifest_path, default_project_dir)
    inherited = {k: os.environ[k] for k in args.inherit_env if k in os.environ}

    if int(args.max_per_gpu) <= 0:
        raise SystemExit("--max-per-gpu must be positive (e.g. 4 for four jobs on GPU 0).")
    gpu_ids = [str(x) for x in args.gpu_ids]
    max_per_gpu = int(args.max_per_gpu)

    if args.max_running <= 0 or args.poll_seconds <= 0:
        raise SystemExit("--max-running and --poll-seconds must be positive")

    running_per_gpu = {g: 0 for g in gpu_ids}
    pending = list(tasks)
    running: dict[str, Running] = {}
    finished_ok: list[str] = []
    finished_fail: list[str] = []

    print(f"[pool] manifest={manifest_path}")
    print(f"[pool] total_tasks={len(tasks)} max_running={args.max_running}")
    print(f"[pool] gpu_ids={','.join(gpu_ids)} max_per_gpu={max_per_gpu}")

    if args.dry_run:
        for t in tasks:
            print(f"[dry-run] {t.unit_name}: {shlex.join(t.command)}")
        return 0

    try:
        while pending or running:
            while pending and len(running) < args.max_running:
                task = pending[0]
                gpu = next_available_gpu(gpu_ids, running_per_gpu, max_per_gpu)
                if gpu is None:
                    break
                launch_task = Task(
                    unit_name=task.unit_name,
                    log_file=task.log_file,
                    command=set_cuda_arg(task.command, args.cuda_arg, gpu),
                    workdir=task.workdir,
                    setenv=task.setenv,
                )

                pending.pop(0)
                popen = launch_one(launch_task, gpu, args.cuda_arg, inherited)
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
                status = " ".join(f"{g}:{running_per_gpu[g]}/{max_per_gpu}" for g in gpu_ids)
                print(
                    f"[pool] running={len(running)} pending={len(pending)} "
                    f"ok={len(finished_ok)} fail={len(finished_fail)} | gpu {status}"
                )
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[pool] interrupted; terminating children...")
        for _n, item in list(running.items()):
            if item.popen.poll() is None:
                item.popen.terminate()
        time.sleep(2)
        for _n, item in list(running.items()):
            if item.popen.poll() is None:
                item.popen.kill()
        return 130

    print(
        f"[summary] finished={len(finished_ok) + len(finished_fail)}/{len(tasks)} "
        f"ok={len(finished_ok)} fail={len(finished_fail)}"
    )
    return 1 if finished_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
