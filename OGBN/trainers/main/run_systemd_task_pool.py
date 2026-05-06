#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONDA_SH = Path("~/anaconda3/etc/profile.d/conda.sh")
DEFAULT_SYSTEMD_USER = os.environ.get("USER", "nobody")
DEFAULT_SYSTEMD_SLICE = f"python-{DEFAULT_SYSTEMD_USER}.slice"


@dataclass(frozen=True)
class Task:
    unit_name: str
    log_file: Path
    command: list[str]
    workdir: Path
    setenv: dict[str, str]


@dataclass(frozen=True)
class ScheduledTask:
    task: Task
    assigned_gpu: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a systemd-backed task pool for OGBN.")
    p.add_argument("--manifest", required=True, help="Task manifest JSONL.")
    p.add_argument("--max-running", type=int, default=8)
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    p.add_argument("--conda-sh", default=str(DEFAULT_CONDA_SH))
    p.add_argument("--conda-env", default="thgfm")
    p.add_argument("--systemd-slice", default=DEFAULT_SYSTEMD_SLICE)
    p.add_argument("--systemd-user", default=DEFAULT_SYSTEMD_USER)
    p.add_argument(
        "--inherit-env",
        nargs="*",
        default=["WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"],
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-ids", nargs="+", default=None)
    p.add_argument("--max-per-gpu", type=int, default=0)
    p.add_argument("--cuda-arg", default="--cuda")
    p.add_argument("--max-load1", type=float, default=0.0)
    p.add_argument("--max-gpu-util", type=float, default=0.0)
    return p.parse_args()


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def read_manifest(path: Path, default_project_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
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


def get_unit_properties(unit_name: str) -> dict[str, str]:
    out = run_command(
        [
            "systemctl",
            "show",
            f"{unit_name}.service",
            "--property=ActiveState,SubState,Result,ExecMainStatus",
        ],
        capture=True,
    )
    props: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v
    return props


def is_unit_running(unit_name: str) -> bool:
    props = get_unit_properties(unit_name)
    return props.get("ActiveState") in {"active", "activating"}


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


def launch_task(
    scheduled: ScheduledTask,
    *,
    conda_sh: Path,
    conda_env: str,
    systemd_slice: str,
    systemd_user: str,
    inherited_env: dict[str, str],
    dry_run: bool,
) -> None:
    task = scheduled.task
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sudo",
        "systemd-run",
        f"--unit={task.unit_name}",
        f"--slice={systemd_slice}",
        f"--property=User={systemd_user}",
        f"--property=WorkingDirectory={task.workdir}",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=MemoryAccounting=yes",
    ]
    merged_env = dict(inherited_env)
    merged_env.update(task.setenv)
    for k, v in merged_env.items():
        cmd.append(f"--setenv={k}={v}")
    shell_cmd = (
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"exec {shlex.join(task.command)} >> {shlex.quote(str(task.log_file))} 2>&1"
    )
    cmd.extend(["/bin/bash", "-lc", shell_cmd])
    print(f"[launch] {task.unit_name} gpu={scheduled.assigned_gpu}")
    print(f"         cmd: {shlex.join(task.command)}")
    print(f"         log: {task.log_file}")
    if dry_run:
        print(f"         dry-run systemd-run: {shlex.join(cmd)}")
        return
    run_command(cmd)


def summarize_finished(task: Task) -> tuple[bool, str]:
    props = get_unit_properties(task.unit_name)
    active = props.get("ActiveState", "unknown")
    result = props.get("Result", "unknown")
    code = props.get("ExecMainStatus", "unknown")
    ok = active == "inactive" and result == "success" and code == "0"
    return ok, f"{task.unit_name}: ActiveState={active} Result={result} Exit={code}"


def can_launch_by_load(max_load1: float) -> tuple[bool, float]:
    load1 = os.getloadavg()[0]
    if max_load1 <= 0:
        return True, load1
    return load1 <= max_load1, load1


def get_gpu_utils(gpu_ids: list[str]) -> dict[str, float]:
    out = run_command(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        capture=True,
    )
    util_map: dict[str, float] = {}
    wanted = set(gpu_ids)
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        idx, util = parts
        if idx not in wanted:
            continue
        try:
            util_map[idx] = float(util)
        except ValueError:
            continue
    return util_map


def can_launch_by_gpu(max_gpu_util: float, gpu_ids: list[str]) -> tuple[bool, dict[str, float]]:
    if max_gpu_util <= 0:
        return True, {}
    util_map = get_gpu_utils(gpu_ids)
    for g in gpu_ids:
        if util_map.get(g, 0.0) > max_gpu_util:
            return False, util_map
    return True, util_map


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    default_project_dir = Path(args.project_dir).expanduser().resolve()
    conda_sh = Path(args.conda_sh).expanduser().resolve()
    if args.max_running <= 0 or args.poll_seconds <= 0:
        raise SystemExit("--max-running and --poll-seconds must be positive")
    if (args.gpu_ids is None and args.max_per_gpu > 0) or (args.gpu_ids is not None and args.max_per_gpu <= 0):
        raise SystemExit("dynamic GPU scheduling requires --gpu-ids with --max-per-gpu > 0")

    tasks = read_manifest(manifest_path, default_project_dir)
    inherited_env = {k: os.environ[k] for k in args.inherit_env if k in os.environ}
    dynamic_gpu = args.gpu_ids is not None and args.max_per_gpu > 0
    gpu_ids = [str(x) for x in (args.gpu_ids or [])]
    running_per_gpu = {g: 0 for g in gpu_ids}
    pending = list(tasks)
    running: dict[str, ScheduledTask] = {}
    finished_ok: list[str] = []
    finished_fail: list[str] = []

    print(f"[pool] manifest={manifest_path}")
    print(f"[pool] total_tasks={len(tasks)} max_running={args.max_running}")
    if dynamic_gpu:
        print(f"[pool] gpu_ids={','.join(gpu_ids)} max_per_gpu={args.max_per_gpu}")

    try:
        while pending or running:
            while pending and len(running) < args.max_running:
                ok_load, load1 = can_launch_by_load(args.max_load1)
                if not ok_load:
                    print(f"[pool] load guard: load1={load1:.2f} > {args.max_load1:.2f}")
                    break
                ok_gpu, util_map = can_launch_by_gpu(args.max_gpu_util, gpu_ids)
                if not ok_gpu:
                    util_txt = " ".join(f"{g}:{util_map.get(g, 0.0):.1f}%" for g in gpu_ids)
                    print(f"[pool] gpu guard: {util_txt} > max_gpu_util={args.max_gpu_util:.1f}")
                    break

                task = pending.pop(0)
                assigned_gpu = ""
                launch_task_obj = task
                if dynamic_gpu:
                    next_gpu = next_available_gpu(gpu_ids, running_per_gpu, args.max_per_gpu)
                    if next_gpu is None:
                        pending.insert(0, task)
                        break
                    launch_task_obj = Task(
                        unit_name=task.unit_name,
                        log_file=task.log_file,
                        command=set_cuda_arg(task.command, args.cuda_arg, next_gpu),
                        workdir=task.workdir,
                        setenv=task.setenv,
                    )
                    assigned_gpu = next_gpu
                scheduled = ScheduledTask(task=launch_task_obj, assigned_gpu=assigned_gpu)
                launch_task(
                    scheduled,
                    conda_sh=conda_sh,
                    conda_env=args.conda_env,
                    systemd_slice=args.systemd_slice,
                    systemd_user=args.systemd_user,
                    inherited_env=inherited_env,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    running[task.unit_name] = scheduled
                    if dynamic_gpu:
                        running_per_gpu[assigned_gpu] += 1

            if args.dry_run:
                break

            finished_now: list[str] = []
            for unit_name, scheduled in running.items():
                if not is_unit_running(unit_name):
                    ok, msg = summarize_finished(scheduled.task)
                    if ok:
                        finished_ok.append(msg)
                        print(f"[done] {msg}")
                    else:
                        finished_fail.append(msg)
                        print(f"[fail] {msg}")
                    if dynamic_gpu:
                        running_per_gpu[scheduled.assigned_gpu] -= 1
                    finished_now.append(unit_name)
            for name in finished_now:
                running.pop(name, None)

            if pending and len(running) < args.max_running:
                continue
            if running:
                print(f"[pool] running={len(running)} pending={len(pending)} done={len(finished_ok)} fail={len(finished_fail)}")
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[pool] interrupted")
        return 130

    print(f"[summary] finished={len(finished_ok) + len(finished_fail)}/{len(tasks)} ok={len(finished_ok)} fail={len(finished_fail)}")
    return 1 if finished_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
