#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
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
    parser = argparse.ArgumentParser(
        description="Run a systemd-backed task pool and keep a fixed number of active jobs.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest path. Each line must contain unit_name, log_file, and command.",
    )
    parser.add_argument(
        "--max-running",
        type=int,
        default=15,
        help="Maximum number of active training jobs to keep running.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Polling interval for checking finished jobs.",
    )
    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_PROJECT_DIR),
        help="Default working directory for tasks when workdir is not set in manifest.",
    )
    parser.add_argument(
        "--conda-sh",
        default=str(DEFAULT_CONDA_SH),
        help="Path to conda.sh used before conda activate.",
    )
    parser.add_argument(
        "--conda-env",
        default="thgfm",
        help="Conda environment name.",
    )
    parser.add_argument(
        "--systemd-slice",
        default=DEFAULT_SYSTEMD_SLICE,
        help="Target systemd slice.",
    )
    parser.add_argument(
        "--systemd-user",
        default=DEFAULT_SYSTEMD_USER,
        help="User property passed to transient systemd services.",
    )
    parser.add_argument(
        "--inherit-env",
        nargs="*",
        default=["WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"],
        help="Environment variables copied from the scheduler process into every task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch commands without submitting them.",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help="GPU IDs used for dynamic scheduling, e.g. --gpu-ids 0 1 2 3.",
    )
    parser.add_argument(
        "--max-per-gpu",
        type=int,
        default=0,
        help="Max concurrent tasks per GPU. Enabled only when > 0 with --gpu-ids.",
    )
    parser.add_argument(
        "--cuda-arg",
        default="--cuda",
        help="Task command argument name for CUDA device ID replacement.",
    )
    parser.add_argument(
        "--max-load1",
        type=float,
        default=0.0,
        help="Pause launching new tasks when 1-min system load is above this value (0 disables).",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=float,
        default=0.0,
        help="Pause launching new tasks when any target GPU utilization(%) is above this value (0 disables).",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path, default_project_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: invalid JSON: {exc}") from exc

            missing = [key for key in ("unit_name", "log_file", "command") if key not in item]
            if missing:
                raise ValueError(
                    f"{manifest_path}:{line_no}: missing required field(s): {', '.join(missing)}"
                )

            command = item["command"]
            if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
                raise ValueError(f"{manifest_path}:{line_no}: command must be a string list")

            workdir_raw = item.get("workdir", str(default_project_dir))
            workdir = Path(workdir_raw)
            if not workdir.is_absolute():
                workdir = default_project_dir / workdir

            log_file = Path(item["log_file"])
            if not log_file.is_absolute():
                log_file = workdir / log_file

            setenv_raw = item.get("setenv", {})
            if not isinstance(setenv_raw, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in setenv_raw.items()
            ):
                raise ValueError(f"{manifest_path}:{line_no}: setenv must be a string-to-string object")

            tasks.append(
                Task(
                    unit_name=str(item["unit_name"]),
                    log_file=log_file,
                    command=command,
                    workdir=workdir,
                    setenv=setenv_raw,
                )
            )

    if not tasks:
        raise ValueError(f"{manifest_path}: no tasks found")

    unit_names = [task.unit_name for task in tasks]
    if len(unit_names) != len(set(unit_names)):
        raise ValueError(f"{manifest_path}: duplicated unit_name detected")

    return tasks


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def get_unit_properties(unit_name: str) -> dict[str, str]:
    result = run_command(
        [
            "systemctl",
            "show",
            f"{unit_name}.service",
            "--property=ActiveState,SubState,Result,ExecMainStatus",
        ],
        capture=True,
    )
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return props


def is_unit_running(unit_name: str) -> bool:
    props = get_unit_properties(unit_name)
    return props.get("ActiveState") in {"active", "activating"}


def set_cuda_arg(command: list[str], cuda_arg: str, gpu_id: str) -> list[str]:
    updated = list(command)
    for idx in range(len(updated) - 1):
        if updated[idx] == cuda_arg:
            updated[idx + 1] = str(gpu_id)
            return updated
    raise ValueError(f"command missing required CUDA arg {cuda_arg!r}: {shlex.join(command)}")


def next_available_gpu(gpu_ids: list[str], running_per_gpu: dict[str, int], max_per_gpu: int) -> str | None:
    candidates = [gpu for gpu in gpu_ids if running_per_gpu.get(gpu, 0) < max_per_gpu]
    if not candidates:
        return None
    return min(candidates, key=lambda gpu: (running_per_gpu.get(gpu, 0), int(gpu) if gpu.isdigit() else gpu))


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

    systemd_cmd = [
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
    for key, value in merged_env.items():
        systemd_cmd.append(f"--setenv={key}={value}")

    shell_cmd = (
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"exec {shlex.join(task.command)} >> {shlex.quote(str(task.log_file))} 2>&1"
    )
    systemd_cmd.extend(["/bin/bash", "-lc", shell_cmd])

    print(f"[launch] {task.unit_name} gpu={scheduled.assigned_gpu}")
    print(f"         log: {task.log_file}")
    print(f"         cmd: {shlex.join(task.command)}")
    if dry_run:
        print(f"         dry-run systemd-run: {shlex.join(systemd_cmd)}")
        return

    run_command(systemd_cmd)


def summarize_finished(task: Task) -> tuple[bool, str]:
    props = get_unit_properties(task.unit_name)
    active_state = props.get("ActiveState", "unknown")
    result = props.get("Result", "unknown")
    exit_code = props.get("ExecMainStatus", "unknown")
    ok = active_state == "inactive" and exit_code == "0" and result == "success"
    message = f"{task.unit_name}: ActiveState={active_state} Result={result} Exit={exit_code}"
    return ok, message


def can_launch_by_load(max_load1: float) -> tuple[bool, float]:
    load1 = os.getloadavg()[0]
    if max_load1 <= 0:
        return True, load1
    return load1 <= max_load1, load1


def get_gpu_utils(gpu_ids: list[str]) -> dict[str, float]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )
    util_map: dict[str, float] = {}
    wanted = set(gpu_ids)
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
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
    for gpu in gpu_ids:
        if util_map.get(gpu, 0.0) > max_gpu_util:
            return False, util_map
    return True, util_map


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    default_project_dir = Path(args.project_dir).expanduser().resolve()
    conda_sh = Path(args.conda_sh).expanduser().resolve()

    if args.max_running <= 0:
        raise SystemExit("--max-running must be positive")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if (args.gpu_ids is None and args.max_per_gpu > 0) or (
        args.gpu_ids is not None and args.max_per_gpu <= 0
    ):
        raise SystemExit("dynamic GPU scheduling requires both --gpu-ids and --max-per-gpu > 0")
    if args.max_per_gpu < 0:
        raise SystemExit("--max-per-gpu must be non-negative")
    if args.max_load1 < 0:
        raise SystemExit("--max-load1 must be non-negative")
    if args.max_gpu_util < 0:
        raise SystemExit("--max-gpu-util must be non-negative")

    tasks = read_manifest(manifest_path, default_project_dir)
    inherited_env = {key: os.environ[key] for key in args.inherit_env if key in os.environ}
    dynamic_gpu = args.gpu_ids is not None and args.max_per_gpu > 0
    gpu_ids = [str(gpu) for gpu in (args.gpu_ids or [])]
    running_per_gpu = {gpu: 0 for gpu in gpu_ids}

    pending = list(tasks)
    running: dict[str, ScheduledTask] = {}
    finished_ok: list[str] = []
    finished_fail: list[str] = []

    print(f"[pool] manifest={manifest_path}")
    print(f"[pool] total_tasks={len(tasks)} max_running={args.max_running}")
    if args.max_load1 > 0:
        print(f"[pool] launch guard: max_load1={args.max_load1}")
    if args.max_gpu_util > 0:
        print(f"[pool] launch guard: max_gpu_util={args.max_gpu_util}")
    if dynamic_gpu:
        print(f"[pool] dynamic_gpu=yes gpu_ids={','.join(gpu_ids)} max_per_gpu={args.max_per_gpu}")
        if args.max_running > len(gpu_ids) * args.max_per_gpu:
            print(
                f"[pool] warning: max_running={args.max_running} exceeds gpu capacity="
                f"{len(gpu_ids) * args.max_per_gpu}; effective concurrency will be capped by GPU slots."
            )

    try:
        while pending or running:
            while pending and len(running) < args.max_running:
                allow_launch, load1 = can_launch_by_load(args.max_load1)
                if not allow_launch:
                    print(
                        f"[pool] load guard active: load1={load1:.2f} > max_load1={args.max_load1:.2f}; "
                        "postpone new launches"
                    )
                    break
                allow_gpu, util_map = can_launch_by_gpu(args.max_gpu_util, gpu_ids)
                if not allow_gpu:
                    util_text = " ".join(f"{gpu}:{util_map.get(gpu, 0.0):.1f}%" for gpu in gpu_ids)
                    print(
                        f"[pool] gpu guard active: {util_text} > max_gpu_util={args.max_gpu_util:.1f}; "
                        "postpone new launches"
                    )
                    break
                task = pending.pop(0)
                assigned_gpu = ""
                scheduled_task = task
                if dynamic_gpu:
                    next_gpu = next_available_gpu(gpu_ids, running_per_gpu, args.max_per_gpu)
                    if next_gpu is None:
                        pending.insert(0, task)
                        break
                    scheduled_task = Task(
                        unit_name=task.unit_name,
                        log_file=task.log_file,
                        command=set_cuda_arg(task.command, args.cuda_arg, next_gpu),
                        workdir=task.workdir,
                        setenv=task.setenv,
                    )
                    assigned_gpu = next_gpu
                scheduled = ScheduledTask(task=scheduled_task, assigned_gpu=assigned_gpu)
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
                    ok, message = summarize_finished(scheduled.task)
                    if ok:
                        finished_ok.append(message)
                        print(f"[done] {message}")
                    else:
                        finished_fail.append(message)
                        print(f"[fail] {message}")
                    if dynamic_gpu:
                        running_per_gpu[scheduled.assigned_gpu] -= 1
                    finished_now.append(unit_name)

            for unit_name in finished_now:
                running.pop(unit_name, None)

            if pending and len(running) < args.max_running:
                continue

            if running:
                print(
                    f"[pool] running={len(running)} pending={len(pending)} "
                    f"finished={len(finished_ok) + len(finished_fail)}"
                )
                if args.max_load1 > 0:
                    load1 = os.getloadavg()[0]
                    print(f"[pool] load1={load1:.2f} max_load1={args.max_load1:.2f}")
                if args.max_gpu_util > 0:
                    util_map = get_gpu_utils(gpu_ids)
                    util_text = " ".join(f"{gpu}:{util_map.get(gpu, 0.0):.1f}%" for gpu in gpu_ids)
                    print(f"[pool] gpu_util {util_text} max_gpu_util={args.max_gpu_util:.1f}")
                if dynamic_gpu:
                    gpu_status = " ".join(f"{gpu}:{running_per_gpu[gpu]}/{args.max_per_gpu}" for gpu in gpu_ids)
                    print(f"[pool] gpu_slots {gpu_status}")
                time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\n[pool] interrupted")
        if running:
            print("[pool] still running units:")
            for unit_name in sorted(running):
                print(f"  - {unit_name}.service")
        return 130

    total_finished = len(finished_ok) + len(finished_fail)
    print(f"[summary] finished={total_finished}/{len(tasks)} ok={len(finished_ok)} fail={len(finished_fail)}")
    if finished_fail:
        print("[summary] failed units:")
        for message in finished_fail:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
