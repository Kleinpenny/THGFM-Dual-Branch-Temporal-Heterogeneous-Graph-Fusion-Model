#!/usr/bin/env python3
"""
Evaluate the six best checkpoints under ``Best_Model_Save/`` using the HTAG / OAG / OGBN evaluators.

Runs **(job × seed)** tasks **concurrently** (subprocess pool). After all finish, prints a summary;
with multiple seeds, each metric is shown as **mean ± (max − min)** across seeds.

Examples::

  # Smoke on GPU 3, single seed
  python scripts/eval_best_model_save.py --smoke --seeds 43 --cuda 3

  # Full eval, two seeds, up to 6 parallel jobs
  python scripts/eval_best_model_save.py --seeds 43,44 --cuda 3 --max-workers 6

  # Disable live streaming (buffer until each job finishes)
  python scripts/eval_best_model_save.py --no-live-progress --seeds 43
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ckpt_tag(path: Path) -> str:
    name = path.name
    return name[:-3] if name.endswith(".pt") else name


def _default_jobs(repo: Path, ckpt_dir: Path) -> list[dict]:
    d = ckpt_dir.resolve()
    return [
        {
            "id": "htag_arxiv",
            "cwd": repo / "HTAG" / "evaluator",
            "script": "eval_pv_htag_arxiv.py",
            "checkpoint": d / "HTAG_Arxiv_sf_r111_s111_s43_lowest_valid_loss.pt",
        },
        {
            "id": "htag_dblp",
            "cwd": repo / "HTAG" / "evaluator",
            "script": "eval_pv_htag_dblp.py",
            "checkpoint": d / "HTAG_DBLP_nglf_r110_s110_s43_lowest_valid_loss.pt",
        },
        {
            "id": "oag_pv",
            "cwd": repo / "OAG" / "evaluator",
            "script": "eval_pv_test_ndcg_mrr.py",
            "checkpoint": d / "PV_OAG_XLNet_sf_r110_s110_s43_lowest_valid_loss",
        },
        {
            "id": "oag_pf_l1",
            "cwd": repo / "OAG" / "evaluator",
            "script": "eval_pf_l1_test_ndcg_mrr.py",
            "checkpoint": d / "PF_OAG_XLNet_L1_tcngsf_r111_s111_s43_L1_lowest_valid_loss",
        },
        {
            "id": "oag_pf_l2",
            "cwd": repo / "OAG" / "evaluator",
            "script": "eval_pf_l2_test_ndcg_mrr.py",
            "checkpoint": d / "PF_XLNet_L2_nglf_r111_s111_s43_L2_lowest_valid_loss",
        },
        {
            "id": "ogbn_mag",
            "cwd": repo / "OGBN" / "evaluator",
            "script": "eval_test_accuracy.py",
            "checkpoint": d / "PV_OGBN_MAG_tcngsf_r111_s111_s43_lowest_valid_loss",
        },
    ]


def _parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty --seeds")
    return out


def _parse_htag_stdout(stdout: str) -> dict[str, float] | None:
    idx = stdout.find("{")
    if idx < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(stdout[idx:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    m: dict[str, float] = {}
    for k in ("valid_acc", "test_acc"):
        if k in obj and isinstance(obj[k], (int, float)):
            m[k] = float(obj[k])
    return m if m else None


def _parse_oag_stdout(stdout: str) -> dict[str, float] | None:
    m: dict[str, float] = {}
    ndcg_m = re.search(r"NDCG=([\d.]+(?:e[-+]\d+)?)", stdout, re.I)
    mrr_m = re.search(r"MRR=([\d.]+(?:e[-+]\d+)?)", stdout, re.I)
    if ndcg_m:
        m["ndcg"] = float(ndcg_m.group(1))
    if mrr_m:
        m["mrr"] = float(mrr_m.group(1))
    return m if m else None


def _parse_ogbn_stdout(stdout: str) -> dict[str, float] | None:
    m: dict[str, float] = {}
    va = re.search(r"valid_acc=([\d.]+(?:e[-+]\d+)?)", stdout, re.I)
    ta = re.search(r"test_acc=([\d.]+(?:e[-+]\d+)?)", stdout, re.I)
    if va:
        m["valid_acc"] = float(va.group(1))
    if ta:
        m["test_acc"] = float(ta.group(1))
    if m:
        if "test_acc" in m:
            m["accuracy"] = m["test_acc"]
        else:
            m["accuracy"] = m["valid_acc"]
        return m
    acc_m = re.search(r"accuracy=([\d.]+(?:e[-+]\d+)?)", stdout, re.I)
    if not acc_m:
        return None
    return {"accuracy": float(acc_m.group(1))}


def _metric_keys_for_job(job_id: str) -> list[str]:
    if job_id in ("htag_arxiv", "htag_dblp"):
        return ["valid_acc", "test_acc"]
    if job_id in ("oag_pv", "oag_pf_l1", "oag_pf_l2"):
        return ["ndcg", "mrr"]
    if job_id == "ogbn_mag":
        return ["valid_acc", "test_acc"]
    return []


def _parse_metrics(job_id: str, stdout: str) -> dict[str, float] | None:
    if job_id in ("htag_arxiv", "htag_dblp"):
        return _parse_htag_stdout(stdout)
    if job_id in ("oag_pv", "oag_pf_l1", "oag_pf_l2"):
        return _parse_oag_stdout(stdout)
    if job_id == "ogbn_mag":
        return _parse_ogbn_stdout(stdout)
    return None


def _format_agg(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.6f}"
    mean_v = sum(values) / len(values)
    spread = max(values) - min(values)
    return f"{mean_v:.6f} \\pm {spread:.6f}"


def _build_cmd(
    job: dict,
    seed: int,
    *,
    cuda: int,
    max_rows: int | None,
    save_root: Path,
    smoke: bool,
    multi_seed: bool,
) -> list[str]:
    ckpt: Path = job["checkpoint"]
    tag = _ckpt_tag(ckpt)
    model_subdir = f"{tag}_seed{seed}" if multi_seed else tag

    extra: list[str] = [
        "--checkpoint",
        str(ckpt),
        "--seed",
        str(int(seed)),
        "--cuda",
        str(int(cuda)),
    ]
    if max_rows is not None:
        extra += ["--max_test_rows", str(int(max_rows))]

    jid = job["id"]
    if jid in ("htag_arxiv", "htag_dblp"):
        if smoke:
            extra += ["--eval_rounds", "1", "--no_save"]
        else:
            extra += ["--save_dir", str(save_root / "HTAG"), "--model_subdir", model_subdir]
    elif jid in ("oag_pv", "oag_pf_l1", "oag_pf_l2"):
        extra += ["--save_dir", str(save_root / "OAG"), "--model_subdir", model_subdir]
    elif jid == "ogbn_mag":
        extra += [
            "--save_dir",
            str(save_root / "OGBN"),
            "--model_subdir",
            model_subdir,
            "--splits",
            "valid,test",
        ]

    return [sys.executable, job["script"], *extra]


def _run_subprocess_with_progress(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    *,
    task_label: str,
    progress_log_file: Path | None,
    use_pty: bool,
) -> tuple[int, str]:
    """
    Run subprocess with live stdout/stderr so tqdm sees a TTY when ``use_pty`` (Unix).

    Also mirrors raw bytes to ``progress_log_file`` (per-task log) when set.
    Returns (returncode, captured_stdout_stderr_text).
    """
    env = {**env, "PYTHONUNBUFFERED": "1"}

    if use_pty:
        import pty

        master_fd, slave_fd = pty.openpty()
        slave_closed = False
        log_fp = None
        proc: subprocess.Popen | None = None
        try:
            if progress_log_file is not None:
                progress_log_file.parent.mkdir(parents=True, exist_ok=True)
                log_fp = progress_log_file.open("wb")

            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
            os.close(slave_fd)
            slave_closed = True

            chunks: list[bytes] = []

            def forward(data: bytes) -> None:
                if not data:
                    return
                chunks.append(data)
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    pass
                if log_fp is not None:
                    log_fp.write(data)
                    log_fp.flush()

            while True:
                r, _, _ = select.select([master_fd], [], [], 0.15)
                if master_fd in r:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    forward(data)
                if proc.poll() is not None:
                    # Drain remaining output after the child exits.
                    while True:
                        r2, _, _ = select.select([master_fd], [], [], 0)
                        if not r2:
                            break
                        try:
                            data = os.read(master_fd, 65536)
                        except OSError:
                            break
                        if not data:
                            break
                        forward(data)
                    break

            rc = int(proc.wait())
            text = b"".join(chunks).decode("utf-8", errors="replace")
            return rc, text
        finally:
            if not slave_closed:
                try:
                    os.close(slave_fd)
                except OSError:
                    pass
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                os.close(master_fd)
            except OSError:
                pass
            if log_fp is not None:
                log_fp.close()

    # Fallback: pipe (tqdm may disable itself when stdout is not a tty).
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=False,
    )
    chunks: list[bytes] = []
    log_fp = None
    try:
        if progress_log_file is not None:
            progress_log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = progress_log_file.open("wb")
        prefix = f"[{task_label}] ".encode("utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            try:
                sys.stdout.buffer.write(prefix)
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                pass
            if log_fp is not None:
                log_fp.write(prefix)
                log_fp.write(line)
                log_fp.flush()
        rc = int(proc.wait())
    finally:
        if log_fp is not None:
            log_fp.close()
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return rc, text


def _run_task(
    job: dict,
    seed: int,
    env: dict[str, str],
    cuda: int,
    max_rows: int | None,
    save_root: Path,
    smoke: bool,
    multi_seed: bool,
    *,
    live_progress: bool,
    live_log_dir: Path | None,
    use_pty: bool,
) -> dict:
    cmd = _build_cmd(
        job,
        seed,
        cuda=cuda,
        max_rows=max_rows,
        save_root=save_root,
        smoke=smoke,
        multi_seed=multi_seed,
    )
    cwd = str(job["cwd"])
    task_label = f"{job['id']} seed={seed}"
    progress_file: Path | None = None
    if live_log_dir is not None:
        safe = re.sub(r"[^\w.\-]+", "_", f"{job['id']}_seed{seed}")
        progress_file = live_log_dir / f"{safe}.log"

    if live_progress:
        rc, out = _run_subprocess_with_progress(
            cmd,
            cwd,
            env,
            task_label=task_label,
            progress_log_file=progress_file,
            use_pty=use_pty,
        )
    else:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env={**env, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            timeout=None,
        )
        rc = int(proc.returncode)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    metrics = _parse_metrics(job["id"], out)
    return {
        "job_id": job["id"],
        "seed": seed,
        "returncode": rc,
        "cmd": cmd,
        "cwd": cwd,
        "metrics": metrics,
        "stdout_tail": out[-4000:],
        "stderr_tail": out[-2000:],
        "progress_log_file": str(progress_file) if progress_file else "",
    }


def main() -> int:
    repo = _repo_root()
    default_ckpt = repo / "Best_Model_Save"

    p = argparse.ArgumentParser(
        description="Concurrent (job×seed) evaluation for Best_Model_Save checkpoints; summary mean ± (max−min)."
    )
    p.add_argument("--ckpt-dir", type=Path, default=default_ckpt, help="Checkpoint directory.")
    p.add_argument(
        "--htag-arxiv-checkpoint",
        type=Path,
        default=None,
        help=(
            "Override the htag_arxiv checkpoint path (use the real .pt basename so HTAG eval infers conv/rota correctly)."
        ),
    )
    p.add_argument(
        "--pf-l2-checkpoint",
        type=Path,
        default=None,
        help="Override the oag_pf_l2 checkpoint path (file or THGFM save without .pt).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds (e.g. 43,44,45). If empty, uses --seed once.",
    )
    p.add_argument("--seed", type=int, default=43, help="Used when --seeds is not set.")
    p.add_argument("--cuda", type=int, default=0, help="GPU id for every subprocess (--cuda).")
    p.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Max concurrent subprocesses (default: 6). Lower if GPU OOM.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: max_test_rows=32; HTAG uses eval_rounds=1 and no_save.",
    )
    p.add_argument("--max-test-rows", type=int, default=None, help="Override smoke row cap when not using --smoke.")
    p.add_argument("--save-root", type=Path, default=None, help="Root for evaluator JSON outputs.")
    p.add_argument("--only", type=str, default="", help="Comma-separated job ids.")
    p.add_argument("--dry-run", action="store_true", help="Print all commands, no execution.")
    p.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Buffer each subprocess until it exits (no live stream; tqdm usually hidden).",
    )
    p.add_argument(
        "--live-log-dir",
        type=Path,
        default=None,
        help="Directory for per-task raw logs (tqdm-safe). Default: <save_root>/_eval_live_logs/<timestamp>.",
    )
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds) if args.seeds.strip() else [int(args.seed)]
    multi_seed = len(seeds) > 1

    jobs = _default_jobs(repo, args.ckpt_dir)
    if args.htag_arxiv_checkpoint is not None:
        hck = args.htag_arxiv_checkpoint.expanduser().resolve()
        if not hck.is_file():
            print(f"--htag-arxiv-checkpoint not found: {hck}", file=sys.stderr)
            return 2
        for j in jobs:
            if j["id"] == "htag_arxiv":
                j["checkpoint"] = hck
                break
    if args.pf_l2_checkpoint is not None:
        p2 = args.pf_l2_checkpoint.expanduser().resolve()
        if not p2.is_file():
            print(f"--pf-l2-checkpoint not found: {p2}", file=sys.stderr)
            return 2
        for j in jobs:
            if j["id"] == "oag_pf_l2":
                j["checkpoint"] = p2
                break
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        jobs = [j for j in jobs if j["id"] in only]
        missing = only - {j["id"] for j in jobs}
        if missing:
            print(f"Unknown --only id(s): {sorted(missing)}", file=sys.stderr)
            print(f"Known ids: {[j['id'] for j in _default_jobs(repo, args.ckpt_dir)]}", file=sys.stderr)
            return 2

    max_rows = 32 if args.smoke else args.max_test_rows
    save_root = args.save_root
    if save_root is None:
        save_root = repo / "Best_Model_Save" / ("eval_results_smoke" if args.smoke else "eval_results")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    tasks: list[tuple[dict, int]] = []
    skipped: list[tuple[str, str]] = []
    for job in jobs:
        ckpt = job["checkpoint"]
        if not ckpt.is_file():
            for sd in seeds:
                skipped.append((job["id"], f"missing checkpoint: {ckpt}"))
            continue
        for sd in seeds:
            tasks.append((job, sd))

    if args.dry_run:
        for job, sd in tasks:
            cmd = _build_cmd(
                job,
                sd,
                cuda=args.cuda,
                max_rows=max_rows,
                save_root=save_root,
                smoke=args.smoke,
                multi_seed=multi_seed,
            )
            print(f"\n[{job['id']} seed={sd}]", flush=True)
            print(f"cd {job['cwd']} && {' '.join(cmd)}", flush=True)
        for jid, reason in skipped:
            print(f"[skip] {jid}: {reason}", flush=True)
        return 0

    if not tasks:
        print("No tasks to run.", flush=True)
        return 1

    save_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(args.max_workers), len(tasks)))

    live_progress = not bool(args.no_live_progress)
    use_pty = live_progress and sys.platform != "win32"
    live_log_dir: Path | None = None
    if live_progress:
        live_log_dir = args.live_log_dir
        if live_log_dir is None:
            live_log_dir = save_root / "_eval_live_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        live_log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    print(f"Running {len(tasks)} task(s) with up to {workers} workers; cuda={args.cuda}; seeds={seeds}", flush=True)
    if live_progress:
        if use_pty:
            print(
                "Live output: child processes use a pseudo-TTY so tqdm progress bars stay enabled; "
                "streams go to this console (parallel jobs may interleave).",
                flush=True,
            )
        else:
            print(
                "Live output: line-buffered pipe (Windows: tqdm may still be disabled by the child).",
                flush=True,
            )
        print(f"Per-task raw logs: {live_log_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _run_task,
                job,
                sd,
                env,
                args.cuda,
                max_rows,
                save_root,
                args.smoke,
                multi_seed,
                live_progress=live_progress,
                live_log_dir=live_log_dir if live_progress else None,
                use_pty=use_pty,
            )
            for job, sd in tasks
        ]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "OK" if r["returncode"] == 0 else f"FAIL({r['returncode']})"
            met = r["metrics"]
            met_s = json.dumps(met, sort_keys=True) if met else "(no metrics parsed)"
            print(f"  [{tag}] {r['job_id']} seed={r['seed']} {met_s}", flush=True)

    failures = [(r["job_id"], r["seed"], r["returncode"]) for r in results if r["returncode"] != 0]
    if skipped:
        print("\nSkipped (missing ckpt):", flush=True)
        for jid, reason in skipped:
            print(f"  {jid}: {reason}", flush=True)

    # Aggregate metrics per job
    by_job: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["returncode"] != 0 or not r["metrics"]:
            continue
        jid = r["job_id"]
        for k, v in r["metrics"].items():
            by_job[jid][k].append(float(v))

    print("\n" + "=" * 72)
    print("SUMMARY (mean ± (max−min) when multiple seeds; else single value)")
    print("=" * 72)
    job_order = [j["id"] for j in _default_jobs(repo, args.ckpt_dir)]
    present_ids = {j["id"] for j in jobs}
    attempted_ids = {job["id"] for job, _ in tasks}
    for jid in job_order:
        if jid not in present_ids or jid not in attempted_ids:
            continue
        keys = _metric_keys_for_job(jid)
        if not keys:
            continue
        parts = [jid]
        for k in keys:
            vals = by_job[jid].get(k, [])
            parts.append(f"{k}={_format_agg(vals)}")
        if all(not by_job[jid].get(k) for k in keys):
            parts.append("(no successful runs)")
        print("  " + "  ".join(parts))

    if failures:
        print("\nFailed tasks:", flush=True)
        for jid, sd, code in sorted(failures):
            print(f"  {jid} seed={sd} exit={code}", flush=True)
        for r in results:
            if r["returncode"] != 0:
                print(f"\n--- stderr tail {r['job_id']} seed={r['seed']} ---\n{r['stderr_tail']}", flush=True)
        return 1

    print("\nAll tasks finished OK.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
