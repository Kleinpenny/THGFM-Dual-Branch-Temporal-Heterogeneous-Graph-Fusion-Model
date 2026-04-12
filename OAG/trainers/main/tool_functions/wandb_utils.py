from __future__ import annotations

from typing import Any


def add_wandb_args(parser, default_project: str = "THGFM"):
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases monitoring.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=default_project,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Optional Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Optional Weights & Biases run name. Defaults to task_name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="",
        help="Optional Weights & Biases run group.",
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=[],
        help="Optional Weights & Biases tags.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )


def _to_scalar(value: Any):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_scalar(v) for k, v in value.items()}
    return str(value)


class _DisabledWandbTracker:
    enabled = False

    def log(self, payload: dict[str, Any], step: int | None = None):
        return None

    def summary(self, payload: dict[str, Any]):
        return None

    def finish(self, exit_code: int = 0):
        return None


class WandbTracker:
    enabled = True

    def __init__(self, wandb_module, run):
        self._wandb = wandb_module
        self._run = run

    def log(self, payload: dict[str, Any], step: int | None = None):
        clean_payload = {str(k): _to_scalar(v) for k, v in payload.items()}
        if step is None:
            self._wandb.log(clean_payload)
        else:
            self._wandb.log(clean_payload, step=step)

    def summary(self, payload: dict[str, Any]):
        for key, value in payload.items():
            self._run.summary[str(key)] = _to_scalar(value)

    def finish(self, exit_code: int = 0):
        self._wandb.finish(exit_code=exit_code)


def init_wandb_tracker(
    args,
    *,
    extra_config: dict[str, Any] | None = None,
    default_group: str | None = None,
    default_job_type: str = "train",
):
    if (not getattr(args, "use_wandb", False)) or getattr(args, "wandb_mode", "online") == "disabled":
        return _DisabledWandbTracker()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or run without --use_wandb."
        ) from exc

    config = {k: _to_scalar(v) for k, v in vars(args).items()}
    if extra_config:
        config.update({str(k): _to_scalar(v) for k, v in extra_config.items()})

    init_kwargs = {
        "project": args.wandb_project,
        "config": config,
        "name": args.wandb_run_name or getattr(args, "task_name", "run"),
        "job_type": default_job_type,
        "mode": args.wandb_mode,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    group = args.wandb_group or default_group
    if group:
        init_kwargs["group"] = group
    if args.wandb_tags:
        init_kwargs["tags"] = list(args.wandb_tags)

    run = wandb.init(**init_kwargs)
    return WandbTracker(wandb, run)
