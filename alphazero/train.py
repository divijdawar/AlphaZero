from __future__ import annotations

import argparse
import json
import os
import tempfile
import webbrowser
import numpy as np
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty
from time import monotonic

from tinygrad import Context, Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import (
    get_parameters, get_state_dict, load_state_dict, safe_load, safe_load_metadata,
    safe_save,
)

from .config import Config
from .mcts import Predict
from .metrics import MetricWindow, RunMetrics, format_metrics, require_plotly
from .nn import NeuralNet
from .selfplay import Sample, WorkerGroup, start_worker, stop_workers

def make_predict(net: NeuralNet, cfg: Config) -> Predict:
    def predict(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with Context(TRAINING=0):
            logits, value = net(Tensor(planes))
            return (
                logits.numpy().astype(np.float32),
                value.numpy().reshape(-1).astype(np.float32),
            )
    return predict

class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("replay buffer capacity must be positive")
        self.data: deque[Sample] = deque(maxlen=capacity)

    def extend(self, samples: Iterable[Sample]) -> None:
        self.data.extend(samples)

    def sample(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n <= 0:
            raise ValueError("sample size must be positive")
        if not self.data:
            raise ValueError("cannot sample from an empty replay buffer")
        items = list(self.data)
        indices = rng.integers(0, len(items), size=n)
        planes = np.stack([items[i].planes for i in indices]).astype(np.float32)
        pi = np.stack([items[i].pi for i in indices]).astype(np.float32)
        z = np.asarray([[items[i].value] for i in indices], dtype=np.float32)
        return planes, pi, z

    def __len__(self) -> int:
        return len(self.data)

def resolve_lr_schedule(cfg: Config) -> Config:
    """Resolve zero-based update milestones once, including in checkpoint config."""
    if cfg.steps < 1 or not cfg.lr or any(not np.isfinite(lr) or lr <= 0 for lr in cfg.lr):
        raise ValueError("steps and learning rates must be positive and finite")
    if cfg.lr_milestones is None:
        if len(cfg.lr) == 1 or cfg.steps < 4:
            cfg = replace(cfg, lr=(cfg.lr[0],), lr_milestones=())
        elif len(cfg.lr) == 4:
            cfg = replace(cfg, lr_milestones=tuple(round(cfg.steps * n / 7) for n in (1, 3, 5)))
        else:
            raise ValueError("provide explicit milestones for a schedule with neither one nor four rates")
    milestones = cfg.lr_milestones
    if (len(milestones) != len(cfg.lr) - 1
            or any(not isinstance(m, int) or not 0 < m < cfg.steps for m in milestones)
            or tuple(sorted(set(milestones))) != milestones):
        raise ValueError("need one increasing milestone per decay, strictly inside the training run")
    return cfg

def lr_for_step(step: int, cfg: Config) -> float:
    cfg = resolve_lr_schedule(cfg)
    schedule_index = sum(step >= milestone for milestone in cfg.lr_milestones)
    return float(cfg.lr[min(schedule_index, len(cfg.lr) - 1)])

def copy_network(learner: NeuralNet, cfg: Config) -> NeuralNet:
    """Copy all model state, including BatchNorm buffers, for a fixed game version."""
    snapshot = NeuralNet(cfg)
    state = {
        name: tensor.clone().realize()
        for name, tensor in get_state_dict(learner).items()
    }
    load_state_dict(snapshot, state, strict=True, verbose=False)
    for tensor in get_state_dict(snapshot).values():
        tensor.is_param_(False)
    return snapshot

def _check_workers(
    actors: WorkerGroup,
    last_activity: list[float],
    timeout: float | None,
) -> None:
    now = monotonic()
    for actor_id, process in enumerate(actors.processes):
        if process.exitcode is not None:
            raise RuntimeError(f"{process.name} exited: {process.exitcode}")
        if timeout is not None and now - last_activity[actor_id] > timeout:
            raise TimeoutError(f"{process.name} made no progress for {timeout}s")

def collect_requests(
    actors: WorkerGroup,
    latest_version: int,
    leases: dict[int, int],
    last_activity: list[float],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Answer control messages; gather predictions without running the network."""
    batches: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    waiting: set[int] = set()
    batch_wait = 0.002  # 2 ms gives nearby worker requests a chance to batch.
    deadline = monotonic() + batch_wait
    # Bound control traffic as well as the time spent gathering requests.
    for _ in range(3 * len(actors.processes)):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        try:
            kind, actor_id, payload = actors.requests.get(timeout=remaining)
        except Empty:
            break
        if not 0 <= actor_id < len(actors.processes):
            raise RuntimeError(f"unknown worker: {actor_id}")
        if actor_id in waiting:
            raise RuntimeError(f"worker {actor_id} has an unanswered prediction")
        last_activity[actor_id] = monotonic()
        reply = actors.replies[actor_id]
        if kind == "acquire":
            if actor_id in leases:
                raise RuntimeError(f"worker {actor_id} already has a network lease")
            leases[actor_id] = latest_version
            reply.put_nowait(("acquire", latest_version))
        elif kind == "release":
            if leases.get(actor_id) != payload:
                raise RuntimeError(f"invalid release from worker {actor_id}")
            del leases[actor_id]
            reply.put_nowait(("release", None))
        elif kind == "predict":
            version, planes = payload
            if leases.get(actor_id) != version:
                raise RuntimeError(f"invalid prediction lease from worker {actor_id}")
            batches[version].append((actor_id, planes))
            waiting.add(actor_id)
            if len(waiting) == len(actors.processes):
                break
        else:
            raise RuntimeError(f"unknown worker request: {kind}")
    return batches

def run_inference(
    actors: WorkerGroup,
    snapshots: dict[int, NeuralNet],
    batches: dict[int, list[tuple[int, np.ndarray]]],
    last_activity: list[float],
    cfg: Config,
) -> None:
    """Batch across workers using the same snapshot, then route their results."""
    for version, requests in batches.items():
        planes = np.concatenate([planes for _, planes in requests], axis=0)
        logits, values = make_predict(snapshots[version], cfg)(planes)
        offset = 0
        for actor_id, actor_planes in requests:
            end = offset + len(actor_planes)
            actors.replies[actor_id].put_nowait((
                "predict", (logits[offset:end], values[offset:end]),
            ))
            last_activity[actor_id] = monotonic()
            offset = end

def retire_snapshots(
    snapshots: dict[int, NeuralNet], leases: dict[int, int], latest_version: int,
) -> None:
    keep = set(leases.values()) | {latest_version}
    for version in list(snapshots):
        if version not in keep:
            del snapshots[version]

def generate_selfplay(
    net: NeuralNet,
    cfg: Config,
    num_games: int,
    *,
    replay: ReplayBuffer,
    seed: int = 0,
) -> int:
    if num_games < 0:
        raise ValueError("num_games cannot be negative")
    if num_games == 0:
        return 0
    if cfg.num_workers < 1:
        raise ValueError("num_workers must be positive")

    actors = start_worker(cfg, seed=seed)
    leases: dict[int, int] = {}
    last_activity = [monotonic()] * cfg.num_workers
    completed = 0
    try:
        while completed < num_games:
            _check_workers(actors, last_activity, timeout=None)
            for _ in range(min(cfg.num_workers, num_games - completed)):
                try:
                    _, _, _, samples = actors.completed_games.get_nowait()
                except Empty:
                    break
                replay.extend(samples)
                completed += 1
            if completed == num_games:
                break
            batches = collect_requests(actors, 0, leases, last_activity)
            run_inference(actors, {0: net}, batches, last_activity, cfg)
    finally:
        stop_workers(actors)
    return completed


def train_step(
    net: NeuralNet,
    opt: optim.Optimizer,
    params: list[Tensor],
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: Config,
    *,
    metrics: dict[str, float] | None = None,
) -> float:
    planes, pi, z = batch
    with Context(TRAINING=1):
        logits, value = net(Tensor(planes))
        policy_loss = -(Tensor(pi) * logits.log_softmax(axis=1)).sum(axis=1).mean()
        value_loss = (Tensor(z) - value).square().mean()
        l2_loss = cfg.l2 * sum((p.square().sum() for p in params), start=Tensor(0.0))
        loss = policy_loss + value_loss + l2_loss

        opt.zero_grad()
        loss.backward()
        # Preserve this batch's losses before mutable parameter buffers change.
        values = Tensor.stack(loss, policy_loss, value_loss, l2_loss).realize() if metrics is not None else None
        opt.step()

    if metrics is None:
        return float(loss.item())
    metrics.update(zip(
        ("loss", "policy_loss", "value_loss", "regularization_loss"),
        map(float, values.numpy()),
    ))
    return metrics["loss"]

def _save_checkpoint(
    net: NeuralNet,
    path: str | os.PathLike[str],
    *,
    opt: optim.Optimizer,
    cfg: Config,
    global_step: int,
    iteration: int,
    step_in_iteration: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opt_state = {k: v for k, v in vars(opt).items() if k not in ("params", "buffers")}
    model_names = {id(tensor): name for name, tensor in get_state_dict(net).items()}
    state = get_state_dict({"model": net, "optimizer": opt_state})
    metadata = {
        "format_version": "1",
        "config": json.dumps(asdict(cfg)),
        "training": json.dumps({
            "global_step": global_step,
            "iteration": iteration,
            "step_in_iteration": step_in_iteration,
        }),
        "optimizer": json.dumps({
            "class": f"{type(opt).__module__}.{type(opt).__qualname__}",
            "options": {
                k: v for k, v in opt_state.items()
                if v is None or isinstance(v, (bool, int, float))
            },
            "parameter_names": [model_names[id(param)] for param in opt.params],
        }),
    }
    Tensor.realize(*state.values())
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        safe_save(state, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)

def load_checkpoint(
    net: NeuralNet,
    path: str | os.PathLike[str],
    *,
    opt: optim.Optimizer,
) -> tuple[Config, dict[str, int]]:
    source, _, header = safe_load_metadata(Path(path))
    metadata = header["__metadata__"]
    if metadata["format_version"] != "1":
        raise ValueError("unsupported checkpoint format")

    optimizer = json.loads(metadata["optimizer"])
    if optimizer["class"] != f"{type(opt).__module__}.{type(opt).__qualname__}":
        raise ValueError("checkpoint optimizer class does not match")
    model_names = {id(tensor): name for name, tensor in get_state_dict(net).items()}
    if optimizer["parameter_names"] != [model_names[id(param)] for param in opt.params]:
        raise ValueError("checkpoint optimizer parameter order does not match")
    if optimizer["options"]["fused"] != opt.fused:
        raise ValueError("checkpoint optimizer fusion setting does not match")

    config = json.loads(metadata["config"])
    for key in ("lr", "lr_milestones", "dirichlet_alpha"):
        if isinstance(config[key], list):
            config[key] = tuple(config[key])
    cfg = Config(**config)
    training = json.loads(metadata["training"])

    opt_state = {k: v for k, v in vars(opt).items() if k not in ("params", "buffers")}
    target = {"model": net, "optimizer": opt_state}
    state = safe_load(source)
    if get_state_dict(target).keys() != state.keys():
        raise ValueError("checkpoint model or optimizer state does not match")
    load_state_dict(target, state, strict=True, verbose=False)
    for key, value in optimizer["options"].items():
        if key != "device":
            setattr(opt, key, value)
    return cfg, training

def run_pipeline(
    cfg: Config,
    *,
    checkpoint_dir: str = "checkpoints",
    seed: int = 0,
    min_replay_size: int = 1024,
    replay_ratio: float = 1.0,
    publish_every: int = 1000,
    max_snapshots: int = 3,
    worker_timeout: float | None = 300.0,
    log_every: int = 100,
    log_seconds: float = 30.0,
    log: Callable[[dict], None] | None = None,
    metrics_dir: str | os.PathLike[str] | None = None,
    open_report: bool = False,
) -> NeuralNet:
    """Keep CPU actors alive while the parent alternates inference and training.

    replay_ratio budgets sample draws per newly generated position. Credits are
    capped at 16 optimizer updates. Each game pins one immutable model snapshot.
    Select the parent's tinygrad device through Context(DEV=...) or the CLI.
    """
    cfg = resolve_lr_schedule(cfg)
    if min(cfg.num_workers, cfg.steps, cfg.batch_size, cfg.max_plies, cfg.num_simulations) < 1:
        raise ValueError("workers, steps, batch size, plies, and simulations must be positive")
    if not 1 <= min_replay_size <= cfg.buffer_size:
        raise ValueError("min_replay_size must be positive and fit in the replay buffer")
    if not np.isfinite(replay_ratio) or replay_ratio <= 0 or publish_every < 1:
        raise ValueError("replay_ratio and publish_every must be positive")
    if max_snapshots < 2 or log_every < 1:
        raise ValueError("need at least two snapshots and positive log_every")
    if not np.isfinite(log_seconds) or log_seconds <= 0:
        raise ValueError("log_seconds must be positive and finite")
    if open_report and metrics_dir is None:
        raise ValueError("open_report requires metrics_dir")
    if worker_timeout is not None and (not np.isfinite(worker_timeout) or worker_timeout <= 0):
        raise ValueError("worker_timeout must be positive or None")

    saved_metrics = None if metrics_dir is None else RunMetrics(metrics_dir, asdict(cfg), {
        "log_every": log_every, "log_seconds": log_seconds, "seed": seed,
        "checkpoint_dir": str(checkpoint_dir), "min_replay_size": min_replay_size,
        "replay_ratio": replay_ratio, "publish_every": publish_every,
        "max_snapshots": max_snapshots, "worker_timeout": worker_timeout,
    })

    Tensor.manual_seed(seed)
    rng = np.random.default_rng(seed)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    learner = NeuralNet(cfg)
    model_tensors = get_parameters(learner)
    params = [p for p in model_tensors if p.is_param]
    # SGD realizes non-trainable buffers alongside each parameter update.
    opt = optim.SGD(model_tensors, lr=float(cfg.lr[0]), momentum=0.9)
    replay = ReplayBuffer(cfg.buffer_size)
    snapshots = {0: copy_network(learner, cfg)}
    latest_version = last_publish_step = global_step = completed_games = 0
    leases: dict[int, int] = {}
    training_credit = 0.0
    started = monotonic()
    window = MetricWindow(started)
    last_learning_rate = lr_for_step(0, cfg)

    def emit_metrics(*, force: bool = False) -> None:
        nonlocal window
        if log is None and saved_metrics is None:
            return
        now = monotonic()
        if not (force or window.updates >= log_every or now - window.started >= log_seconds):
            return
        record = {
            "step": global_step, "total_steps": cfg.steps,
            "elapsed_seconds": now - started,
            "games": completed_games, "samples": len(replay),
            "learning_rate": last_learning_rate,
            "published_version": latest_version,
            "retained_snapshots": len(snapshots),
            "training_credit": training_credit,
            **window.report(now),
        }
        if saved_metrics is not None:
            saved_metrics.write(record)
        if log is not None:
            log(record)
        window = MetricWindow(now)

    def save_checkpoint(filename: str) -> None:
        _save_checkpoint(
            learner, checkpoint_path / filename, opt=opt, cfg=cfg,
            global_step=global_step, iteration=0, step_in_iteration=global_step,
        )

    actors = start_worker(cfg, seed=seed)
    last_activity = [monotonic()] * cfg.num_workers
    try:
        while global_step < cfg.steps:
            _check_workers(actors, last_activity, worker_timeout)
            for _ in range(cfg.num_workers):
                try:
                    actor_id, _, _, samples = actors.completed_games.get_nowait()
                except Empty:
                    break
                replay.extend(samples)
                window.positions += len(samples)
                completed_games += 1
                last_activity[actor_id] = monotonic()
                training_credit = min(
                    16.0, training_credit + len(samples) * replay_ratio / cfg.batch_size,
                )

            batches = collect_requests(
                actors, latest_version, leases, last_activity,
            )
            # Exclude time spent in the parent's GPU/disk work from the worker
            # watchdog: blocked actors cannot make progress until we reply.
            busy_start = monotonic()
            run_inference(actors, snapshots, batches, last_activity, cfg)
            busy_elapsed = monotonic() - busy_start
            last_activity[:] = [min(monotonic(), t + busy_elapsed) for t in last_activity]
            retire_snapshots(snapshots, leases, latest_version)

            if len(replay) < min_replay_size or training_credit < 1:
                # Logging time is parent work, just like inference and saving.
                busy_start = monotonic()
                emit_metrics()
                busy_elapsed = monotonic() - busy_start
                last_activity[:] = [t + busy_elapsed for t in last_activity]
                continue

            busy_start = monotonic()
            last_learning_rate = lr_for_step(global_step, cfg)
            opt.lr.assign(last_learning_rate)
            step_metrics = {} if log is not None or saved_metrics is not None else None
            train_step(learner, opt, params, replay.sample(cfg.batch_size, rng), cfg,
                       metrics=step_metrics)
            if step_metrics is not None:
                window.record_update(step_metrics)
            global_step += 1
            training_credit -= 1

            if (global_step - last_publish_step >= publish_every
                    and len(set(leases.values())) < max_snapshots):
                # Remove an unleased current version before allocating its
                # replacement, keeping snapshot storage within the limit.
                pinned = set(leases.values())
                for version in list(snapshots):
                    if version not in pinned:
                        del snapshots[version]
                snapshots[global_step] = copy_network(learner, cfg)
                latest_version = last_publish_step = global_step

            if cfg.checkpoint > 0 and global_step % cfg.checkpoint == 0:
                save_checkpoint(f"step-{global_step}.safetensors")
            emit_metrics(force=global_step == cfg.steps)
            busy_elapsed = monotonic() - busy_start
            last_activity[:] = [t + busy_elapsed for t in last_activity]
    finally:
        stop_workers(actors)

    save_checkpoint("final.safetensors")
    if saved_metrics is not None:
        report = saved_metrics.finish()
        print(f"Training report: {report}", flush=True)
        if open_report:
            webbrowser.open(report.as_uri())
    return learner


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train AlphaZero with persistent self-play workers")
    parser.add_argument("--device", default="NV", help="tinygrad device (NV for the RTX 4090)")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--workers", type=int, default=defaults.num_workers)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, nargs="+", default=defaults.lr,
                        help="learning rates (default: 0.01, 0.001, 0.0001, 0.00001)")
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=None,
                        help="absolute update indices; default scales decays to --steps")
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    parser.add_argument("--min-replay-size", type=int, default=1024)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--publish-every", type=int, default=1000)
    parser.add_argument("--max-snapshots", type=int, default=3)
    parser.add_argument("--worker-timeout", type=float, default=300.0,
                        help="worker inactivity timeout in seconds; 0 disables it")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-seconds", type=float, default=30.0,
                        help="maximum seconds between progress records, including replay warm-up")
    parser.add_argument("--metrics-dir", help="save metrics and an offline interactive HTML report")
    parser.add_argument("--open-report", action="store_true", help="open the saved report when training finishes")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.open_report and args.metrics_dir is None:
        parser.error("--open-report requires --metrics-dir")
    if args.metrics_dir is not None:
        try:
            require_plotly()
        except RuntimeError as error:
            parser.error(str(error))
    cfg = replace(defaults, steps=args.steps, num_workers=args.workers,
                  batch_size=args.batch_size, buffer_size=args.buffer_size,
                  lr=tuple(args.lr),
                  lr_milestones=None if args.lr_milestones is None else tuple(args.lr_milestones))
    with Context(DEV=args.device):
        run_pipeline(
            cfg, checkpoint_dir=args.checkpoint_dir, seed=args.seed,
            min_replay_size=args.min_replay_size, replay_ratio=args.replay_ratio,
            publish_every=args.publish_every, max_snapshots=args.max_snapshots,
            worker_timeout=None if args.worker_timeout == 0 else args.worker_timeout,
            log_every=args.log_every, log_seconds=args.log_seconds,
            log=lambda record: print(format_metrics(record), flush=True),
            metrics_dir=args.metrics_dir, open_report=args.open_report,
        )


if __name__ == "__main__":
    main()
