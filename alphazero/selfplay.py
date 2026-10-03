from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty, Full
from time import monotonic, sleep
from engine.env import ChessEnv
from .config import Config
from .encode import encode
from .mcts import MCTS, Predict, select_action

__all__ = [
    "Sample", "WorkerGroup", "WorkerStopped", "play_game", "selfplay_worker",
    "start_worker", "stop_workers",
]

@dataclass(frozen=True)
class Sample:
    planes: np.ndarray   # (num_planes, 8, 8) float32
    pi: np.ndarray       # (num_actions,) float32
    value: float         # outcome z for this position's mover

@dataclass(frozen=True)
class WorkerGroup:
    processes: list[BaseProcess]
    requests: Queue
    replies: list[Queue]
    completed_games: Queue
    stop: Event

def start_worker(cfg: Config, seed: int = 0) -> WorkerGroup:
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    requests = ctx.Queue(maxsize=cfg.num_workers)
    replies = [ctx.Queue(maxsize=1) for _ in range(cfg.num_workers)]
    completed_games = ctx.Queue(maxsize=cfg.num_workers)
    seeds = np.random.SeedSequence(seed).spawn(cfg.num_workers)
    processes = []

    for actor_id in range(cfg.num_workers):
        process = ctx.Process(
            target=selfplay_worker,
            args=(
                actor_id,
                cfg,
                seeds[actor_id],
                requests,
                replies[actor_id],
                completed_games,
                stop,
            ),
            name=f"selfplay-{actor_id}",
        )

        process.start()
        processes.append(process)

    return WorkerGroup(
        processes=processes,
        requests=requests,
        replies=replies,
        completed_games=completed_games,
        stop=stop,
    )

def stop_workers(actors: WorkerGroup, timeout: float = 5.0) -> None:
    actors.stop.set()
    queues = [actors.requests, actors.completed_games, *actors.replies]
    deadline = monotonic() + timeout

    try:
        while any(process.is_alive() for process in actors.processes):
            if monotonic() >= deadline:
                break
            # Drain messages so worker queue feeder threads can finish.
            for queue in queues:
                for _ in range(len(actors.processes)):
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
            for process in actors.processes:
                process.join(timeout=0)
            sleep(0.01)

        for process in actors.processes:
            if process.is_alive():
                process.terminate()
        for process in actors.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
    finally:
        for queue in queues:
            queue.cancel_join_thread()
            queue.close()

class WorkerStopped(Exception):
    pass

def selfplay_worker(
    actor_id: int,
    cfg: Config,
    seed: np.random.SeedSequence,
    requests: Queue,
    replies: Queue,
    completed_games: Queue,
    stop: Event,
) -> None:
    rng = np.random.default_rng(seed)
    game_id = 0

    def send(queue: Queue, message) -> None:
        while not stop.is_set():
            try:
                queue.put(message, timeout=0.1)
                return
            except Full:
                pass
        raise WorkerStopped

    def receive():
        while not stop.is_set():
            try:
                return replies.get(timeout=0.1)
            except Empty:
                pass
        raise WorkerStopped

    def rpc(kind: str, payload=None):
        send(requests, (kind, actor_id, payload))
        reply_kind, result = receive()

        if reply_kind == "error":
            raise RuntimeError(result)
        if reply_kind != kind:
            raise RuntimeError(f"Unexpected reply: {reply_kind}")

        return result

    try:
        while not stop.is_set():
            version = rpc("acquire", game_id)

            def predict(
                planes: np.ndarray,
                network_version: int = version,
            ) -> tuple[np.ndarray, np.ndarray]:
                return rpc("predict", (network_version, np.ascontiguousarray(planes)))

            samples = play_game(predict, cfg, rng)
            rpc("release", version)
            send(completed_games, (actor_id, game_id, version, samples))
            game_id += 1

    except WorkerStopped:
        return

def _temperature(ply: int, config: Config) -> float:
    return config.temperature if ply < config.temperature_moves else 0.0

def play_game(predict: Predict, config: Config, rng: np.random.Generator | None = None) -> list[Sample]:
    if rng is None:
        rng = np.random.default_rng()
    env = ChessEnv.startpos()
    mcts = MCTS(predict, config, rng)
    raw: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not env.is_terminal() and env.ply < config.max_plies:
        res = mcts.search(env, add_noise=True)
        raw.append((encode(env, config), res.pi, env.board.turn))
        move, i = select_action(res.root, _temperature(env.ply, config), rng)
        env.step(move)
        mcts.advance(i)

    outcome = env.outcome()
    z = 0 if outcome is None else outcome
    return [Sample(planes=p, pi=pi, value=float(z * player)) for p, pi, player in raw]
