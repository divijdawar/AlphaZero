from dataclasses import dataclass
import numpy as np
from engine.env import ChessEnv
from .config import Config
from .encode import encode
from .mcts import MCTS, Predict, select_action

__all__ = ["Sample", "play_game"]

@dataclass
class Sample:
    planes: np.ndarray   # (num_planes, 8, 8) float32
    pi: np.ndarray       # (num_actions,) float32
    value: float         # outcome z for this position's mover

def _temperature(ply: int, config: Config) -> float:
    return config.temperature if ply < config.temperature_moves else 0.0

def play_game(predict: Predict, config: Config, rng: np.random.Generator | None = None) -> list[Sample]:
    """Play one self-play game; return labeled samples (empty if no moves)."""
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
    z = 0 if outcome is None else outcome   # hit max_plies -> draw
    return [Sample(planes=p, pi=pi, value=float(z * player)) for p, pi, player in raw]
