from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import math
import numpy as np
from engine.env import ChessEnv
from engine.movegen import Move
from .config import Config
from .encode import encode, move_to_index

Predict = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

@dataclass
class SearchResult:
    pi: np.ndarray   # (num_actions,) normalized root visit distribution
    root: Node

class Node:
    __slots__ = (
        "N",
        "P",
        "P0",
        "W",
        "children",
        "env",
        "expanded",
        "move",
        "moves",
        "parent",
        "terminal",
        "value",
    )

    def __init__(self, env: ChessEnv, parent: Node | None = None, move: Move | None = None):
        self.env = env
        self.parent, self.move = parent, move  
        self.children: dict[int, Node] = {}
        self.moves: list[Move] = []
        self.P0 = self.P = self.N = self.W = None   # P0 = raw net priors, P = working priors
        self.terminal = env.is_terminal()
        self.value = env.result_for(env.board.turn) if self.terminal else 0.0
        self.expanded = False

    @property
    def Q(self) -> np.ndarray:
        return np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N > 0)

    def expand(self, logits: np.ndarray, config: Config) -> None:
        self.moves = self.env.legal_moves()
        turn = self.env.board.turn
        idx = np.fromiter(
            (move_to_index(m, turn, config) for m in self.moves),
            dtype=np.int, count=len(self.moves),
        )
        z = logits[idx]
        p = np.exp(z - z.max())
        p /= p.sum()
        self.P0 = p.astype(np.float32)
        self.P = self.P0.copy()
        self.N = np.zeros(len(self.moves), dtype=np.int32)
        self.W = np.zeros(len(self.moves), dtype=np.float32)
        self.expanded = True

def puct_scores(node: Node, c_puct: float) -> np.ndarray:
    if not node.expanded or node.terminal or len(node.moves) == 0:
        raise ValueError("selection requires an expanded, non-terminal node")
    total = int(node.N.sum())
    u = c_puct * node.P * math.sqrt(total) / (1.0 + node.N)
    return node.Q + u

def select_idx(node: Node, c_puct: float) -> int:
    return int(np.argmax(puct_scores(node, c_puct)))

def child(node: Node, i: int) -> Node:
    c = node.children.get(i)
    if c is None:
        env = node.env.clone()
        env.step(node.moves[i])
        c = Node(env, node, node.moves[i])
        node.children[i] = c
    return c

def descend(root: Node, c_puct: float) -> tuple[Node, list[tuple[Node, int]]]:
    node, path = root, []
    while node.expanded and not node.terminal:
        i = select_idx(node, c_puct)
        path.append((node, i))
        node = child(node, i)
    return node, path

def backup(path: list[tuple[Node, int]], v: float) -> None:
    for node, i in reversed(path):
        v = -v
        node.N[i] += 1
        node.W[i] += v

def _dirichlet_alpha(config: Config) -> float:
    a = config.dirichlet_alpha
    return float(a[0]) if isinstance(a, tuple) else float(a)  # chess = index 0

def add_dirichlet_noise(root: Node, config: Config, rng: np.random.Generator) -> None:
    noise = rng.dirichlet([_dirichlet_alpha(config)] * len(root.moves)).astype(np.float32)
    eps = config.dirichlet_epsilon
    root.P = (1.0 - eps) * root.P0 + eps * noise

def run_mcts(
    root_env: ChessEnv,
    predict: Predict,
    config: Config,
    *,
    root: Node | None = None,
    add_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    if rng is None:
        rng = np.random.default_rng()
    if root is None:
        root = Node(root_env)
        logits, _ = predict(encode(root_env, config)[None])
        root.expand(logits[0], config)
    if add_noise:
        add_dirichlet_noise(root, config, rng)

    for _ in range(config.num_simulations):
        leaf, path = descend(root, config.c_puct)
        if leaf.terminal:
            v = leaf.value
        else:
            logits, value = predict(encode(leaf.env, config)[None])
            leaf.expand(logits[0], config)
            v = float(value[0])
        backup(path, v)

    return SearchResult(pi=visits_to_pi(root, config), root=root)

def visits_to_pi(root: Node, config: Config) -> np.ndarray:
    """Dense (num_actions,) distribution from root visit counts."""
    pi = np.zeros(config.num_actions, dtype=np.float32)
    if not root.expanded:
        return pi
    turn = root.env.board.turn
    idx = [move_to_index(m, turn, config) for m in root.moves]
    pi[idx] = root.N
    total = pi.sum()
    if total > 0:
        pi /= total
    return pi

def select_action(root: Node, temperature: float, rng: np.random.Generator | None = None) -> tuple[Move, int]:
    if rng is None:
        rng = np.random.default_rng()
    n = root.N.astype(np.float64)
    if temperature <= 1e-8 or n.sum() == 0:
        i = int(np.argmax(n))
    else:
        p = n ** (1.0 / temperature)
        p /= p.sum()
        i = int(rng.choice(len(n), p=p))
    return root.moves[i], i

class MCTS:
    def __init__(self, predict: Predict, config: Config, rng: np.random.Generator | None = None):
        self.predict = predict
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.root: Node | None = None

    def search(self, env: ChessEnv, *, add_noise: bool = True) -> SearchResult:
        res = run_mcts(
            env, self.predict, self.config,
            root=self.root, add_noise=add_noise, rng=self.rng,
        )
        self.root = res.root
        return res

    def select_action(self, temperature: float) -> tuple[Move, int]:
        return select_action(self.root, temperature, self.rng)

    def advance(self, i: int) -> None:
        if self.root is None:
            return
        c = self.root.children.get(i)
        if c is not None:
            c.parent = None
        self.root = c
