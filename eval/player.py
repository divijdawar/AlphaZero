from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic
import numpy as np

from engine.env import ChessEnv
from engine.movegen import Move
from alphazero.encode import encode
from alphazero.mcts import Node, backup, descend
from .checkpoints import Checkpoint, load_network


def parse_move(env: ChessEnv, uci: str) -> Move:
    for move in env.legal_moves():
        if move.to_uci() == uci:
            return move
    raise ValueError(f"illegal move {uci!r} in {env.to_fen()}")


def history_key(env: ChessEnv) -> tuple:
    return tuple(b.to_fen() for b in env.history)


@dataclass
class Decision:
    uci: str | None
    simulations: int
    elapsed: float
    value: float | None
    pv: list[str]


class ModelPlayer:
    def __init__(self, checkpoint: Checkpoint, device: str, *, warm: bool = True):
        self.checkpoint, self.cfg = checkpoint, checkpoint.config
        self.net, self.predict = load_network(checkpoint, device)
        self.root: Node | None = None
        if warm:
            self.predict(encode(ChessEnv.startpos(), self.cfg)[None])

    def reset(self):
        self.root = None

    def observe(self, uci: str):
        if self.root is None or not self.root.expanded:
            return
        i = next((i for i, m in enumerate(self.root.moves) if m.to_uci() == uci), None)
        self.root = self.root.children.get(i) if i is not None else None
        if self.root is not None:
            self.root.parent = None

    def choose(self, env: ChessEnv, seconds: float | None = 1.0, *,
               simulations: int | None = None, stop: Event | None = None) -> Decision:
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("thinking time must be positive and finite")
        if simulations is not None and simulations < 1:
            raise ValueError("simulations must be positive")
        if seconds is None and simulations is None and stop is None:
            raise ValueError("search needs a time, simulation, or cancellation limit")
        start, count = monotonic(), 0
        deadline = start + seconds if seconds is not None else float("inf")
        if env.is_terminal():
            return Decision(None, 0, monotonic() - start, float(env.result_for(env.board.turn)), [])
        if self.root is None or history_key(self.root.env) != history_key(env):
            self.root = Node(env.clone())
        root = self.root
        if not root.expanded:
            logits, _ = self.predict(encode(root.env, self.cfg)[None])
            root.expand(logits[0], self.cfg)
        # No Dirichlet noise or stochastic move selection in evaluation.
        while monotonic() < deadline and not (stop and stop.is_set()):
            if simulations is not None and count >= simulations:
                break
            leaf, path = descend(root, self.cfg.c_puct)
            if leaf.terminal:
                value = leaf.value
            else:
                logits, values = self.predict(encode(leaf.env, self.cfg)[None])
                leaf.expand(logits[0], self.cfg)
                value = float(values[0])
            backup(path, value)
            count += 1
        # If a deadline expires before any simulation, return the highest-prior legal move.
        i = int(np.argmax(root.N if root.N.sum() else root.P0))
        pv, node = [], root
        for _ in range(12):
            if not node.expanded or node.terminal or not node.moves:
                break
            j = int(np.argmax(node.N if node.N.sum() else node.P0))
            pv.append(node.moves[j].to_uci())
            node = node.children.get(j)
            if node is None:
                break
        value = float(root.Q[i]) if root.N[i] else None
        return Decision(root.moves[i].to_uci(), count, monotonic() - start, value, pv)


class StockfishPlayer:
    def __init__(self, path: str, threads: int, hash_mb: int, syzygy: str = ""):
        import chess.engine
        from .runtime import ROOT, digest
        self.engine = chess.engine.SimpleEngine.popen_uci(path, cwd=str(ROOT), timeout=30)
        self.engine.configure({"Threads": threads, "Hash": hash_mb, "Skill Level": 20,
                               "SyzygyPath": syzygy})
        self.description = {"path": path, "sha256": digest(path), "id": self.engine.id,
                            "threads": threads, "hash_mb": hash_mb, "syzygy": syzygy,
                            "skill_level": 20, "ponder": False}
        self.game = object()

    def reset(self):
        self.game = object()

    def observe(self, uci: str):
        pass

    def choose(self, env: ChessEnv, seconds: float | None = 1.0, *, board=None, clocks=None) -> Decision:
        import chess
        import chess.engine
        if board is None:
            board = chess.Board(env.history[0].to_fen())
            for before, after in zip(env.history, env.history[1:]):
                found = False
                for move in board.legal_moves:
                    b = board.copy()
                    b.push(move)
                    if b.fen(en_passant="fen") == after.to_fen():
                        board.push(move)
                        found = True
                        break
                if not found:
                    raise ValueError("cannot reconstruct move history")
        limit = chess.engine.Limit(time=seconds) if clocks is None else chess.engine.Limit(
            white_clock=clocks[0], black_clock=clocks[1], white_inc=clocks[2], black_inc=clocks[2])
        start = monotonic()
        result = self.engine.play(board, limit, game=self.game, ponder=False, info=chess.engine.INFO_SCORE)
        score = result.info.get("score")
        cp = score.pov(board.turn).score(mate_score=100000) if score else None
        return Decision(result.move.uci() if result.move else None, 0, monotonic() - start,
                        cp, [result.move.uci()] if result.move else [])

    def close(self):
        self.engine.quit()

