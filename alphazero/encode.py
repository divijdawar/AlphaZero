from __future__ import annotations
import numpy as np
from engine.board import *
from engine.env import ChessEnv
from engine.movegen import KNIGHT_DELTAS, Move
from .config import Config

_DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
_DIR_ID = {d: i for i, d in enumerate(_DIRS)}
_KNIGHT_ID = {d: i for i, d in enumerate(KNIGHT_DELTAS)}
_UNDERPROMO = {KNIGHT: 0, BISHOP: 1, ROOK: 2}

def _flip(r: int, turn: int) -> int:
    return 7 - r if turn == BLACK else r

def move_to_index(move: Move, turn: int, config: Config) -> int:
    fr, fc = _flip(move.fr, turn), move.fc
    dr, dc = _flip(move.tr, turn) - fr, move.tc - fc
    base = (fr * 8 + fc) * config.num_policy_planes 

    if move.promo and move.promo is not QUEEN:
        if move.promo not in _UNDERPROMO or dc not in (-1, 0, 1):
            raise ValueError(f"bad underpromotion: {move}")
        # planes 64-72: 3 pieces x (capture-left, push, capture-right)
        return base + 64 + 3 * _UNDERPROMO[move.promo] + (dc + 1)

    if (dr, dc) in _KNIGHT_ID:
        return base + 56 + _KNIGHT_ID[(dr, dc)]

    dist = max(abs(dr), abs(dc))
    unit = ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))
    if dist < 1 or dist > 7 or abs(dr) not in (0, dist) or abs(dc) not in (0, dist):
        raise ValueError(f"move is not a queen/knight displacement: {move}")
    # planes 0-55: (distance-1) * 8 + direction
    return base + (dist - 1) * 8 + _DIR_ID[unit]

def moves_to_mask(moves: list[Move], turn: int, config: Config) -> np.ndarray:
    """Bool mask over 4672 actions with exactly the legal moves set."""
    mask = np.zeros(config.num_actions, dtype=bool)
    for m in moves:
        mask[move_to_index(m, turn, config)] = True
    return mask

def encode(env: ChessEnv, config:Config) -> np.ndarray:
    """Encode a position as float32 planes, shape (steps*14+7, 8, 8).

    Piece planes come from env.frames() (oldest first, padded with the
    initial position). Repetition planes use env.counts with the same
    normalized position_key the draw logic uses. Constant planes describe
    the current board: colour (ones iff black to move), fullmove number,
    perspective-ordered castling rights (P1 = side to move), halfmove clock.
    Count features are scaled to ~[0, 1]; everything else is 0/1.
    """
    step_planes = config.num_piece_planes + config.num_rep_planes
    const = config.history_steps * step_planes 
    planes = np.zeros((const + config.num_const_planes, 8, 8), dtype=np.float32)
    turn = env.board.turn
    black = turn == BLACK

    for t, (board, key) in enumerate(zip(env.frames(config.history_steps), env.frame_keys(config.history_steps))):
        off = t * step_planes
        grid = board.board
        for r in range(8):
            pr = 7 - r if black else r
            for c in range(8):
                v = int(grid[r, c])
                if v == 0:
                    continue
                own = (v > 0) == (turn == WHITE)
                planes[off + (abs(v) - 1) + (0 if own else 6), pr, c] = 1.0
        reps = env.counts[key]  # occurrences incl. current
        if reps >= 2:
            planes[off + 12, :, :] = 1.0  # seen once before -> 2-fold risk
        if reps >= 3:
            planes[off + 13, :, :] = 1.0  # seen twice before -> 3-fold draw

    b = env.board
    if black:
        planes[const + 0, :, :] = 1.0
    planes[const + 1, :, :] = b.fullmove / 100.0
    p1k, p1q, p2k, p2q = (1, 2, 4, 8) if not black else (4, 8, 1, 2)
    if b.castling & p1k:
        planes[const + 2, :, :] = 1.0
    if b.castling & p1q:
        planes[const + 3, :, :] = 1.0
    if b.castling & p2k:
        planes[const + 4, :, :] = 1.0
    if b.castling & p2q:
        planes[const + 5, :, :] = 1.0
    planes[const + 6, :, :] = b.halfmove / 100.0

    return np.ascontiguousarray(planes)
