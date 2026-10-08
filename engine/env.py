from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
from types import MappingProxyType
from .board import *
from .movegen import Move, _apply, is_attacked, legal_moves

FIFTY_MOVE_HALFMOVES = 100  # AlphaZero automatically draws after 50 moves per side.

def _king_square(board: Board, king: int) -> tuple[int, int]:
    for r in range(8):
        for c in range(8):
            if int(board.board[r, c]) == king:
                return r, c
    raise ValueError("king not found")

def _normalize_ep(board: Board, moves: Sequence[Move] | None = None) -> tuple[int, int] | None:
    if board.ep is None:
        return None
    r, c = board.ep
    pr = r + (1 if board.turn == WHITE else -1)
    pawn = PAWN * board.turn
    candidates = [
        Move(pr, fc, r, c) for fc in (c - 1, c + 1)
        if in_bounds(pr, fc) and int(board.board[pr, fc]) == pawn
    ]
    # An adjacent pawn may be pinned, or removing both pawns may expose its king.
    if moves is None:
        moves = legal_moves(board) if candidates else ()
    return board.ep if any(move in moves for move in candidates) else None

def position_key(board: Board, moves: Sequence[Move] | None = None) -> tuple:
    return (board.board.tobytes(), board.turn, board.castling, _normalize_ep(board, moves))

def _insufficient_material(board: Board) -> bool:
    b = board.board
    for pt in (PAWN, ROOK, QUEEN):
        if bool((abs(b) == pt).any()):
            return False
    if int((abs(b) == KNIGHT).sum()) > 0:
        return int(((abs(b) == KNIGHT) | (abs(b) == BISHOP)).sum()) <= 1
    colors = set()
    for r in range(8):
        for c in range(8):
            if abs(int(b[r, c])) == BISHOP:
                colors.add((r + c) & 1)
                if len(colors) > 1:
                    return False
    return True

class ChessEnv:
    def __init__(self, board: Board | None = None, history: Sequence[Board] | None = None) -> None:
        if history:
            if board is not None:
                assert board is history[-1] or board == history[-1], (
                    "board must match history[-1]"
                )
            board = history[-1]
        elif board is None:
            board = Board.startpos()
        self._history = tuple(b.frozen_copy() for b in history) if history else (board.frozen_copy(),)
        self._history_keys = tuple(position_key(b) for b in self._history)
        self._counts = Counter(self._history_keys)
        self._legal_moves: tuple[Move, ...] | None = None
        self._outcome: int | None = None
        self._outcome_checked = False

    @property
    def board(self) -> Board:
        return self._history[-1]

    @property
    def history(self) -> tuple[Board, ...]:
        return self._history

    @property
    def counts(self):
        return MappingProxyType(self._counts)

    @property
    def search_key(self) -> tuple:
        return self._history_keys, self.board.halfmove, self.board.fullmove

    @classmethod
    def startpos(cls) -> ChessEnv:
        return cls()

    @classmethod
    def from_fen(cls, fen: str) -> ChessEnv:
        return cls(Board.from_fen(fen))

    @property
    def ply(self) -> int:
        return len(self.history) - 1

    def legal_moves(self) -> list[Move]:
        return list(self._get_legal_moves())

    def _get_legal_moves(self) -> tuple[Move, ...]:
        if self._legal_moves is None:
            self._legal_moves = tuple(legal_moves(self.board))
        return self._legal_moves

    def step(self, move: Move) -> int | None:
        if self.is_terminal():
            raise ValueError("game is already over")
        if move not in self._get_legal_moves():
            raise ValueError(f"illegal move: {move.to_uci()}")
        return self._step_legal(move)

    def _step_legal(self, move: Move) -> int | None:
        """Apply a move taken from this nonterminal position's legal moves."""
        assert self._outcome_checked and self._outcome is None
        board = _apply(self.board, move).frozen_copy()
        self._history += (board,)
        self._legal_moves = None
        key = position_key(board, self._get_legal_moves())
        self._history_keys += (key,)
        self._counts[key] += 1
        self._outcome_checked = False
        return self.outcome()

    def clone(self) -> ChessEnv:
        clone = object.__new__(ChessEnv)
        clone._history = self._history
        clone._history_keys = self._history_keys
        clone._counts = self._counts.copy()
        clone._legal_moves = self._legal_moves
        clone._outcome = self._outcome
        clone._outcome_checked = self._outcome_checked
        return clone

    def is_terminal(self) -> bool:
        return self.outcome() is not None

    def outcome(self) -> int | None:
        if not self._outcome_checked:
            self._outcome = self._compute_outcome()
            self._outcome_checked = True
        return self._outcome

    def result_for(self, color: int) -> int | None:
        o = self.outcome()
        return None if o is None else o * color

    def frames(self, n: int) -> list[Board]:
        if n <= 1:
            return [self.board]
        tail = self.history[-n:]
        return [self.history[0]] * (n - len(tail)) + list(tail)

    def frame_keys(self, n: int) -> list[tuple]:
        if n <= 1:
            return [self._history_keys[-1]]
        tail = self._history_keys[-n:]
        return [self._history_keys[0]] * (n - len(tail)) + list(tail)

    def to_fen(self) -> str:
        return self.board.to_fen()

    def __repr__(self) -> str:
        return f"ChessEnv({self.to_fen()!r}, ply={self.ply})"

    def _compute_outcome(self) -> int | None:
        if not self._get_legal_moves():
            enemy = -self.board.turn
            kr, kc = _king_square(self.board, KING * self.board.turn)
            return -self.board.turn if is_attacked(self.board, kr, kc, enemy) else 0
        if (
            self.board.halfmove >= FIFTY_MOVE_HALFMOVES
            or self._counts[self._history_keys[-1]] >= 3
            or _insufficient_material(self.board)
        ):
            return 0
        return None
