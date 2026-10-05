from __future__ import annotations
from collections import Counter
from .board import *
from .movegen import Move, is_attacked, legal_moves, make_move

FIFTY_MOVE_HALFMOVES = 100  # AlphaZero automatically draws after 50 moves per side.

def _king_square(board: Board, king: int) -> tuple[int, int]:
    for r in range(8):
        for c in range(8):
            if int(board.board[r, c]) == king:
                return r, c
    raise ValueError("king not found")

def _normalize_ep(board: Board) -> tuple[int, int] | None:
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
    moves = legal_moves(board) if candidates else ()
    return board.ep if any(move in moves for move in candidates) else None

def position_key(board: Board) -> tuple:
    return (board.board.tobytes(), board.turn, board.castling, _normalize_ep(board))

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
    def __init__(self, board: Board | None = None, history: list[Board] | None = None) -> None:
        if history:
            if board is not None:
                assert board is history[-1] or board == history[-1], (
                    "board must match history[-1]"
                )
            board = history[-1]
        elif board is None:
            board = Board.startpos()
        self.board = board
        self.history: list[Board] = list(history) if history else [self.board]
        self.counts: Counter = Counter(position_key(b) for b in self.history)
        self._outcome: int | None = None

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
        return legal_moves(self.board)

    def step(self, move: Move) -> int | None:
        if self._outcome is not None:
            raise ValueError("game is already over")
        if move not in set(self.legal_moves()):
            raise ValueError(f"illegal move: {move.to_uci()}")
        self.board = make_move(self.board, move)
        self.history.append(self.board)
        self.counts[position_key(self.board)] += 1
        self._outcome = self._compute_outcome()
        return self._outcome

    def clone(self) -> ChessEnv:
        return ChessEnv(self.board.copy(), [b.copy() for b in self.history])

    def is_terminal(self) -> bool:
        return self.outcome() is not None

    def outcome(self) -> int | None:
        if self._outcome is None:
            self._outcome = self._compute_outcome()
        return self._outcome

    def result_for(self, color: int) -> int | None:
        o = self.outcome()
        return None if o is None else o * color

    def frames(self, n: int) -> list[Board]:
        if n <= 1:
            return [self.board]
        tail = self.history[-n:]
        return [self.history[0]] * (n - len(tail)) + tail

    def to_fen(self) -> str:
        return self.board.to_fen()

    def __repr__(self) -> str:
        return f"ChessEnv({self.to_fen()!r}, ply={self.ply})"

    def _compute_outcome(self) -> int | None:
        if not legal_moves(self.board):
            enemy = -self.board.turn
            kr, kc = _king_square(self.board, KING * self.board.turn)
            return -self.board.turn if is_attacked(self.board, kr, kc, enemy) else 0
        if (
            self.board.halfmove >= FIFTY_MOVE_HALFMOVES
            or self.counts[position_key(self.board)] >= 3
            or _insufficient_material(self.board)
        ):
            return 0
        return None
