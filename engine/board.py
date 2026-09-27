"""Board representation — pure state, no move generation yet.

Layout:
  board: int8[8,8], row 0 = rank 8, col 0 = file a.
  0 = empty, +P,N,B,R,Q,K = 1..6 white, -1..-6 black. Sign = color.
  turn: +1 white to move, -1 black to move.
  castling: bitmask WK=1, WQ=2, BK=4, BQ=8.
  ep: en-passant target square (r, c) or None. This is the square
      a pawn would land on, e.g. after 1.e4 ep = e3 square.
  halfmove: clock for 50-move rule. fullmove: starts at 1, ++ after black.
"""

from __future__ import annotations
import numpy as np

__all__ = ["ALL_CASTLING", "BISHOP", "BK", "BLACK", "BQ", "EMPTY", "KING", "KNIGHT", "PAWN", "QUEEN", "ROOK", "START_FEN", "WHITE", "WK", "WQ", "Board", "in_bounds", "parse_square", "square_name"]

EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

WK, WQ, BK, BQ = 1, 2, 4, 8
ALL_CASTLING = WK | WQ | BK | BQ

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_FEN_TO_PIECE = {
    "P": PAWN, "N": KNIGHT, "B": BISHOP,
    "R": ROOK, "Q": QUEEN, "K": KING,
}
_PIECE_TO_FEN = {v: k for k, v in _FEN_TO_PIECE.items()}

def square_name(r: int, c: int) -> str:
    if not in_bounds(r, c):
        raise ValueError(f"square out of bounds: {(r, c)}")
    return f"{chr(ord('a') + c)}{8 - r}"

def parse_square(s: str) -> tuple[int, int]:
    if len(s) != 2 or s[0] not in "abcdefgh" or s[1] not in "12345678":
        raise ValueError(f"bad square: {s!r}")
    return 8 - int(s[1]), ord(s[0]) - ord("a")

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8

class Board:
    def __init__(
        self,
        board: np.ndarray | None = None,
        turn: int = WHITE,
        castling: int = ALL_CASTLING,
        ep: tuple[int, int] | None = None,
        halfmove: int = 0,
        fullmove: int = 1,
    ):
        self.board = np.zeros((8, 8), dtype=np.int8) if board is None else board.astype(np.int8, copy=True)
        self.turn = turn
        self.castling = castling
        self.ep = ep
        self.halfmove = halfmove
        self.fullmove = fullmove

    @classmethod
    def startpos(cls) -> Board:
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError(f"bad FEN (need 6 fields): {fen!r}")
        placement, stm, castling_s, ep_s, half_s, full_s = parts

        board = np.zeros((8, 8), dtype=np.int8)
        r, c = 0, 0
        for ch in placement:
            if ch == "/":
                if c != 8:
                    raise ValueError(f"bad FEN row length at row {r}")
                r += 1
                if r > 7:
                    raise ValueError(f"too many FEN rows: {placement!r}")
                c = 0
            elif ch in "12345678":
                n = int(ch)
                if c + n > 8:
                    raise ValueError(f"bad FEN row length at row {r}")
                c += n
            else:
                if r > 7 or c > 7:
                    raise ValueError(f"FEN overflow at {ch}")
                if ch.upper() not in _FEN_TO_PIECE:
                    raise ValueError(f"bad FEN piece: {ch!r}")
                v = _FEN_TO_PIECE[ch.upper()]
                board[r, c] = v if ch.isupper() else -v
                c += 1
        if r != 7 or c != 8:
            raise ValueError(f"bad FEN placement: {placement!r}")

        # sanity: pawns never on rank 1/8, exactly one king per side
        for cc in range(8):
            if abs(int(board[0, cc])) == PAWN or abs(int(board[7, cc])) == PAWN:
                raise ValueError(f"pawn on back rank: {placement!r}")
        if int(np.sum(board == KING)) != 1 or int(np.sum(board == -KING)) != 1:
            raise ValueError(f"FEN must have exactly one king per side: {placement!r}")

        turn = WHITE if stm == "w" else BLACK if stm == "b" else None
        if turn is None:
            raise ValueError(f"bad side to move: {stm!r}")

        castling = 0
        if castling_s != "-":
            for ch in castling_s:
                if ch == "K":
                    castling |= WK
                elif ch == "Q":
                    castling |= WQ
                elif ch == "k":
                    castling |= BK
                elif ch == "q":
                    castling |= BQ
                else:
                    raise ValueError(f"bad castling char: {ch!r}")

        ep = None if ep_s == "-" else parse_square(ep_s)
        if ep is not None:
            # ep square is where a pawn could capture to: rank 6 wtm, rank 3 btm
            ep_rank = 8 - ep[0]
            if (turn == WHITE and ep_rank != 6) or (turn == BLACK and ep_rank != 3):
                raise ValueError(f"bad EP square for side to move: {ep_s!r}")

        try:
            halfmove = int(half_s)
            fullmove = int(full_s)
        except ValueError:
            raise ValueError(f"bad clocks: {half_s!r} {full_s!r}")
        if halfmove < 0 or fullmove < 1:
            raise ValueError(f"bad clocks: {half_s!r} {full_s!r}")

        return cls(board, turn, castling, ep, halfmove, fullmove)

    def to_fen(self) -> str:
        rows = []
        for r in range(8):
            empty = 0
            row = ""
            for c in range(8):
                p = int(self.board[r, c])
                if p == 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    f = _PIECE_TO_FEN[abs(p)]
                    row += f if p > 0 else f.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        castling_s = ""
        if self.castling & WK:
            castling_s += "K"
        if self.castling & WQ:
            castling_s += "Q"
        if self.castling & BK:
            castling_s += "k"
        if self.castling & BQ:
            castling_s += "q"
        return (
            f"{'/'.join(rows)} {'w' if self.turn == WHITE else 'b'} "
            f"{castling_s or '-'} "
            f"{square_name(*self.ep) if self.ep else '-'} "
            f"{self.halfmove} {self.fullmove}"
        )

    def copy(self) -> Board:
        return Board(self.board, self.turn, self.castling, self.ep, self.halfmove, self.fullmove)

    def piece_at(self, r: int, c: int) -> int:
        return int(self.board[r, c])

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Board)
            and self.turn == o.turn
            and self.castling == o.castling
            and self.ep == o.ep
            and self.halfmove == o.halfmove
            and self.fullmove == o.fullmove
            and bool(np.array_equal(self.board, o.board))
        )

    def __repr__(self) -> str:
        return f"Board({self.to_fen()!r})"
