"""Move generation — operates on Board, owns the Move type.

Board stays state-only. Everything here is a pure function of
(Board, Move): attack tests, pseudo-legal gen, make_move, legal filter.
"""

from __future__ import annotations
from dataclasses import dataclass
from .board import *

KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_DELTAS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PROMO_PIECES = (KNIGHT, BISHOP, ROOK, QUEEN)
QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS
UNDERPROMO = {KNIGHT: 0, BISHOP: 1, ROOK: 2}

MOVES = {
    KNIGHT: (KNIGHT_DELTAS, 1),
    KING: (KING_DELTAS, 1),
    BISHOP: (BISHOP_DIRS, 7),
    ROOK: (ROOK_DIRS, 7),
    QUEEN: (ROOK_DIRS + BISHOP_DIRS, 7),
}

_CORNER_RIGHT = {(7, 7): WK, (7, 0): WQ, (0, 7): BK, (0, 0): BQ}

@dataclass(frozen=True)
class Move:
    fr: int
    fc: int
    tr: int
    tc: int
    promo: int = 0  # 0 = none, else KNIGHT/BISHOP/ROOK/QUEEN

    def to_uci(self) -> str:
        s = square_name(self.fr, self.fc) + square_name(self.tr, self.tc)
        if self.promo:
            s += {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}[self.promo]
        return s

def _check_kings(board: Board) -> None:
    if int((board.board == KING).sum()) != 1 or int((board.board == -KING).sum()) != 1:
        raise ValueError("both kings must be on the board")

def _find_king(board: Board, king: int) -> tuple[int, int]:
    for r in range(8):
        for c in range(8):
            if int(board.board[r, c]) == king:
                return r, c
    raise ValueError("king not found")

def _strip_corner_right(cast: int, r: int, c: int) -> int:
    return cast & ~_CORNER_RIGHT.get((r, c), 0)

def _promo_ok(piece: int, move: Move, last: int) -> bool:
    is_pawn = abs(piece) == PAWN
    if move.promo:
        return is_pawn and move.tr == last and move.promo in PROMO_PIECES
    return not (is_pawn and move.tr == last)

def is_attacked(board: Board, r: int, c: int, by_color: int) -> bool:
    dr = 1 if by_color == WHITE else -1
    for dc in (-1, 1):
        rr, cc = r + dr, c + dc
        if in_bounds(rr, cc) and int(board.board[rr, cc]) == PAWN * by_color:
            return True
    for deltas, piece in ((KNIGHT_DELTAS, KNIGHT), (KING_DELTAS, KING)):
        for dr, dc in deltas:
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc) and int(board.board[rr, cc]) == piece * by_color:
                return True
    for dirs, sliders in ((ROOK_DIRS, (ROOK, QUEEN)), (BISHOP_DIRS, (BISHOP, QUEEN))):
        for dr, dc in dirs:
            for step in range(1, 8):
                rr, cc = r + dr * step, c + dc * step
                if not in_bounds(rr, cc):
                    break
                p = int(board.board[rr, cc])
                if p == EMPTY:
                    continue
                if (p > 0) == (by_color > 0) and abs(p) in sliders:
                    return True
                break
    return False

def piece_moves(board: Board, fr: int, fc: int, out: list[Move]) -> None:
    piece = int(board.board[fr, fc])
    if abs(piece) == PAWN:
        pawn_moves(board, fr, fc, out)
        return
    dirs, max_steps = MOVES[abs(piece)]
    for dr, dc in dirs:
        for step in range(1, max_steps + 1):
            r, c = fr + dr * step, fc + dc * step
            if not in_bounds(r, c):
                break
            target = int(board.board[r, c])
            if target == EMPTY:
                out.append(Move(fr, fc, r, c))
                continue
            if (target > 0) != (piece > 0):
                out.append(Move(fr, fc, r, c))
            break

def _pawn_target(out: list[Move], fr: int, fc: int, tr: int, tc: int, last: int) -> None:
    if tr == last:
        for p in PROMO_PIECES:
            out.append(Move(fr, fc, tr, tc, p))
    else:
        out.append(Move(fr, fc, tr, tc))

def pawn_moves(board: Board, fr: int, fc: int, out: list[Move]) -> None:
    piece = int(board.board[fr, fc])
    color = WHITE if piece > 0 else BLACK
    dr = -1 if color == WHITE else 1
    start = 6 if color == WHITE else 1
    last = 0 if color == WHITE else 7
    r = fr + dr
    if in_bounds(r, fc) and int(board.board[r, fc]) == EMPTY:
        _pawn_target(out, fr, fc, r, fc, last)
        r2 = fr + 2 * dr
        if fr == start and int(board.board[r2, fc]) == EMPTY:
            out.append(Move(fr, fc, r2, fc))
    for dc in (-1, 1):
        c = fc + dc
        if not in_bounds(r, c):
            continue
        target = int(board.board[r, c])
        if target != EMPTY and (target > 0) != (color > 0):
            _pawn_target(out, fr, fc, r, c, last)
        elif board.ep == (r, c):
            out.append(Move(fr, fc, r, c))

def castle_moves(board: Board, out: list[Move]) -> None:
    color = board.turn
    row, rights = (7, (WK, WQ)) if color == WHITE else (0, (BK, BQ))
    if int(board.board[row, 4]) != KING * color:
        return
    for right, empties, rook_col, dest_col in (
        (rights[0], (5, 6), 7, 6),
        (rights[1], (1, 2, 3), 0, 2),
    ):
        if board.castling & right and all(int(board.board[row, c]) == EMPTY for c in empties) and int(board.board[row, rook_col]) == ROOK * color:
            out.append(Move(row, 4, row, dest_col))

def pseudo_legal_moves(board: Board) -> list[Move]:
    _check_kings(board)
    return _gen_pseudo(board)

def _gen_pseudo(board: Board) -> list[Move]:
    color = board.turn
    out: list[Move] = []
    for r in range(8):
        for c in range(8):
            p = int(board.board[r, c])
            if p == EMPTY or (p > 0) != (color == WHITE):
                continue
            piece_moves(board, r, c, out)
    castle_moves(board, out)
    return out

def legal_moves(board: Board) -> list[Move]:
    _check_kings(board)
    enemy = -board.turn
    kr, kc = _find_king(board, KING * board.turn)
    out: list[Move] = []
    for move in _gen_pseudo(board):
        mover = abs(int(board.board[move.fr, move.fc]))
        if mover == KING and abs(move.tc - move.fc) == 2:
            if is_attacked(board, move.fr, move.fc, enemy):
                continue
            step = 1 if move.tc > move.fc else -1
            if is_attacked(board, move.fr, move.fc + step, enemy):
                continue
        nb = _apply(board, move)
        dr, dc = (move.tr, move.tc) if mover == KING else (kr, kc)
        if not is_attacked(nb, dr, dc, enemy):
            out.append(move)
    return out

def make_move(board: Board, move: Move) -> Board:
    _check_kings(board)
    if not (in_bounds(move.fr, move.fc) and in_bounds(move.tr, move.tc)):
        raise ValueError(f"move out of bounds: {move}")
    piece = int(board.board[move.fr, move.fc])
    if piece == EMPTY or (piece > 0) != (board.turn == WHITE):
        raise ValueError(f"no mover piece: {move}")
    color = board.turn
    last_rank = 0 if color == WHITE else 7
    if not _promo_ok(piece, move, last_rank):
        raise ValueError(f"bad promotion: {move}")
    if move not in _gen_pseudo(board):
        raise ValueError(f"illegal move: {move.to_uci()}")
    return _apply(board, move)

def _apply(board: Board, move: Move) -> Board:
    piece = int(board.board[move.fr, move.fc])
    color = board.turn
    ptype = abs(piece)
    is_pawn = ptype == PAWN
    target = int(board.board[move.tr, move.tc])
    is_ep = is_pawn and board.ep == (move.tr, move.tc) and move.tc != move.fc and target == EMPTY
    nb = board.copy()
    nb.board[move.tr, move.tc] = move.promo * color if move.promo else piece
    nb.board[move.fr, move.fc] = EMPTY
    if is_ep:
        nb.board[move.fr, move.tc] = EMPTY
    if ptype == KING and abs(move.tc - move.fc) == 2:
        if move.tc > move.fc:
            nb.board[move.fr, 5] = nb.board[move.fr, 7]
            nb.board[move.fr, 7] = EMPTY
        else:
            nb.board[move.fr, 3] = nb.board[move.fr, 0]
            nb.board[move.fr, 0] = EMPTY
    cast = nb.castling
    if ptype == KING:
        cast &= ~(WK | WQ) if color == WHITE else ~(BK | BQ)
    if ptype == ROOK:
        cast = _strip_corner_right(cast, move.fr, move.fc)
    if target != EMPTY or is_ep:
        cast = _strip_corner_right(cast, move.tr, move.tc)
    nb.castling = cast
    if is_pawn and abs(move.tr - move.fr) == 2:
        nb.ep = ((move.fr + move.tr) // 2, move.fc)
    else:
        nb.ep = None
    if is_pawn or target != EMPTY or is_ep:
        nb.halfmove = 0
    else:
        nb.halfmove = board.halfmove + 1
    if color == BLACK:
        nb.fullmove = board.fullmove + 1
    nb.turn = -color
    return nb

def is_legal(board: Board, move: Move) -> bool:
    _check_kings(board)
    if not (in_bounds(move.fr, move.fc) and in_bounds(move.tr, move.tc)):
        return False
    if (move.fr, move.fc) == (move.tr, move.tc):
        return False
    piece = int(board.board[move.fr, move.fc])
    if piece == EMPTY or (piece > 0) != (board.turn == WHITE):
        return False
    target = int(board.board[move.tr, move.tc])
    if target != EMPTY and (target > 0) == (board.turn == WHITE):
        return False
    last = 0 if board.turn == WHITE else 7
    if not _promo_ok(piece, move, last):
        return False
    return move in set(legal_moves(board))
