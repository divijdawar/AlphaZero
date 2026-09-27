"""Tests for engine.board.

Covered suites:
  Suite 1 — coordinates (square_name / parse_square / in_bounds)
  Suite 2 — initial position (startpos / default construction)
  Suite 6 — copy semantics (MCTS-critical)
  Suite 8 — repr / hash surface
"""

import numpy as np
import pytest

from engine.board import *

def test_square_name_corners():
    assert square_name(0, 0) == "a8"
    assert square_name(0, 7) == "h8"
    assert square_name(7, 0) == "a1"
    assert square_name(7, 7) == "h1"

def test_square_name_spot_checks():
    # (row 0 = rank 8, col 0 = file a)
    assert square_name(4, 4) == "e4"
    assert square_name(6, 4) == "e2"
    assert square_name(1, 0) == "a7"
    assert square_name(3, 7) == "h5"

def test_parse_square_corners():
    assert parse_square("a8") == (0, 0)
    assert parse_square("h8") == (0, 7)
    assert parse_square("a1") == (7, 0)
    assert parse_square("h1") == (7, 7)

def test_square_roundtrip_all_64():
    for r in range(8):
        for c in range(8):
            assert parse_square(square_name(r, c)) == (r, c)

def test_name_roundtrip_all_64():
    files = "abcdefgh"
    for rank in range(1, 9):
        for f in files:
            name = f"{f}{rank}"
            r, c = parse_square(name)
            assert square_name(r, c) == name

def test_in_bounds_all_64_true():
    for r in range(8):
        for c in range(8):
            assert in_bounds(r, c) is True

def test_in_bounds_outside_false():
    assert in_bounds(-1, 0) is False
    assert in_bounds(0, -1) is False
    assert in_bounds(0, 8) is False
    assert in_bounds(8, 0) is False
    assert in_bounds(8, 8) is False
    assert in_bounds(-1, -1) is False
    assert in_bounds(3, -5) is False
    assert in_bounds(100, 100) is False

def test_parse_square_invalid_rank_raises():
    for bad in ("e9", "e0", "a9", "h0"):
        with pytest.raises(ValueError):
            parse_square(bad)

def test_parse_square_invalid_file_raises():
    for bad in ("i1", "z3", "A1", "11"):
        with pytest.raises(ValueError):
            parse_square(bad)

def test_parse_square_malformed_raises():
    for bad in ("", "e", "e11", "ee"):
        with pytest.raises(ValueError):
            parse_square(bad)

# ---------------------------------------------------------------------------
# Suite 2 — Initial position: startpos() and default construction
# ---------------------------------------------------------------------------

def test_startpos_fen_roundtrip():
    assert Board.startpos().to_fen() == START_FEN

def test_startpos_black_back_rank():
    b = Board.startpos()
    expected = [-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING, -BISHOP, -KNIGHT, -ROOK]
    assert [b.piece_at(0, c) for c in range(8)] == expected

def test_startpos_black_pawns():
    b = Board.startpos()
    assert [b.piece_at(1, c) for c in range(8)] == [-PAWN] * 8

def test_startpos_white_pawns():
    b = Board.startpos()
    assert [b.piece_at(6, c) for c in range(8)] == [PAWN] * 8

def test_startpos_white_back_rank():
    b = Board.startpos()
    expected = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]
    assert [b.piece_at(7, c) for c in range(8)] == expected

def test_startpos_middle_empty():
    b = Board.startpos()
    for r in range(2, 6):
        for c in range(8):
            assert b.piece_at(r, c) == 0

def test_startpos_metadata():
    b = Board.startpos()
    assert b.turn == WHITE
    assert b.castling == ALL_CASTLING
    assert b.ep is None
    assert b.halfmove == 0
    assert b.fullmove == 1

def test_startpos_array_shape_and_dtype():
    b = Board.startpos()
    assert b.board.shape == (8, 8)
    assert b.board.dtype == np.int8

def test_startpos_piece_counts():
    b = Board.startpos()
    flat = b.board.ravel()
    assert int(np.count_nonzero(flat > 0)) == 16  # white
    assert int(np.count_nonzero(flat < 0)) == 16  # black
    assert int(np.count_nonzero(flat == 0)) == 32  # empty

def test_default_board_is_empty_not_startpos():
    b = Board()
    assert int(np.count_nonzero(b.board)) == 0
    assert b != Board.startpos()

def test_startpos_kings_on_e_files():
    b = Board.startpos()
    assert b.piece_at(0, 4) == -KING  # e8
    assert b.piece_at(7, 4) == KING  # e1

def test_black_constant_is_minus_one():
    # Pin the sign convention white=+1 / black=-1 used across the engine.
    assert WHITE == 1
    assert BLACK == -1

# ---------------------------------------------------------------------------
# Suite 6 — copy semantics (MCTS clones boards heavily, so aliasing here is fatal)
# ---------------------------------------------------------------------------

def test_copy_equals_original():
    b = Board.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert b.copy() == b

def test_copy_is_distinct_object():
    b = Board.startpos()
    c = b.copy()
    assert c is not b
    assert c.board is not b.board

def test_copy_board_array_is_independent():
    b = Board.startpos()
    c = b.copy()
    c.board[0, 0] = 0
    assert b.piece_at(0, 0) == -ROOK
    assert c.piece_at(0, 0) == 0

def test_copy_placing_piece_does_not_leak():
    b = Board.startpos()
    c = b.copy()
    c.board[4, 4] = QUEEN
    assert b.piece_at(4, 4) == 0

def test_copy_metadata_is_independent():
    b = Board.startpos()
    c = b.copy()
    c.turn = BLACK
    c.castling = 0
    c.ep = (5, 4)
    c.halfmove = 7
    c.fullmove = 12
    assert b.turn == WHITE
    assert b.castling == ALL_CASTLING
    assert b.ep is None
    assert b.halfmove == 0
    assert b.fullmove == 1

def test_copy_preserves_every_field():
    b = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12")
    c = b.copy()
    assert c.turn == b.turn
    assert c.castling == b.castling
    assert c.ep == b.ep
    assert c.halfmove == b.halfmove
    assert c.fullmove == b.fullmove
    assert np.array_equal(c.board, b.board)

def test_constructor_copies_input_array():
    arr = np.zeros((8, 8), dtype=np.int8)
    b = Board(arr)
    arr[0, 0] = QUEEN
    assert b.piece_at(0, 0) == 0

def test_constructor_casts_input_to_int8():
    arr = np.zeros((8, 8), dtype=np.int16)
    arr[0, 0] = ROOK
    b = Board(arr)
    assert b.board.dtype == np.int8

# ---------------------------------------------------------------------------
# Suite 8 — repr
# ---------------------------------------------------------------------------

def test_repr_contains_fen():
    assert START_FEN in repr(Board.startpos())

def test_repr_matches_to_fen_format():
    b = Board.startpos()
    assert repr(b) == f"Board({b.to_fen()!r})"

def test_repr_reflects_all_state_for_black():
    b = Board.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert repr(b) == f"Board({b.to_fen()!r})"

# ---------------------------------------------------------------------------
# Suite 8 — hash
# ---------------------------------------------------------------------------

def test_board_hash_is_none():
    assert Board.__hash__ is None

def test_board_is_unhashable():
    with pytest.raises(TypeError):
        hash(Board.startpos())

def test_board_cannot_be_a_set_member():
    with pytest.raises(TypeError):
        {Board.startpos()}
