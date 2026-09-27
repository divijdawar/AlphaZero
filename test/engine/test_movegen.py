import pytest

from engine.board import *
from engine.movegen import *

def perft(board, depth):
    if depth == 0:
        return 1
    moves = legal_moves(board)
    if depth == 1:
        return len(moves)
    return sum(perft(make_move(board, m), depth - 1) for m in moves)

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

def test_startpos_legal_count():
    assert len(legal_moves(Board.startpos())) == 20

def test_startpos_is_legal_delegates():
    b = Board.startpos()
    assert is_legal(b, Move(6, 4, 4, 4))
    assert is_legal(b, Move(7, 6, 5, 5))
    assert not is_legal(b, Move(6, 4, 3, 4))
    assert not is_legal(b, Move(7, 4, 6, 4))

def test_knight_geometry_count():
    b = Board.from_fen("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
    assert len(legal_moves(b)) == 5 + 3

def test_rook_blocked_by_own_king():
    b = Board.from_fen("4k3/8/8/8/8/8/4P3/4K2R w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 7, 7, 5) in moves
    assert Move(7, 7, 7, 4) not in moves
    assert Move(7, 7, 0, 7) in moves

def test_bishop_slider_stops_at_enemy():
    b = Board.from_fen("4k3/8/8/8/3p4/8/8/4K1B1 w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 6, 6, 5) in moves
    assert Move(7, 6, 5, 4) in moves
    assert Move(7, 6, 4, 3) in moves
    assert Move(7, 6, 3, 2) not in moves

def test_pawn_promotion_four_options():
    b = Board.from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    promos = [m for m in legal_moves(b) if (m.fr, m.fc) == (1, 0)]
    assert len(promos) == 4
    assert {m.promo for m in promos} == set(PROMO_PIECES)

def test_promotion_applied():
    b = Board.from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    nb = make_move(b, Move(1, 0, 0, 0, QUEEN))
    assert int(nb.board[0, 0]) == QUEEN
    assert int(nb.board[1, 0]) == EMPTY
    assert nb.ep is None
    assert nb.halfmove == 0
    assert nb.turn == BLACK

def test_castling_both_sides_available():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) in moves
    assert Move(7, 4, 7, 2) in moves

def test_castling_transit_square_attacked():
    b = Board.from_fen("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) in moves

def test_castling_applies_rook_lift_and_rights():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    nb = make_move(b, Move(7, 4, 7, 6))
    assert int(nb.board[7, 6]) == KING
    assert int(nb.board[7, 5]) == ROOK
    assert int(nb.board[7, 7]) == EMPTY
    assert int(nb.board[7, 4]) == EMPTY
    assert nb.castling == (BK | BQ)

def test_rook_move_strips_only_that_right():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    nb = make_move(b, Move(7, 7, 6, 7))
    assert not (nb.castling & WK)
    assert nb.castling & WQ

def test_en_passant_available():
    b = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert Move(3, 4, 2, 3) in set(legal_moves(b))

def test_en_passant_applied():
    b = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    nb = make_move(b, Move(3, 4, 2, 3))
    assert int(nb.board[2, 3]) == PAWN
    assert int(nb.board[3, 4]) == EMPTY
    assert int(nb.board[3, 3]) == EMPTY
    assert nb.ep is None
    assert nb.halfmove == 0

def test_en_passant_discovered_check_is_illegal():
    b = Board.from_fen("k7/8/8/K2Pp2r/8/8/8/8 w - e6 0 1")
    assert Move(3, 3, 2, 4) not in set(legal_moves(b))
    assert not is_legal(b, Move(3, 3, 2, 4))

def test_cannot_leave_king_in_check():
    b = Board.from_fen("4k3/8/8/8/8/8/4r3/4K1N1 w - - 0 1")
    assert not is_legal(b, Move(7, 6, 5, 5))
    assert is_legal(b, Move(7, 4, 6, 4))

def test_double_push_sets_ep_target():
    b = Board.startpos()
    nb = make_move(b, Move(6, 4, 4, 4))
    assert nb.ep == (5, 4)
    nb2 = make_move(b, Move(6, 4, 5, 4))
    assert nb2.ep is None

def test_make_move_clocks_and_turn():
    b = Board.startpos()
    nb = make_move(b, Move(7, 6, 5, 5))
    assert nb.halfmove == 1
    assert nb.fullmove == 1
    assert nb.turn == BLACK
    nb2 = make_move(nb, Move(1, 4, 3, 4))
    assert nb2.fullmove == 2
    assert nb2.turn == WHITE
    assert nb2.halfmove == 0

def test_perft_startpos():
    b = Board.startpos()
    assert perft(b, 1) == 20
    assert perft(b, 2) == 400
    assert perft(b, 3) == 8902

def test_perft_kiwipete():
    b = Board.from_fen(KIWIPETE)
    assert perft(b, 1) == 48
    assert perft(b, 2) == 2039

def test_black_pawn_single_and_double():
    b = Board.from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    pawn_moves = {m for m in legal_moves(b) if (m.fr, m.fc) == (1, 4)}
    assert Move(1, 4, 2, 4) in pawn_moves
    assert Move(1, 4, 3, 4) in pawn_moves
    assert len(pawn_moves) == 2

def test_black_pawn_captures_and_push():
    b = Board.from_fen("4k3/8/8/3p4/2P1P3/8/8/4K3 b - - 0 1")
    pawn_moves = {m for m in legal_moves(b) if (m.fr, m.fc) == (3, 3)}
    assert Move(3, 3, 4, 3) in pawn_moves
    assert Move(3, 3, 4, 2) in pawn_moves
    assert Move(3, 3, 4, 4) in pawn_moves
    assert len(pawn_moves) == 3

def test_black_double_push_sets_ep_target():
    b = Board.from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    nb = make_move(b, Move(1, 4, 3, 4))
    assert nb.ep == (2, 4)
    assert nb.halfmove == 0
    assert nb.turn == WHITE
    nb2 = make_move(b, Move(1, 4, 2, 4))
    assert nb2.ep is None

def test_black_promotion_four_options():
    b = Board.from_fen("4K3/8/8/8/8/8/p7/4k3 b - - 0 1")
    promos = [m for m in legal_moves(b) if (m.fr, m.fc) == (6, 0)]
    assert len(promos) == 4
    assert {m.promo for m in promos} == set(PROMO_PIECES)

def test_black_promotion_applied():
    b = Board.from_fen("4K3/8/8/8/8/8/p7/4k3 b - - 0 1")
    nb = make_move(b, Move(6, 0, 7, 0, QUEEN))
    assert int(nb.board[7, 0]) == -QUEEN
    assert int(nb.board[6, 0]) == EMPTY
    assert nb.ep is None
    assert nb.halfmove == 0
    assert nb.turn == WHITE

def test_pawn_single_blocked_by_own_piece():
    b = Board.from_fen("4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1")
    blocked = [m for m in legal_moves(b) if (m.fr, m.fc) == (6, 4)]
    assert blocked == []

def test_pawn_forward_blocked_by_enemy():
    b = Board.from_fen("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1")
    blocked = [m for m in legal_moves(b) if (m.fr, m.fc) == (6, 4)]
    assert blocked == []

def test_pawn_double_push_destination_blocked():
    b = Board.from_fen("4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(6, 4, 5, 4) in moves
    assert Move(6, 4, 4, 4) not in moves

def test_pawn_no_double_push_from_non_start():
    b = Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    moves = {m for m in legal_moves(b) if (m.fr, m.fc) == (4, 4)}
    assert Move(4, 4, 3, 4) in moves
    assert Move(4, 4, 2, 4) not in moves
    assert len(moves) == 1

def test_pawn_cannot_move_backward_or_sideways():
    b = Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    assert is_legal(b, Move(4, 4, 3, 4))
    assert not is_legal(b, Move(4, 4, 5, 4))
    assert not is_legal(b, Move(4, 4, 4, 3))
    assert not is_legal(b, Move(4, 4, 4, 5))

def test_pawn_promotion_validation_rejects():
    b = Board.from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    with pytest.raises(ValueError):
        make_move(b, Move(1, 0, 0, 0))
    with pytest.raises(ValueError):
        make_move(b, Move(1, 0, 0, 0, KING))
    assert not is_legal(b, Move(1, 0, 0, 0))
    assert not is_legal(b, Move(1, 0, 0, 0, KING))

def test_non_pawn_promotion_flag_rejected():
    b = Board.from_fen("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
    with pytest.raises(ValueError):
        make_move(b, Move(7, 6, 5, 5, QUEEN))
    assert not is_legal(b, Move(7, 6, 5, 5, QUEEN))

def test_capture_promotion_twelve_options():
    b = Board.from_fen("r1r3k1/1P6/8/8/8/8/8/4K3 w - - 0 1")
    pawn_moves = [m for m in legal_moves(b) if (m.fr, m.fc) == (1, 1)]
    assert len(pawn_moves) == 12
    for tr, tc in ((0, 1), (0, 0), (0, 2)):
        promos = {m.promo for m in pawn_moves if (m.tr, m.tc) == (tr, tc)}
        assert promos == set(PROMO_PIECES)

def test_underpromotion_knight_applied():
    b = Board.from_fen("r1r3k1/1P6/8/8/8/8/8/4K3 w - - 0 1")
    nb = make_move(b, Move(1, 1, 0, 1, KNIGHT))
    assert int(nb.board[0, 1]) == KNIGHT
    assert int(nb.board[1, 1]) == EMPTY
    assert nb.halfmove == 0
    assert nb.turn == BLACK

def test_capture_promotion_applied():
    b = Board.from_fen("r1r3k1/1P6/8/8/8/8/8/4K3 w - - 0 1")
    nb = make_move(b, Move(1, 1, 0, 0, KNIGHT))
    assert int(nb.board[0, 0]) == KNIGHT
    assert int(nb.board[1, 1]) == EMPTY
    nb2 = make_move(b, Move(1, 1, 0, 2, BISHOP))
    assert int(nb2.board[0, 2]) == BISHOP
    assert int(nb2.board[1, 1]) == EMPTY

def test_castling_while_in_check_illegal():
    b = Board.from_fen("4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) not in moves
    assert not is_legal(b, Move(7, 4, 7, 6))
    assert not is_legal(b, Move(7, 4, 7, 2))

def test_castling_kingside_destination_attacked():
    b = Board.from_fen("6rk/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) in moves

def test_castling_queenside_destination_attacked():
    b = Board.from_fen("2r3k1/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 2) not in moves
    assert Move(7, 4, 7, 6) in moves

def test_castling_blocked_by_pieces():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R2BK1NR w KQkq - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) not in moves

def test_castling_requires_rights_flag():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) not in moves

def test_castling_requires_rook_on_corner():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K3 w KQkq - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 7, 6) not in moves
    assert Move(7, 4, 7, 2) in moves

def test_king_move_strips_both_rights():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    nb = make_move(b, Move(7, 4, 7, 5))
    assert not (nb.castling & WK)
    assert not (nb.castling & WQ)
    assert nb.castling & BK
    assert nb.castling & BQ

def test_capture_on_corner_strips_right():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    nb = make_move(b, Move(7, 0, 0, 0))
    assert nb.castling & WK
    assert not (nb.castling & WQ)
    assert nb.castling & BK
    assert not (nb.castling & BQ)

def test_black_castling_both_sides_available():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    moves = set(legal_moves(b))
    assert Move(0, 4, 0, 6) in moves
    assert Move(0, 4, 0, 2) in moves

def test_black_castling_transit_attacked():
    b = Board.from_fen("r3k2r/8/8/8/8/8/8/R3KR2 b kq - 0 1")
    moves = set(legal_moves(b))
    assert Move(0, 4, 0, 6) not in moves
    assert Move(0, 4, 0, 2) in moves

def test_pinned_rook_stays_on_file():
    b = Board.from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(6, 4, 6, 5) not in moves
    assert Move(6, 4, 6, 3) not in moves
    assert Move(6, 4, 5, 4) in moves
    assert Move(6, 4, 0, 4) in moves

def test_single_check_block_or_king_move():
    b = Board.from_fen("4r1k1/8/8/8/8/8/3B4/4K3 w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(6, 3, 5, 4) in moves
    assert Move(6, 3, 5, 2) not in moves
    assert Move(7, 4, 7, 5) in moves
    assert Move(7, 4, 6, 4) not in moves

def test_double_check_only_king_moves():
    b = Board.from_fen("4r1k1/8/8/8/8/2b5/8/4K1N1 w - - 0 1")
    moves = set(legal_moves(b))
    knight_moves = [m for m in moves if (m.fr, m.fc) == (7, 6)]
    assert knight_moves == []
    assert Move(7, 4, 7, 5) in moves
    assert Move(7, 4, 6, 4) not in moves

def test_king_cannot_move_into_check():
    b = Board.from_fen("6k1/8/8/8/8/8/r7/4K3 w - - 0 1")
    moves = set(legal_moves(b))
    assert Move(7, 4, 6, 4) not in moves
    assert Move(7, 4, 7, 3) in moves
    assert not is_legal(b, Move(7, 4, 6, 4))

def test_is_attacked_pawn_knight_king():
    b = Board.from_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
    assert is_attacked(b, 2, 3, WHITE)
    assert is_attacked(b, 2, 5, WHITE)
    assert not is_attacked(b, 2, 4, WHITE)
    assert not is_attacked(b, 2, 3, BLACK)
    bn = Board.from_fen("4k3/8/8/4N3/8/8/8/4K3 w - - 0 1")
    assert is_attacked(bn, 1, 3, WHITE)
    assert not is_attacked(bn, 2, 4, WHITE)
    bk = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert is_attacked(bk, 6, 4, WHITE)
    assert is_attacked(bk, 7, 3, WHITE)
    assert not is_attacked(bk, 5, 4, WHITE)

def test_is_attacked_slider_blocked():
    b = Board.from_fen("4k3/8/8/8/8/8/3N4/2B1K3 w - - 0 1")
    assert is_attacked(b, 6, 1, WHITE)
    assert not is_attacked(b, 5, 4, WHITE)
    br = Board.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert is_attacked(br, 0, 0, WHITE)
    assert is_attacked(br, 7, 1, WHITE)
    assert not is_attacked(br, 6, 1, WHITE)

def test_pseudo_legal_includes_pinned_sideways_move():
    b = Board.from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    pseudo = set(pseudo_legal_moves(b))
    legal = set(legal_moves(b))
    assert Move(6, 4, 6, 5) in pseudo
    assert Move(6, 4, 6, 5) not in legal

def test_make_move_rejects_pawn_teleport():
    b = Board.startpos()
    with pytest.raises(ValueError):
        make_move(b, Move(6, 0, 0, 0, QUEEN))

def test_make_move_rejects_blocked_slider_path():
    b = Board.startpos()
    with pytest.raises(ValueError):
        make_move(b, Move(7, 0, 0, 0))

def test_make_move_rejects_wrong_piece_geometry():
    b = Board.startpos()
    with pytest.raises(ValueError):
        make_move(b, Move(6, 4, 4, 5))

def test_make_move_rejects_zero_length_move():
    b = Board.startpos()
    with pytest.raises(ValueError):
        make_move(b, Move(6, 4, 6, 4))

def test_make_move_accepts_pinned_piece_move():
    b = Board.from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    nb = make_move(b, Move(6, 4, 6, 5))
    assert Move(6, 4, 6, 5) not in set(legal_moves(b))
    assert int(nb.board[6, 5]) == ROOK

def test_make_move_accepts_all_pseudo_legal_generated():
    b = Board.startpos()
    for m in pseudo_legal_moves(b):
        make_move(b, m)
