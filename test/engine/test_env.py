import pytest

from engine.board import *
from engine.movegen import Move
from engine.env import ChessEnv, _insufficient_material, _king_square

def _mv(uci):
    (fr, fc), (tr, tc) = parse_square(uci[:2]), parse_square(uci[2:4])
    promo = {"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}.get(uci[4:5], 0)
    return Move(fr, fc, tr, tc, promo)

def _play(env, moves):
    out = None
    for m in moves:
        out = env.step(_mv(m))
    return out

def test_insufficient_material_draws():
    for fen in (
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KB2 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
        "4kb2/8/8/8/8/8/8/2BK4 w - - 0 1",
    ):
        assert _insufficient_material(Board.from_fen(fen)) is True

def test_sufficient_material_not_draws():
    for fen in (
        "4kb2/8/8/8/8/8/8/4KB2 w - - 0 1",
        "4kn2/8/8/8/8/8/8/4KN2 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KR2 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1",
    ):
        assert _insufficient_material(Board.from_fen(fen)) is False

def test_king_square_finds_kings():
    b = Board.startpos()
    assert _king_square(b, KING) == (7, 4)
    assert _king_square(b, -KING) == (0, 4)

def test_king_square_missing_raises():
    with pytest.raises(ValueError):
        _king_square(Board(), KING)

def test_fools_mate_by_play():
    env = ChessEnv.startpos()
    assert _play(env, ["f2f3", "e7e5", "g2g4"]) is None
    assert env.step(_mv("d8h4")) == -1
    assert env.is_terminal()

def test_fools_mate_from_fen():
    env = ChessEnv.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert env.outcome() == -1
    assert env.result_for(WHITE) == -1
    assert env.result_for(BLACK) == 1

def test_back_rank_mate():
    env = ChessEnv.from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    assert env.step(_mv("a1a8")) == 1
    assert env.result_for(WHITE) == 1
    assert env.result_for(BLACK) == -1

def test_stalemate():
    env = ChessEnv.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert env.outcome() == 0
    assert env.result_for(WHITE) == 0
    assert env.result_for(BLACK) == 0

def test_seventyfive_move_rule_boundary():
    assert ChessEnv.from_fen("4k3/8/8/8/8/8/8/R3K2R w - - 150 120").outcome() == 0
    fresh = ChessEnv.from_fen("4k3/8/8/8/8/8/8/R3K2R w - - 149 120")
    assert fresh.outcome() is None
    assert not fresh.is_terminal()

def test_seventyfive_move_reached_by_step():
    env = ChessEnv.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 149 100")
    assert env.step(_mv("h1h2")) == 0

def test_threefold_repetition():
    env = ChessEnv.startpos()
    assert _play(env, ["g1f3", "g8f6", "f3g1", "f6g8"]) is None
    assert _play(env, ["g1f3", "g8f6", "f3g1", "f6g8"]) == 0

def test_bare_kings_draw():
    env = ChessEnv.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert env.outcome() == 0
    assert env.is_terminal()

def test_default_construction_is_startpos():
    env = ChessEnv()
    assert env.board == Board.startpos()
    assert env.ply == 0
    assert len(env.history) == 1
    assert env.outcome() is None
    assert not env.is_terminal()
    assert env.to_fen() == START_FEN

def test_from_fen_construction():
    env = ChessEnv.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert env.board == Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert env.ply == 0
    assert len(env.history) == 1

def test_history_constructor_uses_last():
    h = [Board.startpos()]
    env = ChessEnv(history=h)
    assert env.board is h[-1]
    assert env.history == h
    assert env.history is not h
    b = Board.startpos()
    env2 = ChessEnv(board=b.copy(), history=[b])
    assert env2.board == b

def test_history_mismatch_raises():
    with pytest.raises(AssertionError):
        ChessEnv(board=Board.startpos(), history=[Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")])

def test_empty_history_falls_back_to_startpos():
    env = ChessEnv(history=[])
    assert env.board == Board.startpos()
    assert env.ply == 0

def test_step_updates_history_and_ply():
    env = ChessEnv.startpos()
    env.step(_mv("e2e4"))
    env.step(_mv("e7e5"))
    assert env.ply == 2
    assert len(env.history) == 3
    assert env.board is env.history[-1]
    assert sum(env.counts.values()) == len(env.history)

def test_history_reconstruction_preserves_counts():
    env = ChessEnv.startpos()
    _play(env, ["g1f3", "g8f6", "f3g1", "f6g8"])
    env2 = ChessEnv(history=env.history)
    assert env2.outcome() is None
    assert _play(env2, ["g1f3", "g8f6", "f3g1", "f6g8"]) == 0

def test_repr_contains_fen_and_ply():
    env = ChessEnv.startpos()
    assert START_FEN in repr(env)
    assert "ply=0" in repr(env)
    env.step(_mv("e2e4"))
    assert "ply=1" in repr(env)

def test_clone_matches_state():
    env = ChessEnv.startpos()
    _play(env, ["e2e4", "e7e5"])
    c = env.clone()
    assert c.board == env.board
    assert c.board is not env.board
    assert c.ply == env.ply
    assert c.history == env.history

def test_clone_is_independent():
    env = ChessEnv.startpos()
    _play(env, ["e2e4", "e7e5"])
    c = env.clone()
    c.step(_mv("g1f3"))
    assert env.ply == 2
    assert len(env.history) == 3
    assert c.ply == 3
    c.board.board[0, 0] = EMPTY
    assert env.board.board[0, 0] != EMPTY

def test_clone_preserves_repetition_counts():
    env = ChessEnv.startpos()
    _play(env, ["g1f3", "g8f6", "f3g1", "f6g8"])
    c = env.clone()
    assert _play(c, ["g1f3", "g8f6", "f3g1", "f6g8"]) == 0
    assert env.outcome() is None

def test_clone_of_terminal():
    env = ChessEnv.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    c = env.clone()
    assert c.outcome() == -1
    assert c.is_terminal()

def test_frames_single():
    env = ChessEnv.startpos()
    assert env.frames(0) == [env.board]
    assert env.frames(1) == [env.board]

def test_frames_padding():
    env = ChessEnv.startpos()
    f = env.frames(8)
    assert len(f) == 8
    assert all(b == env.history[0] for b in f)
    assert f[-1] is env.board

def test_frames_window():
    env = ChessEnv.startpos()
    _play(env, ["e2e4", "e7e5", "g1f3"])
    f = env.frames(8)
    assert len(f) == 8
    assert f[:5] == [env.history[0]] * 5
    assert f[5:] == env.history[1:]
    assert env.frames(4) == env.history
    assert env.frames(2) == env.history[-2:]
