import numpy as np
import pytest

from engine.board import BLACK, WHITE
from engine.env import ChessEnv
from engine.movegen import Move
from alphazero.config import Config
from alphazero.encode import encode, move_to_index
from alphazero.mcts import (
    MCTS,
    Node,
    add_dirichlet_noise,
    backup,
    puct_scores,
    run_mcts,
    select_action,
    select_idx,
)

CFG = Config(num_simulations=20)

def fake_node(P, N, W):
    n = Node(ChessEnv.startpos())
    n.moves = [None] * len(P)
    n.P0 = np.asarray(P, dtype=np.float32)
    n.P = n.P0.copy()
    n.N = np.asarray(N, dtype=np.int32)
    n.W = np.asarray(W, dtype=np.float32)
    n.expanded = True
    return n

def uniform_predict(cfg):
    calls = {"n": 0}

    def predict(planes):
        calls["n"] += 1
        B = planes.shape[0]
        return np.zeros((B, cfg.num_actions), np.float32), np.zeros(B, np.float32)

    return predict, calls

def mv(uci):
    from engine.board import BISHOP, KNIGHT, QUEEN, ROOK

    return Move(
        8 - int(uci[1]), ord(uci[0]) - 97, 8 - int(uci[3]), ord(uci[2]) - 97,
        {"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}.get(uci[4:5], 0),
    )

# ---------------------------------------------------------------- selection

def test_select_prefers_prior_on_tie():
    n = fake_node([0.1, 0.7, 0.2], [1, 1, 1], [0, 0, 0])
    assert select_idx(n, 1.0) == 1

def test_select_prefers_value_on_tie():
    n = fake_node([0.3, 0.3, 0.3], [2, 2, 2], [0, 2, 4])
    assert select_idx(n, 1.0) == 2

def test_exploration_bonus_decreases_with_visits():
    n = fake_node([0.3, 0.3, 0.3], [0, 5, 10], [0, 0, 0])
    s = puct_scores(n, 1.0)
    assert s[0] > s[1] > s[2]

def test_unvisited_edges_remain_selectable():
    n = fake_node([0.5, 0.5], [3, 0], [0, 0])
    assert select_idx(n, 1.0) == 1  # N=0 edge gets the larger U

def test_q_dominates_after_many_visits():
    n = fake_node([0.98, 0.01, 0.01], [1000, 0, 0], [1000.0, 0, 0])
    assert select_idx(n, 1.0) == 0

def test_fresh_node_picks_first_index():
    # sqrt(total): total==0 => U==0 and Q==0 => argmax degenerates to index 0.
    n = fake_node([0.1, 0.7, 0.2], [0, 0, 0], [0, 0, 0])
    assert select_idx(n, 1.0) == 0

def test_selection_is_deterministic():
    n = fake_node([0.2, 0.2, 0.2], [1, 1, 1], [0, 0, 0])
    assert len({select_idx(n, 1.0) for _ in range(10)}) == 1

def test_scores_length_and_finiteness():
    n = fake_node([0.2, 0.3, 0.5], [1, 2, 3], [0, 1, -1])
    s = puct_scores(n, 1.0)
    assert s.shape == (3,) and np.isfinite(s).all()

def test_selection_guard_raises():
    with pytest.raises(ValueError):
        puct_scores(Node(ChessEnv.startpos()), 1.0)  # unexpanded

    term = Node(ChessEnv.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert term.terminal
    with pytest.raises(ValueError):
        puct_scores(term, 1.0)

# ------------------------------------------------------------- tree building

def test_visit_accounting():
    predict, _ = uniform_predict(CFG)
    res = run_mcts(ChessEnv.startpos(), predict, CFG, rng=np.random.default_rng(0))
    assert int(res.root.N.sum()) == CFG.num_simulations
    assert res.pi.sum() == pytest.approx(1.0)
    assert (res.pi > 0).sum() == CFG.num_simulations  # distinct root edges visited

def test_pi_is_normalized_and_masks_illegal():
    env = ChessEnv.startpos()
    predict, _ = uniform_predict(CFG)
    res = run_mcts(env, predict, CFG, rng=np.random.default_rng(0))

    legal = np.array([move_to_index(m, env.board.turn, CFG) for m in env.legal_moves()])
    illegal = np.setdiff1d(np.arange(CFG.num_actions), legal)

    assert res.pi.sum() == pytest.approx(1.0)
    assert np.all(res.pi[illegal] == 0.0)

def test_expanded_node_invariants():
    predict, _ = uniform_predict(CFG)
    res = run_mcts(ChessEnv.startpos(), predict, CFG, rng=np.random.default_rng(0))

    def walk(node):
        if node.expanded:
            assert len(node.N) == len(node.moves) == len(node.P)
            assert node.P.sum() == pytest.approx(1.0, abs=1e-5)
            assert node.P0.sum() == pytest.approx(1.0, abs=1e-5)
        for c in node.children.values():
            walk(c)

    walk(res.root)

def test_children_materialized_lazily():
    cfg = Config(num_simulations=1)
    predict, calls = uniform_predict(cfg)
    res = run_mcts(ChessEnv.startpos(), predict, cfg, rng=np.random.default_rng(0))
    assert len(res.root.children) == 1                 # only the traversed edge
    (c,) = res.root.children.values()
    assert c.expanded                                   # leaf got expanded
    assert calls["n"] == 2                              # root + one leaf

def test_terminal_leaf_skips_predict():
    # Ra1-a8 is checkmate. The first sim (fresh node, sqrt(total)=0) picks index 0
    # (a non-mate move) and expands it; every later sim follows the high prior to
    # the mate, whose child is terminal -> no network call for those leaves.
    cfg = Config(num_simulations=10)
    env = ChessEnv.from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    mate_idx = move_to_index(mv("a1a8"), WHITE, cfg)
    calls = {"n": 0}

    def predict(planes):
        calls["n"] += 1
        logits = np.zeros((planes.shape[0], cfg.num_actions), np.float32)
        logits[:, mate_idx] = 100.0
        return logits, np.zeros(planes.shape[0], np.float32)

    res = run_mcts(env, predict, cfg, rng=np.random.default_rng(0))
    assert calls["n"] == 2                              # root + first (non-mate) leaf only
    i = list(res.root.moves).index(mv("a1a8"))
    assert res.root.N[i] == 9
    assert res.root.Q[i] == pytest.approx(1.0)          # win for the side to move

def test_terminal_root_zero_policy_without_predict():
    cfg = Config(num_simulations=8)
    calls = {"n": 0}

    def predict(planes):
        calls["n"] += 1
        return np.zeros((planes.shape[0], cfg.num_actions), np.float32), np.zeros(planes.shape[0], np.float32)

    for fen, value in (
        ("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1", -1),   # back-rank checkmate
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),      # stalemate
    ):
        calls["n"] = 0
        res = run_mcts(ChessEnv.from_fen(fen), predict, cfg, rng=np.random.default_rng(0))
        assert calls["n"] == 0                       # no network call
        assert res.root.terminal
        assert res.root.value == value               # result for the side to move
        assert np.all(res.pi == 0.0)                 # zero policy

def test_forced_mate_assigns_positive_value_to_winning_edge():
    cfg = Config(num_simulations=32)
    env0 = ChessEnv.from_fen("3k4/8/8/8/8/8/1R6/R6K w - - 0 1")

    boost = {encode(env0, cfg).tobytes(): [move_to_index(mv("b2b7"), WHITE, cfg)]}
    lifted = env0.clone()
    lifted.step(mv("b2b7"))
    boost[encode(lifted, cfg).tobytes()] = [
        move_to_index(mv("d8c8"), BLACK, cfg), move_to_index(mv("d8e8"), BLACK, cfg),
    ]
    for king_uci in ("d8c8", "d8e8"):
        after = lifted.clone()
        after.step(mv(king_uci))
        boost[encode(after, cfg).tobytes()] = [move_to_index(mv("a1a8"), WHITE, cfg)]

    def predict(planes):
        logits = np.zeros((planes.shape[0], cfg.num_actions), np.float32)
        for i in boost.get(planes[0].tobytes(), ()):
            logits[0, i] = 50.0
        return logits, np.zeros(planes.shape[0], np.float32)

    res = run_mcts(env0, predict, cfg, rng=np.random.default_rng(0))
    i = list(res.root.moves).index(mv("b2b7"))
    assert res.root.N[i] > 0
    assert res.root.Q[i] > 0.0                       # winning edge valued positively
    assert int(np.argmax(res.root.Q)) == i

def test_backup_sign_flip():
    n0 = fake_node([0.5, 0.5], [0, 0], [0, 0])
    n1 = fake_node([0.5, 0.5], [0, 0], [0, 0])
    backup([(n0, 0), (n1, 1)], v=1.0)
    assert n1.N[1] == 1 and n1.W[1] == pytest.approx(-1.0)  # immediate parent flips
    assert n0.N[0] == 1 and n0.W[0] == pytest.approx(1.0)   # two plies up: original sign

# ------------------------------------------------------------------- noise

def test_dirichlet_noise_idempotent_from_p0():
    n = fake_node([0.5, 0.3, 0.2], [0, 0, 0], [0, 0, 0])
    add_dirichlet_noise(n, CFG, np.random.default_rng(7))
    p_first = n.P.copy()
    add_dirichlet_noise(n, CFG, np.random.default_rng(7))  # rebuilt from P0, not cumulative
    assert np.allclose(p_first, n.P)
    assert not np.allclose(p_first, n.P0)
    assert n.P.sum() == pytest.approx(1.0, abs=1e-5)

# ----------------------------------------------------------------- facade

def test_facade_search_and_reuse():
    cfg = Config(num_simulations=8)
    predict, _ = uniform_predict(cfg)
    m = MCTS(predict, cfg, rng=np.random.default_rng(0))

    res = m.search(ChessEnv.startpos(), add_noise=True)
    assert m.root is res.root and res.root.expanded

    move, i = m.select_action(temperature=0.0)
    assert move == res.root.moves[int(np.argmax(res.root.N))]

    m.advance(i)
    assert m.root is not None and m.root.parent is None
    assert m.root.move == move

    res2 = m.search(ChessEnv.startpos(), add_noise=True)
    assert int(res2.root.N.sum()) >= cfg.num_simulations

def test_reused_root_tracks_advance_and_resets_on_mismatch():
    cfg = Config(num_simulations=4)
    predict, calls = uniform_predict(cfg)
    m = MCTS(predict, cfg, rng=np.random.default_rng(0))

    m.search(ChessEnv.startpos(), add_noise=False)
    _, i = m.select_action(temperature=0.0)
    m.advance(i)
    assert m.root is not None

    # same position as the advanced root: subtree reused, only leaves evaluated
    n_before = calls["n"]
    res = m.search(m.root.env, add_noise=False)
    assert res.root is m.root
    assert calls["n"] - n_before == cfg.num_simulations

    # different position: stale root discarded and re-evaluated from scratch
    n_before = calls["n"]
    res = m.search(ChessEnv.startpos(), add_noise=False)
    assert res.root.move is None and res.root.parent is None
    assert calls["n"] - n_before == cfg.num_simulations + 1
    assert int(res.root.N.sum()) == cfg.num_simulations

def test_temperature_zero_is_greedy():
    cfg = Config(num_simulations=8)
    predict, _ = uniform_predict(cfg)
    res = run_mcts(ChessEnv.startpos(), predict, cfg, rng=np.random.default_rng(0))
    move, i = select_action(res.root, 0.0, np.random.default_rng(0))
    assert move == res.root.moves[i]
    assert i == int(np.argmax(res.root.N))

def test_search_owns_root_and_invalidates_clock_or_history_changes():
    cfg = Config(num_simulations=2)
    predict, _ = uniform_predict(cfg)
    mcts = MCTS(predict, cfg, np.random.default_rng(0))
    env = ChessEnv.startpos()
    root = mcts.search(env, add_noise=False).root
    env.step(mv("g1f3"))
    assert root.env.ply == 0 and root.env.to_fen() != env.to_fen()
    for uci in ("g8f6", "f3g1", "f6g8"):
        env.step(mv(uci))
    root = mcts.search(env, add_noise=False).root
    no_history = ChessEnv.from_fen(env.to_fen())
    assert mcts.search(no_history, add_noise=False).root is not root
    board = no_history.board.copy()
    board.halfmove = 100
    root = mcts.search(ChessEnv(board), add_noise=False).root
    assert root.terminal and not root.expanded and root.value == 0
