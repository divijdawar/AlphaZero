from __future__ import annotations

from dataclasses import asdict
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request

import chess
import chess.engine
import chess.pgn
import numpy as np
import pytest
from tinygrad import Context, Tensor
from tinygrad.nn.state import get_state_dict, safe_load, safe_save
from alphazero.config import Config
from alphazero.encode import encode
from engine.env import ChessEnv
from eval.checkpoints import discover, inspect_checkpoint, load_network
from eval.player import ModelPlayer, parse_move, Decision
from eval.arena import Settings, play_game, schedule
from eval.human import HumanGame, make_handler
from eval.runtime import ROOT, output_path, write_json
from eval.ratings import fit


def test_loader_preserves_predictions_and_batchnorm(checkpoint):
    path, original, cfg = checkpoint
    loaded, predict = load_network(inspect_checkpoint(path), "CPU:CLANG")
    x = encode(ChessEnv.startpos(), cfg)[None]
    with Context(DEV="CPU:CLANG", TRAINING=0):
        logits, value = original(Tensor(x))
        expected = logits.numpy(), value.numpy().reshape(-1)
    actual = predict(x)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    for name, tensor in get_state_dict(loaded).items():
        np.testing.assert_array_equal(tensor.numpy(), get_state_dict(original)[name].numpy())
        assert not tensor.is_param


def test_final_alias_deduplicates_weights_not_metadata(checkpoint, tmp_path):
    path, _, cfg = checkpoint
    safe_save(safe_load(str(path)), str(tmp_path / "final.safetensors"), metadata={
        "format_version": "1", "config": json.dumps(asdict(cfg)), "training": json.dumps({"global_step": 20})})
    players = discover([str(tmp_path)])
    assert len(players) == 1
    assert players[0].step == 20
    assert players[0].path.name == "final.safetensors"


def fake_player():
    p = ModelPlayer.__new__(ModelPlayer)
    p.cfg, p.root = Config(), None
    p.predict = lambda planes: (np.zeros((len(planes), 4672), np.float32), np.zeros(len(planes), np.float32))
    return p


def test_search_cancellation_and_exact_simulation_cap():
    player = fake_player()
    env = ChessEnv.startpos()
    stopped = threading.Event()
    stopped.set()
    first = player.choose(env, None, stop=stopped)
    assert first.simulations == 0
    assert first.uci in [m.to_uci() for m in env.legal_moves()]
    result = player.choose(env, None, simulations=4)
    assert result.simulations == 4
    assert player.root.N.sum() == 4
    np.testing.assert_array_equal(player.root.P, player.root.P0)


def test_search_clock_and_history_invalidation():
    player = fake_player()
    env = ChessEnv.startpos()
    player.choose(env, None, simulations=2)
    old = player.root
    same_board_different_clock = ChessEnv.from_fen(env.to_fen().replace("0 1", "9 1"))
    player.choose(same_board_different_clock, None, simulations=2)
    assert player.root is not old
    before = time.monotonic()
    result = player.choose(env, 0.002)
    assert time.monotonic() - before < 0.5
    assert result.uci


def test_observe_both_players_moves_preserves_reuse():
    player, env = fake_player(), ChessEnv.startpos()
    for _ in range(3):
        d = player.choose(env, None, simulations=5)
        env.step(parse_move(env, d.uci))
        player.observe(d.uci)
        if player.root:
            assert [b.to_fen() for b in player.root.env.history] == [b.to_fen() for b in env.history]


def test_terminal_and_underpromotion():
    player = fake_player()
    assert player.choose(ChessEnv.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), .1).uci is None
    env = ChessEnv.from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")
    move = parse_move(env, "a7a8n")
    env.step(move)
    assert int(env.board.board[0, 0]) == 2


class ScriptPlayer:
    def __init__(self, moves):
        self.moves = moves
    def reset(self):
        self.i = 0
    def observe(self, uci):
        pass
    def choose(self, env, seconds, **kwargs):
        m = self.moves[self.i]
        self.i += 1
        return Decision(m, 1, 0.0, 0, [m])


def test_complete_game_pgn_and_board_differential():
    record = play_game(ScriptPlayer(["f2f3", "g2g4"]), ScriptPlayer(["e7e5", "d8h4"]),
                       "white", "black", Settings(), {"name": "initial"}, "mate")
    assert record["result"] == "0-1"
    assert record["termination"] == "checkmate"
    game = chess.pgn.read_game(io.StringIO(record["pgn"]))
    assert not game.errors
    assert game.end().board().is_checkmate()
    assert len(record["trace"]) == 4


def test_opening_history_repetition_and_ply_cap():
    opening = {"moves": ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]}
    r = play_game(ScriptPlayer([]), ScriptPlayer(["f6g8"]), "white", "black", Settings(), opening, "repetition")
    assert r["termination"] == "threefold automatic draw"
    assert r["result"] == "1/2-1/2"
    r = play_game(ScriptPlayer(["e2e4"]), ScriptPlayer([]), "white", "black", Settings(max_plies=1), {}, "cap")
    assert r["termination"] == "custom ply-cap draw"


def test_schedule_stable_balanced_connected():
    names, baseline = ["az-a", "az-b", "az-c"], "sf8"
    openings = [{"name": "initial"}, {"name": "e4", "moves": ["e2e4"]}]
    first = schedule(names, baseline, 2, openings, 42)
    extended = schedule(names, baseline, 3, openings, 42)
    assert {x[0] for x in first}.issubset({x[0] for x in extended})
    for i in range(0, len(first), 2):
        assert first[i][1:3] == first[i + 1][2:0:-1]
        assert first[i][3] == first[i + 1][3]


def test_output_escape_and_symlink_rejected(tmp_path):
    with pytest.raises(ValueError):
        output_path("../alphazero/new.py")
    symlink = tmp_path / "escape"
    symlink.symlink_to(ROOT.parent)
    with pytest.raises(ValueError):
        output_path(symlink / "README.md")


def test_bayeselo_real_binary_connected_fit_and_anchor(tmp_path):
    run = tmp_path / "ratings"
    (run / "games").mkdir(parents=True)
    write_json(run / "manifest.json", {"baseline_name": "sf8", "settings": {"seconds": 1},
                                     "checkpoints": [{"name": "az-a", "step": 10, "path": "synthetic"}]})
    for i in range(20):
        w, b = ("az-a", "sf8") if i % 2 else ("sf8", "az-a")
        result = "1/2-1/2" if i < 10 else "1-0" if w == "az-a" else "0-1"
        pgn = f'[White "{w}"]\n[Black "{b}"]\n[Result "{result}"]\n\n{result}'
        write_json(run / "games" / f"{i}.json", {"white": w, "black": b, "result": result, "pgn": pgn})
    report = fit(run)
    rows = {r["name"]: r for r in report["ratings"]}
    assert rows["az-a"]["elo"] > rows["sf8"]["elo"] == 0
    assert rows["az-a"]["draws"] == rows["az-a"]["wins"] == 10
    assert rows["az-a"]["lower95"] < rows["az-a"]["elo"] < rows["az-a"]["upper95"]
    assert (run / "report.html").exists()
    anchored = fit(run, anchor_elo=3000, anchor_source="test reference, artificial")
    assert next(r for r in anchored["ratings"] if r["name"] == "sf8")["elo"] == 3000
    with pytest.raises(ValueError):
        fit(run, anchor_elo=3000)


def test_bayeselo_rejects_disconnected_games(tmp_path):
    run = tmp_path / "disconnected"
    (run / "games").mkdir(parents=True)
    write_json(run / "manifest.json", {"baseline_name": "sf8", "settings": {}, "checkpoints": []})
    write_json(run / "games/1.json", {"white": "a", "black": "b", "result": "1-0", "pgn": ""})
    with pytest.raises(ValueError, match="connected"):
        fit(run)


def test_browser_http_move_and_stale_revision(checkpoint, tmp_path):
    from http.server import ThreadingHTTPServer
    player = ModelPlayer(inspect_checkpoint(checkpoint[0]), "CPU:CLANG")
    game = HumanGame(player, str(tmp_path / "human"), .001, simulations=1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(game))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        state = json.load(urllib.request.urlopen(base + "/api/state"))
        body = {"id": state["id"], "revision": state["revision"], "move": "e2e4"}
        request = urllib.request.Request(base + "/api/move", json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        updated = json.load(urllib.request.urlopen(request))
        assert len(updated["history"]) == 2 and updated["turn"] == "white"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 409
        game = chess.pgn.read_game(io.StringIO(updated["pgn"]))
        assert not game.errors
        assert len(list(game.mainline_moves())) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_uci_real_engine_protocol(checkpoint):
    engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-B", "-m", "eval", "uci", "--checkpoint",
                str(checkpoint[0]), "--device", "CPU:CLANG"], cwd=ROOT.parent, timeout=20)
    try:
        engine.ping()
        board = chess.Board()
        board.push_uci("e2e4")
        result = engine.play(board, chess.engine.Limit(nodes=2))
        assert result.move in board.legal_moves
        analysis = engine.analysis(board)
        time.sleep(.15)
        engine.ping()
        analysis.stop()
        result = analysis.wait()
        assert result.move in board.legal_moves
        terminal = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        result = engine.play(terminal, chess.engine.Limit(time=.01))
        assert not result.move  # python-chess represents bestmove 0000 as a null Move.
    finally:
        engine.quit()
