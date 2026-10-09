from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import uuid
import webbrowser

import chess
import chess.pgn
from engine.env import ChessEnv
from .player import ModelPlayer, parse_move
from .runtime import ROOT, output_path, write_json


class HumanGame:
    def __init__(self, player, output: str, seconds: float, simulations=None):
        self.player, self.output = player, output_path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seconds, self.simulations = seconds, simulations
        self.lock = threading.Lock()
        self.env, self.board = ChessEnv.startpos(), chess.Board()
        self.id, self.revision, self.human, self.override = uuid.uuid4().hex, 0, 1, None
        self.last = None

    def result(self):
        outcome = self.override if self.override is not None else self.env.outcome()
        return {None: "*", 1: "1-0", -1: "0-1", 0: "1/2-1/2"}[outcome]

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({"Event": "Human vs AlphaZero", "White": "Human" if self.human == 1 else self.player.checkpoint.name,
                             "Black": "Human" if self.human == -1 else self.player.checkpoint.name,
                             "Result": self.result()})
        return str(game) + "\n"

    def save(self):
        write_json(self.output / f"game-{self.id}.json", {"checkpoint": self.player.checkpoint.manifest(),
                   "seconds": self.seconds, "moves": [m.uci() for m in self.board.move_stack], "result": self.result()})
        # Atomic PGN replacement avoids losing the latest playable state on interruption.
        target = self.output / f"game-{self.id}.pgn"
        temporary = target.with_suffix(".pgn.tmp")
        temporary.write_text(self.pgn())
        temporary.replace(target)

    def advance(self, uci):
        move = parse_move(self.env, uci)
        cmove = chess.Move.from_uci(uci)
        if cmove not in self.board.legal_moves:
            raise RuntimeError("chess implementations disagree")
        self.env.step(move)
        self.board.push(cmove)
        self.player.observe(uci)
        self.revision += 1
        self.save()

    def computer(self):
        if self.result() == "*" and self.env.board.turn != self.human:
            d = self.player.choose(self.env, self.seconds, simulations=self.simulations)
            self.last = {"move": d.uci, "seconds": d.elapsed, "simulations": d.simulations, "pv": d.pv}
            if d.uci is None:
                raise RuntimeError("model returned no move")
            self.advance(d.uci)

    def state(self):
        history, b = [], self.board.root()
        for m in self.board.move_stack:
            history.append(b.san(m))
            b.push(m)
        pieces = {chess.square_name(s): p.symbol() for s, p in self.board.piece_map().items()}
        return {"id": self.id, "revision": self.revision, "fen": self.env.to_fen(), "pieces": pieces,
                "turn": "white" if self.env.board.turn == 1 else "black", "human": "white" if self.human == 1 else "black",
                "legal": [m.to_uci() for m in self.env.legal_moves()] if self.result() == "*" else [],
                "result": self.result(), "check": self.board.is_check(), "history": history, "last": self.last,
                "checkpoint": self.player.checkpoint.path.name, "player": self.player.checkpoint.name,
                "seconds": self.seconds, "pgn": self.pgn()}

    def new(self, color: str, seconds: float):
        if color not in ("white", "black") or not math.isfinite(seconds) or not 0 < seconds <= 3600:
            raise ValueError("choose white/black and a thinking time between 0 and 3600 seconds")
        self.env, self.board = ChessEnv.startpos(), chess.Board()
        self.id, self.revision, self.human = uuid.uuid4().hex, 0, 1 if color == "white" else -1
        self.override, self.last, self.seconds = None, None, seconds
        self.player.reset()
        self.computer()
        self.save()
        return self.state()


def make_handler(game: HumanGame):
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, status=200, content="application/json"):
            data = json.dumps(value).encode() if content == "application/json" else value.encode()
            self.send_response(status)
            self.send_header("Content-Type", content + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.send((ROOT / "web/index.html").read_text(), content="text/html")
            elif self.path == "/api/state":
                with game.lock:
                    self.send(game.state())
            elif self.path == "/api/pgn":
                with game.lock:
                    self.send(game.pgn(), content="application/x-chess-pgn")
            else:
                self.send({"error": "not found"}, 404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("invalid request size")
                origin = self.headers.get("Origin")
                if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                    self.send({"error": "origin rejected"}, 403)
                    return
                body = json.loads(self.rfile.read(length))
                with game.lock:
                    if self.path == "/api/new":
                        state = game.new(body.get("color", "white"), float(body.get("seconds", game.seconds)))
                    elif self.path in ("/api/move", "/api/resign", "/api/retry"):
                        if body.get("id") != game.id or body.get("revision") != game.revision:
                            self.send({"error": "game changed; refresh the board", "state": game.state()}, 409)
                            return
                        if game.result() != "*":
                            raise ValueError("game is already over")
                        if self.path == "/api/resign":
                            game.override = -game.human
                            game.revision += 1
                            game.save()
                        elif self.path == "/api/move":
                            if game.env.board.turn != game.human:
                                raise ValueError("wait for the model's move")
                            game.advance(body["move"])
                            game.computer()
                        else:
                            game.computer()
                        state = game.state()
                    else:
                        self.send({"error": "not found"}, 404)
                        return
                    self.send(state)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send({"error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001 — HTTP boundary reports inference failures to the UI
                self.send({"error": str(exc), "state": game.state()}, 500)

        def log_message(self, format, *args):
            pass
    return Handler


def play(checkpoint, args):
    seconds = args.seconds if args.seconds is not None else 1.0
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("thinking time must be positive and finite")
    player = ModelPlayer(checkpoint, args.device)
    game = HumanGame(player, args.output, seconds, args.simulations)
    game.new(args.color, seconds)
    if args.interface == "terminal":
        print("Enter UCI moves (e2e4, e7e8q), pgn, resign, or quit.")
        while game.result() == "*":
            print(game.board.unicode(borders=True))
            try:
                move = input(f'{game.state()["turn"]} to move > ').strip().lower()
            except EOFError:
                break
            if move == "quit":
                break
            if move == "pgn":
                print(game.pgn())
                continue
            if move == "resign":
                game.override = -game.human
                game.save()
                break
            try:
                game.advance(move)
                game.computer()
            except ValueError as exc:
                print(exc)
        print("Result:", game.result())
        print("PGN:", game.output / f"game-{game.id}.pgn")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Play {checkpoint.path.name} at {url} — Ctrl+C to stop", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()

