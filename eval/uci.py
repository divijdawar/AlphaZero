"""UCI adapter with a responsive command reader and cancellable search thread."""
from __future__ import annotations

import sys
import threading
from engine.env import ChessEnv
from .player import ModelPlayer, parse_move


def serve(checkpoint, device: str, default_seconds: float, simulation_cap: int | None):
    output_lock = threading.Lock()
    def emit(line):
        with output_lock:
            print(line, flush=True)

    state = {"player": None, "error": None}
    ready = threading.Event()
    def initialize():
        try:
            state["player"] = ModelPlayer(checkpoint, device)
        except Exception as exc:  # noqa: BLE001 — keep UCI output valid when inference fails
            state["error"] = str(exc)
        finally:
            ready.set()
    loader = threading.Thread(target=initialize, daemon=True)
    loader.start()
    env, stop, worker = ChessEnv.startpos(), threading.Event(), None

    def cancel():
        nonlocal worker
        stop.set()
        if worker:
            worker.join()
            worker = None

    def search(position, seconds, nodes, cancel_event, searchmoves):
        ready.wait()
        try:
            if state["error"]:
                raise RuntimeError(state["error"])
            player = state["player"]
            result = player.choose(position, seconds, simulations=nodes, stop=cancel_event)
            if searchmoves and result.uci not in searchmoves:
                raise ValueError("searchmoves is not supported; omit it")
            info = f'info nodes {result.simulations} time {int(result.elapsed * 1000)}'
            if result.pv:
                info += ' pv ' + ' '.join(result.pv)
            emit(info)
            emit('bestmove ' + (result.uci or '0000'))
        except Exception as exc:  # noqa: BLE001 — keep UCI output valid when inference fails
            emit('info string error: ' + str(exc).replace('\n', ' '))
            emit('bestmove 0000')

    try:
        for line in sys.stdin:
            tokens = line.strip().split()
            if not tokens:
                continue
            command = tokens[0]
            try:
                if command == "uci":
                    emit('id name AlphaZero-' + checkpoint.name)
                    emit('id author local AlphaZero eval')
                    emit('option name MoveTime type spin default 1000 min 1 max 3600000')
                    emit('uciok')
                elif command == "isready":
                    def acknowledge():
                        ready.wait()
                        if state["error"]:
                            emit('info string initialization failed: ' + state["error"])
                        emit('readyok')
                    threading.Thread(target=acknowledge, daemon=True).start()
                elif command == "setoption":
                    if tokens[1:3] == ["name", "MoveTime"] and "value" in tokens:
                        default_seconds = int(tokens[tokens.index("value") + 1]) / 1000
                        if default_seconds <= 0:
                            raise ValueError("MoveTime must be positive")
                elif command == "ucinewgame":
                    cancel()
                    ready.wait()
                    if state["player"]:
                        state["player"].reset()
                    env = ChessEnv.startpos()
                elif command == "position":
                    cancel()
                    split = tokens.index("moves") if "moves" in tokens else len(tokens)
                    candidate = ChessEnv.startpos() if tokens[1] == "startpos" else ChessEnv.from_fen(' '.join(tokens[2:split]))
                    for move in tokens[split + 1:]:
                        candidate.step(parse_move(candidate, move))
                    env = candidate
                elif command == "go":
                    cancel()
                    stop = threading.Event()
                    def parameter(name, tokens=tokens):
                        return int(tokens[tokens.index(name) + 1]) if name in tokens else None
                    nodes = parameter("nodes") or simulation_cap
                    seconds = default_seconds
                    if "infinite" in tokens or ("nodes" in tokens and "movetime" not in tokens):
                        seconds = None
                    elif "movetime" in tokens:
                        seconds = max(0.001, parameter("movetime") / 1000)
                    elif "wtime" in tokens or "btime" in tokens:
                        remaining = parameter("wtime" if env.board.turn == 1 else "btime")
                        if remaining is not None:
                            seconds = max(0.001, remaining / 20000)
                    if "searchmoves" in tokens:
                        raise ValueError("searchmoves is unsupported")
                    worker = threading.Thread(target=search, args=(env.clone(), seconds, nodes, stop, []), daemon=True)
                    worker.start()
                elif command == "stop":
                    cancel()
                elif command == "quit":
                    break
            except (ValueError, IndexError) as exc:
                emit('info string error: ' + str(exc).replace('\n', ' '))
    finally:
        cancel()
