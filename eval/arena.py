from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
from time import monotonic

import chess
import chess.pgn
from engine.env import ChessEnv, position_key
from .checkpoints import Checkpoint
from .player import ModelPlayer, StockfishPlayer, parse_move
from .runtime import digest, output_path, write_json


@dataclass(frozen=True)
class Settings:
    profile: str = "checkpoint"
    seconds: float = 1.0
    main_seconds: float | None = None
    increment: float = 0.0
    max_plies: int = 512
    simulations: int | None = None
    resignation: bool = False
    model_resign_value: float = -0.9
    stockfish_resign_cp: int = -900
    stockfish_resign_moves: int = 10
    device: str = "CPU:CLANG"
    seed: int = 0


def load_openings(path: str | None) -> list[dict]:
    if path is None:
        return [{"name": "initial", "fen": chess.STARTING_FEN, "moves": []}]
    p = Path(path).expanduser().resolve(strict=True)
    if p.suffix.lower() == ".pgn":
        result = []
        with p.open() as f:
            while (game := chess.pgn.read_game(f)) is not None:
                if game.errors:
                    raise ValueError(f"invalid opening PGN: {game.errors}")
                result.append({"name": game.headers.get("Event", "opening"),
                               "fen": game.board().fen(en_passant="fen"),
                               "moves": [m.uci() for m in game.mainline_moves()]})
    else:
        result = json.loads(p.read_text())
    if not isinstance(result, list) or not result:
        raise ValueError("opening suite must contain at least one opening")
    for item in result:
        env = ChessEnv.from_fen(item.get("fen", chess.STARTING_FEN))
        for move in item.get("moves", []):
            env.step(parse_move(env, move))
        if env.is_terminal():
            raise ValueError("opening suite contains a terminal position")
    return result


def play_game(white, black, white_name: str, black_name: str, settings: Settings,
              opening: dict, game_id: str) -> dict:
    fen = opening.get("fen", chess.STARTING_FEN)
    env, board = ChessEnv.from_fen(fen), chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    game.headers.update({"Event": f"AlphaZero evaluation {settings.profile}", "White": white_name,
                         "Black": black_name, "Round": game_id, "Opening": str(opening.get("name", "initial"))})
    game.headers["TimeControl"] = (f'{settings.main_seconds:g}+{settings.increment:g}' if settings.main_seconds is not None
                                   else f'1/{settings.seconds:g}')
    node, moves = game, []
    for uci in opening.get("moves", []):
        env.step(parse_move(env, uci))
        cmove = chess.Move.from_uci(uci)
        board.push(cmove)
        node = node.add_variation(cmove)
        moves.append(uci)
    white.reset()
    black.reset()
    clocks = [settings.main_seconds, settings.main_seconds]
    traces, bad_moves = [], {1: 0, -1: 0}
    termination, outcome = None, None
    while not env.is_terminal() and env.ply < settings.max_plies:
        color = env.board.turn
        index = 0 if color == 1 else 1
        player = white if color == 1 else black
        seconds = settings.seconds if settings.main_seconds is None else max(0.001, clocks[index] / 20)
        start = monotonic()
        if isinstance(player, StockfishPlayer):
            decision = player.choose(env, seconds, board=board, clocks=None if settings.main_seconds is None
                                      else (clocks[0], clocks[1], settings.increment))
        else:
            decision = player.choose(env, seconds, simulations=settings.simulations)
        elapsed = monotonic() - start
        if settings.main_seconds is not None:
            clocks[index] -= elapsed
            if clocks[index] <= 0:
                outcome, termination = -color, "time forfeiture"
                break
            clocks[index] += settings.increment
        traces.append({"ply": env.ply, "player": white_name if color == 1 else black_name,
                       **asdict(decision), "wall_seconds": elapsed})
        if decision.uci is None:
            raise RuntimeError("player returned no move in a nonterminal position")
        if settings.resignation and decision.value is not None:
            if isinstance(player, StockfishPlayer):
                bad_moves[color] = bad_moves[color] + 1 if decision.value <= settings.stockfish_resign_cp else 0
                resign = bad_moves[color] >= settings.stockfish_resign_moves
            else:
                resign = decision.value < settings.model_resign_value
            if resign:
                outcome, termination = -color, "resignation"
                break
        move = parse_move(env, decision.uci)
        cmove = chess.Move.from_uci(decision.uci)
        if cmove not in board.legal_moves:
            raise RuntimeError(f"repo and python-chess disagree on legal move {decision.uci}")
        env.step(move)
        board.push(cmove)
        if board.fen(en_passant="fen") != env.to_fen():
            raise RuntimeError("repo and python-chess board states diverged")
        node = node.add_variation(cmove)
        node.comment = f"{decision.simulations} simulations, {elapsed:.3f}s"
        moves.append(decision.uci)
        white.observe(decision.uci)
        black.observe(decision.uci)
    if outcome is None:
        outcome = env.outcome()
        if outcome is None:
            outcome, termination = 0, "512-ply rule" if settings.max_plies == 512 else "custom ply-cap draw"
        elif outcome:
            termination = "checkmate"
        elif not env.legal_moves():
            termination = "stalemate"
        elif env.board.halfmove >= 100:
            termination = "50-move automatic draw"
        elif env.counts[position_key(env.board)] >= 3:
            termination = "threefold automatic draw"
        else:
            termination = "insufficient material"
    result = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[outcome]
    game.headers.update({"Result": result, "Termination": termination})
    return {"id": game_id, "white": white_name, "black": black_name, "result": result,
            "termination": termination, "opening": opening, "moves": moves, "trace": traces,
            "final_fen": env.to_fen(), "pgn": str(game)}


def schedule(names: list[str], baseline: str, pairs: int, openings: list[dict], seed: int,
             final: bool = False) -> list[tuple]:
    matchups = [(name, baseline) for name in names]
    if not final:
        for i, name in enumerate(names):
            opponents = {i - 1, i - 2, 0} - {i}
            matchups.extend((name, names[j]) for j in sorted(opponents) if 0 <= j < i)
    rng = random.Random(seed)
    order = list(openings)
    rng.shuffle(order)
    jobs = []
    for a, b in matchups:
        for n in range(pairs):
            opening = order[n % len(order)]
            for w, bl in ((a, b), (b, a)):
                gid = hashlib.sha256(json.dumps([w, bl, n, opening], sort_keys=True).encode()).hexdigest()[:24]
                jobs.append((gid, w, bl, opening))
    return jobs


def run_matches(checkpoints: list[Checkpoint], settings: Settings, *, run: str, stockfish: str,
                threads: int, hash_mb: int, syzygy: str, pairs: int, openings: list[dict],
                final: bool = False, log=print) -> Path:
    target = output_path(run)
    target.mkdir(parents=True, exist_ok=True)
    (target / "games").mkdir(exist_ok=True)
    engine = StockfishPlayer(stockfish, threads, hash_mb, syzygy)
    if not engine.description["id"].get("name", "").startswith("Stockfish 8"):
        engine.close()
        raise ValueError("the published reference profile requires Stockfish 8; run eval/run setup for the bundled binary")
    baseline = "sf8-" + hashlib.sha256(json.dumps(engine.description, sort_keys=True).encode()).hexdigest()[:12]
    identity = {"search": asdict(settings), "engine": engine.description, "openings": openings,
                "final": final, "platform": platform.platform()}
    # Include the exact inference/rules sources, even when the working tree is dirty.
    from .runtime import REPO
    identity["repo_sources"] = {str(p.relative_to(REPO)): digest(p)
                               for folder in ("engine", "alphazero", "eval")
                               for p in sorted((REPO / folder).glob("*.py")) if p.name != "setup.py"}
    manifest_path = target / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old and old["settings"] != identity:
        engine.close()
        raise ValueError("run settings or source code changed; use a new --output to avoid mixing ratings")
    combined = {c["name"]: c for c in old["checkpoints"]} if old else {}
    combined.update({c.name: c.manifest() for c in checkpoints})
    manifest = {"settings": identity, "baseline_name": baseline, "checkpoints": list(combined.values()),
                "requested_pairs": pairs}
    write_json(manifest_path, manifest)
    models = {c.name: c for c in checkpoints}
    cache = {}
    try:
        jobs = schedule(list(models), baseline, pairs, openings, settings.seed, final)
        for index, (gid, w, b, opening) in enumerate(jobs):
            game_path = target / "games" / f"{gid}.json"
            if game_path.exists():
                continue
            # Retain only the two players needed for this pairing to bound accelerator memory.
            for name in list(cache):
                if name not in (w, b):
                    del cache[name]
            for name in (w, b):
                if name != baseline and name not in cache:
                    cache[name] = ModelPlayer(models[name], settings.device)
            log(f"game {index + 1}/{len(jobs)}: {w} vs {b}")
            record = play_game(engine if w == baseline else cache[w], engine if b == baseline else cache[b],
                               w, b, settings, opening, gid)
            write_json(game_path, record)
            log(f'{record["result"]}: {record["termination"]}')
    finally:
        engine.close()
    return target
