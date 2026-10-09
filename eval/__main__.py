from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from .runtime import ROOT, configure


def main(argv=None) -> int:
    configure()
    parser = argparse.ArgumentParser(description="Isolated AlphaZero checkpoint Elo and play tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="install chess and build Stockfish 8/BayesElo inside eval/vendor")
    commands.add_parser("audit", help="compare files outside eval against the implementation baseline")
    inspect = commands.add_parser("inspect", help="show configuration and identity without loading an optimizer")
    inspect.add_argument("checkpoints", nargs="+")
    for name in ("evaluate", "final", "play", "uci"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", nargs="+" if name == "evaluate" else None, required=True,
                       help="safetensors path; evaluate also accepts checkpoint directories")
        p.add_argument("--config", help="Config JSON for metadata-free weights only")
        p.add_argument("--device", default="CPU:CLANG", help="tinygrad backend; CPU:CLANG is portable, METAL/NV are optional")
        p.add_argument("--seconds", type=float, default=None, help="thinking seconds per move")
        p.add_argument("--simulations", type=int, help="optional development simulation cap")
        if name in ("evaluate", "final"):
            p.add_argument("--stockfish", default=str(ROOT / "vendor/bin/stockfish8"))
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--hash-mb", type=int, default=128)
            p.add_argument("--syzygy", default="", help="optional read-only tablebase path")
            p.add_argument("--bayeselo", default=str(ROOT / "vendor/bin/bayeselo"))
            p.add_argument("--games", type=int, default=None, help="even number of games per matchup")
            p.add_argument("--max-plies", type=int, default=512)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--openings", help="opening suite as PGN or JSON")
            p.add_argument("--resignation", action=argparse.BooleanOptionalAction, default=None)
            p.add_argument("--output", default="runs/checkpoints" if name == "evaluate" else "runs/final")
            p.add_argument("--anchor-elo", type=float)
            p.add_argument("--anchor-source", default="")
            if name == "final":
                p.add_argument("--protocol", choices=("2017", "2018"), default="2017")
                p.add_argument("--main-seconds", type=float, help="override 2018 main clock (development)")
                p.add_argument("--increment", type=float, default=None)
        if name == "play":
            p.add_argument("--interface", choices=("browser", "terminal"), default="browser")
            p.add_argument("--color", choices=("white", "black"), default="white")
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--no-open", action="store_true")
            p.add_argument("--output", default="runs/human")
    rate = commands.add_parser("rate", help="refit saved games without rerunning matches")
    rate.add_argument("--run", required=True)
    rate.add_argument("--bayeselo", default=str(ROOT / "vendor/bin/bayeselo"))
    rate.add_argument("--anchor-elo", type=float)
    rate.add_argument("--anchor-source", default="")
    args = parser.parse_args(argv)
    if hasattr(args, "device"):
        os.environ["DEV"] = args.device
    try:
        if args.command == "setup":
            from .setup import setup
            print(json.dumps(setup(), indent=2))
        elif args.command == "audit":
            from .runtime import isolation_audit
            result = isolation_audit()
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        elif args.command == "inspect":
            from .checkpoints import discover
            print(json.dumps([c.manifest() for c in discover(args.checkpoints)], indent=2))
        elif args.command == "rate":
            from .ratings import fit
            report = fit(Path(args.run), bayeselo=args.bayeselo,
                         anchor_elo=args.anchor_elo, anchor_source=args.anchor_source)
            print(json.dumps(report, indent=2))
        else:
            if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
                raise ValueError("thinking time must be positive and finite")
            if args.simulations is not None and args.simulations < 1:
                raise ValueError("simulation cap must be positive")
            from .checkpoints import discover
            from alphazero.config import Config
            config = Config(**json.loads(Path(args.config).read_text())) if args.config else None
            paths = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
            checkpoints = discover(paths, config)
            if args.command in ("evaluate", "final"):
                from .arena import Settings, load_openings, run_matches
                from .ratings import fit
                final = args.command == "final"
                if final and len(checkpoints) != 1:
                    raise ValueError("final evaluation needs exactly one checkpoint")
                protocol = args.protocol if final else "checkpoint"
                seconds = args.seconds if args.seconds is not None else (60.0 if protocol == "2017" else 1.0)
                games = args.games if args.games is not None else (1000 if protocol == "2018" else 100 if final else 20)
                main_clock = (args.main_seconds if args.main_seconds is not None else 10800.0) if protocol == "2018" else None
                increment = (args.increment if args.increment is not None else 15.0) if protocol == "2018" else 0.0
                if games < 2 or games % 2 or args.threads < 1 or args.hash_mb < 1 or args.max_plies < 1:
                    raise ValueError("games must be positive/even; threads, hash, and max-plies must be positive")
                if not math.isfinite(seconds) or seconds <= 0 or (main_clock is not None and (not math.isfinite(main_clock) or main_clock <= 0)):
                    raise ValueError("thinking time and main clock must be finite and positive")
                if not math.isfinite(increment) or increment < 0:
                    raise ValueError("increment must be finite and nonnegative")
                if args.simulations is not None and args.simulations < 1:
                    raise ValueError("simulation cap must be positive")
                if args.anchor_elo is not None and (not math.isfinite(args.anchor_elo) or not args.anchor_source.strip()):
                    raise ValueError("a finite --anchor-elo requires --anchor-source")
                settings = Settings(profile=protocol, seconds=seconds, main_seconds=main_clock, increment=increment,
                                    max_plies=args.max_plies, simulations=args.simulations,
                                    resignation=args.resignation if args.resignation is not None else final,
                                    stockfish_resign_cp=-650 if protocol == "2018" else -900,
                                    stockfish_resign_moves=4 if protocol == "2018" else 10,
                                    device=args.device, seed=args.seed)
                run = run_matches(checkpoints, settings, run=args.output, stockfish=str(Path(args.stockfish).expanduser().resolve(strict=True)),
                                  threads=args.threads, hash_mb=args.hash_mb, syzygy=args.syzygy,
                                  pairs=games // 2, openings=load_openings(args.openings), final=final)
                report = fit(run, bayeselo=args.bayeselo, anchor_elo=args.anchor_elo, anchor_source=args.anchor_source)
                print(f"Report: {run / 'report.html'}")
                for r in report["ratings"]:
                    print(f'{r["name"]}: Elo {r["elo"]}, 95% [{r["lower95"]}, {r["upper95"]}], {r["games"]} games' +
                          (' (provisional)' if r["provisional"] else ''))
            elif args.command == "uci":
                from .uci import serve
                serve(checkpoints[0], args.device, args.seconds if args.seconds is not None else 1.0, args.simulations)
            else:
                from .human import play
                play(checkpoints[0], args)
    except KeyboardInterrupt:
        print("Stopped. Completed evaluation games are retained; rerun the same command to resume.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
