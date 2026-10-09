# Checkpoint Elo and human play

Everything added for evaluation lives under `eval/`. Training and chess-engine code are imported read-only. Checkpoints can be read from anywhere; caches, installed evaluation dependencies, binaries, match records, reports and human games stay under this directory. No training hook or checkpoint promotion gate was added.

## Quick start

Run these commands from the repository root. Replace the checkpoint paths with your actual `.safetensors` files.

```sh
cd /Users/divij/Desktop/mcts/alphazero

# Already completed on this machine; rerun on a fresh checkout.
eval/run setup

# Confirm the checkpoint's saved architecture, training step and weight identity.
eval/run inspect /absolute/path/to/checkpoint.safetensors

# Assign a checkpoint Elo: 20 games against full-strength Stockfish 8,
# balanced colors, one second per move, then an actual BayesElo fit.
eval/run evaluate --checkpoint /absolute/path/to/checkpoint.safetensors

# Open the local browser chessboard and play that checkpoint.
eval/run play --checkpoint /absolute/path/to/checkpoint.safetensors

# Final trained model: your selected 2017 default, 100 games, 60 seconds/move.
eval/run final --checkpoint /absolute/path/to/final.safetensors
```

`evaluate` saves its default HTML report to `eval/runs/checkpoints/report.html`; `final` uses `eval/runs/final/report.html`. Both also save `ratings.json`, `ratings.csv`, `games.pgn`, individual game JSON with move/search traces, raw BayesElo output, the fitting commands, and a reproducibility manifest. The report includes wins, draws, losses, approximate 95% rating intervals and an Elo-versus-training-step chart when step metadata is available.

The browser opens `http://127.0.0.1:8765/`. Select a piece and its destination. You can play either color, change thinking time, flip the board, choose promotions, resign, start another game and save PGN. Games are also saved automatically under `eval/runs/human/`. Stop the server with Ctrl+C. Use `--no-open`, `--port 8766` or `--interface terminal` when needed.

## What the rating means

The default scale sets **this run's Stockfish 8 baseline to Elo 0**. A model rated -300 is 300 Elo below that benchmark under the recorded hardware and search conditions. This is a relative engine rating; it is not a FIDE, Chess.com or Lichess rating.

AlphaZero's published training plot fitted checkpoint tournament results jointly using BayesElo and anchored the baseline to a published rating. To put your results on an external scale, supply a documented baseline value whose conditions you intend to use:

```sh
eval/run rate --run runs/checkpoints \
  --anchor-elo YOUR_DOCUMENTED_STOCKFISH8_ELO \
  --anchor-source 'Rating-list URL, version, hardware and time control'
```

Replace the placeholder with a number. The software requires its source and records it; it does not assume an arbitrary Stockfish rating. Re-anchoring changes the rating offset, not the match evidence or hardware comparability. `rate` reads the existing run and does not replay games.

Small samples and all-win/all-loss results are marked provisional. The virtual-draw prior keeps those fits finite, but an all-loss result against Stockfish cannot reliably establish how far below Stockfish your model plays. Include multiple checkpoint opponents for a useful training comparison and collect more games. Repeated deterministic games from the same start are correlated; game count alone does not guarantee independent evidence. An opening suite supplies additional starting positions.

## Follow the AlphaZero measurement method

The [2017 paper, Methods/Evaluation](https://arxiv.org/pdf/1712.01815) describes two separate experiments: a one-second-per-move tournament among training iterations and a baseline, fitted jointly with BayesElo, and a 100-game final head-to-head match at one minute per move. The latter reports W/D/L, rather than constituting the training curve itself. This implementation retains that distinction and also fits the final match to provide the final-model Elo you requested.

- **Checkpoint tournament:** one second per move, full-strength Stockfish 8, paired colors, greedy root visit-count selection, no exploration noise or pondering. Defaults to 20 games per matchup as a practical local starting point; this game count and the sparse pairing schedule are implementation choices. Each checkpoint plays Stockfish, its previous two checkpoints and the initial checkpoint where applicable. All completed games are fitted together.
- **2017 final:** 100 games, 50 as each color, 60 seconds per move, initial board, resignation enabled. Stockfish resigns at -900 centipawns for ten consecutive own moves. The model resigns below search value -0.9, the expected-outcome conversion of the paper's 5% win-rate threshold.
- **2018 alternative:** `--protocol 2018` uses 1,000 games with 10,800 seconds plus 15 seconds per move. Its Stockfish resignation setting is -650 centipawns for four consecutive own moves. The model clock allocator spends approximately remaining time / 20 per move; this allocator is a local implementation choice, not a reproduction of DeepMind's internal allocator. See the [2018 publication and supplement](https://doi.org/10.1126/science.aar6404).
- **BayesElo:** uses the compiled upstream program, virtual-draw prior 2, advantage 32.8, draw Elo 97.3, scale 1 and 95% covariance intervals. These estimator settings are recorded explicitly; the papers do not publish all estimator options or their full checkpoint game schedule.

Local defaults use one Stockfish thread and 128 MB hash, and `CPU:CLANG` for tinygrad. The 2017 experiment used 64 Stockfish threads / 1 GB hash and four TPUs for AlphaZero; the 2018 experiment used 44 CPU cores / 32 GB hash and tablebases. Your local run therefore follows the measurement approach, with its own measured benchmark conditions. It does not reproduce the published AlphaZero rating or hardware.

For the original 2017 Stockfish resource settings, on a machine that supports them:

```sh
eval/run final --checkpoint /absolute/path/to/final.safetensors \
  --threads 64 --hash-mb 1024 --output runs/final-2017-64threads
```

The optional 2018 profile:

```sh
eval/run final --checkpoint /absolute/path/to/final.safetensors \
  --protocol 2018 --threads 44 --hash-mb 32768 \
  --syzygy /absolute/path/to/tablebases --output runs/final-2018
```

`--device METAL` or `--device NV` can select an available tinygrad accelerator. Those backends were not verified here; CPU inference was verified. Model loading/JIT warmup occurs before playing. Search stops at the time budget between inference calls; an in-flight leaf inference can overrun it, which is especially relevant for a large model on a CPU. Wall time and simulations are recorded per move. The 2018 profile enforces total clocks, including inference overruns. Fixed-time profiles use search budgets rather than adjudicating per-move overruns as forfeits.

Games follow the repository's automatic repetition, 50-move and insufficient-material rules and draw after 512 plies. Optional `--max-plies`, `--seconds`, `--simulations`, `--main-seconds`, `--increment`, `--no-resignation` and custom openings alter the benchmark and are recorded. Development overrides should not be presented as the original published experiment.

## Several checkpoints and resuming

```sh
# Directories discover their top-level .safetensors checkpoints.
eval/run evaluate --checkpoint /absolute/path/to/checkpoints \
  --games 100 --output runs/training-curve

# Explicit files also work; final weights identical to the last checkpoint
# are deduplicated by tensor contents, including batch-normalization state.
eval/run evaluate --checkpoint /path/step-100.safetensors /path/step-200.safetensors \
  /path/final.safetensors --output runs/selected-checkpoints
```

Run the same command again after interruption to resume completed game records. Only a partially played game is replayed. You can increase `--games` or add checkpoints and refit the expanded connected tournament. Use the complete checkpoint set when extending a training curve so its ordering stays consistent. Use a new `--output` whenever engine/search settings, openings or inference/rules source code changes; incompatible runs are rejected rather than pooled.

Output paths, including `rate --run`, are relative to `eval/`. Absolute output paths must still resolve within `eval/`. Checkpoint/config/opening/tablebase inputs may be outside it.

An opening suite can be PGN games containing just their opening moves, or a JSON list:

```json
[
  {"name": "initial", "moves": []},
  {"name": "queen-pawn", "moves": ["d2d4", "d7d5", "c2c4"]}
]
```

An optional `fen` is the initial position before that item's moves. Moves preserve repetition and input history. Both colors play each sampled opening. Pass `--openings /path/openings.json` and use a separate output directory.

## UCI and checkpoint compatibility

```sh
eval/run uci --checkpoint /absolute/path/to/checkpoint.safetensors --seconds 1
```

Configure a standard chess GUI to launch the absolute `eval/run` executable with these arguments: `uci --checkpoint /absolute/path/to/checkpoint.safetensors`. GUIs that require an argument-free engine path can use a small executable wrapper saved inside `eval/`:

```sh
#!/bin/sh
exec /Users/divij/Desktop/mcts/alphazero/eval/run uci \
  --checkpoint /absolute/path/to/checkpoint.safetensors
```

The adapter supports `uci`, `isready`, `ucinewgame`, `position startpos/fen ... moves ...`, `go movetime/nodes/infinite/wtime/btime`, `stop`, `quit`, and a `MoveTime` option. It reports legal `bestmove`, search nodes/time and PV, with a null move for terminal positions. It does not support `searchmoves`, pondering or MultiPV. Clocked UCI searches allocate remaining time / 20; increments and `movestogo` are not used by this allocator.

The repo's training checkpoints supply architecture and training-step metadata and contain `model.*` plus optional `optimizer.*` tensors. Only inference tensors are loaded; optimizer state is ignored and batch normalization is frozen. Metadata-free weight files require `--config /path/architecture.json` on evaluate/final/play/uci, using the exact saved `alphazero.config.Config` architecture. Unsupported formats, shape mismatches and non-finite predictions fail explicitly.

## Files and verification

- `run`, `__main__.py`, `runtime.py`: CLI, isolated caches/outputs and isolation audit.
- `setup.py`, `requirements.txt`, `.gitignore`: pinned local dependencies, official Stockfish 8 source and upstream BayesElo builds; generated data stays ignored. Setup needs a C++ compiler and the repo's existing `.venv`; it does not edit project dependency files or install globally.
- `checkpoints.py`, `player.py`: checkpoint discovery/loading, deterministic timed MCTS and Stockfish UCI adapter.
- `arena.py`, `ratings.py`: balanced/resumable matches, provenance, joint BayesElo ratings, CSV/JSON/HTML reports.
- `human.py`, `uci.py`, `web/index.html`: local browser/terminal play, game saving and UCI serving.
- `tests/`: inference parity and frozen BN, checkpoint identity, MCTS limits/history, legal moves/promotions, game outcomes, pairing, output confinement, real BayesElo fitting, browser HTTP state and real UCI interoperability.
- `vendor/`, `work/`, `runs/`: locally built tools/dependencies, disposable verification fixtures/caches and your result data.

Verified using a small synthetic checkpoint: 14 tests, scoped Ruff lint, two full Stockfish games ending in checkmate, both final-profile paths with shortened development settings, game-record resumption and browser human-move/model-reply interaction. The synthetic checkpoint is not a trained model and its test ratings do not describe your checkpoint. Full 100-game/60-second and 1,000-game/long-clock experiments have not been run.

```sh
PYTHONDONTWRITEBYTECODE=1 DEV=CPU:CLANG .venv/bin/python -B -m pytest eval/tests \
  -q -p no:cacheprovider --basetemp=eval/work/pytest
.venv/bin/ruff check eval --no-cache --exclude eval/vendor,eval/work,eval/runs
eval/run audit
```

`audit` compares files outside `eval/` against the snapshot taken before this implementation. Existing uncommitted changes outside the folder were preserved. It checks this repository, not every file on the computer.
