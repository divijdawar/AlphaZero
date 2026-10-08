# AlphaZero

*A lean implementation of AlphaZero for chess*

This repo is an implementation of the paper [AlphaZero](https://arxiv.org/abs/1712.01815) in pure Python + tinygrad. AlphaZero is a reinforcement-learning system from DeepMind that learned chess, shogi, and Go entirely through self-play, starting from random play and receiving only the rules of the game. It combines a neural network + Monte Carlo Tree Search (MCTS) + self-play.

## Training metrics

Inference and training use persistent TinyJit captures by default. Pass
`--no-jit` to run without captures. Each model snapshot owns its inference
captures, which are released when that snapshot retires.

`ChessEnv` boards and histories are read-only and shared safely by search clones.
To edit a position, use `env.board.copy()` and construct a new `ChessEnv` from it.

Training prints interval-average total, policy, value, and regularization losses,
learning rate, completed games, replay size, and wall-time throughput. Progress is
printed every 100 updates or 30 seconds by default, including during replay
warm-up (losses appear as `—` until an update completes).

```bash
uv run python -m alphazero.train --log-every 20 --log-seconds 10
```

To save an interactive report, install the optional plotting dependency:

```bash
uv sync --extra metrics
uv run --extra metrics python -m alphazero.train \
  --log-every 20 --log-seconds 10 --metrics-dir runs --open-report
```

Each run gets a unique directory under `runs/` containing `config.json`, flushed
`metrics.jsonl` records, and an end-of-run `report.html`. The report works offline:
hover for values, zoom or pan, click legend entries to toggle curves, and use the
camera button to export an image. `--open-report` is optional; the report path is
printed at the end. Losses are plotted against training step, and throughput
against elapsed time. Self-play throughput counts positions received from
completed games, so short intervals can be bursty.

Without `--metrics-dir`, Plotly is not required. After an interrupted run, rebuild
a report from the records already written:

```bash
uv run --extra metrics python -m alphazero.metrics \
  runs/<run-directory>/metrics.jsonl --open-report
```

Use `--output path/to/report.html` to choose the regenerated report's location.
