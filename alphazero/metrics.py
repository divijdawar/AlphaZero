"""Training metric summaries and optional, offline interactive reports."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import webbrowser
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

LOSS_KEYS = ("loss", "policy_loss", "value_loss", "regularization_loss")


@dataclass
class MetricWindow:
    started: float
    updates: int = 0
    positions: int = 0
    losses: dict[str, float] = field(default_factory=lambda: dict.fromkeys(LOSS_KEYS, 0.0))

    def record_update(self, metrics: dict[str, float]) -> None:
        self.updates += 1
        for key in LOSS_KEYS:
            self.losses[key] += metrics[key]

    def report(self, now: float) -> dict:
        elapsed = max(now - self.started, 1e-9)
        return {
            **{key: self.losses[key] / self.updates if self.updates else None for key in LOSS_KEYS},
            "window_updates": self.updates,
            "window_positions": self.positions,
            "window_seconds": elapsed,
            "training_updates_per_second": self.updates / elapsed,
            "selfplay_positions_per_second": self.positions / elapsed,
        }


def format_metrics(record: dict) -> str:
    def loss(key: str) -> str:
        return "—" if record[key] is None else f"{record[key]:.3f}"

    phase = "  warming replay" if record["step"] == 0 else ""
    return (
        f"step={record['step']}/{record['total_steps']}  elapsed={record['elapsed_seconds']:.1f}s"
        f"  lr={record['learning_rate']:.6g}{phase}\n"
        f"loss={loss('loss')}  policy={loss('policy_loss')}  value={loss('value_loss')}"
        f"  regularization={loss('regularization_loss')}\n"
        f"updates/s={record['training_updates_per_second']:.2f}"
        f"  selfplay_positions/s={record['selfplay_positions_per_second']:.2f}"
        f"  games={record['games']}  replay={record['samples']}"
    )


def require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError("Interactive reports require Plotly. Install with 'uv sync --extra metrics'.") from error
    return go, make_subplots


class RunMetrics:
    def __init__(self, root: str | Path, config: dict, settings: dict):
        require_plotly()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ-")
        self.directory = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        self.path = self.directory / "metrics.jsonl"
        self.path.touch()
        (self.directory / "config.json").write_text(
            json.dumps({"config": config, "settings": settings}, indent=2) + "\n", encoding="utf-8",
        )

    def write(self, record: dict) -> None:
        # Close each record promptly so it survives an interrupted training run.
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
            output.flush()

    def finish(self) -> Path:
        return build_report(self.path, self.directory / "report.html")


def read_records(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            # A hard stop can leave an incomplete final write; earlier rows remain usable.
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError(f"Invalid metrics JSON on line {index + 1}") from error
    return records


def build_figure(records: list[dict]):
    go, make_subplots = require_plotly()
    figure = make_subplots(
        rows=4, cols=1,
        specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("Losses · interval averages", "Learning rate", "Throughput", "Games and replay"),
        vertical_spacing=0.08,
    )

    def add(key: str, label: str, row: int, x: str = "step", secondary_y: bool | None = None):
        points = [r for r in records if r.get(key) is not None and math.isfinite(r[key])]
        figure.add_trace(go.Scatter(
            x=[r[x] for r in points], y=[r[key] for r in points],
            name=label, mode="lines+markers", marker={"size": 4},
            connectgaps=False,
        ), row=row, col=1, secondary_y=secondary_y)

    for key, label in zip(LOSS_KEYS, ("Total loss", "Policy loss", "Value loss", "Regularization loss")):
        add(key, label, 1)
    add("learning_rate", "Learning rate", 2)
    add("training_updates_per_second", "Updates/s", 3, "elapsed_seconds", False)
    add("selfplay_positions_per_second", "Self-play positions/s", 3, "elapsed_seconds", True)
    add("games", "Completed games", 4, secondary_y=False)
    add("samples", "Replay positions", 4, secondary_y=True)
    for row in (1, 2, 4):
        figure.update_xaxes(title_text="Training step", row=row, col=1)
    figure.update_xaxes(title_text="Elapsed seconds", row=3, col=1)
    figure.update_yaxes(title_text="Loss", row=1, col=1)
    figure.update_yaxes(title_text="Learning rate", type="log", row=2, col=1)
    figure.update_yaxes(title_text="Updates/s", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Positions/s", row=3, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Games", row=4, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Positions", row=4, col=1, secondary_y=True)
    figure.update_layout(
        title="Training metrics", template="plotly_white", height=1250,
        hovermode="x unified", margin={"t": 100, "r": 160},
        legend={"groupclick": "toggleitem"},
    )
    if not any(r.get("loss") is not None for r in records):
        figure.add_annotation(text="No training updates recorded", x=0.5, y=1.04,
                              xref="paper", yref="paper", showarrow=False)
    return figure


def build_report(source: str | Path, output: str | Path) -> Path:
    figure = build_figure(read_records(source))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True, auto_open=False,
                      config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild an offline training metrics report")
    parser.add_argument("input", type=Path, help="saved metrics.jsonl")
    parser.add_argument("--output", type=Path, help="default: report.html beside the input")
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(args.input, args.output or args.input.with_name("report.html"))
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Training report: {report}", flush=True)
    if args.open_report:
        webbrowser.open(report.as_uri())


if __name__ == "__main__":
    main()
