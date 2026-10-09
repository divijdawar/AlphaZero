from __future__ import annotations

import csv
from datetime import datetime, UTC
import html
import json
from pathlib import Path
import re
import subprocess

from .runtime import ROOT, digest, output_path, write_json


def fit(run: Path, *, bayeselo: str | None = None, anchor_elo: float | None = None,
        anchor_source: str = "", prior: float = 2.0) -> dict:
    run = output_path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((run / "games").glob("*.json"))]
    records = [r for r in records if r["result"] != "*"]
    if not records:
        raise ValueError("no completed games to rate")
    if anchor_elo is not None and not anchor_source.strip():
        raise ValueError("--anchor-elo requires --anchor-source with the published scale/conditions")
    anchor = manifest["baseline_name"]
    graph = {}
    stats = {}
    for r in records:
        for name, opponent, white in ((r["white"], r["black"], True), (r["black"], r["white"], False)):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("invalid PGN player identifier")
            graph.setdefault(name, set()).add(opponent)
            s = stats.setdefault(name, {"wins": 0, "draws": 0, "losses": 0})
            outcome = "draws" if r["result"] == "1/2-1/2" else (
                "wins" if (r["result"] == "1-0") == white else "losses")
            s[outcome] += 1
    reached, todo = set(), [anchor]
    while todo:
        node = todo.pop()
        if node not in reached:
            reached.add(node)
            todo.extend(graph.get(node, set()) - reached)
    if set(graph) - reached or anchor not in graph:
        raise ValueError("rated games must form a connected graph with the Stockfish baseline")
    (run / "games.pgn").write_text("\n\n".join(r["pgn"] for r in records) + "\n")
    executable = str(Path(bayeselo or ROOT / "vendor/bin/bayeselo").resolve(strict=True))
    commands = "\n".join([
        "prompt off", "readpgn games.pgn", "elo", f"prior {prior}", "advantage 32.8", "drawelo 97.3",
        "confidence 0.95", "mm 0 0", "scale 1", "covariance",
        f"offset {anchor_elo if anchor_elo is not None else 0} {anchor}", "ratings", "x", "x", "",
    ])
    result = subprocess.run([executable], input=commands, text=True, capture_output=True,
                            cwd=run, timeout=120, check=True)
    (run / "bayeselo-commands.txt").write_text(commands)
    (run / "bayeselo-output.txt").write_text(result.stdout + result.stderr)
    pattern = re.compile(r"^\s*\d+\s+(\S+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+%", re.MULTILINE)
    rows = []
    checkpoints = {c["name"]: c for c in manifest["checkpoints"]}
    for name, elo, plus, minus, games in pattern.findall(result.stdout):
        s = stats[name]
        n = sum(s.values())
        if int(games) != n:
            raise RuntimeError("BayesElo game counts differ from saved match records")
        rows.append({"name": name, "step": checkpoints.get(name, {}).get("step"),
                     "checkpoint": checkpoints.get(name, {}).get("path"), "elo": int(elo),
                     "delta_vs_stockfish": int(elo) - (anchor_elo or 0),
                     "lower95": int(elo) - int(minus), "upper95": int(elo) + int(plus),
                     "games": n, **s, "score": (s["wins"] + 0.5 * s["draws"]) / n,
                     "provisional": n < 100 or s["wins"] == n or s["losses"] == n})
    if {r["name"] for r in rows} != set(stats):
        raise RuntimeError("could not parse a complete BayesElo rating table; inspect bayeselo-output.txt")
    baseline = next(r["elo"] for r in rows if r["name"] == anchor)
    for row in rows:
        row["delta_vs_stockfish"] = row["elo"] - baseline
    report = {"created_utc": datetime.now(UTC).isoformat(), "completed_games": len(records),
              "profile": manifest["settings"], "anchor": {"name": anchor, "elo": anchor_elo,
              "source": anchor_source, "scale": "published anchor" if anchor_elo is not None else "Stockfish baseline = 0; relative Elo"},
              "estimator": {"tool": "BayesElo", "binary_sha256": digest(executable), "prior_virtual_draws": prior,
              "advantage": 32.8, "drawelo": 97.3, "scale": 1, "confidence": 0.95,
              "uncertainty": "BayesElo covariance Hessian approximation, conditional on benchmark and anchor"},
              "ratings": sorted(rows, key=lambda r: r["elo"], reverse=True)}
    write_json(run / "ratings.json", report)
    versions = run / "reports"
    versions.mkdir(exist_ok=True)
    write_json(versions / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + ".json"), report)
    with (run / "ratings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(report["ratings"])
    render_report(run, report)
    return report


def render_report(run: Path, report: dict):
    rows = report["ratings"]
    points = sorted([r for r in rows if r["step"] is not None], key=lambda r: r["step"])
    svg = ""
    if points:
        maxstep = max(1, max(r["step"] for r in points))
        low, high = min(r["lower95"] for r in points), max(r["upper95"] for r in points)
        spread = max(high - low, 1)
        coords = [(60 + 800 * r["step"] / maxstep, 310 - 260 * (r["elo"] - low) / spread) for r in points]
        svg = '<svg viewBox="0 0 920 370" role="img" aria-label="Checkpoint Elo versus training step">'
        svg += '<path d="M60 35 V310 H870" fill="none" stroke="#718096"/>'
        svg += '<text x="15" y="25">Elo</text><text x="700" y="355">Training step</text>'
        svg += '<polyline fill="none" stroke="#38bdf8" stroke-width="3" points="' + ' '.join(f'{x},{y}' for x, y in coords) + '"/>'
        for r, (x, y) in zip(points, coords):
            ymin = 310 - 260 * (r["upper95"] - low) / spread
            ymax = 310 - 260 * (r["lower95"] - low) / spread
            svg += f'<path d="M{x} {ymin} V{ymax}" stroke="#94a3b8"/><circle cx="{x}" cy="{y}" r="5" fill="#38bdf8"><title>step {r["step"]}: {r["elo"]} [{r["lower95"]}, {r["upper95"]}]</title></circle>'
        svg += f'<text x="65" y="335">0</text><text x="830" y="335">{maxstep}</text><text x="12" y="310">{low}</text><text x="12" y="50">{high}</text></svg>'
    cells = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        r["name"], r["step"] if r["step"] is not None else "baseline", r["elo"],
        f'{r["lower95"]} … {r["upper95"]}', r["wins"], r["draws"], r["losses"],
        "provisional" if r["provisional"] else "rated")) + '</tr>' for r in rows)
    (run / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Checkpoint Elo</title>'
        '<style>body{background:#0f172a;color:#e2e8f0;font:16px system-ui;max-width:1050px;margin:40px auto;padding:20px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #334155}svg{width:100%;fill:#e2e8f0}pre{white-space:pre-wrap}</style>'
        '<h1>Checkpoint Elo</h1><p>' + html.escape(report["anchor"]["scale"]) + '</p>' + svg +
        '<table><tr><th>Player</th><th>Step</th><th>Elo</th><th>95% interval</th><th>W</th><th>D</th><th>L</th><th>Status</th></tr>' + cells + '</table>'
        '<p>One-sided or small samples are provisional. Intervals are approximate and conditional on the recorded benchmark.</p><h2>Match settings</h2><pre>' +
        html.escape(json.dumps(report["profile"], indent=2)) + '</pre>')

