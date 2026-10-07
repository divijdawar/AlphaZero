import json

import numpy as np
import pytest

from alphazero.metrics import (
    LOSS_KEYS, MetricWindow, RunMetrics, build_figure, build_report,
    format_metrics, read_records,
)
from alphazero.train import run_pipeline, train_step
from test.alphazero.test_pipeline import tiny_config


def record(step=2, **changes):
    return {
        "step": step, "total_steps": 3, "elapsed_seconds": 10.0,
        "loss": 3.5, "policy_loss": 2.0, "value_loss": 1.0,
        "regularization_loss": 0.5, "learning_rate": 0.01,
        "training_updates_per_second": 0.2,
        "selfplay_positions_per_second": 4.0,
        "games": 2, "samples": 40,
        **changes,
    }


def test_interval_averages_and_wall_time_throughput():
    window = MetricWindow(100.0)
    window.positions += 40
    window.record_update(dict(zip(LOSS_KEYS, (3.5, 2.0, 1.0, 0.5))))
    window.record_update(dict(zip(LOSS_KEYS, (7.0, 4.0, 2.0, 1.0))))
    report = window.report(110.0)
    assert report["loss"] == 5.25
    assert report["policy_loss"] == 3.0
    assert report["value_loss"] == 1.5
    assert report["regularization_loss"] == 0.75
    assert report["training_updates_per_second"] == 0.2
    assert report["selfplay_positions_per_second"] == 4.0
    assert report["window_updates"] == 2
    assert report["window_seconds"] == 10.0


def test_warmup_has_no_loss_and_readable_terminal_output():
    report = MetricWindow(0.0).report(10.0)
    assert all(report[key] is None for key in LOSS_KEYS)
    assert report["training_updates_per_second"] == 0.0
    text = format_metrics(record(step=0, **report))
    assert "warming replay" in text
    assert "loss=—" in text
    assert "updates/s=0.00" in text


def test_loss_metrics_preserve_float_return_and_training_state():
    from tinygrad.nn import optim
    from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
    from alphazero.nn import NeuralNet

    cfg = tiny_config(batch_size=2)
    net, reference = NeuralNet(cfg), NeuralNet(cfg)
    state = {name: tensor.clone().realize() for name, tensor in get_state_dict(net).items()}
    load_state_dict(reference, state, strict=True, verbose=False)
    tensors, reference_tensors = get_parameters(net), get_parameters(reference)
    opt = optim.SGD(tensors, lr=0.01, momentum=0.9)
    reference_opt = optim.SGD(reference_tensors, lr=0.01, momentum=0.9)
    planes = np.random.default_rng(42).standard_normal((2, 119, 8, 8)).astype(np.float32)
    pi = np.zeros((2, cfg.num_actions), np.float32)
    pi[:, 0] = 1
    batch = planes, pi, np.zeros((2, 1), np.float32)
    metrics = {}
    for _ in range(3):
        loss = train_step(net, opt, [p for p in tensors if p.is_param], batch, cfg, metrics=metrics)
        train_step(reference, reference_opt, [p for p in reference_tensors if p.is_param], batch, cfg)
    assert isinstance(loss, float)
    assert loss == metrics["loss"]
    assert loss == pytest.approx(sum(metrics[key] for key in LOSS_KEYS[1:]), rel=1e-6)
    actual_reference = get_state_dict(reference)
    for name, tensor in get_state_dict(net).items():
        np.testing.assert_allclose(tensor.numpy(), actual_reference[name].numpy(), rtol=1e-5, atol=1e-6)


def test_unique_runs_flush_records_and_persist_configuration(tmp_path):
    pytest.importorskip("plotly")
    first = RunMetrics(tmp_path, {"steps": 3}, {"log_every": 2})
    second = RunMetrics(tmp_path, {"steps": 3}, {"log_every": 2})
    assert first.directory != second.directory
    assert json.loads((first.directory / "config.json").read_text()) == {
        "config": {"steps": 3}, "settings": {"log_every": 2},
    }
    first.write(record())
    # Read while the run object is still alive, without any finish/close call.
    assert read_records(first.path) == [record()]


def test_interrupted_final_json_write_keeps_earlier_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(record()) + '\n{"step":')
    assert read_records(path) == [record()]
    path.write_text(json.dumps(record()) + '\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        read_records(path)


def test_report_series_axes_warmup_and_embedded_javascript(tmp_path):
    pytest.importorskip("plotly")
    warmup = record(step=0, **dict.fromkeys(LOSS_KEYS))
    records = [warmup, record()]
    figure = build_figure(records)
    traces = {trace.name: trace for trace in figure.data}
    assert list(traces["Total loss"].x) == [2]
    assert list(traces["Total loss"].y) == [3.5]
    assert list(traces["Updates/s"].x) == [10.0, 10.0]
    assert traces["Updates/s"].xaxis != traces["Total loss"].xaxis
    assert figure.layout.xaxis3.title.text == "Elapsed seconds"
    assert figure.layout.xaxis.title.text == "Training step"
    source = tmp_path / "metrics.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    output = build_report(source, tmp_path / "report.html")
    html = output.read_text()
    assert "Plotly.newPlot" in html
    assert "plotly.js" in html
    assert '<script src="' not in html
    assert "Policy loss" in html and "Replay positions" in html
    assert "scrollZoom" in html


def test_empty_report_is_usable(tmp_path):
    pytest.importorskip("plotly")
    source = tmp_path / "metrics.jsonl"
    source.touch()
    assert "No training updates recorded" in build_report(source, tmp_path / "report.html").read_text()


@pytest.mark.parametrize("with_callback", [True, False])
def test_saved_pipeline_final_record_and_partial_interval(tmp_path, monkeypatch, with_callback):
    pytest.importorskip("plotly")
    import alphazero.train as training

    opened = []
    monkeypatch.setattr(training.webbrowser, "open", opened.append)
    cfg = tiny_config(steps=3, batch_size=2, num_workers=1, checkpoint=0,
                      max_plies=2, num_simulations=1)
    received = []
    run_pipeline(cfg, min_replay_size=1, checkpoint_dir=str(tmp_path / "checkpoints"),
                 metrics_dir=tmp_path / "runs", log_every=2,
                 log=received.append if with_callback else None, open_report=with_callback)
    run = next((tmp_path / "runs").iterdir())
    saved = read_records(run / "metrics.jsonl")
    assert [row["step"] for row in saved] == [2, 3]
    assert [row["window_updates"] for row in saved] == [2, 1]
    assert received == (saved if with_callback else [])
    assert opened == ([(run / "report.html").resolve().as_uri()] if with_callback else [])
    config = json.loads((run / "config.json").read_text())
    assert config["config"]["lr_milestones"] == []
    assert config["settings"]["log_every"] == 2
    assert (run / "report.html").is_file()
    for row in saved:
        assert row["loss"] == pytest.approx(sum(row[key] for key in LOSS_KEYS[1:]), rel=1e-6)
        assert row["elapsed_seconds"] > 0


def test_interruption_preserves_logged_records_for_regeneration(tmp_path):
    pytest.importorskip("plotly")
    cfg = tiny_config(steps=3, batch_size=2, num_workers=1, checkpoint=0,
                      max_plies=2, num_simulations=1)

    def interrupt(record):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(cfg, min_replay_size=1, checkpoint_dir=str(tmp_path / "checkpoints"),
                     metrics_dir=tmp_path / "runs", log_every=1, log=interrupt)
    run = next((tmp_path / "runs").iterdir())
    assert [row["step"] for row in read_records(run / "metrics.jsonl")] == [1]
    assert not (run / "report.html").exists()
    assert build_report(run / "metrics.jsonl", run / "report.html").exists()


def test_time_threshold_emits_during_warmup_and_after_final_update(tmp_path, monkeypatch):
    import alphazero.train as training

    original = training.collect_requests
    clock = [0.0]

    def collect(*args, **kwargs):
        clock[0] += 1.0
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "collect_requests", collect)
    monkeypatch.setattr(training, "monotonic", lambda: clock[0])
    cfg = tiny_config(steps=1, batch_size=2, num_workers=1, checkpoint=0,
                      max_plies=2, num_simulations=1)
    received = []
    run_pipeline(cfg, min_replay_size=1, checkpoint_dir=str(tmp_path),
                 log_every=100, log_seconds=1.0, log=received.append)
    assert received[0]["step"] == 0
    assert received[0]["loss"] is None
    assert received[-1]["step"] == 1
    assert received[-1]["loss"] is not None


def test_missing_plotly_fails_before_starting_training(tmp_path, monkeypatch):
    import builtins
    import alphazero.train as training

    original_import = builtins.__import__

    def missing_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_plotly)
    monkeypatch.setattr(training, "start_worker", lambda *a, **k: pytest.fail("training started"))
    with pytest.raises(RuntimeError, match="uv sync --extra metrics"):
        run_pipeline(tiny_config(), min_replay_size=1, metrics_dir=tmp_path / "runs",
                     checkpoint_dir=str(tmp_path / "checkpoints"))
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_log_seconds(seconds, tmp_path):
    with pytest.raises(ValueError, match="log_seconds"):
        run_pipeline(tiny_config(), min_replay_size=1, log_seconds=seconds,
                     checkpoint_dir=str(tmp_path))


def test_open_report_requires_saved_metrics(tmp_path):
    with pytest.raises(ValueError, match="open_report requires metrics_dir"):
        run_pipeline(tiny_config(), min_replay_size=1, open_report=True,
                     checkpoint_dir=str(tmp_path))
