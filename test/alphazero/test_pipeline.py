import numpy as np

from alphazero.config import Config
from alphazero.selfplay import play_game
from alphazero.train import ReplayBuffer, lr_for_step, make_predict, run_pipeline, train_step
from alphazero.nn import NeuralNet

def tiny_config(**over):
    base = {
        "num_block": 2, "num_filters": 8, "policy_head_filters": 8,
        "value_head_filters": 8, "value_hidden": 8,
        "num_simulations": 2, "max_plies": 4, "temperature_moves": 100,
        "batch_size": 4, "buffer_size": 1000, "checkpoint": 1000,
    }
    base.update(over)
    return Config(**base)

def test_play_game_produces_valid_samples():
    cfg = tiny_config()
    net = NeuralNet(cfg)
    samples = play_game(make_predict(net, cfg), cfg, np.random.default_rng(0))

    assert 0 < len(samples) <= cfg.max_plies
    for s in samples:
        assert s.planes.shape == (119, 8, 8) and s.planes.dtype == np.float32
        assert s.pi.shape == (cfg.num_actions,)
        assert abs(s.pi.sum() - 1.0) < 1e-5          # normalized visit distribution
        assert s.value in (-1.0, 0.0, 1.0)           # outcome from mover's view

def test_replay_buffer_sample_shapes():
    cfg = tiny_config()
    net = NeuralNet(cfg)
    samples = play_game(make_predict(net, cfg), cfg, np.random.default_rng(0))
    buf = ReplayBuffer(100)
    buf.extend(samples)
    planes, pi, z = buf.sample(4, np.random.default_rng(1))
    assert planes.shape == (4, 119, 8, 8)
    assert pi.shape == (4, cfg.num_actions)
    assert z.shape == (4, 1)

def test_train_step_is_finite():
    cfg = tiny_config()
    net = NeuralNet(cfg)
    samples = play_game(make_predict(net, cfg), cfg, np.random.default_rng(0))
    buf = ReplayBuffer(100)
    buf.extend(samples)

    from tinygrad.nn import optim
    from tinygrad.nn.state import get_parameters

    params = [p for p in get_parameters(net) if p.is_param]
    opt = optim.SGD(params, lr=0.1, momentum=0.9)
    loss = train_step(net, opt, params, buf.sample(4, np.random.default_rng(1)), cfg)
    assert np.isfinite(loss)

def test_lr_schedule():
    cfg = tiny_config(lr=(0.2, 0.02, 0.002, 0.0002))
    assert lr_for_step(0, cfg) == 0.2
    assert lr_for_step(100_000, cfg) == 0.02
    assert lr_for_step(300_000, cfg) == 0.002
    assert lr_for_step(500_000, cfg) == 0.0002

def test_pipeline_smoke(tmp_path):
    cfg = tiny_config()
    net = run_pipeline(
        cfg, iters=2, games_per_iter=1, train_steps_per_iter=2,
        seed=0, checkpoint_dir=str(tmp_path), log=None,
    )
    assert (tmp_path / "final.safetensors").exists()
    # the net is usable after training
    from tinygrad import Tensor

    logits, _ = net(Tensor(np.zeros((1, 119, 8, 8), np.float32)))
    assert tuple(logits.shape) == (1, cfg.num_actions)

def test_pipeline_two_iters_exact_counts_and_reload(tmp_path):
    from tinygrad import Context, Tensor
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load

    cfg = tiny_config(checkpoint=2)
    metrics = []
    net = run_pipeline(
        cfg, iters=2, games_per_iter=1, train_steps_per_iter=2,
        seed=0, checkpoint_dir=str(tmp_path), log=metrics.append,
    )

    assert [m["iteration"] for m in metrics] == [1, 2]
    assert [m["games"] for m in metrics] == [1, 1]
    assert [m["step"] for m in metrics] == [2, 4]
    for m in metrics:
        assert m["loss"] is not None and np.isfinite(m["loss"])

    assert 0 < metrics[0]["samples"] <= cfg.max_plies
    assert metrics[0]["samples"] < metrics[1]["samples"] <= 2 * cfg.max_plies

    assert sorted(p.name for p in tmp_path.glob("*.safetensors")) == [
        "final.safetensors", "step-2.safetensors", "step-4.safetensors",
    ]

    saved = safe_load(str(tmp_path / "final.safetensors"))
    x = np.zeros((1, 119, 8, 8), np.float32)

    def infer(m):
        with Context(TRAINING=0):
            logits, value = m(Tensor(x))
            return logits.numpy(), value.numpy()

    fresh = NeuralNet(cfg)
    load_state_dict(fresh, saved, strict=True, verbose=False)
    after = {k: v.numpy() for k, v in get_state_dict(fresh).items()}
    assert set(after) == set(saved)
    for k, v in saved.items():
        np.testing.assert_allclose(after[k], v.numpy(), err_msg=k)

    logits_trained, value_trained = infer(net)
    logits_loaded, value_loaded = infer(fresh)
    np.testing.assert_allclose(logits_loaded, logits_trained, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(value_loaded, value_trained, rtol=1e-4, atol=1e-5)
