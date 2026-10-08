import numpy as np
import pytest

from alphazero.config import Config
from alphazero.selfplay import play_game
from alphazero.train import ReplayBuffer, lr_for_step, make_predict, make_train_step, resolve_lr_schedule, run_pipeline, train_step
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

    model_tensors = get_parameters(net)
    params = [p for p in model_tensors if p.is_param]
    opt = optim.SGD(model_tensors, lr=0.1, momentum=0.9)
    loss = train_step(net, opt, params, buf.sample(4, np.random.default_rng(1)), cfg)
    assert np.isfinite(loss)

def test_optimizer_realizes_batchnorm_updates(monkeypatch):
    from tinygrad import Tensor
    from tinygrad.nn import optim
    from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict

    cfg = tiny_config(batch_size=2)
    net = NeuralNet(cfg)
    reference = NeuralNet(cfg)
    initial = {name: tensor.clone().realize() for name, tensor in get_state_dict(net).items()}
    load_state_dict(reference, initial, strict=True, verbose=False)
    model_tensors = get_parameters(net)
    params = [p for p in model_tensors if p.is_param]
    opt = optim.SGD(model_tensors, lr=0.01, momentum=0.9, fused=False)
    reference_tensors = get_parameters(reference)
    reference_params = [p for p in reference_tensors if p.is_param]
    reference_buffers = [p for p in reference_tensors if not p.is_param]

    reference_opt = optim.SGD(reference_params, lr=0.01, momentum=0.9, fused=False)
    reference_step = reference_opt.step

    def explicit_buffer_step():
        # Compute statistics from the forward pass before weights change.
        Tensor.realize(*reference_buffers)
        reference_step()

    monkeypatch.setattr(reference_opt, "step", explicit_buffer_step)
    planes = np.random.default_rng(0).standard_normal((2, 119, 8, 8)).astype(np.float32)
    pi = np.zeros((2, cfg.num_actions), np.float32)
    pi[:, 0] = 1
    batch = planes, pi, np.zeros((2, 1), np.float32)
    for _ in range(10):
        train_step(net, opt, params, batch, cfg)
        train_step(reference, reference_opt, reference_params, batch, cfg)

    reference_state = get_state_dict(reference)
    # Read candidate statistics only at the end: reading them each step would
    # realize them and conceal the deferred-update bug.
    for name, tensor in get_state_dict(net).items():
        if not tensor.is_param:
            np.testing.assert_allclose(tensor.numpy(), reference_state[name].numpy(),
                                       rtol=1e-5, atol=1e-6, err_msg=name)
            if name.endswith("num_batches_tracked"):
                assert tensor.item() == 10

def test_jit_predictions_reuse_bounded_captures_and_own_outputs(monkeypatch):
    import alphazero.train as training
    from tinygrad import Tensor, TinyJit
    from tinygrad.nn.state import get_state_dict

    Tensor.manual_seed(5)
    cfg = tiny_config(num_filters=4, value_hidden=4)
    net = NeuralNet(cfg)
    before = {n: t.numpy().copy() for n, t in get_state_dict(net).items() if not t.is_param}
    captures = []

    def capture(function):
        compiled = TinyJit(function)
        captures.append(compiled)
        return compiled

    monkeypatch.setattr(training, "TinyJit", capture)
    predict, reference = make_predict(net, cfg), make_predict(net, cfg, jit=False)
    rng = np.random.default_rng(3)
    previous = []
    for size in (1, 2, 3, 5, 9, 19) * 3:
        planes = rng.normal(size=(size, 119, 8, 8)).astype(np.float32)[:, :, :, ::-1]
        actual, expected = predict(planes), reference(planes)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-6)
            assert a.dtype == np.float32
        previous.append((actual, tuple(a.copy() for a in actual)))
    assert len(captures) == 5 and all(c.cnt >= 3 for c in captures)
    for actual, saved in previous:
        for a, e in zip(actual, saved):
            np.testing.assert_array_equal(a, e)
    for name, value in before.items():
        np.testing.assert_array_equal(get_state_dict(net)[name].numpy(), value)

@pytest.mark.parametrize("fused,constant_lr", [(False, False), (True, False), (False, True)])
def test_jit_training_matches_eager_through_lr_change(monkeypatch, fused, constant_lr):
    from tinygrad import Tensor
    from tinygrad.nn import optim
    from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict

    monkeypatch.setenv("CONST_LR", str(int(constant_lr)))
    Tensor.manual_seed(7)
    cfg = tiny_config(num_filters=4, value_hidden=4, batch_size=2)
    net, reference = NeuralNet(cfg), NeuralNet(cfg)
    initial = {n: t.clone().realize() for n, t in get_state_dict(net).items()}
    load_state_dict(reference, initial, strict=True, verbose=False)
    tensors, reference_tensors = get_parameters(net), get_parameters(reference)
    params = [p for p in tensors if p.is_param]
    reference_params = [p for p in reference_tensors if p.is_param]
    opt = optim.SGD(tensors, lr=.01, momentum=.9, fused=fused)
    reference_opt = optim.SGD(reference_tensors, lr=.01, momentum=.9, fused=fused)
    compiled = make_train_step(net, opt, params, cfg)
    make_train_step(reference, reference_opt, reference_params, cfg, jit=False)
    gradients, reference_gradients = [], []

    def observe(optimizer, output):
        original = optimizer.step

        def step():
            # Captured copies expose this execution's gradients, rather than
            # inspecting Python .grad references left by the capture call.
            output[:] = [p.grad.clone().realize() for p in optimizer.params]
            original()

        monkeypatch.setattr(optimizer, "step", step)

    observe(opt, gradients)
    observe(reference_opt, reference_gradients)
    lr_buffer = opt.lr.uop
    rng = np.random.default_rng(8)
    for i in range(64):
        if i == 32:
            opt.lr.assign(.001).realize()
            reference_opt.lr.assign(.001).realize()
        planes = rng.normal(size=(2, 119, 8, 8)).astype(np.float32)
        pi = np.zeros((2, cfg.num_actions), np.float32)
        pi[np.arange(2), rng.integers(cfg.num_actions, size=2)] = 1
        batch = planes, pi, rng.uniform(-1, 1, (2, 1)).astype(np.float32)
        actual, expected = {}, {}
        train_step(net, opt, params, batch, cfg, metrics=actual, tensor_step=compiled)
        train_step(reference, reference_opt, reference_params, batch, cfg, metrics=expected)
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-5, atol=1e-6)
        for a, e in zip(gradients, reference_gradients):
            np.testing.assert_allclose(a.numpy(), e.numpy(), rtol=1e-5, atol=1e-6)
    assert compiled.cnt == 64 and opt.lr.uop is lr_buffer
    # Delay model and BN reads until all updates finish.
    expected_state = get_state_dict(reference)
    for name, tensor in get_state_dict(net).items():
        np.testing.assert_allclose(tensor.numpy(), expected_state[name].numpy(), rtol=1e-5, atol=1e-6, err_msg=name)
        if name.endswith("num_batches_tracked"):
            assert tensor.item() == 64
    for a, e in zip(opt.b, reference_opt.b):
        np.testing.assert_allclose(a.numpy(), e.numpy(), rtol=1e-5, atol=1e-6)

def test_persistent_predictors_route_versions_and_retire_together():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from alphazero.train import retire_snapshots, run_inference

    actors = SimpleNamespace(replies=[Mock(), Mock(), Mock()])
    predictors = {v: Mock(side_effect=lambda x, v=v: (x[:, :1, 0, 0] + v, x[:, 0, 0, 0] - v)) for v in (0, 7, 11)}
    batches = {7: [(2, np.full((2, 1, 1, 1), 3, np.float32)), (0, np.full((1, 1, 1, 1), 5, np.float32))],
               0: [(1, np.full((1, 1, 1, 1), 2, np.float32))]}
    for _ in range(2):
        run_inference(actors, predictors, batches, [0.] * 3)
    for actor, policy, value in ((0, [12], [-2]), (1, [2], [2]), (2, [10, 10], [-4, -4])):
        kind, (p, v) = actors.replies[actor].put_nowait.call_args.args[0]
        assert kind == "predict"
        np.testing.assert_array_equal(p[:, 0], policy)
        np.testing.assert_array_equal(v, value)
    assert predictors[0].call_count == predictors[7].call_count == 2
    snapshots = dict.fromkeys(predictors)
    retire_snapshots(snapshots, {2: 7}, 11, predictors)
    assert snapshots.keys() == predictors.keys() == {7, 11}

def test_lr_schedule():
    cfg = tiny_config(steps=700_000, lr=(0.2, 0.02, 0.002, 0.0002),
                      lr_milestones=(100_000, 300_000, 500_000))
    assert lr_for_step(0, cfg) == 0.2
    assert lr_for_step(100_000, cfg) == 0.02
    assert lr_for_step(300_000, cfg) == 0.002
    assert lr_for_step(500_000, cfg) == 0.0002

@pytest.mark.parametrize("steps,milestones", [
    (100_000, (14_286, 42_857, 71_429)),
    (70_000, (10_000, 30_000, 50_000)),
    (4, (1, 2, 3)),
])
def test_default_lr_schedule_scales_and_decays_at_boundaries(steps, milestones):
    cfg = resolve_lr_schedule(tiny_config(steps=steps))
    assert cfg.lr == (0.01, 0.001, 0.0001, 0.00001)
    assert cfg.lr_milestones == milestones
    for i, milestone in enumerate(milestones):
        assert lr_for_step(milestone - 1, cfg) == cfg.lr[i]
        assert lr_for_step(milestone, cfg) == cfg.lr[i + 1]
    assert lr_for_step(steps - 1, cfg) == cfg.lr[-1]

def test_short_and_single_rate_runs_use_constant_lr():
    for cfg in (tiny_config(steps=3), tiny_config(lr=(0.03,))):
        resolved = resolve_lr_schedule(cfg)
        assert resolved.lr_milestones == ()
        assert lr_for_step(0, resolved) == lr_for_step(cfg.steps - 1, resolved) == cfg.lr[0]

@pytest.mark.parametrize("overrides", [
    {"lr": ()}, {"lr": (float("nan"),)}, {"lr": (0.0,)},
    {"lr": (0.01, 0.001)}, {"steps": 0},
    {"lr_milestones": (1, 2)}, {"lr_milestones": (1, 1, 2)},
    {"lr_milestones": (2, 1, 3)}, {"lr_milestones": (0, 1, 2)},
    {"lr_milestones": (1, 2, 100_000)}, {"lr_milestones": (1, 2, 3.5)},
])
def test_invalid_lr_schedule_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_lr_schedule(tiny_config(**overrides))

@pytest.mark.parametrize("momentum", [0.01, 0.1, 0.99])
def test_configured_bn_momentum_reaches_every_layer(momentum):
    net = NeuralNet(tiny_config(bn_momentum=momentum))
    norms = [net.stem_bn, net.pol_bn, net.val_bn]
    norms.extend(bn for block in net.blocks for bn in (block.bn1, block.bn2))
    assert all(bn.momentum == momentum for bn in norms)
    assert Config().bn_momentum == 0.1

def test_pipeline_uses_and_saves_resolved_lr_schedule(tmp_path):
    import json
    from tinygrad.nn.state import safe_load_metadata

    cfg = tiny_config(steps=4, batch_size=2, num_workers=1, checkpoint=0,
                      max_plies=2, num_simulations=1)
    metrics = []
    net = run_pipeline(cfg, min_replay_size=1, seed=0, checkpoint_dir=str(tmp_path),
                       log_every=1, log=metrics.append)
    assert [m["learning_rate"] for m in metrics] == list(cfg.lr)
    assert net.cfg.lr_milestones == (1, 2, 3)
    _, _, header = safe_load_metadata(tmp_path / "final.safetensors")
    saved_config = json.loads(header["__metadata__"]["config"])
    assert saved_config["lr_milestones"] == [1, 2, 3]

@pytest.mark.parametrize("jit", [False, True])
def test_pipeline_smoke(tmp_path, jit):
    cfg = tiny_config(steps=2, num_workers=1, checkpoint=0)
    net = run_pipeline(
        cfg, min_replay_size=1,
        seed=0, checkpoint_dir=str(tmp_path), log=None, jit=jit,
    )
    assert (tmp_path / "final.safetensors").exists()
    # the net is usable after training
    from tinygrad import Tensor

    logits, _ = net(Tensor(np.zeros((1, 119, 8, 8), np.float32)))
    assert tuple(logits.shape) == (1, cfg.num_actions)

def test_pipeline_steps_checkpoints_and_reload(tmp_path):
    from tinygrad import Context, Tensor
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load

    cfg = tiny_config(steps=4, num_workers=1, checkpoint=2)
    metrics = []
    net = run_pipeline(
        cfg, min_replay_size=1, log_every=1,
        seed=0, checkpoint_dir=str(tmp_path), log=metrics.append,
        publish_every=1, max_snapshots=2,
    )

    updates = [m for m in metrics if m["window_updates"]]
    assert [m["step"] for m in updates] == [1, 2, 3, 4]
    assert [m["games"] for m in updates] == sorted(m["games"] for m in updates)
    for m in updates:
        assert m["loss"] is not None and np.isfinite(m["loss"])
        assert 0 < m["samples"] <= min(cfg.buffer_size, m["games"] * cfg.max_plies)
        assert m["retained_snapshots"] <= 2

    assert sorted(p.name for p in tmp_path.glob("*.safetensors")) == [
        "final.safetensors", "step-2.safetensors", "step-4.safetensors",
    ]

    saved = {
        k.removeprefix("model."): v
        for k, v in safe_load(str(tmp_path / "final.safetensors")).items()
        if k.startswith("model.")
    }
    earlier = safe_load(str(tmp_path / "step-2.safetensors"))
    assert not np.array_equal(earlier["model.pol2.weight"].numpy(), saved["pol2.weight"].numpy())
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

@pytest.mark.parametrize("fused", [None, False, True])
def test_load_checkpoint_restores_training_state(tmp_path, fused):
    import json
    from tinygrad.nn import optim
    from tinygrad.nn.state import get_parameters, get_state_dict, safe_load_metadata
    from alphazero.train import _save_checkpoint, load_checkpoint

    cfg = tiny_config(batch_size=2)
    net = NeuralNet(cfg)
    tensors = get_parameters(net)
    params = [p for p in tensors if p.is_param]
    fusion = {} if fused is None else {"fused": fused}
    opt = optim.SGD(tensors, lr=0.03, momentum=0.8, nesterov=True, **fusion)
    planes = np.random.default_rng(0).standard_normal((2, 119, 8, 8)).astype(np.float32)
    pi = np.zeros((2, cfg.num_actions), dtype=np.float32)
    pi[:, 0] = 1
    batch = planes, pi, np.zeros((2, 1), np.float32)
    train_step(net, opt, params, batch, cfg)

    path = tmp_path / "checkpoint.safetensors"
    _save_checkpoint(net, path, opt=opt, cfg=cfg, global_step=1,
                     iteration=1, step_in_iteration=1)
    _, _, header = safe_load_metadata(path)
    assert json.loads(header["__metadata__"]["optimizer"])["options"]["fused"] is bool(opt.fused)
    loaded_net = NeuralNet(cfg)
    loaded_tensors = get_parameters(loaded_net)
    loaded_opt = optim.SGD(loaded_tensors, lr=0.1, momentum=0.9, **fusion)
    loaded_cfg, training = load_checkpoint(loaded_net, path, opt=loaded_opt)

    assert loaded_cfg == cfg
    assert training == {"global_step": 1, "iteration": 1, "step_in_iteration": 1}
    assert type(loaded_opt) is type(opt)
    assert (loaded_opt.momentum, loaded_opt.nesterov) == (0.8, True)
    assert loaded_opt.lr.item() == pytest.approx(0.03)
    expected = get_state_dict({"model": net, "optimizer": opt})
    actual = get_state_dict({"model": loaded_net, "optimizer": loaded_opt})
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name].numpy(), expected[name].numpy())

    train_step(net, opt, params, batch, cfg)
    train_step(loaded_net, loaded_opt, [p for p in loaded_tensors if p.is_param], batch, cfg)
    expected = get_state_dict({"model": net, "optimizer": opt})
    actual = get_state_dict({"model": loaded_net, "optimizer": loaded_opt})
    for name in expected:
        np.testing.assert_allclose(actual[name].numpy(), expected[name].numpy(), rtol=1e-5, atol=1e-6, err_msg=name)
