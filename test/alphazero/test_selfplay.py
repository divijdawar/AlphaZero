from queue import Empty, Full
from threading import Event
from unittest.mock import Mock, call

import numpy as np
import pytest

from alphazero import selfplay
from alphazero.config import Config


@pytest.fixture
def worker_io():
    return Mock(), Mock(), Mock(), Event()


def test_worker_request_reply_cycle(monkeypatch, worker_io):
    requests, replies, completed, stop = worker_io
    cfg = Config()
    planes = np.arange(119 * 64, dtype=np.float32).reshape(1, 119, 8, 8)[:, :, :, ::-1]
    prediction = (np.zeros((1, cfg.num_actions), np.float32), np.zeros(1, np.float32))
    samples = [selfplay.Sample(planes[0], prediction[0][0], 0.0)]
    replies.get.side_effect = [
        reply for version in (7, 11)
        for reply in (("acquire", version), ("predict", prediction), ("release", None))
    ]
    trace = Mock()
    trace.attach_mock(requests, "requests")
    trace.attach_mock(completed, "completed")

    def play(predict, config, rng):
        assert config is cfg and isinstance(rng, np.random.Generator)
        assert predict(planes) is prediction
        return samples

    def publish(message, timeout):
        if message[1] == 1:
            stop.set()

    monkeypatch.setattr(selfplay, "play_game", play)
    completed.put.side_effect = publish
    selfplay.selfplay_worker(2, cfg, np.random.SeedSequence(0), *worker_io)

    assert [c[0] for c in trace.mock_calls] == (
        ["requests.put"] * 3 + ["completed.put"]
    ) * 2
    for game_id, version in enumerate((7, 11)):
        acquire, predict, release = requests.put.call_args_list[3 * game_id:3 * game_id + 3]
        assert acquire.args[0] == ("acquire", 2, game_id)
        kind, actor_id, (network_version, batch) = predict.args[0]
        assert (kind, actor_id, network_version) == ("predict", 2, version)
        assert batch.flags.c_contiguous
        np.testing.assert_array_equal(batch, planes)
        assert release.args[0] == ("release", 2, version)
        message = completed.put.call_args_list[game_id].args[0]
        assert message[:3] == (2, game_id, version)
        assert message[3] is samples


def test_worker_retries_queues_and_stops(monkeypatch, worker_io):
    requests, replies, completed, stop = worker_io
    requests.put.side_effect = [Full, None]

    def stop_waiting(timeout):
        if replies.get.call_count == 2:
            stop.set()
        raise Empty

    replies.get.side_effect = stop_waiting
    play = Mock()
    monkeypatch.setattr(selfplay, "play_game", play)
    selfplay.selfplay_worker(2, Config(), np.random.SeedSequence(0), *worker_io)

    assert requests.put.call_args_list == [call(("acquire", 2, 0), timeout=0.1)] * 2
    assert replies.get.call_count == 2
    play.assert_not_called()
    completed.put.assert_not_called()


def test_worker_rejects_bad_replies(monkeypatch, worker_io):
    _, replies, completed, _ = worker_io
    play = Mock()
    monkeypatch.setattr(selfplay, "play_game", play)
    for reply, message in (
        (("error", "unknown network version"), "unknown network version"),
        (("predict", None), "Unexpected reply: predict"),
    ):
        replies.get.return_value = reply
        with pytest.raises(RuntimeError, match=message):
            selfplay.selfplay_worker(2, Config(), np.random.SeedSequence(0), *worker_io)
    play.assert_not_called()
    completed.put.assert_not_called()


def test_start_worker_wires_actors(monkeypatch):
    cfg = Config(num_workers=2)
    ctx = Mock()
    queues = [Mock() for _ in range(4)]
    processes = [Mock(), Mock()]
    ctx.Queue.side_effect = queues
    ctx.Process.side_effect = processes
    get_context = Mock(return_value=ctx)
    monkeypatch.setattr(selfplay.mp, "get_context", get_context)

    actors = selfplay.start_worker(cfg, seed=17)

    get_context.assert_called_once_with("spawn")
    ctx.Event.assert_called_once_with()
    assert ctx.Queue.call_args_list == [call(maxsize=n) for n in (2, 1, 1, 2)]
    assert actors.processes == processes
    assert actors.requests is queues[0] and actors.completed_games is queues[3]
    assert actors.replies == queues[1:3] and actors.stop is ctx.Event.return_value
    seeds = np.random.SeedSequence(17).spawn(2)
    for i, creation in enumerate(ctx.Process.call_args_list):
        assert creation.kwargs["target"] is selfplay.selfplay_worker
        assert creation.kwargs["name"] == f"selfplay-{i}"
        actor_id, config, seed, requests, reply, completed, stop = creation.kwargs["args"]
        assert actor_id == i and config is cfg
        assert requests is actors.requests and reply is actors.replies[i]
        assert completed is actors.completed_games and stop is actors.stop
        np.testing.assert_array_equal(seed.generate_state(4), seeds[i].generate_state(4))
        processes[i].start.assert_called_once_with()


def test_stop_workers_drains_and_cleans_up(monkeypatch):
    alive = [True, True, True]
    processes = [Mock() for _ in alive]
    for i, process in enumerate(processes):
        process.is_alive.side_effect = lambda i=i: alive[i]
    processes[0].join.side_effect = lambda **_: alive.__setitem__(0, False)
    processes[1].terminate.side_effect = lambda: alive.__setitem__(1, False)
    processes[2].kill.side_effect = lambda: alive.__setitem__(2, False)
    queues = [Mock() for _ in range(5)]
    for queue in queues:
        queue.get_nowait.side_effect = [object(), Empty]
    actors = selfplay.WorkerGroup(processes, queues[0], queues[2:], queues[1], Event())
    monkeypatch.setattr(selfplay, "monotonic", Mock(side_effect=[0.0, 0.0, 1.0]))
    monkeypatch.setattr(selfplay, "sleep", Mock())

    selfplay.stop_workers(actors, timeout=1.0)

    assert actors.stop.is_set() and alive == [False, False, False]
    processes[0].terminate.assert_not_called()
    processes[0].kill.assert_not_called()
    processes[1].terminate.assert_called_once_with()
    processes[1].kill.assert_not_called()
    processes[2].terminate.assert_called_once_with()
    processes[2].kill.assert_called_once_with()
    for process in processes[:2]:
        assert process.join.call_args_list == [call(timeout=0), call(timeout=1)]
    assert processes[2].join.call_args_list == [call(timeout=0), call(timeout=1), call()]
    for queue in queues:
        assert queue.get_nowait.call_count == 2
        queue.cancel_join_thread.assert_called_once_with()
        queue.close.assert_called_once_with()
