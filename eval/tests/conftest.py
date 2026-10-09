import os
from eval.runtime import configure

configure()
os.environ["DEV"] = "CPU:CLANG"

import pytest
from dataclasses import asdict
import json
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict, safe_save
from alphazero.config import Config
from alphazero.nn import NeuralNet


@pytest.fixture
def checkpoint(tmp_path):
    cfg = Config(num_block=2, num_filters=8, value_hidden=16)
    Tensor.manual_seed(991)
    net = NeuralNet(cfg)
    path = tmp_path / "step-10.safetensors"
    net.stem_bn.running_mean.assign(Tensor.full(net.stem_bn.running_mean.shape, 0.2)).realize()
    state = get_state_dict({"model": net})
    state["optimizer.test"] = Tensor([7.0])
    safe_save(state, str(path), metadata={"format_version": "1", "config": json.dumps(asdict(cfg)),
              "training": json.dumps({"global_step": 10})})
    return path, net, cfg

