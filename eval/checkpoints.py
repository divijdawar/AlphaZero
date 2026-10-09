from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct

from alphazero.config import Config


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    config: Config
    step: int
    identity: str

    @property
    def name(self) -> str:
        return f"az-{self.identity[:16]}"

    def manifest(self) -> dict:
        return {"path": str(self.path), "step": self.step, "weights_sha256": self.identity,
                "name": self.name, "config": asdict(self.config)}


def inspect_checkpoint(path: str | Path, config: Config | None = None) -> Checkpoint:
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        header = json.loads(f.read(size))
        metadata = header.get("__metadata__", {})
        if metadata.get("format_version", "1") != "1":
            raise ValueError("unsupported checkpoint format_version")
        if config is None:
            if "config" not in metadata:
                raise ValueError("checkpoint needs Config metadata (or --config architecture.json)")
            cfg = json.loads(metadata["config"])
            for key in ("lr", "lr_milestones", "dirichlet_alpha"):
                if isinstance(cfg.get(key), list):
                    cfg[key] = tuple(cfg[key])
            config = Config(**cfg)
        if (config.board_size, config.num_policy_planes, config.num_actions) != (8, 73, 4672):
            raise ValueError("checkpoint must use the repo's 8x8, 73-plane chess encoding")
        prefixed = any(k.startswith("model.") for k in header)
        weights = {k: v for k, v in header.items() if k != "__metadata__"
                   and (k.startswith("model.") if prefixed else not k.startswith("optimizer."))}
        if not weights:
            raise ValueError("checkpoint contains no model tensors")
        architecture = {k: v for k, v in asdict(config).items() if k in (
            "board_size", "history_steps", "num_piece_planes", "num_rep_planes", "num_const_planes",
            "num_policy_planes", "num_actions", "num_block", "num_filters", "conv_kernel",
            "policy_head_filters", "value_head_filters", "value_hidden", "bn_momentum")}
        h = hashlib.sha256(json.dumps(architecture, sort_keys=True).encode())
        file_size = path.stat().st_size
        for name, tensor in sorted(weights.items()):
            begin, end = tensor["data_offsets"]
            if not 0 <= begin <= end <= file_size - 8 - size:
                raise ValueError(f"invalid tensor offset: {name}")
            h.update(json.dumps([name.removeprefix("model."), tensor["dtype"], tensor["shape"]]).encode())
            f.seek(8 + size + begin)
            remaining = end - begin
            while remaining:
                block = f.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("truncated tensor data")
                h.update(block)
                remaining -= len(block)
    progress = json.loads(metadata.get("training", "{}"))
    return Checkpoint(path, config, int(progress.get("global_step", 0)), h.hexdigest())


def load_network(checkpoint: Checkpoint, device: str):
    from tinygrad import Context, Tensor
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load
    from alphazero.nn import NeuralNet
    import numpy as np

    with Context(DEV=device, TRAINING=0):
        net = NeuralNet(checkpoint.config)
        state = safe_load(str(checkpoint.path))
        prefixed = any(k.startswith("model.") for k in state)
        weights = {k.removeprefix("model."): v for k, v in state.items()
                   if (k.startswith("model.") if prefixed else not k.startswith("optimizer."))}
        load_state_dict(net, weights, strict=True, verbose=False)
        for t in get_state_dict(net).values():
            t.is_param_(False)

    def predict(planes):
        with Context(DEV=device, TRAINING=0):
            logits, value = net(Tensor(planes))
            logits, value = logits.numpy(), value.numpy().reshape(-1)
        if logits.shape != (len(planes), checkpoint.config.num_actions) or value.shape != (len(planes),):
            raise ValueError("invalid network output shape")
        if not np.isfinite(logits).all() or not np.isfinite(value).all():
            raise ValueError("checkpoint produced non-finite predictions")
        return logits.astype(np.float32), value.astype(np.float32)

    return net, predict


def discover(paths: list[str], config: Config | None = None) -> list[Checkpoint]:
    found = []
    for name in paths:
        path = Path(name).expanduser()
        candidates = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        found.extend(inspect_checkpoint(p, config) for p in candidates)
    unique = {c.identity: c for c in sorted(found, key=lambda c: (c.step, c.path.name == "final.safetensors"))}
    if not unique:
        raise ValueError("no .safetensors checkpoints found")
    return sorted(unique.values(), key=lambda c: (c.step, c.identity))
