from __future__ import annotations
from tinygrad import Tensor, nn
from .config import Config

__all__ = ["NeuralNet", "ResidualBlock", "num_input_planes"]

def num_input_planes(cfg: Config) -> int:
    """Trunk input channels: T*(12 piece + 2 rep) + 7 constant = 119."""
    return cfg.history_steps * (cfg.num_piece_planes + cfg.num_rep_planes) + cfg.num_const_planes

class ResidualBlock:

    def __init__(self, num_filters: int, kernel_size: int = 3, bn_momentum: float = 0.1):
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters, momentum=bn_momentum)

    def __call__(self, x: Tensor) -> Tensor:
        return (self.bn2(self.conv2(self.bn1(self.conv1(x)).relu())) + x).relu()

class NeuralNet:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        f = cfg.num_filters
        self.stem = nn.Conv2d(num_input_planes(cfg), f, cfg.conv_kernel, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(f, momentum=cfg.bn_momentum)
        self.blocks = [ResidualBlock(f, cfg.conv_kernel, cfg.bn_momentum) for _ in range(cfg.num_block - 1)]
        # policy head: 1x1 conv -> BN -> ReLU -> 1x1 conv to 73 action planes
        self.pol1 = nn.Conv2d(f, cfg.policy_head_filters, 1, bias=False)
        self.pol_bn = nn.BatchNorm2d(cfg.policy_head_filters, momentum=cfg.bn_momentum)
        self.pol2 = nn.Conv2d(cfg.policy_head_filters, cfg.num_policy_planes, 1, bias=True)
        # value head: 1x1 conv -> BN -> ReLU -> FC(value_hidden) -> ReLU -> FC(1) -> tanh
        self.val1 = nn.Conv2d(f, cfg.value_head_filters, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(cfg.value_head_filters, momentum=cfg.bn_momentum)
        self.val_fc1 = nn.Linear(64 * cfg.value_head_filters, cfg.value_hidden)
        self.val_fc2 = nn.Linear(cfg.value_hidden, 1)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.stem_bn(self.stem(x)).relu()
        for blk in self.blocks:
            h = blk(h)
        logits = self.pol2(self.pol_bn(self.pol1(h)).relu()).permute(0, 2, 3, 1).flatten(1)
        v = self.val_bn(self.val1(h)).relu().flatten(1)
        return logits, self.val_fc2(self.val_fc1(v).relu()).tanh()
