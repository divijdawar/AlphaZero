from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    #representation
    board_size: int = 8
    history_steps: int = 8
    num_piece_planes: int = 12      # 6 own + 6 opp
    num_rep_planes: int = 2
    num_const_planes:int = 7
    num_policy_planes: int = 73     # 56 queen + 8 knight + 9 underpromo
    num_workers: int = 2
    num_actions: int = 4672

    #network
    num_block: int = 20
    num_filters: int = 256
    conv_kernel: int = 3
    policy_head_filters: int = 2
    value_head_filters: int = 1
    value_hidden: int = 256
    bn_momentum: float = 0.1   # new-batch weight: retain 90% of running statistics

    #search
    num_simulations: int = 100
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 30     # tau=1 for first N plies, greedy after

    #selfplay
    max_plies: int = 512            # overlong games scored as a draw
    buffer_size: int = 50_000

    #training
    batch_size: int = 64
    steps: int = 100_000
    l2: float = 1e-4
    lr: tuple[float, ...] = (0.01, 0.001, 0.0001, 0.00001)
    checkpoint: int = 2000
    lr_milestones: tuple[int, ...] | None = None  # None scales decays to steps
