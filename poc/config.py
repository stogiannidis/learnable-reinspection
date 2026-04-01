from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Grid
    grid_size: int = 8
    n_colors: int = 6
    n_shapes: int = 4
    min_objects: int = 2
    max_objects: int = 5

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_decoder_layers: int = 4
    n_queries: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1

    # Data
    n_train: int = 50000
    n_val: int = 5000
    n_test: int = 5000

    # Training
    batch_size: int = 256
    n_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.05

    # Evaluation
    top_k_attn: int = 5

    # Misc
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    output_dir: str = "poc/figures"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        self.n_cells = self.grid_size ** 2          # 64
        self.n_objects = self.n_colors * self.n_shapes  # 24
        self.max_q_len = 11   # type A: 11 tokens, type B: 5 (padded to 11)
        self.max_ans_len = 5  # type A: 2 + 3 pad, type B: 5
        # Full sequence length for decoder
        self.seq_len = self.n_cells + self.n_queries + self.max_q_len + self.max_ans_len
