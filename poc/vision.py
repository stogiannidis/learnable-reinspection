"""Symbolic grid encoder: color + shape + row + col embeddings."""
import torch
import torch.nn as nn

from .config import Config


class VisionEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.grid_size = config.grid_size
        d = config.d_model
        assert d % 4 == 0, "d_model must be divisible by 4 for vision encoder"
        d4 = d // 4

        # +1 for the "empty cell" token
        self.color_embed = nn.Embedding(config.n_colors + 1, d4)
        self.shape_embed = nn.Embedding(config.n_shapes + 1, d4)
        self.row_embed   = nn.Embedding(config.grid_size, d4)
        self.col_embed   = nn.Embedding(config.grid_size, d4)
        self.proj = nn.Linear(d, d)

    def forward(self, color_ids: torch.Tensor, shape_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            color_ids: (B, N_cells) int64  [0..n_colors, where n_colors = empty]
            shape_ids: (B, N_cells) int64  [0..n_shapes, where n_shapes = empty]
        Returns:
            V: (B, N_cells, d_model)
        """
        B, N = color_ids.shape
        device = color_ids.device

        idx = torch.arange(N, device=device)
        rows = (idx // self.grid_size).unsqueeze(0).expand(B, -1)  # (B, N)
        cols = (idx %  self.grid_size).unsqueeze(0).expand(B, -1)  # (B, N)

        v = torch.cat([
            self.color_embed(color_ids),  # (B, N, d//4)
            self.shape_embed(shape_ids),  # (B, N, d//4)
            self.row_embed(rows),         # (B, N, d//4)
            self.col_embed(cols),         # (B, N, d//4)
        ], dim=-1)                        # (B, N, d)
        return self.proj(v)               # (B, N, d)
