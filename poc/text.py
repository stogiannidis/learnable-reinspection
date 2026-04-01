"""Text encoder: token embedding + sinusoidal-style learned positional embedding."""
import torch
import torch.nn as nn

from .config import Config
from .vocab import Vocab


class TextEncoder(nn.Module):
    def __init__(self, config: Config, vocab: Vocab):
        super().__init__()
        # max position covers max(max_q_len, max_ans_len) = 11
        max_pos = max(config.max_q_len, config.max_ans_len) + 2
        self.embedding = nn.Embedding(vocab.size, config.d_model, padding_idx=vocab.PAD)
        self.pos_embedding = nn.Embedding(max_pos, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) int64
        Returns:
            (B, L, d_model)
        """
        L = token_ids.shape[1]
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)  # (1, L)
        return self.dropout(self.embedding(token_ids) + self.pos_embedding(pos))
