"""Re-Inspection Module (Section 3 of reinspection_module_formulation).

Stage 1 — Task Conditioning:
    Q_task = LN( base_queries + CrossAttn(Q=base_queries, KV=T) )
    Q_task = Q_task + FFN( LN(Q_task) )

Stage 2 — Visual Re-Inspection:
    R = LN( Q_task + CrossAttn(Q=Q_task, KV=V) )
    R = R + FFN( LN(R) )

Returns R, A_task (query→text), A_vis (query→vision).
"""
import torch
import torch.nn as nn

from .config import Config
from .attention import MultiHeadCrossAttention


def _make_ffn(d_model: int, ffn_mult: int, dropout: float):
    return nn.Sequential(
        nn.Linear(d_model, d_model * ffn_mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * ffn_mult, d_model),
        nn.Dropout(dropout),
    )


class ReInspectionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        d = config.d_model
        h = config.n_heads
        nq = config.n_queries
        ffn = config.ffn_mult
        dp = config.dropout

        self.base_queries = nn.Parameter(torch.randn(nq, d) * 0.02)

        # Stage 1: task conditioning
        self.norm_q0 = nn.LayerNorm(d)          # pre-norm on queries before cross-attn
        self.cross_text = MultiHeadCrossAttention(d, h, dp)
        self.norm1 = nn.LayerNorm(d)
        self.ffn1 = _make_ffn(d, ffn, dp)

        # Stage 2: visual re-inspection
        self.norm_q1 = nn.LayerNorm(d)          # pre-norm on Q_task before cross-attn
        self.cross_vis = MultiHeadCrossAttention(d, h, dp)
        self.norm2 = nn.LayerNorm(d)
        self.ffn2 = _make_ffn(d, ffn, dp)

    def forward(self, V: torch.Tensor, T: torch.Tensor):
        """
        Args:
            V: (B, N_cells, d)   vision tokens
            T: (B, q_len, d)     question tokens
        Returns:
            R:      (B, N_q, d)         re-inspection tokens
            A_task: (B, N_q, q_len)     mean-head attention query→text
            A_vis:  (B, N_q, N_cells)   mean-head attention query→vision
        """
        B = V.shape[0]
        q = self.base_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, d)

        # --- Stage 1: task conditioning ---
        attn_out, A_task = self.cross_text(self.norm_q0(q), T)   # A_task: (B,H,N_q,q_len)
        q = q + attn_out
        q = q + self.ffn1(self.norm1(q))
        Q_task = q

        # --- Stage 2: visual re-inspection ---
        attn_out, A_vis = self.cross_vis(self.norm_q1(Q_task), V)  # A_vis: (B,H,N_q,N_cells)
        q = Q_task + attn_out
        R = q + self.ffn2(self.norm2(q))

        # Average over heads for interpretability
        return R, A_task.mean(dim=1), A_vis.mean(dim=1)
