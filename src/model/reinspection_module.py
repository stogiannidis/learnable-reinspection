"""Re-inspection module: two-stage bottleneck cross-attention over text then vision.

Learned queries first attend to language tokens in a low-dimensional bottleneck,
then to vision tokens, with residual FFN blocks.  This design biases the module
toward question-conditioned visual aggregation before features are injected back
into the language-model hidden states.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ReInspectionConfig


class BottleneckCrossAttention(nn.Module):
    """Multi-head cross-attention in the bottleneck width ``d_r``.

    When ``need_weights`` is False, uses memory-efficient scaled dot-product
    attention; otherwise materializes softmax weights for supervision or metrics.
    """

    def __init__(
        self,
        d_r: int,
        n_heads: int,
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        """Build projection layers and attention dropout.

        Args:
            d_r: Bottleneck hidden size; must divide ``n_heads``.
            n_heads: Number of attention heads.
            dropout: Dropout probability on attention probabilities (training only).
            dtype: Optional module dtype (for example ``torch.bfloat16``).
        """
        super().__init__()
        assert d_r % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_r // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_r, d_r, dtype=dtype)
        self.k_proj = nn.Linear(d_r, d_r, dtype=dtype)
        self.v_proj = nn.Linear(d_r, d_r, dtype=dtype)
        self.out_proj = nn.Linear(d_r, d_r, dtype=dtype)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.BoolTensor] = None,
        need_weights: bool = False,
    ):
        """Apply cross-attention from queries ``q`` to key/value ``kv``.

        Args:
            q: Query tensor of shape ``(B, S_q, d_r)``.
            kv: Key/value tensor of shape ``(B, S_kv, d_r)``.
            kv_mask: Boolean mask with shape ``(B, S_kv)``; ``True`` keeps a position.
            need_weights: If True, return attention weights; if False, use fused SDPA.

        Returns:
            Tuple ``(out, attn_weights)`` where ``out`` has shape ``(B, S_q, d_r)``
            and ``attn_weights`` is ``None`` or ``(B, H, S_q, S_kv)``.
        """
        B, S_q, _ = q.shape
        S_kv = kv.shape[1]
        H, Dh = self.n_heads, self.d_head

        q_ = self.q_proj(q).reshape(B, S_q, H, Dh).transpose(1, 2)
        k_ = self.k_proj(kv).reshape(B, S_kv, H, Dh).transpose(1, 2)
        v_ = self.v_proj(kv).reshape(B, S_kv, H, Dh).transpose(1, 2)

        attn_bias = None
        if kv_mask is not None:
            attn_bias = kv_mask[:, None, None, :].to(dtype=q_.dtype)
            attn_bias = (1.0 - attn_bias) * torch.finfo(q_.dtype).min

        if not need_weights:
            dp = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=attn_bias, dropout_p=dp, scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(B, S_q, -1)
            out = self.out_proj(out)
            return out, None

        scores = (q_ @ k_.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        attn_weights = torch.softmax(scores, dim=-1)
        out = (self.attn_drop(attn_weights) @ v_).transpose(1, 2).reshape(B, S_q, -1)
        out = self.out_proj(out)
        return out, attn_weights


def _make_ffn(d: int, mult: int, dropout: float, dtype: Optional[torch.dtype] = None):
    """Two-layer GELU feed-forward with dropout, width ``d * mult`` hidden."""
    return nn.Sequential(
        nn.Linear(d, d * mult, dtype=dtype),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d * mult, d, dtype=dtype),
        nn.Dropout(dropout),
    )


class ReInspectionModule(nn.Module):
    """Two-stage cross-attention with down/up projections to the LM width ``d_model``."""

    def __init__(self, config: ReInspectionConfig, dtype: Optional[torch.dtype] = None):
        """Allocate queries, projections, attention blocks, and FFNs from ``config``.

        Args:
            config: Global hyperparameters (widths, heads, FFN multiplier, query count).
            dtype: Optional parameter dtype for mixed-precision training.
        """
        super().__init__()
        d = config.d_model
        d_r = config.d_bottleneck
        h = config.n_heads
        ffn = config.ffn_mult
        dp = config.dropout
        nq = config.n_queries

        self.base_queries = nn.Parameter(
            torch.randn(nq, d_r, dtype=dtype) * math.sqrt(2.0 / (d + d_r))
        )

        self.W_down_t = nn.Linear(d, d_r, bias=False, dtype=dtype)
        self.W_down_v = nn.Linear(d, d_r, bias=False, dtype=dtype)
        self.W_up = nn.Linear(d_r, d, bias=False, dtype=dtype)

        self.norm_q0 = nn.LayerNorm(d_r, dtype=dtype)
        self.cross_text = BottleneckCrossAttention(d_r, h, dp, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_r, dtype=dtype)
        self.ffn1 = _make_ffn(d_r, ffn, dp, dtype=dtype)

        self.norm_q1 = nn.LayerNorm(d_r, dtype=dtype)
        self.cross_vis = BottleneckCrossAttention(d_r, h, dp, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_r, dtype=dtype)
        self.ffn2 = _make_ffn(d_r, ffn, dp, dtype=dtype)

        self._init_weights()

    def _init_weights(self):
        """Xavier-initialize linear layers; zero the up-projection for stable residuals."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.W_up.weight)

    def forward(
        self,
        V: torch.Tensor,
        T: torch.Tensor,
        V_mask: Optional[torch.BoolTensor] = None,
        T_mask: Optional[torch.BoolTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        text_mask: Optional[torch.BoolTensor] = None,
        need_weights: bool = False,
    ):
        """Run text-then-vision cross-attention and project back to ``d_model``.

        Args:
            V: Vision hidden states ``(B, S_v, d_model)``.
            T: Text hidden states ``(B, S_t, d_model)``.
            V_mask: Optional vision padding mask ``(B, S_v)`` (alias: ``vision_mask``).
            T_mask: Optional text padding mask ``(B, S_t)`` (alias: ``text_mask``).
            vision_mask: Deprecated alias for ``V_mask``; used if ``V_mask`` is None.
            text_mask: Deprecated alias for ``T_mask``; used if ``T_mask`` is None.
            need_weights: Whether to materialize attention distributions.

        Returns:
            Tuple ``(R, A_task, A_vis, R_r, Q_task, T_down)`` where ``R`` is the
            residual in ``d_model``, ``A_*`` are mean-pooled attention maps (or
            ``None``), ``R_r`` is the full bottleneck state before up-projection,
            ``Q_task`` is the query tensor after text interaction (before vision
            cross-attn), and ``T_down`` is text in bottleneck width ``(B, S_t, d_r)``.
        """
        if V_mask is None:
            V_mask = vision_mask
        if T_mask is None:
            T_mask = text_mask

        input_dtype = V.dtype
        module_dtype = self.W_down_t.weight.dtype
        if T.dtype != module_dtype:
            T = T.to(module_dtype)
        if V.dtype != module_dtype:
            V = V.to(module_dtype)

        B = V.shape[0]
        T_down = self.W_down_t(T)
        V_down = self.W_down_v(V)
        q = self.base_queries.unsqueeze(0).expand(B, -1, -1)

        attn_out, A_task = self.cross_text(
            self.norm_q0(q), T_down, kv_mask=T_mask, need_weights=need_weights,
        )
        q = q + attn_out
        q = q + self.ffn1(self.norm1(q))
        Q_task = q

        attn_out, A_vis = self.cross_vis(
            self.norm_q1(Q_task), V_down, kv_mask=V_mask, need_weights=need_weights,
        )
        q = Q_task + attn_out
        R_r = q + self.ffn2(self.norm2(q))

        R = self.W_up(R_r)
        if R.dtype != input_dtype:
            R = R.to(input_dtype)

        A_task_out = A_task.mean(dim=1) if A_task is not None else None
        A_vis_out = A_vis.mean(dim=1) if A_vis is not None else None
        # Q_task: queries after text cross-attn + FFN, before vision (for query–text contrastive).
        return R, A_task_out, A_vis_out, R_r, Q_task, T_down

    def count_parameters(self):
        """Return the number of trainable scalar parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
