"""Attention primitives: MHSA, MHCA, TransformerBlock, CausalTransformer."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x:    (B, S, D)
            mask: (1, 1, S, S) additive mask, -inf for masked positions
        Returns:
            out:          (B, S, D)
            attn_weights: (B, H, S, S)  post-softmax, no dropout applied
        """
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, S, Dh)

        scale = Dh ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, S, S)
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        out = (self.attn_drop(attn_weights) @ v).transpose(1, 2).reshape(B, S, D)
        out = self.out(out)
        return out, attn_weights


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj  = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out     = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """
        Args:
            q:  (B, S_q,  D)
            kv: (B, S_kv, D)
        Returns:
            out:          (B, S_q, D)
            attn_weights: (B, H, S_q, S_kv)
        """
        B, S_q, D = q.shape
        S_kv = kv.shape[1]
        H, Dh = self.n_heads, self.d_head

        q_  = self.q_proj(q).reshape(B, S_q, H, Dh).transpose(1, 2)      # (B,H,S_q,Dh)
        kv_ = self.kv_proj(kv).reshape(B, S_kv, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv_.unbind(0)                                                # each (B,H,S_kv,Dh)

        scale = Dh ** -0.5
        scores = (q_ @ k.transpose(-2, -1)) * scale   # (B, H, S_q, S_kv)
        attn_weights = torch.softmax(scores, dim=-1)
        out = (self.attn_drop(attn_weights) @ v).transpose(1, 2).reshape(B, S_q, D)
        out = self.out(out)
        return out, attn_weights


class TransformerBlock(nn.Module):
    """Pre-LayerNorm causal transformer block (self-attention + FFN)."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask=None):
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class CausalTransformer(nn.Module):
    """Stack of TransformerBlocks; returns hidden states + per-layer attention."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x:    (B, S, D)
            mask: (1, 1, S, S) causal additive mask
        Returns:
            x:        (B, S, D)
            all_attn: list of (B, H, S, S), one per layer
        """
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            all_attn.append(attn)
        return self.norm(x), all_attn
