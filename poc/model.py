"""BaselineModel and ReInspectionModel."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .vocab import Vocab
from .vision import VisionEncoder
from .text import TextEncoder
from .attention import CausalTransformer
from .reinspection import ReInspectionModule


def _make_causal_mask(seq_len: int, device) -> torch.Tensor:
    """Upper-triangular additive mask; shape (1, 1, S, S)."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class BaselineModel(nn.Module):
    """
    Sequence: [V(64) | L(N_q) | Q(q_len) | A(ans_len)]

    L are static learnable tokens that CANNOT attend to Q (causal mask),
    giving them no task-specific information.
    """
    def __init__(self, config: Config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.vision_encoder    = VisionEncoder(config)
        self.text_encoder      = TextEncoder(config, vocab)
        self.learnable_tokens  = nn.Parameter(torch.randn(config.n_queries, config.d_model) * 0.02)
        self.decoder           = CausalTransformer(
            config.d_model, config.n_heads, config.n_decoder_layers,
            config.ffn_mult, config.dropout,
        )
        self.out_proj = nn.Linear(config.d_model, vocab.size, bias=False)
        # Weight tying
        self.out_proj.weight = self.text_encoder.embedding.weight

        # Offsets for slicing the sequence
        self._vis_len    = config.n_cells      # 64
        self._prefix_len = config.n_queries    # 8
        self._q_len      = config.max_q_len    # 11
        self._ans_len    = config.max_ans_len  # 5

    @property
    def _ans_start(self):
        return self._vis_len + self._prefix_len + self._q_len  # 83

    def forward(self, batch, return_attn=False):
        color_ids     = batch['color_ids']       # (B, 64)
        shape_ids     = batch['shape_ids']       # (B, 64)
        question_ids  = batch['question_ids']    # (B, 11)
        ans_input_ids = batch['ans_input_ids']   # (B, 5)
        ans_target_ids = batch['ans_target_ids'] # (B, 5)
        B = color_ids.shape[0]

        V = self.vision_encoder(color_ids, shape_ids)                    # (B, 64, d)
        T = self.text_encoder(question_ids)                              # (B, 11, d)
        L = self.learnable_tokens.unsqueeze(0).expand(B, -1, -1)        # (B,  8, d)
        A = self.text_encoder(ans_input_ids)                             # (B,  5, d)

        # Assemble: [V | L | Q | A]
        seq = torch.cat([V, L, T, A], dim=1)                            # (B, 88, d)
        S = seq.shape[1]

        mask = _make_causal_mask(S, seq.device)
        out, all_attn = self.decoder(seq, mask)                         # (B, 88, d)

        # Predict answer tokens from positions [ans_start : ans_start + ans_len]
        ans_start = self._ans_start
        logits = self.out_proj(out[:, ans_start:ans_start + self._ans_len])  # (B, 5, V)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab.size),
            ans_target_ids.reshape(-1),
            ignore_index=self.vocab.PAD,
        )

        if return_attn:
            # L→V attention from the last decoder layer
            last_attn = all_attn[-1]  # (B, H, S, S)
            l_s = self._vis_len
            l_e = self._vis_len + self._prefix_len
            lv_attn = last_attn[:, :, l_s:l_e, 0:self._vis_len]  # (B, H, N_q, 64)
            attn_map = lv_attn.mean(dim=(1, 2))                   # (B, 64)
            # Normalize so values sum to 1 over cells
            attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
            return logits, loss, attn_map

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReInspectionModel(nn.Module):
    """
    R = ReInspectionModule(V, T_question)   [explicit cross-attention to both]
    Sequence: [V(64) | Q(q_len) | R(N_q) | A(ans_len)]

    R is computed from both V and Q, giving it task-conditioned spatial focus.
    """
    def __init__(self, config: Config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.vision_encoder    = VisionEncoder(config)
        self.text_encoder      = TextEncoder(config, vocab)
        self.reinspection      = ReInspectionModule(config)
        self.decoder           = CausalTransformer(
            config.d_model, config.n_heads, config.n_decoder_layers,
            config.ffn_mult, config.dropout,
        )
        self.out_proj = nn.Linear(config.d_model, vocab.size, bias=False)
        self.out_proj.weight = self.text_encoder.embedding.weight

        self._vis_len    = config.n_cells
        self._prefix_len = config.n_queries
        self._q_len      = config.max_q_len
        self._ans_len    = config.max_ans_len

    @property
    def _ans_start(self):
        # [V | Q | R | A]: V=64, Q=11, R=8  → A starts at 83
        return self._vis_len + self._q_len + self._prefix_len  # 83

    def forward(self, batch, return_attn=False):
        color_ids      = batch['color_ids']
        shape_ids      = batch['shape_ids']
        question_ids   = batch['question_ids']
        ans_input_ids  = batch['ans_input_ids']
        ans_target_ids = batch['ans_target_ids']
        B = color_ids.shape[0]

        V = self.vision_encoder(color_ids, shape_ids)   # (B, 64, d)
        T = self.text_encoder(question_ids)              # (B, 11, d)
        A = self.text_encoder(ans_input_ids)             # (B,  5, d)

        R, A_task, A_vis = self.reinspection(V, T)       # (B, 8, d), (B,8,11), (B,8,64)

        # Assemble: [V | Q | R | A]
        seq = torch.cat([V, T, R, A], dim=1)             # (B, 88, d)
        S = seq.shape[1]

        mask = _make_causal_mask(S, seq.device)
        out, _ = self.decoder(seq, mask)

        ans_start = self._ans_start
        logits = self.out_proj(out[:, ans_start:ans_start + self._ans_len])  # (B, 5, V)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab.size),
            ans_target_ids.reshape(-1),
            ignore_index=self.vocab.PAD,
        )

        if return_attn:
            # A_vis: (B, N_q, 64) — aggregate over queries
            attn_map = A_vis.mean(dim=1)                                    # (B, 64)
            attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
            return logits, loss, attn_map

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
