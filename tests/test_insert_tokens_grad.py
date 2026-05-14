"""Smoke tests: R-token insert keeps autograd path for LM-style loss."""

import torch

from src.config import ReInspectionConfig
from src.model.reinspection_module import ReInspectionModule


def _stack_insert_embeds(
    inputs_embeds: torch.Tensor,
    R: torch.Tensor,
    insert_pos: int,
) -> torch.Tensor:
    """Mirror backend `_insert_tokens` embedding construction (fixed pos per row)."""
    B = inputs_embeds.shape[0]
    rows = []
    for b in range(B):
        rows.append(
            torch.cat(
                [
                    inputs_embeds[b, :insert_pos].detach(),
                    R[b],
                    inputs_embeds[b, insert_pos:].detach(),
                ],
                dim=0,
            )
        )
    return torch.stack(rows, dim=0)


def test_cat_insert_r_receives_grad_backbone_detached():
    B, L, D, n_q = 2, 8, 32, 4
    pos = 3
    ie = torch.randn(B, L, D, requires_grad=True)
    r = torch.randn(B, n_q, D, requires_grad=True)
    new = _stack_insert_embeds(ie, r, pos)
    new.sum().backward()
    assert r.grad is not None
    assert r.grad.abs().sum() > 0
    assert ie.grad is None


def test_reinspection_w_up_grad_through_insert_pattern():
    cfg = ReInspectionConfig(
        d_model=128,
        d_bottleneck=32,
        n_queries=4,
        n_selector_queries=2,
        n_heads=4,
        ffn_mult=2,
    )
    m = ReInspectionModule(cfg)
    # Default W_up is zeros: CE-style grad hits W_up but ∂R/∂R_r is 0, so nothing below W_up
    # gets signal. Use small random W_up to smoke-test the full value path.
    with torch.no_grad():
        m.W_up.weight.normal_(0, 0.02)
    B, nv, nt, L = 2, 10, 12, 20
    pos = 5
    V = torch.randn(B, nv, 128)
    T = torch.randn(B, nt, 128)
    vm = torch.ones(B, nv, dtype=torch.bool)
    tm = torch.ones(B, nt, dtype=torch.bool)
    R = m(V, T, V_mask=vm, T_mask=tm, need_weights=False)[0]
    ie = torch.randn(B, L, 128)
    new_embeds = _stack_insert_embeds(ie, R, pos)
    new_embeds.sum().backward()
    assert m.W_up.weight.grad is not None
    assert m.W_up.weight.grad.abs().sum() > 0
    assert m.cross_vis.v_proj.weight.grad is not None
    assert m.cross_vis.v_proj.weight.grad.abs().sum() > 0
    assert m.ffn2[0].weight.grad is not None
    assert m.ffn2[0].weight.grad.abs().sum() > 0
