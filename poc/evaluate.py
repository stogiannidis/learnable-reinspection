"""Evaluation metrics: accuracy, attention entropy, IoU, mass on target."""
import math
import torch
import numpy as np

from .config import Config


# ── per-sample attention metrics ─────────────────────────────────────────────

def attn_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over 64 grid cells.  (B,64) -> (B,)"""
    p = attn.clamp(min=1e-9)
    return -(p * p.log()).sum(dim=-1)


def attn_iou(attn: torch.Tensor, target_cells: list, k: int) -> torch.Tensor:
    """Binarise top-k attention cells; IoU with ground-truth cells.
    attn: (B,64), target_cells: list[list[int]] -> (B,)
    """
    B = attn.shape[0]
    ious = torch.zeros(B, device=attn.device)
    top_k = attn.topk(k, dim=-1).indices  # (B, k)
    for i in range(B):
        pred_set   = set(top_k[i].tolist())
        gt_set     = set(target_cells[i])
        intersect  = len(pred_set & gt_set)
        union      = len(pred_set | gt_set)
        ious[i]    = intersect / max(union, 1)
    return ious


def attn_mass(attn: torch.Tensor, target_cells: list) -> torch.Tensor:
    """Sum of attention values on target cells.  (B,64) -> (B,)"""
    B = attn.shape[0]
    masses = torch.zeros(B, device=attn.device)
    for i in range(B):
        for c in target_cells[i]:
            masses[i] += attn[i, c]
    return masses


# ── accuracy ─────────────────────────────────────────────────────────────────

def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """Exact-match accuracy: all non-PAD tokens correct. (B,L,V), (B,L) -> scalar."""
    preds = logits.argmax(dim=-1)      # (B, L)
    mask  = targets != pad_id          # (B, L)
    # A sample is correct iff every non-PAD position matches
    correct = ((preds == targets) | ~mask).all(dim=-1)  # (B,)
    return correct.float().mean().item()


# ── epoch-level aggregation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_epoch_metrics(model, loader, config: Config, device) -> dict:
    model.eval()
    pad_id = model.vocab.PAD
    k      = config.top_k_attn

    losses, accs, entropies, ious, masses = [], [], [], [], []

    for batch in loader:
        batch_dev = {key: val.to(device) if hasattr(val, 'to') else val
                     for key, val in batch.items()}
        logits, loss, attn_map = model(batch_dev, return_attn=True)
        target_cells = batch['target_cells']   # list[list[int]] (stays on CPU)

        losses.append(loss.item())
        accs.append(batch_accuracy(logits, batch_dev['ans_target_ids'], pad_id))
        entropies.extend(attn_entropy(attn_map).tolist())
        ious.extend(attn_iou(attn_map, target_cells, k).tolist())
        masses.extend(attn_mass(attn_map, target_cells).tolist())

    return {
        'loss':    float(np.mean(losses)),
        'acc':     float(np.mean(accs)),
        'entropy': float(np.mean(entropies)),
        'iou':     float(np.mean(ious)),
        'mass':    float(np.mean(masses)),
    }


@torch.no_grad()
def compute_test_metrics(model, loader, config: Config, device) -> dict:
    """Like compute_epoch_metrics but also splits by question type."""
    model.eval()
    pad_id = model.vocab.PAD
    k      = config.top_k_attn

    results = {'A': [], 'B': [], 'all': []}

    for batch in loader:
        batch_dev = {key: val.to(device) if hasattr(val, 'to') else val
                     for key, val in batch.items()}
        logits, loss, attn_map = model(batch_dev, return_attn=True)
        target_cells = batch['target_cells']
        q_types      = batch['q_types']
        targets      = batch_dev['ans_target_ids']

        preds  = logits.argmax(dim=-1)
        mask   = targets != pad_id
        correct = ((preds == targets) | ~mask).all(dim=-1)  # (B,)
        ent    = attn_entropy(attn_map)
        iou_v  = attn_iou(attn_map, target_cells, k)
        mass_v = attn_mass(attn_map, target_cells)

        for i, qt in enumerate(q_types):
            rec = {
                'correct': correct[i].item(),
                'entropy': ent[i].item(),
                'iou':     iou_v[i].item(),
                'mass':    mass_v[i].item(),
            }
            results[qt].append(rec)
            results['all'].append(rec)

    def _agg(lst):
        if not lst:
            return {}
        return {
            'acc':     float(np.mean([r['correct'] for r in lst])),
            'entropy': float(np.mean([r['entropy'] for r in lst])),
            'iou':     float(np.mean([r['iou']     for r in lst])),
            'mass':    float(np.mean([r['mass']     for r in lst])),
        }

    return {qt: _agg(results[qt]) for qt in ('A', 'B', 'all')}
