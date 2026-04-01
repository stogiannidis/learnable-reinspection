"""Training loop with cosine LR schedule and warmup."""
import math
import torch
import torch.nn as nn
from torch.optim import AdamW

from .config import Config
from .evaluate import compute_epoch_metrics


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model, train_loader, val_loader, config: Config, device, verbose=True):
    """Train model for config.n_epochs; return history dict."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    total_steps  = config.n_epochs * len(train_loader)
    warmup_steps = int(total_steps * config.warmup_frac)

    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_entropy': [], 'val_iou': [], 'val_mass': [],
    }
    best_val_acc = -1.0
    best_state   = None
    step = 0

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            # Update LR
            lr = _cosine_lr(step, total_steps, warmup_steps, config.lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            optimizer.zero_grad()
            _, loss = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            step       += 1

        train_loss = epoch_loss / n_batches
        metrics    = compute_epoch_metrics(model, val_loader, config, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['val_acc'].append(metrics['acc'])
        history['val_entropy'].append(metrics['entropy'])
        history['val_iou'].append(metrics['iou'])
        history['val_mass'].append(metrics['mass'])

        if metrics['acc'] > best_val_acc:
            best_val_acc = metrics['acc']
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{config.n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={metrics['loss']:.4f}  "
                f"val_acc={metrics['acc']:.4f}  entropy={metrics['entropy']:.3f}"
            )

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history
