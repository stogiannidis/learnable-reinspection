"""Paper-ready figures (saved to config.output_dir)."""
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

from .config import Config
from .vocab import COLORS, SHAPES


# Colour name → matplotlib colour
_MPL_COLORS = {
    'red': 'red', 'green': 'limegreen', 'blue': 'dodgerblue',
    'yellow': 'gold', 'orange': 'darkorange', 'purple': 'mediumpurple',
}

# Shape → matplotlib marker
_MPL_MARKERS = {
    'circle': 'o', 'square': 's', 'triangle': '^', 'star': '*',
}


def _attn_heatmap(attn_64, ax, title, color_ids, shape_ids, target_cells,
                  grid_size=8, cmap='Reds'):
    """Draw one 8×8 grid panel with attention overlay."""
    heat = attn_64.reshape(grid_size, grid_size)
    im = ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=heat.max(), alpha=0.8, aspect='equal')
    ax.set_title(title, fontsize=7, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])

    # Object markers
    for cell_idx in range(grid_size * grid_size):
        c = int(color_ids[cell_idx])
        s = int(shape_ids[cell_idx])
        if c < len(COLORS):  # non-empty
            row = cell_idx // grid_size
            col = cell_idx % grid_size
            ax.plot(col, row,
                    marker=_MPL_MARKERS[SHAPES[s]],
                    color=_MPL_COLORS[COLORS[c]],
                    markersize=12, markeredgecolor='black', markeredgewidth=0.8,
                    zorder=3)

    # Highlight target cells with red border
    for cell in target_cells:
        r = cell // grid_size
        c_idx = cell % grid_size
        rect = patches.Rectangle(
            (c_idx - 0.5, r - 0.5), 1, 1,
            linewidth=2, edgecolor='red', facecolor='none', zorder=4,
        )
        ax.add_patch(rect)

    return im


def _question_str(sample: dict) -> str:
    from .vocab import get_vocab
    vocab = get_vocab()
    q = vocab.decode(sample['question_ids'].tolist())
    return ' '.join(q)


def figure1_attention_panels(baseline_model, reinsp_model, test_samples, config, device,
                              n_panels=4):
    """Figure 1: 2×n_panels grid showing baseline vs ours attention maps."""
    os.makedirs(config.output_dir, exist_ok=True)
    from .data import collate_fn

    baseline_model.eval(); reinsp_model.eval()

    # Pick examples: n_panels//2 type A, n_panels//2 type B
    type_a = [s for s in test_samples if s['q_type'] == 'A'][:n_panels // 2]
    type_b = [s for s in test_samples if s['q_type'] == 'B'][:n_panels // 2]
    selected = type_a + type_b

    fig, axes = plt.subplots(2, n_panels, figsize=(n_panels * 3, 7))
    row_labels = ['Baseline (L)', 'Ours (R)']

    for col_i, sample in enumerate(selected):
        batch = collate_fn([sample])
        batch_dev = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        with torch.no_grad():
            _, _, attn_base  = baseline_model(batch_dev, return_attn=True)
            _, _, attn_ours  = reinsp_model(batch_dev, return_attn=True)

        attn_b = attn_base[0].cpu().numpy()
        attn_o = attn_ours[0].cpu().numpy()
        color_ids   = sample['color_ids']
        shape_ids   = sample['shape_ids']
        target_cells = sample['target_cells']

        from .vocab import get_vocab
        vocab = get_vocab()
        q_str = ' '.join(vocab.decode(sample['question_ids'].tolist()))

        _attn_heatmap(attn_b, axes[0, col_i], q_str[:60],
                      color_ids, shape_ids, target_cells, config.grid_size)
        _attn_heatmap(attn_o, axes[1, col_i], '',
                      color_ids, shape_ids, target_cells, config.grid_size)

    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=10, fontweight='bold')

    fig.suptitle('Attention Maps: Baseline vs Re-Inspection Module', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(config.output_dir, 'figure1_attention_panels.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")


def figure2_entropy_curves(histories_base, histories_ours, config):
    """Figure 2: attention entropy over training epochs (3-seed bands)."""
    os.makedirs(config.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    def _plot(histories, label, color):
        data = np.array([h['val_entropy'] for h in histories])  # (n_seeds, n_epochs)
        mean = data.mean(0)
        std  = data.std(0)
        xs   = np.arange(1, len(mean) + 1)
        ax.plot(xs, mean, label=label, color=color, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    _plot(histories_base, 'Baseline (L)', 'steelblue')
    _plot(histories_ours, 'Ours (R)',     'darkorange')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Attention Entropy (nats)')
    ax.set_title('Attention Entropy over Training'); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.output_dir, 'figure2_entropy_curves.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")


def figure3_accuracy_curves(histories_base, histories_ours, config):
    """Figure 3: validation accuracy over epochs."""
    os.makedirs(config.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    def _plot(histories, label, color):
        data = np.array([h['val_acc'] for h in histories])
        mean = data.mean(0)
        std  = data.std(0)
        xs   = np.arange(1, len(mean) + 1)
        ax.plot(xs, mean, label=label, color=color, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    _plot(histories_base, 'Baseline (L)', 'steelblue')
    _plot(histories_ours, 'Ours (R)',     'darkorange')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy')
    ax.set_title('Accuracy over Training'); ax.legend()
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.output_dir, 'figure3_accuracy_curves.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")


def figure4_attention_mass(test_metrics_base, test_metrics_ours, config):
    """Figure 4: bar chart of attention mass on target cells (by question type)."""
    os.makedirs(config.output_dir, exist_ok=True)

    labels = ['Type A\nBaseline', 'Type A\nOurs', 'Type B\nBaseline', 'Type B\nOurs',
              'All\nBaseline', 'All\nOurs']
    values = [
        np.mean([m['A']['mass'] for m in test_metrics_base]),
        np.mean([m['A']['mass'] for m in test_metrics_ours]),
        np.mean([m['B']['mass'] for m in test_metrics_base]),
        np.mean([m['B']['mass'] for m in test_metrics_ours]),
        np.mean([m['all']['mass'] for m in test_metrics_base]),
        np.mean([m['all']['mass'] for m in test_metrics_ours]),
    ]
    errors = [
        np.std([m['A']['mass'] for m in test_metrics_base]),
        np.std([m['A']['mass'] for m in test_metrics_ours]),
        np.std([m['B']['mass'] for m in test_metrics_base]),
        np.std([m['B']['mass'] for m in test_metrics_ours]),
        np.std([m['all']['mass'] for m in test_metrics_base]),
        np.std([m['all']['mass'] for m in test_metrics_ours]),
    ]
    colors = ['steelblue', 'darkorange'] * 3

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, yerr=errors, color=colors, capsize=4, width=0.6, alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Attention Mass on Target Cells')
    ax.set_title('Attention Mass on Ground-Truth Target Cells')
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, axis='y', alpha=0.3)

    # Legend patch
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='steelblue', label='Baseline'),
                        Patch(color='darkorange', label='Ours')], loc='upper right')
    plt.tight_layout()
    path = os.path.join(config.output_dir, 'figure4_attention_mass.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")


def figure5_per_query(reinsp_model, test_samples, config, device, n_examples=2):
    """Figure 5 (optional): Per-query attention maps for type A examples."""
    os.makedirs(config.output_dir, exist_ok=True)
    from .data import collate_fn
    reinsp_model.eval()

    samples = [s for s in test_samples if s['q_type'] == 'A'][:n_examples]
    if not samples:
        return

    nq = config.n_queries
    fig, axes = plt.subplots(n_examples, nq, figsize=(nq * 2, n_examples * 2.5))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row_i, sample in enumerate(samples):
        batch = collate_fn([sample])
        batch_dev = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        with torch.no_grad():
            V = reinsp_model.vision_encoder(batch_dev['color_ids'], batch_dev['shape_ids'])
            T = reinsp_model.text_encoder(batch_dev['question_ids'])
            _, _, A_vis = reinsp_model.reinspection(V, T)  # (1, N_q, 64)

        A_vis_np = A_vis[0].cpu().numpy()   # (N_q, 64)

        for q_i in range(nq):
            ax = axes[row_i, q_i]
            heat = A_vis_np[q_i].reshape(config.grid_size, config.grid_size)
            ax.imshow(heat, cmap='Blues', vmin=0, vmax=heat.max(), aspect='equal')
            ax.set_title(f'Q{q_i}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if q_i == 0:
                from .vocab import get_vocab
                q_str = ' '.join(get_vocab().decode(sample['question_ids'].tolist()))
                ax.set_ylabel(q_str[:30], fontsize=6)

    fig.suptitle('Per-Query Attention Maps (Type A)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(config.output_dir, 'figure5_per_query.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")
