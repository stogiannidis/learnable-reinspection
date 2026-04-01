"""CLI entry point.

Usage:
    python -m poc.run --model both --epochs 30 --seeds 42 123 456
"""
import argparse
import os
import torch
import numpy as np

from .config import Config
from .vocab import get_vocab
from .data import get_dataloaders
from .model import BaselineModel, ReInspectionModel
from .train import train_model
from .evaluate import compute_test_metrics
from .visualize import (
    figure1_attention_panels,
    figure2_entropy_curves,
    figure3_accuracy_curves,
    figure4_attention_mass,
    figure5_per_query,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Grid PoC for Re-Inspection Module')
    parser.add_argument('--model', choices=['baseline', 'ours', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=None, help='Override config.n_epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to run (default: 42 123 456)')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--no-fig5', action='store_true', help='Skip optional per-query figure')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.seeds is not None:
        config.seeds = args.seeds
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"model={args.model!r}  seeds={config.seeds}  epochs={config.n_epochs}")

    vocab = get_vocab()

    # Generate data once; reused across seeds via cache
    data_cache: dict = {}
    print("Generating datasets...")
    _, _, test_loader, splits = get_dataloaders(config, vocab, cache=data_cache)
    test_samples = splits['test']

    histories_base: list = []
    histories_ours: list = []
    test_metrics_base: list = []
    test_metrics_ours: list = []

    # Keep the first-seed models for visualisation (no re-training needed)
    vis_model_b = None
    vis_model_r = None

    run_baseline = args.model in ('baseline', 'both')
    run_ours     = args.model in ('ours',     'both')

    for i, seed in enumerate(config.seeds):
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        set_seed(seed)

        train_loader, val_loader, _, _ = get_dataloaders(config, vocab, cache=data_cache)

        if run_baseline:
            print("\n--- Baseline ---")
            model_b = BaselineModel(config, vocab)
            print(f"  Parameters: {model_b.count_parameters():,}")
            hist_b = train_model(model_b, train_loader, val_loader, config, device)
            metrics_b = compute_test_metrics(model_b, test_loader, config, device)
            histories_base.append(hist_b)
            test_metrics_base.append(metrics_b)
            _print_test(metrics_b)
            if i == 0:
                vis_model_b = model_b  # save for figures

        if run_ours:
            print("\n--- Re-Inspection Model ---")
            model_r = ReInspectionModel(config, vocab)
            print(f"  Parameters: {model_r.count_parameters():,}")
            hist_r = train_model(model_r, train_loader, val_loader, config, device)
            metrics_r = compute_test_metrics(model_r, test_loader, config, device)
            histories_ours.append(hist_r)
            test_metrics_ours.append(metrics_r)
            _print_test(metrics_r)
            if i == 0:
                vis_model_r = model_r  # save for figures

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    if vis_model_b is not None and vis_model_r is not None:
        figure1_attention_panels(vis_model_b, vis_model_r, test_samples, config, device)
        figure4_attention_mass(test_metrics_base, test_metrics_ours, config)
        if not args.no_fig5:
            figure5_per_query(vis_model_r, test_samples, config, device)

    # Entropy / accuracy curves (need both to compare)
    if histories_base and histories_ours:
        figure2_entropy_curves(histories_base, histories_ours, config)
        figure3_accuracy_curves(histories_base, histories_ours, config)
    elif histories_base:
        figure2_entropy_curves(histories_base, histories_base, config)
        figure3_accuracy_curves(histories_base, histories_base, config)
    elif histories_ours:
        figure2_entropy_curves(histories_ours, histories_ours, config)
        figure3_accuracy_curves(histories_ours, histories_ours, config)

    print(f"\nAll figures saved to {config.output_dir}/")
    print("Done.")


def _print_test(m):
    print(f"  Test acc: A={m['A']['acc']:.4f}  B={m['B']['acc']:.4f}  "
          f"all={m['all']['acc']:.4f}  entropy={m['all']['entropy']:.3f}  "
          f"mass={m['all']['mass']:.4f}")


if __name__ == '__main__':
    main()
