"""Unified training entry: ``python -m reinspection_vlm.train --backend qwen3vl|internvl3 --stage 1|2``."""

import argparse

from reinspection_vlm.train_common import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["qwen3vl", "internvl3"])
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="reinspection-vlm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config, output_dir=args.output_dir)
    run_training(args.backend, args.stage, config, args)


if __name__ == "__main__":
    main()
