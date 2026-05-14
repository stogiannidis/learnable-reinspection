"""Training entry point for DeepSpeed + Hydra.

Launched as ``deepspeed --module src.train`` with Hydra overrides (for example
``stage=stage1``, ``data_root=...``).  The Hydra ``main`` builds a
:class:`~src.config.ReInspectionConfig` from the merged YAML and forwards to
:class:`~src.training.trainer.run_training`.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_util import strip_deepspeed_local_rank_argv

strip_deepspeed_local_rank_argv()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Instantiate config from Hydra and start the distributed training loop.

    Args:
        cfg: Merged Hydra configuration (``DictConfig``) for this run.
    """
    from src.config import ReInspectionConfig
    from src.training.trainer import run_training

    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    run_training(config)


if __name__ == "__main__":
    main()
