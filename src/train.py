"""Training entry point (Hydra): ``deepspeed --module src.train stage=stage1 data_root=...``"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_util import strip_deepspeed_local_rank_argv

strip_deepspeed_local_rank_argv()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.config import ReInspectionConfig
    from src.training.trainer import run_training

    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    run_training(config)


if __name__ == "__main__":
    main()
