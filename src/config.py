"""Unified configuration for Re-Inspection VLM training (Qwen3-VL and InternVL3)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReInspectionConfig:
    # ------------------------------------------------------------------ #
    # Runtime (set via Hydra overrides or config file)                    #
    # ------------------------------------------------------------------ #
    backend: str = "internvl3"          # qwen3vl | internvl3
    stage: int = 1                      # 1 | 2
    data_root: str = "/data/datasets"
    output_dir: str = "models"
    deepspeed_config: Optional[str] = None
    stage1_checkpoint: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 300
    experiment_name: Optional[str] = None
    num_workers: int = 4

    # Eval-specific
    checkpoint_dir: Optional[str] = None
    lora_checkpoint_dir: Optional[str] = None
    output_file: str = "eval_results.json"
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "vsr", "gqa_spatial",
            "3dsrbench", "mindcube", "blink", "srbench",
        ]
    )
    eval_condition: str = "reinspection"
    eval_compare: bool = False
    max_samples: int = -1

    # ------------------------------------------------------------------ #
    # Re-Inspection Module                                                 #
    # ------------------------------------------------------------------ #
    d_model: int = 4096
    d_bottleneck: int = 512
    n_queries: int = 64
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    processor_name_or_path: Optional[str] = None
    image_seq_length: int = 256
    answer_ignore_index: int = -100

    # ------------------------------------------------------------------ #
    # LoRA (Stage 2)                                                       #
    # ------------------------------------------------------------------ #
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # ------------------------------------------------------------------ #
    # Stage 1                                                              #
    # ------------------------------------------------------------------ #
    stage1_lr_module: float = 1e-4
    stage1_lr_merger: float = 1e-5
    stage1_lr_projector: float = 1e-5
    stage1_epochs: int = 5
    stage1_batch_size: int = 8
    stage1_grad_accum: int = 8
    stage1_warmup_ratio: float = 0.03
    stage1_warmup_steps: Optional[int] = None
    stage1_attn_loss_weight: float = 1.0
    stage1_train_projector: bool = False
    stage1_attn_loss_type: str = "focal"
    stage1_aux_loss: str = "both"  # "attn" | "grounding" | "both"
    stage1_grounding_loss_weight: float = 5.0
    stage1_grounding_l1_weight: float = 5.0
    stage1_grounding_giou_weight: float = 2.0
    stage1_grounding_warmup_steps: int = 100

    # ------------------------------------------------------------------ #
    # Stage 2                                                              #
    # ------------------------------------------------------------------ #
    stage2_lr_module: float = 5e-5
    stage2_lr_lora: float = 2e-5
    stage2_epochs: int = 10
    stage2_batch_size: int = 4
    stage2_grad_accum: int = 8
    stage2_warmup_ratio: float = 0.03
    stage2_warmup_steps: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 4 * 28 * 28
    crop_to_patches_stage1: bool = False
    crop_to_patches_stage2: bool = True
    system_prompt: str = "You are a helpful assistant."

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.d_bottleneck % self.n_heads != 0:
            raise ValueError(
                f"d_bottleneck ({self.d_bottleneck}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_head = self.d_bottleneck // self.n_heads

    @property
    def processor_path(self) -> str:
        return self.processor_name_or_path or self.model_name_or_path
