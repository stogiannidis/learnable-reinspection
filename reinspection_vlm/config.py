"""Unified configuration for Re-Inspection VLM training (Qwen3-VL and InternVL3)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReInspectionConfig:
    # Re-Inspection Module dimensions
    d_model: int = 4096
    d_bottleneck: int = 256
    n_queries: int = 32
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0

    # Model
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    processor_name_or_path: Optional[str] = None
    image_seq_length: int = 256
    answer_ignore_index: int = -100

    # LoRA config (Stage 2)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training Stage 1 (grounding warm-up)
    stage1_lr_module: float = 1e-4
    stage1_lr_merger: float = 1e-5
    stage1_lr_projector: float = 1e-5
    stage1_epochs: int = 5
    stage1_batch_size: int = 4
    stage1_grad_accum: int = 8
    stage1_warmup_ratio: float = 0.03
    stage1_warmup_steps: Optional[int] = None
    stage1_attn_loss_weight: float = 10.0
    stage1_train_projector: bool = False
    # "focal" = Qwen-style focal + diversification; "kl" = KL (InternVL default)
    stage1_attn_loss_type: str = "focal"

    # Training Stage 2 (spatial reasoning)
    stage2_lr_module: float = 5e-5
    stage2_lr_lora: float = 2e-5
    stage2_epochs: int = 10
    stage2_batch_size: int = 4
    stage2_grad_accum: int = 8
    stage2_warmup_ratio: float = 0.03
    stage2_warmup_steps: Optional[int] = None

    # Data + prompting (InternVL-specific options)
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 4 * 28 * 28
    crop_to_patches_stage1: bool = False
    crop_to_patches_stage2: bool = True
    system_prompt: str = "You are a helpful assistant."

    # Misc
    seed: int = 42
    output_dir: str = "outputs"
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
