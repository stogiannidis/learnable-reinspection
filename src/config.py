"""Unified configuration for Re-Inspection VLM training and evaluation.

This module defines a single dataclass consumed by Hydra-resolved YAML plus
CLI overrides.  Fields group runtime paths, model geometry (bottleneck
attention), two-stage optimization hyperparameters, and data preprocessing
flags shared across backends (InternVL3, Qwen2.5-VL, Gemma 4).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReInspectionConfig:
    """Resolved experiment configuration for training or benchmark evaluation.

    Attributes are intentionally flat so Hydra can override any field by name.
    Stage 1 optimizes the re-inspection module (and optional grounding head) on
    referring-expression data; stage 2 freezes the vision stack and attaches
    LoRA to the language model while continuing to train the module.

    The derived attribute ``d_head`` is computed in ``__post_init__`` and must
    satisfy ``d_bottleneck % n_heads == 0`` for multi-head attention in the
    bottleneck space.
    """

    # ------------------------------------------------------------------ #
    # Runtime (set via Hydra overrides or config file)                    #
    # ------------------------------------------------------------------ #
    backend: str = "internvl3"          # internvl3 | qwen25vl | gemma4
    stage: int = 1                      # 1 | 2
    data_root: str = "/data/datasets"
    coco_images_dir: Optional[str] = "/data/datasets/coco/images/train2014"
    output_dir: str = "models"
    deepspeed_config: Optional[str] = None
    stage1_checkpoint: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 300
    wandb_gradient_log_interval: int = 1
    experiment_name: Optional[str] = None
    # Optional pbar.io cloud progress (https://pbar.io/docs/integrations).
    # When enabled, set PBAR_API_KEY in the environment if your account requires it.
    pbar_enabled: bool = False
    pbar_api_url: str = "https://pbar.io/api"
    pbar_update_interval: float = 0.5
    num_workers: int = 4

    # Eval-specific
    checkpoint_dir: Optional[str] = None
    lora_checkpoint_dir: Optional[str] = None
    output_file: str = "eval_results.json"
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "vsr", "gqa_spatial", "whatsup",
            "3dsrbench", "mindcube", "blink", "srbench",
            "qspatial", "embspatial",
        ]
    )
    eval_condition: str = "reinspection"
    eval_compare: bool = False
    frozen_cache_file: Optional[str] = None
    max_samples: int = -1

    # ------------------------------------------------------------------ #
    # Re-Inspection Module                                                 #
    # ------------------------------------------------------------------ #
    d_model: int = 4096
    d_bottleneck: int = 512
    n_queries: int = 64
    n_selector_queries: int = 8
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_name_or_path: str = "OpenGVLab/InternVL3-8B-hf"
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
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
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
    stage1_max_steps: Optional[int] = None
    # ---- Stage 1 auxiliary losses (each independently toggleable) ----
    # Stage 1 is a pure grounding stage; LM CE / NTP is disabled by default.
    stage1_use_lm_ce: bool = False
    # Attention KL between selector-token attn_vis and overlap-area target.
    stage1_use_attn_loss: bool = True
    stage1_attn_loss_weight: float = 1.0
    stage1_attn_loss_type: str = "kl"
    stage1_attn_small_box_weight: bool = True
    # Box regression head on selector tokens: L1 + GIoU (DETR weights).
    stage1_use_grounding_loss: bool = True
    stage1_grounding_loss_weight: float = 1.0
    stage1_grounding_l1_weight: float = 5.0
    stage1_grounding_giou_weight: float = 2.0
    stage1_grounding_warmup_steps: int = 100
    # ROI feature grounding: cosine between pooled content tokens and ROI-pooled
    # frozen vision features. Encourages content tokens to carry region info.
    stage1_use_roi_feature_loss: bool = True
    stage1_roi_feature_loss_weight: float = 0.2
    # Symmetric query–text InfoNCE in bottleneck space.
    stage1_use_query_text_infonce: bool = False
    stage1_query_text_infonce_weight: float = 0.1
    stage1_query_text_infonce_temperature: float = 0.07
    stage1_train_projector: bool = False

    # ------------------------------------------------------------------ #
    # Stage 2                                                              #
    # ------------------------------------------------------------------ #
    stage2_lr_module: float = 5e-5
    stage2_lr_lora: float = 2e-5
    stage2_epochs: int = 6
    stage2_batch_size: int = 4
    stage2_grad_accum: int = 8
    stage2_warmup_ratio: float = 0.03
    stage2_warmup_steps: Optional[int] = None
    # Keep the Stage-1 ROI encoder adaptable during instruction tuning unless
    # an experiment explicitly freezes it.
    stage2_train_reinspection: bool = True
    # Optional weak grounding stream mixed into Stage 2. Disabled unless both
    # the master weight and cadence are positive.
    stage2_aux_grounding_weight: float = 0.0
    stage2_aux_attn_loss_weight: float = 1.0
    stage2_aux_roi_feature_loss_weight: float = 0.2
    stage2_aux_every_n_steps: int = 0
    stage2_aux_batch_size: Optional[int] = None
    stage2_aux_stage1_dataset_names: Optional[List[str]] = None
    stage2_aux_stage1_extra_datasets: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 4 * 28 * 28
    crop_to_patches_stage1: bool = False
    crop_to_patches_stage2: bool = True
    system_prompt: str = "You are a helpful assistant."
    val_split_ratio: float = 0.05
    val_batch_size: Optional[int] = None
    stage1_dataset_names: Optional[List[str]] = None  # None → registry.stage1_defaults()
    # Extra stage-1 datasets appended to the base list (e.g. [grit, grefcoco]).
    stage1_extra_datasets: Optional[List[str]] = None
    stage2_dataset_names: Optional[List[str]] = None  # None → registry.stage2_defaults()

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate head divisibility and set per-head bottleneck width.

        Raises:
            ValueError: If ``d_bottleneck`` is not divisible by ``n_heads``.
        """
        if self.d_bottleneck % self.n_heads != 0:
            raise ValueError(
                f"d_bottleneck ({self.d_bottleneck}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0 < self.n_selector_queries < self.n_queries:
            raise ValueError(
                f"n_selector_queries ({self.n_selector_queries}) must be in (0, n_queries={self.n_queries})"
            )
        self.d_head = self.d_bottleneck // self.n_heads

    @property
    def processor_path(self) -> str:
        """Hugging Face id or local path used to load tokenizer/processor.

        Falls back to ``model_name_or_path`` when a separate processor checkpoint
        is not specified (typical for weight-tied processor bundles).
        """
        return self.processor_name_or_path or self.model_name_or_path
