"""SFT training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SFTTrainConfig:
    # Paths
    checkpoint_path: str = ""
    tokenizer_path: str = "checkpoints/tokenizer"
    output_dir: str = "checkpoints/sft"

    # Dataset
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    eval_dataset_split: str = "test_sft"
    max_samples: int = -1  # -1 = use all

    # Dataset source: "huggingface" (default) or "modelscope" (for restricted regions).
    # Set ms_dataset_name to the ModelScope repo ID if it differs from dataset_name
    # (e.g. "AI-ModelScope/ultrachat_200k"). Leave empty to try dataset_name as-is.
    dataset_source: str = "huggingface"
    ms_dataset_name: str = ""

    # Model / sequence
    max_seq_len: int = 2048

    # Optimisation
    num_epochs: int = 1
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    max_lr: float = 2e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"

    # Logging / saving
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 500
    use_wandb: bool = False
    wandb_project: str = "unbox-ai-sft"
    wandb_run_name: str = ""

    # SFTTrainer-specific
    packing: bool = False
