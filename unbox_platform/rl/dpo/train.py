"""DPO training entry point using TRL's DPOTrainer.

Usage:
    python -m unbox_platform.rl.dpo.train --config configs/rl/dpo.yaml

The reference model is a frozen copy of the SFT checkpoint. TRL manages it
automatically when ref_model=None is passed to DPOTrainer.

Dataset format: HuggingFaceH4/ultrafeedback_binarized
  chosen / rejected columns are lists of message dicts with "role" / "content".
  The string "prompt" column is dropped so TRL's extract_prompt can split the
  conversation correctly into a list-format prompt + assistant response.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from unbox_platform.model.hf_adapter import UnboxForCausalLM


@dataclass
class DPOTrainConfig:
    # Paths
    model_path: str = "checkpoints/sft/basic/checkpoint-3000"
    output_dir: str = "checkpoints/rl/dpo"

    # Dataset
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    max_samples: int = -1  # -1 = use all

    # DPO hyperparameters
    beta: float = 0.1
    loss_type: str = "sigmoid"
    max_length: int = 2048

    # Training
    num_epochs: int = 1
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        # PyYAML parses bare scientific notation (e.g. 5e-7) as str, not float.
        # Coerce here so a missing decimal point in the YAML doesn't silently crash.
        self.learning_rate = float(self.learning_rate)
        self.beta = float(self.beta)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip = float(self.grad_clip)
        self.beta1 = float(self.beta1)
        self.beta2 = float(self.beta2)

    # Logging / saving
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 500
    use_wandb: bool = False
    wandb_project: str = "unbox-ai-dpo"
    wandb_run_name: str = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = DPOTrainConfig(**{k: v for k, v in raw.items() if k in DPOTrainConfig.__dataclass_fields__})

    import torch
    model = UnboxForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16)
    # Load the reference model explicitly rather than letting TRL derive it from the
    # checkpoint. TRL's create_model_from_path does getattr(transformers, arch_name)
    # which can't find our custom UnboxForCausalLM class.
    ref_model = UnboxForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset_name, split=cfg.train_split)
    eval_dataset = load_dataset(cfg.dataset_name, split=cfg.eval_split)

    # Drop the string "prompt" column — ultrafeedback has a string prompt that conflicts
    # with TRL's conversational processing. TRL's extract_prompt will re-derive a
    # list-format prompt from the shared prefix of chosen/rejected.
    drop_cols = [c for c in ["prompt", "prompt_id", "messages", "score_chosen", "score_rejected"]
                 if c in dataset.column_names]
    dataset = dataset.remove_columns(drop_cols)
    eval_drop = [c for c in drop_cols if c in eval_dataset.column_names]
    eval_dataset = eval_dataset.remove_columns(eval_drop)

    if cfg.max_samples > 0:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
        eval_dataset = eval_dataset.select(range(min(cfg.max_samples // 10 + 1, len(eval_dataset))))

    if cfg.use_wandb and cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    training_args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.beta1,
        adam_beta2=cfg.beta2,
        max_grad_norm=cfg.grad_clip,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=(cfg.dtype == "bfloat16"),
        fp16=(cfg.dtype == "float16"),
        logging_steps=cfg.log_every_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_every_steps,
        save_strategy="steps",
        save_steps=cfg.save_every_steps,
        load_best_model_at_end=False,
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.wandb_run_name or None,
        # DPO-specific
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    resume_from = get_last_checkpoint(cfg.output_dir) if Path(cfg.output_dir).exists() else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"DPO complete. Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
