"""SFT training entry point using TRL's SFTTrainer.

Usage:
    python -m unbox_platform.sft.train --config configs/sft/basic.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from unbox_platform.model.hf_adapter import UnboxConfig, UnboxForCausalLM

from .config import SFTTrainConfig


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: SFTTrainConfig) -> UnboxForCausalLM:
    if cfg.checkpoint_path:
        # Load from a raw unbox pretraining checkpoint (.pt)
        unbox_cfg = UnboxConfig()  # defaults; overridden by saved config on HF checkpoints
        model = UnboxForCausalLM.from_unbox_checkpoint(
            cfg.checkpoint_path,
            unbox_cfg,
            device="cpu",
        )
    else:
        # Fresh model with default config (useful for smoke tests)
        model = UnboxForCausalLM(UnboxConfig())
    return model


def build_tokenizer(cfg: SFTTrainConfig) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_path)
    # SFTTrainer needs pad_token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = load_config(args.config)
    cfg = SFTTrainConfig(**{k: v for k, v in raw.items() if k in SFTTrainConfig.__dataclass_fields__})

    model = build_model(cfg)
    tokenizer = build_tokenizer(cfg)

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    eval_dataset = load_dataset(cfg.dataset_name, split=cfg.eval_dataset_split)

    if cfg.max_samples > 0:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
        eval_dataset = eval_dataset.select(range(min(cfg.max_samples // 10 + 1, len(eval_dataset))))

    # Keep only the "messages" column so TRL uses the chat-template path rather
    # than the prompt+completion path (triggered when a "prompt" column is present).
    dataset = dataset.select_columns(["messages"])
    eval_dataset = eval_dataset.select_columns(["messages"])

    if cfg.use_wandb and cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    # SFTConfig extends TrainingArguments with SFT-specific params (max_length, packing, etc.)
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        learning_rate=cfg.max_lr,
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
        # SFT-specific
        max_length=cfg.max_seq_len,
        packing=cfg.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"SFT complete. Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
