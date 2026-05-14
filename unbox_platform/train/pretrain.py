"""
Pretraining entry point.

Usage:
    # Single GPU
    python -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from unbox_platform.data import DataConfig, build_dataloader
from unbox_platform.model import ModelConfig, Transformer
from unbox_platform.tokenizer import Tokenizer

from .checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .config import ParallelConfig, TrainConfig
from .parallel import get_device, init_distributed, setup_megatron, setup_model
from .scheduler import get_lr


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    config: TrainConfig,
    rank: int,
    start_step: int = 0,
    start_epoch: int = 0,
) -> None:
    device = get_device(rank)
    dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
    use_amp = dtype in (torch.bfloat16, torch.float16)

    estimated_chunks = train_loader.dataset.estimate_num_chunks()
    total_steps = (estimated_chunks // config.batch_size) * config.num_epochs // config.grad_accumulation_steps
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    output_dir = Path(config.output_dir)
    step = start_step
    t0 = time.time()

    wandb_run = None
    if rank == 0 and config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config={
                "total_steps": total_steps,
                "effective_batch_size": config.effective_batch_size(),
                "max_lr": config.max_lr,
                "dtype": config.dtype,
            },
            resume="allow",
        )

    for epoch in range(start_epoch, config.num_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        accum_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                _, loss = model(input_ids, labels)
                loss = loss / config.grad_accumulation_steps

            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            is_accumulation_step = (batch_idx + 1) % config.grad_accumulation_steps == 0
            if not is_accumulation_step:
                continue

            # Update LR
            current_lr = get_lr(
                step, total_steps, config.max_lr, config.min_lr_ratio, config.warmup_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Gradient clip + optimizer step
            if dtype == torch.float16:
                scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            if dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            step += 1

            if rank == 0 and step % config.log_every_steps == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    config.batch_size
                    * config.grad_accumulation_steps
                    * config.max_seq_len
                    * config.log_every_steps
                    / dt
                )
                print(
                    f"step={step} epoch={epoch} loss={accum_loss:.4f} "
                    f"lr={current_lr:.2e} tokens/s={tokens_per_sec:.0f} "
                    f"elapsed={dt:.1f}s"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": accum_loss,
                        "train/lr": current_lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch,
                    }, step=step)
                t0 = time.time()

            accum_loss = 0.0

            if rank == 0 and step % config.eval_every_steps == 0:
                eval_loss = evaluate(model, eval_loader, device, dtype, use_amp)
                ppl = math.exp(min(eval_loss, 20))
                print(f"  eval step={step} loss={eval_loss:.4f} perplexity={ppl:.2f}")
                if wandb_run is not None:
                    wandb_run.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": ppl,
                    }, step=step)

            if rank == 0 and step % config.save_every_steps == 0:
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step,
                    epoch,
                    accum_loss,
                )

    if rank == 0:
        save_checkpoint(output_dir, model, optimizer, step, config.num_epochs, accum_loss)
        print(f"Training complete. Final checkpoint at step {step}.")
        if wandb_run is not None:
            wandb_run.finish()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            _, loss = model(input_ids, labels)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    train_cfg = TrainConfig(**{k: v for k, v in cfg_dict.get("train", {}).items()
                               if k in TrainConfig.__dataclass_fields__})
    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
    data_cfg = DataConfig(**{k: v for k, v in cfg_dict.get("data", {}).items()
                             if k in DataConfig.__dataclass_fields__})
    parallel_cfg = ParallelConfig(**{k: v for k, v in cfg_dict.get("parallel", {}).items()
                                     if k in ParallelConfig.__dataclass_fields__})
    train_cfg.parallel = parallel_cfg

    rank, world_size = init_distributed()
    setup_megatron(parallel_cfg, rank, world_size)
    device = get_device(rank)

    tokenizer = Tokenizer(train_cfg.tokenizer_path)

    model = Transformer(model_cfg)
    if rank == 0:
        print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")

    model = setup_model(model, parallel_cfg, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.max_lr,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=torch.cuda.is_available(),
    )

    train_loader = build_dataloader(
        tokenizer, data_cfg, split="train",
        batch_size=train_cfg.batch_size,
        rank=rank, world_size=world_size,
    )
    eval_loader = build_dataloader(
        tokenizer, data_cfg, split="eval",
        batch_size=train_cfg.batch_size,
        rank=rank, world_size=world_size,
    )

    start_step, start_epoch = 0, 0
    ckpt_path = find_latest_checkpoint(train_cfg.output_dir)
    if ckpt_path is not None:
        if rank == 0:
            print(f"Resuming from {ckpt_path}")
        state = load_checkpoint(ckpt_path, model, optimizer, device=device)
        start_step = state["step"]
        start_epoch = state["epoch"]

    train(
        model, optimizer, train_loader, eval_loader,
        train_cfg, rank,
        start_step=start_step, start_epoch=start_epoch,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
