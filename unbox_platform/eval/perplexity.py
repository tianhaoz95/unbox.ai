"""
Compute perplexity on a held-out dataset split.

Usage:
    python -m platform.eval.perplexity \
        --checkpoint checkpoints/pretrain/760m/latest.pt \
        --tokenizer  checkpoints/tokenizer \
        --config     configs/pretrain/760m.yaml \
        --max-batches 200
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from unbox_platform.data import DataConfig, build_dataloader
from unbox_platform.model import ModelConfig, Transformer
from unbox_platform.tokenizer import Tokenizer
from unbox_platform.train.checkpoint import load_checkpoint


@torch.no_grad()
def compute_perplexity(
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    data_config: DataConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    max_batches: int | None = None,
) -> float:
    tokenizer = Tokenizer(tokenizer_path)

    model = Transformer(model_config)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model = model.to(device)
    model.eval()

    eval_loader = build_dataloader(
        tokenizer, data_config, split="eval", batch_size=batch_size
    )

    total_loss = 0.0
    total_tokens = 0
    use_amp = dtype in (torch.bfloat16, torch.float16)

    for i, batch in enumerate(eval_loader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            _, loss = model(input_ids, labels)

        # loss is mean over non-ignored tokens; recover sum for accurate perplexity
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
    data_cfg = DataConfig(**{k: v for k, v in cfg_dict.get("data", {}).items()
                             if k in DataConfig.__dataclass_fields__})

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    ppl = compute_perplexity(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        model_config=model_cfg,
        data_config=data_cfg,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
