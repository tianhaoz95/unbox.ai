"""
Qualitative sampling: load a checkpoint and generate text from a prompt.

Usage:
    python -m platform.eval.sample \
        --checkpoint checkpoints/pretrain/760m/latest.pt \
        --tokenizer  checkpoints/tokenizer \
        --config     configs/pretrain/760m.yaml \
        --prompt     "The theory of relativity states that"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from unbox_platform.model import ModelConfig, Transformer
from unbox_platform.tokenizer import Tokenizer
from unbox_platform.train.checkpoint import load_checkpoint


def sample(
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    tokenizer = Tokenizer(tokenizer_path)

    model = Transformer(model_config)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model = model.to(device).to(dtype)
    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
    )

    generated = output_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    result = sample(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        model_config=model_cfg,
        prompt=args.prompt,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(result)


if __name__ == "__main__":
    main()
