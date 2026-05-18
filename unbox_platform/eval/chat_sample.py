"""Chat-format sampling from an SFT (HF) checkpoint.

Usage:
    python -m unbox_platform.eval.chat_sample \\
        --checkpoint checkpoints/sft/basic/checkpoint-3000 \\
        --prompt "What is the capital of China?"

Two pitfalls this script handles explicitly:
- add_special_tokens=False when encoding chat-template output (the tokenizer
  would otherwise append a spurious <|eos|>, causing the model to predict the
  next-conversation header "user" instead of the answer).
- use_cache=False in generate() because UnboxForCausalLM does not implement
  the KV cache; with use_cache=True HF passes only the latest token on steps
  2+, destroying context and producing degenerate output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from unbox_platform.model.hf_adapter import UnboxForCausalLM


def chat_sample(
    checkpoint: str | Path,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = UnboxForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False: the chat template already contains all needed special tokens;
    # letting the tokenizer add its own would append a trailing <|eos|> and break generation.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=False,  # KV cache not implemented; see module docstring
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat-format generation from an SFT checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Prompt     : {args.prompt}")
    print("-" * 60)
    response = chat_sample(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(response)


if __name__ == "__main__":
    main()
