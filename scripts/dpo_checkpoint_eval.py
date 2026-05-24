"""Generate responses from SFT baseline + all DPO checkpoints for comparison report."""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from unbox_platform.model.hf_adapter import UnboxForCausalLM

PROMPTS = [
    "Explain the difference between a list and a tuple in Python in one sentence.",
    "I have 10 minutes to study for an exam. What should I do?",
    "Is it better to use tabs or spaces for Python indentation? Give me a direct answer.",
]

CHECKPOINTS = [
    ("SFT baseline", "checkpoints/sft/basic"),
    ("DPO step 500",  "checkpoints/rl/dpo/checkpoint-500"),
    ("DPO step 1000", "checkpoints/rl/dpo/checkpoint-1000"),
    ("DPO step 1500", "checkpoints/rl/dpo/checkpoint-1500"),
    ("DPO step 2000", "checkpoints/rl/dpo/checkpoint-2000"),
    ("DPO step 2500", "checkpoints/rl/dpo/checkpoint-2500"),
    ("DPO step 2868", "checkpoints/rl/dpo/checkpoint-2868"),
]

METRICS = {
    "SFT baseline":  {"step": 0,    "epoch": 0,     "loss": None,   "margins": None,   "accuracy": None},
    "DPO step 500":  {"step": 500,  "epoch": 0.523, "loss": 0.6915, "margins": 0.004063, "accuracy": 0.4719},
    "DPO step 1000": {"step": 1000, "epoch": 1.046, "loss": 0.6857, "margins": 0.01608,  "accuracy": 0.5750},
    "DPO step 1500": {"step": 1500, "epoch": 1.569, "loss": 0.6867, "margins": 0.01397,  "accuracy": 0.5578},
    "DPO step 2000": {"step": 2000, "epoch": 2.092, "loss": 0.6869, "margins": 0.01353,  "accuracy": 0.5797},
    "DPO step 2500": {"step": 2500, "epoch": 2.616, "loss": 0.6837, "margins": 0.02012,  "accuracy": 0.6219},
    "DPO step 2868": {"step": 2868, "epoch": 3.0,   "loss": 0.6819, "margins": 0.024,    "accuracy": 0.6016},
}

# Preference pairs per optimizer step: 8 GPUs × batch 2 × grad_accum 4 = 64
PAIRS_PER_STEP = 64


CHAT_TEMPLATE = open("checkpoints/sft/basic/chat_template.jinja").read()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    messages = [{"role": "user", "content": prompt}]
    tokenizer.chat_template = CHAT_TEMPLATE
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
    # Strip trailing EOS added by tokenizer — model should generate after the prompt, not after EOS
    if input_ids[0, -1] == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    results = {}
    for label, ckpt_path in CHECKPOINTS:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"SKIP {label}: {ckpt_path} not found", file=sys.stderr)
            continue
        print(f"Loading {label} ...", file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = UnboxForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16
        ).to("cuda:0")
        model.eval()
        responses = []
        for prompt in PROMPTS:
            print(f"  Generating for prompt: {prompt[:50]}...", file=sys.stderr, flush=True)
            responses.append(generate(model, tokenizer, prompt))
        results[label] = responses
        del model
        torch.cuda.empty_cache()

    # Output JSON for report generation
    print(json.dumps({"results": results, "metrics": METRICS, "prompts": PROMPTS,
                      "pairs_per_step": PAIRS_PER_STEP}))


if __name__ == "__main__":
    main()
