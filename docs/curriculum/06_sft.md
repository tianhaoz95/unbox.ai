# 6 · Supervised Fine-Tuning (SFT)

## What SFT does

Supervised fine-tuning teaches the base model to follow instructions. The model is trained on a dataset of (prompt, response) pairs, and the loss is computed only on the response tokens — the model is rewarded for producing the correct response given the prompt.

After SFT, a model that was generating expository FineWeb-Edu text will instead answer questions, follow formats, and stay on-topic.

## What SFT cannot do

**SFT teaches format, not facts.** If the base model never absorbed a fact during pre-training, SFT cannot inject it. A model that doesn't know Beijing is China's capital will, after SFT, give a fluent, on-topic, confidently wrong answer about China's capital.

This is the core reason pre-training matters — SFT is a thin layer on top of whatever knowledge the base model already has.

## Framework: TRL

We use [TRL](https://github.com/huggingface/trl)'s `SFTTrainer` + `SFTConfig`. TRL is the industry standard for offline supervised fine-tuning, providing:

- Gradient checkpointing
- Mixed precision
- Evaluation loops
- W&B integration
- Chat template application (automatic from a `"messages"` column)

The alternative — a custom training loop — would duplicate TRL's infrastructure for no benefit.

## Dataset: UltraChat 200k

[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) is a dataset of ~200k multi-turn conversations curated for instruction tuning. Each example has a `"messages"` column containing a list of `{"role": ..., "content": ...}` turns.

TRL applies our ChatML chat template automatically when it sees the `"messages"` column:

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

!!! note "Dataset column filtering"
    UltraChat 200k has three columns: `"prompt"`, `"prompt_id"`, and `"messages"`. TRL 1.4 detects the `"prompt"` column and tries the prompt+completion path instead of the chat-template path — causing a `KeyError: 'completion'`. Fix: strip to `["messages"]` only before passing to the trainer.

## Training configuration

| Parameter | Value |
|---|---|
| Base checkpoint | `checkpoints/pretrain/760m/latest.pt` |
| Dataset | UltraChat 200k (`train_sft` / `test_sft`) |
| Epochs | 1 |
| Batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Max LR | 2e-5 (10× lower than pre-training) |
| LR schedule | Cosine |
| Warmup steps | 100 |
| Max sequence length | 2048 |
| Dtype | bfloat16 |

The learning rate is much lower than pre-training (2e-5 vs. 3e-4) — we want to adjust the model's behaviour, not overwrite its pre-trained representations.

## Running SFT

```bash
.venv/bin/python -m unbox_platform.sft.train --config configs/sft/basic.yaml
```

## The experimental rationale

We ran SFT at step 15,000 of pre-training (~20% of the token budget) as an experiment — not because the base model is fully trained, but to answer the question: *how much pre-training is needed before SFT produces useful results?*

If SFT at step 15,000 produces usable instruction-following behaviour, we can stop pre-training early. If not, we resume pre-training from the step 15,000 checkpoint and try again later.

The cost of the experiment is a few hours. The upside is potentially saving days of pre-training compute.

## W&B tracking

SFT metrics are logged to the `unbox-ai-sft` W&B project, separate from the pre-training project. Key metrics: `train/loss`, `train/mean_token_accuracy`, `eval/loss`.
