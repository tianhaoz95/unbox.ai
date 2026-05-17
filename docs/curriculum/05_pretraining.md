# 5 · Pre-training

## What pre-training does

Pre-training is the process of teaching the model to predict the next token across a massive corpus of text. The model sees billions of tokens and adjusts its weights to assign higher probability to likely continuations. By the end of pre-training, the model has compressed the statistical patterns of the training corpus into its parameters — it can generate fluent text but has no concept of following instructions.

## Training configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 4 | Per-device |
| Gradient accumulation | 8 | Effective batch = 32 sequences |
| Max sequence length | 2048 | |
| Max learning rate | 3e-4 | Standard for ~1B models |
| LR schedule | Cosine with warmup | |
| Warmup steps | 2,000 | |
| Min LR ratio | 0.1 | Floor at 3e-5 |
| Weight decay | 0.1 | |
| β₁, β₂ | 0.9, 0.95 | AdamW betas |
| Gradient clip | 1.0 | |
| Dtype | bfloat16 | |
| Optimizer | AdamW (fused) | CUDA fused kernel when available |

## Learning rate schedule

```
LR
▲
│      ╭──────────────────────────╮
│     ╱                            ╲
│    ╱                              ╲___________
│___╱
└──────────────────────────────────────────────▶ step
   warmup (2k)        cosine decay             floor
```

The cosine schedule smoothly decays the learning rate from `max_lr` to `max_lr × min_lr_ratio`. The warmup period prevents large gradient updates at the start of training when the model weights are random.

## Mixed precision: bfloat16

Training uses bfloat16 for forward and backward passes via `torch.amp.autocast`. bfloat16:

- Has the same exponent range as float32 → no overflow/underflow issues
- Requires half the memory of float32 → larger batches
- Is natively accelerated on modern NVIDIA GPUs (A100, H100, GB10)

The optimizer states (momentum, variance) are kept in float32 for numerical stability.

## Distributed training

The training loop uses Megatron-Core for distributed state management. The current run uses DDP (data parallel) with a single GPU — the `setup_model()` abstraction is designed so tensor and pipeline parallelism are additive later.

## Checkpointing

Checkpoints are saved every 1,000 steps to `checkpoints/pretrain/760m/`. Each checkpoint contains:

- Model state dict
- Optimizer state dict
- Step and epoch counters
- Latest loss value

A `latest.json` pointer tracks the most recent checkpoint for easy resume.

**Resume correctness:** on resume, the number of chunks already consumed (`step × batch_size × grad_accumulation_steps`) is computed and the dataset iterator skips ahead, ensuring no data is seen twice within an epoch.

## Training progress

| Step | Eval Loss | Perplexity | Notes |
|---|---|---|---|
| 2,500 | 3.89 | 49.0 | Early — loss falling fast |
| 5,000 | 3.48 | 32.3 | Warmup complete |
| 7,500 | 3.31 | 27.3 | |
| 10,000 | 3.22 | 25.1 | First sanity check — coherent text |
| 15,000 | 3.10 | 22.2 | Second sanity check — steady decline |

## Sanity checks

We stop training periodically and sample from the checkpoint to verify the model is producing coherent text. See the [reports section](../reports/sanity_check_step10000.md) for full results.

**Step 10,000 sample** (`The best way to learn programming is`):
> *The Science of Teaching, by Christine R. Noyes, Ph.D., is one of the best scientific curricula. Students who are interested in the science of teaching and learning… will be well prepared to pursue their studies in this highly interactive and effective environment.*

Grammatically correct, on-topic style. Factual hallucination expected at this stage.

## What pre-training produces

A **base model**: it can generate fluent, coherent text but has no concept of instruction following. Prompting it with a question produces more text in the style of the question, not an answer. Instruction following requires SFT.
