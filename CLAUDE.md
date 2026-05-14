# CLAUDE.md / GEMINI.md / AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Vision

unbox.ai is a lightweight LLM research and experimentation platform. The guiding principle is **simplicity over peak performance**: target ~50% of industry throughput/efficiency while keeping implementations minimal enough to iterate rapidly on novel experiments. Every component should be readable, hackable, and self-contained.

## Architecture Overview

The repository has two top-level directories with distinct roles:

```
unbox_platform/   # Stable, fully functional system — the "parts bin"
  model/          # Model architectures (Transformer variants, attention mechanisms, positional encodings)
  data/           # Data pipeline (curation, tokenization, dataset loading, batching, packing)
  tokenizer/      # Tokenizer training (BPE, SentencePiece) — produces reusable tokenizer artifacts
  train/          # Pre-training and distributed training (3D parallelism, mixed precision)
  sft/            # Supervised fine-tuning: data formatting, loss masking, training loop
  rl/             # RL post-training: PPO, DPO, GRPO, reward modeling
  distill/        # Knowledge distillation: logit matching, hidden state distillation, reasoning transfer
  infer/          # Inference server (continuous batching, KV cache management, sampling)
  eval/           # Evaluation: perplexity, benchmark harness, model comparison
  utils/          # Logging, config, profiling

unbox/            # Live research lab — experiments with unknown or non-universal outcomes
  <paper_or_topic>/   # Each experiment is self-contained
```

Note: the package is named `unbox_platform` (not `platform`) to avoid shadowing Python's stdlib `platform` module.

**`unbox_platform/`** is a fully functional system comparable to Megatron-LM or SGLang in scope, but designed as Lego pieces: every subsystem works end-to-end, and every component is individually importable. The primary target is autoregressive language models.

**`unbox/`** is where cutting-edge research lives. An experiment imports primitives from `unbox_platform`, assembles them with minimal boilerplate, and runs. `unbox/` code is not held to the same quality or generality bar as `unbox_platform/` — it can be messy, half-validated, or domain-specific (e.g., diffusion LMs). It is expected to stay in `unbox/` permanently if its results are correct but not reusable across the core AR platform.

### Component Granularity in `unbox_platform/`

Components are provided at two levels:
- **Atoms**: individual ops (`RMSNorm`, `RotaryEmbedding`, `CausalSelfAttention`, `FFNSwiGLU`)
- **Molecules**: standard compositions (`TransformerBlock`, `LLaMA`, `GPT2`) built from atoms

Molecules serve as reference implementations. The atoms are the real value — an `unbox/` experiment can swap one atom out of a molecule with minimal code.

### Graduation from `unbox/` to `unbox_platform/`

A component graduates when:
1. It is reusable across multiple AR experiments (not just correct in isolation)
2. It has been validated with confirmed results
3. Its presence simplifies future `unbox/` work rather than adding complexity

Components that are correct but domain-specific (e.g., useful only for diffusion models, not AR) **do not graduate**. The boundary is about generality to the core platform, not quality.

**Complexity audit**: periodically review whether `unbox_platform/` primitives are actually being reused across `unbox/` experiments. Primitives that aren't being pulled out are candidates for removal or consolidation.

## Design Principles

- **Minimal abstractions**: prefer flat, explicit code over deep class hierarchies. A researcher should be able to read any file top-to-bottom and understand what it does.
- **Config-driven**: all experiments are specified via YAML/TOML config files (no argparse spaghetti). Configs are typed with `dataclasses` or Pydantic.
- **Build on, don't reinvent**: industry-standard frameworks (TBD after high-level design) are first-class dependencies. Use their primitives directly (checkpointing, mixed precision, distributed collectives) rather than reimplementing them. `/platform` code lives above this layer, not below it.
- **Single-file components where possible**: an attention module, a scheduler, a sampler — each should live in one file that can be read and modified independently.

## Parallelism Strategy

The platform targets **3D parallelism** as used in industry for models at the 30B+ scale. All three axes must be composable from the start — retrofitting TP or PP into a data-parallel-only design requires rewriting the training loop.

| Strategy | Purpose | Implementation target |
|---|---|---|
| Data Parallel (ZeRO) | Shard optimizer state, gradients, parameters across DP ranks | DeepSpeed ZeRO or FSDP2 (TBD) |
| Tensor Parallel (TP) | Split individual layers (attention heads, FFN columns) across GPUs within a node | Megatron-LM-style column/row linear splits |
| Pipeline Parallel (PP) | Partition layers across nodes; overlap compute and communication | 1F1B schedule |
| Sequence Parallel | Distribute sequence dimension for long-context training | Ring attention or Ulysses |

The MVP uses DDP for simplicity. The parallelism abstraction (`setup_model(model, parallel_config)`) must be designed so TP and PP are additive, not architectural rewrites.

**Framework stack: PyTorch + Megatron-Core.** `megatron-core` (PyPI) is a standalone library extracted from Megatron-LM that provides TP, PP, SP, and a built-in distributed optimizer (ZeRO-2/3 equivalent) — no full repo clone, no DeepSpeed dependency. This keeps the dependency tree shallow while covering everything needed up to 30B+ scale. Using it from the MVP ensures the parallelism abstraction is designed correctly from the start.

## Inference Infrastructure

Target: a simplified vLLM/SGLang-style server.
- **PagedAttention**-style KV cache (blocks of fixed size, free-list allocator)
- **Continuous batching**: scheduler that fills a batch with waiting + running sequences each step
- **Sampling**: greedy, top-p, top-k, temperature — no exotic samplers unless research needs it
- Serve via a minimal FastAPI endpoint; no need for production-grade OpenAI-compatible server

## Development Setup

```bash
# Install in editable mode with all dev deps
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/platform/model/test_attention.py -v

# Run tests matching a pattern
pytest tests/ -k "test_flash_attn" -v

# Lint + type check
ruff check unbox_platform/ unbox/
mypy unbox_platform/ unbox/
```

## Running Experiments

Platform entry points:

```bash
# Pre-training
python -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

# SFT
python -m unbox_platform.sft.train --config configs/sft/basic.yaml

# RL post-training
python -m unbox_platform.rl.train --config configs/rl/ppo.yaml

# Distributed launch (torchrun)
torchrun --nproc_per_node=8 -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

# Inference server
python -m unbox_platform.infer.server --config configs/infer/serve.yaml

# Download FineWeb-Edu sample-10BT
python -m unbox_platform.data.prepare --output data/fineweb_edu_10bt.jsonl

# Train tokenizer
python -m unbox_platform.tokenizer.train --data data/fineweb_edu_10bt.jsonl --output checkpoints/tokenizer

# Evaluate perplexity
python -m unbox_platform.eval.perplexity --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer checkpoints/tokenizer --config configs/pretrain/760m.yaml

# Qualitative sampling
python -m unbox_platform.eval.sample --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer checkpoints/tokenizer --config configs/pretrain/760m.yaml \
    --prompt "The theory of relativity states that"
```

`/unbox` experiments have their own entry points and configs colocated within the experiment directory.

## Testing Conventions

- Unit tests mirror the source tree: `tests/platform/model/` tests `unbox_platform/model/`, etc.
- Tests that require GPUs are marked `@pytest.mark.gpu` and skipped in CPU-only CI.
- For distributed tests, use `torch.distributed` with `gloo` backend on CPU to keep them runnable without a multi-GPU machine.
- Numerical correctness tests compare against a naive reference implementation, not against HuggingFace outputs (avoids version drift).

## Key Tensions to Navigate

- **Research speed vs. correctness**: prefer readable reference implementations first; optimize only when a bottleneck is proven.
- **Generality vs. simplicity**: resist adding config knobs for hypothetical use cases. Add them when a second experiment actually needs them.
- **Logging**: use structured logging (key=value pairs) so experiment results are easy to parse and compare without a full MLflow/W&B setup.
