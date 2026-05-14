# unbox.ai

A lightweight LLM research and experimentation platform. The guiding principle is **simplicity over peak performance** — target ~50% of industry throughput while keeping implementations minimal enough to iterate rapidly on novel experiments.

## Structure

```
unbox_platform/   # Stable, fully functional system — the "parts bin"
  model/          # Transformer architecture (RMSNorm, RoPE, GQA, SwiGLU, Flash Attention)
  tokenizer/      # BPE tokenizer training and wrapper
  data/           # Pretraining data pipeline (JSONL, HuggingFace datasets)
  train/          # Pretraining loop with Megatron-Core parallelism
  eval/           # Perplexity evaluation and qualitative sampling
  sft/            # (planned) Supervised fine-tuning
  rl/             # (planned) RL post-training: PPO, DPO, GRPO
  distill/        # (planned) Knowledge distillation
  infer/          # (planned) Inference server

unbox/            # Live research lab — self-contained experiments
configs/          # YAML configs for all training runs
tests/            # Unit tests mirroring unbox_platform/
third_party/      # Reference implementations (minimind, etc.)
design/           # Architecture and design documents
```

## Setup

We use [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync --extra dev

# Activate
source .venv/bin/activate
```

Requires Python ≥ 3.10. For GPU training, ensure your system has a CUDA-compatible PyTorch installed — `uv sync` will pull the CPU build by default; override with:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

To enable [Weights & Biases](https://wandb.ai) experiment tracking:

```bash
uv pip install -e ".[logging]"
wandb login
```

Then set `use_wandb: true` in your training config (e.g. `configs/pretrain/760m.yaml`). Metrics logged: `train/loss`, `train/lr`, `train/tokens_per_sec`, `eval/loss`, `eval/perplexity`.

## MVP: Pretraining a ~760M Parameter Model

### 1. Download data

```bash
python -m unbox_platform.data.prepare --output data/fineweb_edu_10bt.jsonl
```

Downloads FineWeb-Edu `sample-10BT` (~10B tokens, ~30-40GB) from HuggingFace.

### 2. Train tokenizer

```bash
python -m unbox_platform.tokenizer.train \
    --data data/fineweb_edu_10bt.jsonl \
    --output checkpoints/tokenizer
```

Trains a 32K ByteLevel BPE tokenizer on the corpus.

### 3. Pretrain

```bash
# Single GPU
python -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

# Multi-GPU (e.g. 8 GPUs on DGX Spark)
torchrun --nproc_per_node=8 -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml
```

Target config (`configs/pretrain/760m.yaml`): hidden=1792, 24 layers, GQA heads=16/8, vocab=32K, seq=2048, ~760M parameters. Expected to complete in ~72 hours on a DGX Spark.

### 4. Evaluate

```bash
# Perplexity on held-out split
python -m unbox_platform.eval.perplexity \
    --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer  checkpoints/tokenizer \
    --config     configs/pretrain/760m.yaml

# Qualitative sampling
python -m unbox_platform.eval.sample \
    --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer  checkpoints/tokenizer \
    --config     configs/pretrain/760m.yaml \
    --prompt     "The theory of relativity states that"
```

## Running Tests

```bash
# All tests (GPU tests auto-skipped if no CUDA)
pytest tests/

# Single file
pytest tests/platform/model/test_model.py -v

# Pattern match
pytest tests/ -k "test_attention" -v
```

## Design Philosophy

- **`unbox_platform/`** is the stable parts bin — fully functional, tested, Lego-piece granularity. Primarily targets autoregressive LMs.
- **`unbox/`** is the live lab — messy, half-validated, domain-specific experiments are fine here. Import atoms from `unbox_platform`, assemble, run.
- A component graduates from `unbox/` to `unbox_platform/` when it is reusable across multiple AR experiments and validated. Domain-specific results (e.g. diffusion LM attention) stay in `unbox/` permanently.
- Framework stack: **PyTorch + Megatron-Core** for 3D parallelism (TP + PP + ZeRO) up to 30B+ scale.

See [`design/mvp_analysis.md`](design/mvp_analysis.md) for the full MVP rationale and [`CLAUDE.md`](CLAUDE.md) for agent guidance.
