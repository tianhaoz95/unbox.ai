# CLAUDE.md / GEMINI.md / AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Vision

unbox.ai is a lightweight LLM research and experimentation platform. The guiding principle is **simplicity over peak performance**: target ~50% of industry throughput/efficiency while keeping implementations minimal enough to iterate rapidly on novel experiments. Every component should be readable, hackable, and self-contained.

## Architecture Overview

The repository has two top-level directories with distinct roles:

```
unbox_platform/   # Stable, fully functional system — the "parts bin"
  model/          # Model architectures + HF-compatible adapter (UnboxForCausalLM / UnboxConfig)
  data/           # Data pipeline (curation, tokenization, dataset loading, batching, packing)
  tokenizer/      # Tokenizer training (BPE, SentencePiece) — produces reusable tokenizer artifacts
  train/          # Pre-training and distributed training (3D parallelism, mixed precision)
  sft/            # Supervised fine-tuning via TRL's SFTTrainer; dataset must have a "messages" column
  rl/             # RL post-training: TRL for offline (DPO, GRPO); OpenRLHF for online PPO
  distill/        # Knowledge distillation: logit matching, hidden state distillation, reasoning transfer
  infer/          # Fully-fledged inference engine: disaggregated prefill-decode, NCCL TP, ZMQ IPC
    kernels/      # Inference-specific Triton kernels (paged decode attention, fused RMSNorm, RoPE, SwiGLU); stays here unless a kernel proves reusable in training
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
- **Build on, don't reinvent**: industry-standard frameworks are first-class dependencies. Use their primitives directly rather than reimplementing them. `/platform` code lives above this layer, not below it. See the Framework Stack section for the resolved choices.
- **Single-file components where possible**: an attention module, a scheduler, a sampler — each should live in one file that can be read and modified independently.

## Framework Stack

Resolved framework choices by subsystem. Do not re-litigate these without a concrete reason.

| Subsystem | Framework | Rationale |
|---|---|---|
| Pre-training | PyTorch + Megatron-Core | TP/PP/SP built-in; no full repo clone needed |
| HF compatibility | `transformers.PreTrainedModel` adapter (`UnboxForCausalLM`) | Lets TRL, PEFT, and eval harnesses consume the model without modifying core model code |
| SFT | TRL `SFTTrainer` + `SFTConfig` | Industry standard for offline supervised fine-tuning; chat-template application is automatic from a "messages" column |
| Offline RL (DPO, GRPO) | TRL | See `design/rl_posttraining.md` |
| Online RL (PPO) | OpenRLHF | See `design/rl_posttraining.md` |
| RLVR (verifiable rewards) | mini-sglang rollout + Megatron-Core | See `design/rlvr.md` |
| Distillation | TRL `GKDTrainer` | See `design/distillation.md` |
| Inference kernels | Triton | Python-native, readable, sufficient at 50% throughput target; do not use CUDA |
| Inference IPC | ZMQ | Frontend-worker and KV-transfer communication; no RDMA, no zero-copy |
| Inference TP | NCCL | Same all-reduce collectives as training TP, applied at serve time |

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

Target: a **fully-fledged industry-standard inference engine**, comparable to SGLang or vLLM in architectural scope. Performance is explicitly not a goal — up to 50% slower than production systems is acceptable. Every major subsystem that exists in production engines must exist here, implemented readably rather than optimally.

### Serving modes

The engine supports two modes, selected via `InferConfig.mode`:

```python
mode: Literal["unified", "disaggregated"] = "unified"
```

**Unified mode** (default) — a single worker pool handles both prefill and decode in the same scheduler loop. The router is trivial (round-robin load balancing). No KV transfer. Right for single-GPU development, small-scale serving, and as the starting point for implementation.

**Disaggregated mode** — separate prefill and decode worker pools, coordinated by a router that tracks KV handoff. Right for multi-node production deployments where prefill (compute-bound) and decode (memory-bandwidth-bound) need independent scaling.

The components that are **identical** in both modes: `config`, `messaging`, `sampling`, `kvcache`, `kernels`, `engine`, `distributed`, `server`. The difference is purely topological — how workers are coordinated, not how they compute.

The components that **branch** on mode:
- `scheduler.py` — unified runs one loop handling both chunked prefill and decode batching; disaggregated splits these into two specialized schedulers
- `worker.py` — reads `worker_type: Literal["unified", "prefill", "decode"]` and sets up accordingly
- `router.py` — disaggregated routes to two pools and tracks KV transfer; unified load-balances one pool and skips KV transfer

Build and validate unified mode end-to-end first, then layer disaggregated on top.

### Architecture: disaggregated prefill-decode

In disaggregated mode, separate worker pools handle the two phases, coordinated by a central router. This reflects how production systems are deployed at scale (prefill is compute-bound, decode is memory-bandwidth-bound; mixing them on the same GPU is suboptimal).

```
HTTP client
    │
    ▼
FastAPI frontend  (unbox_platform/infer/server.py)
    │  ZMQ PUSH/PULL
    ▼
Router / dispatcher  (unbox_platform/infer/router.py)
    │                       │
    ▼                       ▼
Prefill workers         Decode workers
(one or more GPUs)      (one or more GPUs)
    │  KV cache transfer (ZMQ)
    └──────────────────────▶│
                            ▼
                    Token stream back to router → client
```

Each worker is a separate Python process. Workers within a pool use **NCCL** for tensor parallelism (splitting attention heads and FFN columns across GPUs). The frontend and workers communicate exclusively via **ZMQ** (push/pull for requests, pub/sub for results).

### Component map

```
unbox_platform/infer/
  kernels/        # Triton kernels (see below)
  kvcache.py      # Paged KV cache: block allocator, radix-tree prefix cache, LRU eviction
  scheduler.py    # Per-worker scheduler: continuous batching, chunked prefill, decode batching
  engine.py       # Single-worker forward pass: loads model, runs attention + FFN, samples
  worker.py       # Worker process entry point: prefill worker or decode worker
  router.py       # Dispatcher: accepts requests, routes to prefill pool, tracks KV transfer, streams tokens
  server.py       # FastAPI HTTP layer: POST /generate, POST /v1/chat/completions (OpenAI-compatible)
  distributed.py  # NCCL tensor-parallel setup for multi-GPU workers
  messaging.py    # ZMQ socket abstractions (request envelope, KV transfer, result streaming)
  config.py       # InferConfig dataclass: worker counts, TP degree, block size, etc.
  sampling.py     # Greedy, top-k, top-p, temperature sampling
  tokenizer.py    # Prompt tokenization and incremental streaming detokenization
```

### Key subsystem details

**KV cache** — paged/blocked layout (fixed-size blocks, free-list allocator) with a radix tree for prefix caching and LRU eviction. The block table is the interface between the scheduler and the paged decode attention kernel.

**Scheduler** — in unified mode, one scheduler loop handles both chunked prefill and decode batching on the same worker. In disaggregated mode, prefill workers run a chunked-prefill scheduler (fill a compute budget per step) and decode workers run a decode-batching scheduler (batch all in-flight sequences). The router coordinates phase handoff in disaggregated mode only.

**KV transfer** — after a prefill worker finishes a request, it serialises the KV blocks and sends them to an assigned decode worker via ZMQ. This is the disaggregation boundary. Simplicity over speed: no RDMA, no zero-copy, just ZMQ byte transfer.

**Tensor parallelism** — within a worker pool, each GPU rank holds a shard of the weight matrices (column-parallel for Q/K/V/gate projections, row-parallel for output projections). NCCL all-reduce after each row-parallel layer. Same Megatron-style split as training TP, but applied at inference time.

**Tokenizer / detokenizer** — `tokenizer.py` owns both directions of text↔token conversion and runs in the server process (CPU-bound, no process isolation needed at this throughput target). On the input side it tokenizes the prompt text into token IDs before the request is pushed to the router. On the output side it holds a per-request token ID buffer and handles incremental detokenization: because tokens can represent multi-character or split UTF-8 sequences, individual token IDs cannot be decoded independently. The standard approach is to append each new token ID to the buffer and re-decode the full suffix each step, emitting only the newly confirmed text. The buffer is keyed by request ID and lives alongside the SSE emission loop in the server. The tokenizer wraps `unbox_platform/tokenizer/` (or a HuggingFace `PreTrainedTokenizerFast`) and is loaded once at server startup.

**Server** — FastAPI with OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`). Streaming via SSE. Frontend is async; model work is in worker processes. The server owns the tokenizer instance and the per-request detokenization buffers.

### Kernel language: Triton

All custom kernels are written in **Triton** (not CUDA). Triton is Python-native, readable without GPU architecture expertise, and sufficient for the ~50% throughput target. The performance gap vs. hand-tuned CUDA only appears in the last ~10% of optimization headroom, which is not a goal here.

Do not write CUDA kernels. If a kernel cannot be expressed cleanly in Triton, that is a signal the abstraction is too complex for this codebase.

### Kernel placement

Custom ops live in `unbox_platform/infer/kernels/`. Training uses PyTorch's built-in `scaled_dot_product_attention` (which dispatches to Flash Attention) and needs no custom kernels at this throughput target. Inference-specific ops — paged attention with block-table indirection, fused RMSNorm, fused RoPE, fused SwiGLU — stay inside `infer/kernels/` because they have no training consumer. If a kernel later proves reusable across training and inference, graduate it to a top-level `unbox_platform/kernels/` package at that point.

**Do not reimplement Flash Attention.** `F.scaled_dot_product_attention` already dispatches to it for prefill. The kernel that must be written is paged decode attention — single-token Q attending to a block-table KV cache — which PyTorch's SDPA cannot express.

**Planned kernel roadmap** (in priority order):
1. Paged decode attention — unblocks continuous batching
2. Fused RMSNorm — eliminates a memory roundtrip on every norm layer
3. Fused SwiGLU (silu + elementwise multiply) — simple, meaningful gain in the FFN
4. Fused RoPE — applies rotation in-place to Q/K before attention

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for environment management. The virtual environment lives at `.venv/` in the repo root.

**All commands must use the `.venv` binaries directly — never the system Python or any globally installed tools.** Prefix every `python`, `pytest`, `ruff`, `mypy`, and `torchrun` invocation with `.venv/bin/` (or activate first). This ensures the correct package versions are used and avoids contaminating or depending on the system environment.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all deps including dev
uv sync --extra dev

# Activate the environment (optional — activating means you can omit .venv/bin/ prefixes)
source .venv/bin/activate

# Install in editable mode with all dev deps (alternative to uv sync)
uv pip install -e ".[dev]"

# Optional: install wandb for experiment tracking
uv pip install -e ".[logging]"

# Run all tests (use venv pytest)
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/platform/model/test_model.py -v

# Run tests matching a pattern
.venv/bin/pytest tests/ -k "test_model_forward" -v

# Lint + type check (use venv ruff/mypy)
.venv/bin/ruff check unbox_platform/ unbox/
.venv/bin/mypy unbox_platform/ unbox/
```

## Running Experiments

Platform entry points — always use `.venv/bin/python` and `.venv/bin/torchrun`:

```bash
# Pre-training — wandb stores data locally; sync to cloud after training with: wandb sync wandb/
WANDB_MODE=offline .venv/bin/torchrun --nproc_per_node=8 -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

# SFT — same offline wandb pattern; torchrun handles 8-GPU DDP via HF Trainer
WANDB_MODE=offline .venv/bin/torchrun --nproc_per_node=8 -m unbox_platform.sft.train --config configs/sft/basic.yaml

# RL post-training
.venv/bin/python -m unbox_platform.rl.train --config configs/rl/ppo.yaml

# Upload wandb offline runs after training
.venv/bin/wandb sync wandb/

# Inference server
.venv/bin/python -m unbox_platform.infer.server --config configs/infer/serve.yaml

# Download FineWeb-Edu sample-10BT
.venv/bin/python -m unbox_platform.data.prepare --output data/fineweb_edu_10bt.jsonl

# Train tokenizer
.venv/bin/python -m unbox_platform.tokenizer.train --data data/fineweb_edu_10bt.jsonl --output checkpoints/tokenizer

# Evaluate perplexity
.venv/bin/python -m unbox_platform.eval.perplexity --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer checkpoints/tokenizer --config configs/pretrain/760m.yaml

# Qualitative sampling
.venv/bin/python -m unbox_platform.eval.sample --checkpoint checkpoints/pretrain/760m/latest.pt \
    --tokenizer checkpoints/tokenizer --config configs/pretrain/760m.yaml \
    --prompt "The theory of relativity states that"
```

`/unbox` experiments have their own entry points and configs colocated within the experiment directory.

## Testing Conventions

- Unit tests mirror the source tree: `tests/platform/model/` tests `unbox_platform/model/`, etc.
- Tests that require GPUs are marked `@pytest.mark.gpu` and skipped in CPU-only CI.
- For distributed tests, use `torch.distributed` with `gloo` backend on CPU to keep them runnable without a multi-GPU machine.
- Numerical correctness tests compare against a naive reference implementation, not against HuggingFace outputs (avoids version drift).

## ModelScope Checkpoint Upload and Download

Checkpoints are published to ModelScope for cross-machine resume and sharing.
ModelScope is installed in the venv: `.venv/bin/modelscope`.

**Published models:**
- `tianhaoz95/unbox-760m-base` — pretrain checkpoint (`latest.pt`, native format)
- `tianhaoz95/unbox-760m-sft` — SFT checkpoint (HF format, step 3000)

**Upload a checkpoint:**
```bash
# Upload a single file (e.g. pretrain latest.pt)
.venv/bin/modelscope upload tianhaoz95/unbox-760m-base \
    checkpoints/pretrain/760m/latest.pt latest.pt \
    --commit-message "description"

# Upload an entire directory (e.g. HF-format SFT checkpoint)
.venv/bin/modelscope upload tianhaoz95/unbox-760m-sft \
    checkpoints/sft/basic/checkpoint-3000 . \
    --commit-message "description"
```

The `upload` command creates the repo automatically if it does not exist.

**Download and resume training on another machine:**
```bash
# Pretrain resume — downloads latest.pt + tokenizer
.venv/bin/modelscope download tianhaoz95/unbox-760m-base \
    --local_dir checkpoints/pretrain/760m
mkdir -p checkpoints/tokenizer
cp checkpoints/pretrain/760m/tokenizer/* checkpoints/tokenizer/
# The pretrain script auto-detects the checkpoint and skips already-seen data
.venv/bin/python -m unbox_platform.train.pretrain --config configs/pretrain/760m.yaml

# SFT resume — downloads full HF checkpoint directory
.venv/bin/modelscope download tianhaoz95/unbox-760m-sft \
    --local_dir checkpoints/sft/basic/checkpoint-3000
# HF Trainer auto-resumes from step 3000 including mid-epoch data skip
WANDB_MODE=offline .venv/bin/torchrun --nproc_per_node=8 -m unbox_platform.sft.train --config configs/sft/basic.yaml
```

Both checkpoints include optimizer and scheduler state — training resumes exactly
where it left off. The pretrain skip position is derived from the saved step count;
the SFT skip uses HF Trainer's native `skip_first_batches`.

## Publishing Reports and Design Docs to GitHub Pages

`docs/reports` and `docs/design` are symlinks to the root-level `reports/` and `design/` directories. Any `.md` file added to either directory is automatically included in the site on the next push — no copying, no nav entries required.

To publish:
```bash
# just add the file and push
git add reports/my_report.md
git commit -m "add report"
git push
```

GitHub Actions rebuilds and deploys automatically.

## Key Tensions to Navigate

- **Research speed vs. correctness**: prefer readable reference implementations first; optimize only when a bottleneck is proven.
- **Generality vs. simplicity**: resist adding config knobs for hypothetical use cases. Add them when a second experiment actually needs them.
- **Logging**: use structured logging (key=value pairs) so experiment results are easy to parse and compare without a full MLflow/W&B setup.
