# 8 · Inference Engine

## Goals

The inference engine targets **architectural parity with SGLang/vLLM**, not performance parity. Up to 50% slower than production systems is acceptable — every subsystem that exists in a production engine must exist here, implemented readably rather than optimally.

## Architecture: disaggregated prefill-decode

The engine separates the two phases of LLM inference into distinct worker pools:

```
HTTP client
    │
    ▼
FastAPI frontend  (server.py)
    │  ZMQ PUSH/PULL
    ▼
Router  (router.py)
    │                     │
    ▼                     ▼
Prefill workers       Decode workers
(compute-bound)       (memory-bandwidth-bound)
    │  KV cache transfer (ZMQ)
    └─────────────────────▶│
                           ▼
                   Token stream → router → client
```

**Why disaggregate?**

- **Prefill** processes the full prompt in parallel — one forward pass over T tokens. It is compute-bound (GPU utilisation ≈ 100%).
- **Decode** generates one token per step, attending to the growing KV cache. It is memory-bandwidth-bound (GPU utilisation ≈ 10–20%).

Mixing both on the same GPU means the memory-bound decode step wastes the GPU's compute capacity. Separate pools allow each to be scaled independently.

## KV cache: paged allocation

Storing the KV cache as one contiguous buffer per sequence wastes GPU memory when sequences have different lengths. The paged approach, introduced by vLLM, treats the KV cache like virtual memory:

- A pool of fixed-size **blocks** (e.g., 16 tokens × num_kv_heads × head_dim) is pre-allocated
- Each sequence has a **block table** mapping logical positions to physical block indices
- Completed sequences release their blocks back to the free list

A **radix tree** sits on top of the block allocator for **prefix caching** — if two requests share the same prompt prefix, they share the same KV blocks (no recomputation).

```
Free list: [block 0, block 3, block 7, ...]

Sequence A: [block 1][block 4][block 9]
Sequence B: [block 1][block 4][block 2]  ← shares prefix blocks with A
```

## Scheduler: continuous batching

Without continuous batching, the engine would either serve one request at a time or wait for a full batch — both waste throughput. Continuous batching allows new requests to join mid-batch as existing sequences complete:

1. Each step, the scheduler fills a batch: prefill tokens for new requests, one decode token per in-flight sequence
2. After the forward pass, completed sequences (hit EOS or `max_tokens`) are retired
3. New requests immediately take their slots

**Chunked prefill** splits long prompts across multiple steps to prevent a single large prefill from blocking the decode queue.

## Tensor parallelism: NCCL

Within each worker pool, GPUs split the model weights:

- **Column-parallel**: Q, K, V, gate, up projections split across GPUs along the output dimension
- **Row-parallel**: O, down projections split along the input dimension, followed by NCCL all-reduce

This is the same Megatron-style TP used in training, applied at serve time.

## IPC: ZMQ

All inter-process communication uses ZMQ:

- **Frontend → router**: PUSH/PULL socket pair, one message per request
- **Router → prefill workers**: distributes requests to the prefill pool
- **Prefill → decode**: serialised KV block transfer after prefill completes
- **Decode → router**: token stream as each token is generated

Simplicity over speed: no RDMA, no zero-copy, just ZMQ byte transfer.

## Triton kernels

Custom operations live in `unbox_platform/infer/kernels/`, written in **Triton** (not CUDA).

### Paged decode attention

The critical kernel: single-token Q attending to a block-table KV cache. PyTorch's `scaled_dot_product_attention` cannot express this because it has no concept of a block table.

```
Q: [1, num_heads, head_dim]
KV blocks: [num_blocks, 2, block_size, num_kv_heads, head_dim]
Block table: [seq_len // block_size]
→ attention output: [1, num_heads, head_dim]
```

### Fused RMSNorm

Fuses the RMS computation and weight multiplication into one kernel — eliminates a memory roundtrip on every norm layer.

### Fused SwiGLU

Fuses `silu(gate) * up` into one kernel, saving a write and read of the intermediate activation tensor.

### Fused RoPE

Applies the rotary embedding in-place to Q and K before attention, saving a write+read of the full activation tensors.

!!! info "Flash Attention for prefill"
    Prefill uses standard `F.scaled_dot_product_attention(is_causal=True)`, which dispatches to Flash Attention via PyTorch's C++ backend. No custom kernel needed for prefill.

## Component map

```
unbox_platform/infer/
  kernels/
    paged_attention.py   # Triton paged decode attention
    rms_norm.py          # Triton fused RMSNorm
    swiglu.py            # Triton fused SwiGLU
    rope.py              # Triton fused RoPE
  kvcache.py             # Block allocator, radix tree, LRU eviction
  scheduler.py           # Per-worker continuous batching loop
  engine.py              # Single-worker forward pass
  worker.py              # Prefill / decode worker process entry point
  router.py              # Request dispatcher, KV transfer coordinator
  server.py              # FastAPI: POST /v1/chat/completions, GET /v1/models
  distributed.py         # NCCL TP setup
  messaging.py           # ZMQ socket abstractions
  sampling.py            # Greedy, top-k, top-p, temperature
  config.py              # InferConfig dataclass
```

## Reference

The KV cache design (radix tree with LRU eviction) and the prefill/decode scheduler split are informed by [mini-sglang](https://github.com/sgl-project/mini-sglang). Mini-sglang delegates attention to FlashInfer — we write our own Triton kernel instead.

## Status

Planned. Implementation will begin after pre-training and SFT are validated.
