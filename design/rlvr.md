# RLVR Design: Disaggregated Rollout + Megatron-Core Training

## What RLVR Is

**RLVR** (Reinforcement Learning with Verifiable Rewards) trains the model to produce outputs that are correct according to a deterministic verifier — rather than a trained reward model or human rater. The canonical tasks are:

- **Math** — exact answer match against a ground-truth numeric answer (GSM8K, MATH)
- **Code** — execution against unit tests (HumanEval, LiveCodeBench)
- **Logic / formal reasoning** — proof verification

The key distinction from standard RLHF: the reward signal is **ground truth**, not a trained approximation. This eliminates reward hacking against a flawed reward model and makes the signal perfectly calibrated. DeepSeek-R1, Qwen-QwQ, and Kimi k1.5 are all trained primarily on RLVR rather than RLHF.

---

## Why a Custom Implementation

TRL's `GRPOTrainer` does rollout **in-process** — the same GPU that runs the forward+backward pass also runs generation. This is simple but has two problems:

1. **Throughput waste** — during rollout, the training framework sits idle; during the backward pass, the inference KV cache is idle. Neither side achieves its maximum utilisation.
2. **No KV cache reuse** — in-process generation uses a naive loop without paged attention or continuous batching. Generating 8–16 responses per prompt is slow.

A disaggregated design runs a **dedicated inference engine** for rollout and a **dedicated training process** for gradient updates. Each achieves high utilisation on its own workload. This is the architecture used by verl, OpenRLHF's latest design, and production RLVR systems at DeepSeek and Qwen.

We implement this with:
- **Rollout**: mini-sglang (initially); our own inference engine (once built)
- **Training**: Megatron-Core (same training backbone as pre-training)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RLVR Training Loop                   │
│                                                         │
│  ┌───────────────────┐      ┌────────────────────────┐  │
│  │  Rollout Engine   │      │   Training Process     │  │
│  │  (mini-sglang /   │      │   (Megatron-Core)      │  │
│  │   unbox infer)    │      │                        │  │
│  │                   │      │  1. Receive rollouts   │  │
│  │  • Paged KV cache │      │  2. Score with reward  │  │
│  │  • Cont. batching │      │  3. Compute advantages │  │
│  │  • Generate N     │◄─────│  4. Forward + backward │  │
│  │    responses/prompt│     │  5. AdamW step         │  │
│  │                   │      │  6. Sync weights →     │  │
│  └───────────────────┘      └────────────────────────┘  │
│           │                           ▲                  │
│           │    rollout buffer         │ weight sync      │
│           └───────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

The two processes communicate via two channels:
- **Rollout buffer** — the inference engine pushes `(prompt, [response_1..N], [reward_1..N])` tuples to a queue consumed by the training process
- **Weight sync** — after each gradient update, training pushes updated weights to the inference engine

---

## Training Algorithm

The policy update algorithm is **GRPO** (Group Relative Policy Optimization), not PPO. Reasons:

- No critic network required — GRPO uses the group mean as a baseline
- Simpler than PPO for verifiable rewards where variance is low
- Avoids the instability of a jointly-trained value function when rewards are sparse/binary
- Used by DeepSeek-R1 and Qwen-QwQ

**GRPO update (per step):**

```
For each prompt p in batch:
    Generate N responses: [r_1, ..., r_N] from policy π_θ
    Score each: [s_1, ..., s_N] = verifier(r_i, p)
    Compute advantage: A_i = (s_i - mean(s)) / std(s)

Policy gradient objective (clipped, PPO-style):
    L = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
        - β * KL(π_θ || π_ref)

where ratio = π_θ(r_i | p) / π_θ_old(r_i | p)
and π_ref is the frozen SFT model (KL reference)
```

---

## Weight Sync Strategy

Three options in increasing complexity. Implement in order.

### Phase 1: Checkpoint-based (implement first)

After each gradient update, the training process saves a checkpoint to a shared path. The inference engine polls for a new checkpoint and reloads weights before the next rollout batch.

```
Training → checkpoint/rlvr/policy_step_{N}.pt
Rollout engine: poll every K steps, reload if new checkpoint found
```

**Pros:** Zero IPC complexity; uses the same checkpoint format as pre-training.  
**Cons:** Adds ~2–5s per sync for a 760M model. Acceptable for research.  
**When to use:** Always, until throughput becomes a bottleneck.

### Phase 2: Shared CUDA memory (single-node)

Training and inference run on the same node. After the optimizer step, parameters are written to a shared memory region; the inference engine maps the same region and reads directly.

```python
# training side
for name, param in model.named_parameters():
    shared_buf[name].copy_(param.data)

# inference side
for name, param in model.named_parameters():
    param.data.copy_(shared_buf[name])
```

**Pros:** Sub-millisecond sync; no disk I/O.  
**Cons:** Single-node only; requires careful synchronisation to avoid reading mid-update.

### Phase 3: NCCL broadcast (multi-node)

Training ranks broadcast updated parameters to inference worker ranks using NCCL all-broadcast. This is what verl uses for multi-node RLVR at scale.

```python
# after optimizer step, on each training rank
for param in model.parameters():
    dist.broadcast(param.data, src=0, group=sync_group)
```

**Pros:** Fast (~seconds for 760M, < 1s for weight-only broadcast); multi-node capable.  
**Cons:** Requires a shared process group between training and inference processes; significant setup complexity.

---

## Rollout Engine Interface

The rollout backend is behind a thin interface so it can be swapped:

```python
class RolloutBackend(Protocol):
    def generate(
        self,
        prompts: list[str],
        n: int,              # responses per prompt
        max_new_tokens: int,
        temperature: float,
    ) -> list[list[str]]:   # [num_prompts, n]
        ...

    def load_weights(self, checkpoint_path: str) -> None:
        ...
```

**Phase 1 — mini-sglang backend:**

```python
class MiniSGLangBackend(RolloutBackend):
    def __init__(self, model_path: str, ...):
        self.engine = sglang.Engine(model_path=model_path, ...)
```

mini-sglang provides paged attention, continuous batching, and RadixAttention prefix caching out of the box. It does not support custom model architectures directly — the model must be exported to HF format first (`UnboxForCausalLM.save_pretrained()`).

**Phase 2 — unbox inference engine backend:**

Once `unbox_platform/infer/` is implemented, replace the backend:

```python
class UnboxInferBackend(RolloutBackend):
    def __init__(self, config: InferConfig):
        self.server = UnboxInferServer(config)
```

This gives us full control: custom attention kernels, block table, disaggregated prefill-decode for the rollout pool.

---

## Reward Function Interface

```python
class RewardFn(Protocol):
    def __call__(
        self,
        prompts: list[str],
        responses: list[str],
        ground_truths: list[str],
    ) -> list[float]:
        ...
```

**Built-in reward functions:**

| Name | Task | Signal |
|---|---|---|
| `exact_match` | Math (GSM8K) | 1.0 if final answer matches, 0.0 otherwise |
| `partial_math` | Math | 0.5 for correct reasoning steps, 1.0 for correct answer |
| `code_execution` | Code (HumanEval) | Fraction of unit tests passing |
| `format_compliance` | Any | 1.0 if output matches required format (JSON, chain-of-thought tags, etc.) |

Reward functions are composable — a weighted sum of `exact_match + format_compliance` is a common starting point.

---

## Comparison with Existing Approaches

| Approach | Rollout | Training | Weight sync | Notes |
|---|---|---|---|---|
| **TRL GRPOTrainer** | In-process | PyTorch | None (same process) | Simplest; low throughput |
| **OpenRLHF PPO** | vLLM | DeepSpeed | NCCL | Requires reward model; full PPO |
| **verl** | vLLM / SGLang | Megatron / FSDP | NCCL | Production-grade; complex setup |
| **This design (Phase 1)** | mini-sglang | Megatron-Core | Checkpoint | Simple, readable, research-grade |
| **This design (Phase 2)** | unbox infer | Megatron-Core | NCCL | Full-stack ownership |

This design occupies the same architectural niche as verl but targets readability over throughput, consistent with the platform philosophy.

---

## Implementation Plan

### Phase 1 (initial)

- `unbox_platform/rl/rlvr/rollout.py` — `RolloutBackend` protocol + `MiniSGLangBackend`
- `unbox_platform/rl/rlvr/reward.py` — `RewardFn` protocol + built-in reward functions
- `unbox_platform/rl/rlvr/buffer.py` — rollout buffer (in-memory queue of `(prompt, responses, rewards)`)
- `unbox_platform/rl/rlvr/trainer.py` — GRPO training loop over Megatron-Core model
- `unbox_platform/rl/rlvr/weight_sync.py` — checkpoint-based sync (Phase 1)
- `configs/rl/rlvr_gsm8k.yaml` — GSM8K config as the reference experiment

Entry point:
```bash
.venv/bin/python -m unbox_platform.rl.rlvr.train --config configs/rl/rlvr_gsm8k.yaml
```

### Phase 2 (after inference engine is built)

- Add `UnboxInferBackend` to `rollout.py`
- Add `NcclWeightSync` to `weight_sync.py`
- Validate end-to-end on GSM8K with the unbox inference engine as rollout backend

---

## Prerequisites

| Prerequisite | Status |
|---|---|
| SFT model (policy initialisation) | In progress |
| mini-sglang installed and tested | Not done |
| GSM8K dataset + answer extraction | Not done |
| Megatron-Core GRPO gradient computation | Not done |
| `UnboxForCausalLM.save_pretrained()` in HF format | Done (via `from_pretrained`) |

---

## Open Questions

- **KL coefficient β:** How strongly to penalise deviation from the SFT reference? Typical values are 0.01–0.1. Too low → reward hacking; too high → no learning.
- **N (responses per prompt):** 8 is standard for GRPO. More reduces variance but increases rollout cost.
- **Reward sparsity on math:** Binary exact-match is very sparse for a 760M model at early training. Is partial-credit reward (correct reasoning steps) needed to get a learning signal?
- **Chain-of-thought format:** Does GSM8K training require enforcing a reasoning format (e.g., `<think>...</think>` tags) or is unstructured generation sufficient?
- **mini-sglang model compatibility:** mini-sglang loads HF models; our `UnboxForCausalLM` must export cleanly and be loadable by mini-sglang's engine. Needs a compatibility test before Phase 1 can start.
