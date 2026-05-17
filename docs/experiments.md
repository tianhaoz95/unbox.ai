# Planned Experiments

This is a living list of research experiments to run once the core platform is validated. Each experiment lives (or will live) in `unbox/` and imports primitives from `unbox_platform/`.

## Architecture experiments

| Experiment | Description | Status |
|---|---|---|
| **Mixture of Experts (MoE)** | Replace dense FFN with sparse MoE routing; compare quality vs. compute at equal parameter count | Planned |
| **Sliding window attention** | Replace full causal attention with sliding window (Mistral-style) for long-context efficiency | Planned |
| **Multi-head latent attention (MLA)** | DeepSeek-V2's KV cache compression via low-rank projection | Planned |
| **Diff Transformer** | Replace softmax attention with differential attention for better noise cancellation | Planned |

## Training experiments

| Experiment | Description | Status |
|---|---|---|
| **Muon optimizer** | Orthogonalised gradient descent for the weight matrices; compare convergence vs. AdamW | Planned |
| **YaRN long-context scaling** | Extend effective context length beyond `max_seq_len` via RoPE frequency interpolation | Planned |
| **Warmup-stable-decay (WSD) schedule** | Alternative to cosine decay with a stable phase for mid-training checkpoints | Planned |

## Data experiments

| Experiment | Description | Status |
|---|---|---|
| **Curriculum learning** | Sort training examples by difficulty; train on easy examples first | Planned |
| **Domain upsampling** | Oversample high-quality domains (math, code, science) during training | Planned |
| **Synthetic data augmentation** | Generate synthetic QA pairs from FineWeb-Edu using a teacher model | Planned |

## RL experiments

| Experiment | Description | Status |
|---|---|---|
| **GRPO on math** | Use GRPO with a verifiable reward (exact answer match) to improve mathematical reasoning | Planned |
| **Constitutional AI** | Self-critique loop: model critiques its own responses, then revises | Planned |
| **Length penalty** | Add a length reward signal to reduce verbosity without sacrificing quality | Planned |

## Inference experiments

| Experiment | Description | Status |
|---|---|---|
| **Speculative decoding** | Use a small draft model to propose tokens; verify with the large model in parallel | Planned |
| **Quantization (INT8, FP8)** | Post-training quantization of weights and/or activations; measure quality degradation | Planned |
| **Prefix caching** | Benchmark the KV cache hit rate improvement from radix-tree prefix sharing on real workloads | Planned |

---

!!! tip "Contributing an experiment"
    Create a directory under `unbox/<experiment_name>/` with a self-contained `README.md`, entry point, and config. Import primitives from `unbox_platform/` rather than copying code. An experiment does not need to be generalisable — it only needs to be correct and reproducible.
