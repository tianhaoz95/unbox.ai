# 4 · Model Architecture

## Overview

The model is a **760M-parameter decoder-only Transformer** — the same family as GPT, Llama, and Qwen. Every component is a standard building block from the modern LLM literature, chosen for readability and correctness rather than novelty.

```
Input token IDs
      ↓
Token embeddings  (vocab_size × hidden_size)
      ↓
24 × TransformerBlock
      ├── RMSNorm
      ├── Grouped Query Attention (GQA) + RoPE
      └── SwiGLU FFN
      ↓
RMSNorm
      ↓
LM head  (hidden_size × vocab_size, tied to embeddings)
      ↓
Logits → loss (cross-entropy) or next token
```

## Configuration

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 1792 | Embedding and residual stream dimension |
| `num_layers` | 24 | Transformer blocks |
| `num_heads` | 16 | Query heads |
| `num_kv_heads` | 8 | Key/value heads (GQA 2:1 ratio) |
| `ffn_intermediate_size` | 4864 | SwiGLU hidden dimension |
| `vocab_size` | 32,768 | |
| `max_seq_len` | 2048 | |
| `rope_theta` | 500,000 | Long-context RoPE base |
| **Total parameters** | **~760M** | |

## Components

### RMSNorm

Pre-normalization applied before each attention and FFN sublayer. Simpler than LayerNorm — no mean subtraction, just RMS scaling:

```python
def forward(self, x):
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm * self.weight
```

Pre-norm (applied before the sublayer, not after) stabilises training at large scale and is standard in all modern LLMs.

### Grouped Query Attention (GQA)

Standard multi-head attention uses one K/V head per Q head — expensive in memory during inference. GQA uses fewer K/V heads shared across multiple Q heads:

- 16 Q heads, 8 KV heads → 2:1 ratio
- KV cache is 2× smaller than full MHA
- Attention quality is negligibly different from MHA for this scale

The K/V heads are expanded to match Q heads via `repeat_interleave` before the attention dot product.

### Rotary Position Embedding (RoPE)

RoPE encodes position by rotating Q and K vectors in the complex plane before the attention dot product. Unlike learned position embeddings, RoPE:

- Generalises to sequence lengths beyond those seen in training
- Is applied directly to Q/K, not added to embeddings
- Preserves relative position information

We use `rope_theta=500,000` (vs. the original 10,000) — a higher base extends effective context length, following the Llama 3 scaling approach.

!!! warning "The `freqs_cis` buffer"
    RoPE frequencies are precomputed as a complex64 buffer (`freqs_cis`). This buffer cannot survive a dtype cast to bfloat16 — PyTorch silently drops imaginary parts, destroying all positional information. `Transformer.to()` is overridden to remove the buffer before casting and recompute it on the correct device.

### SwiGLU Feed-Forward

The FFN uses the SwiGLU activation from the Llama family:

```python
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

SwiGLU has two linear projections (gate and up) gated by a SiLU activation, followed by a down projection. This outperforms standard ReLU FFNs and is the default in Llama, Qwen, and Mistral.

### Tied Embeddings

The input embedding matrix and the output LM head weight matrix are shared (`lm_head.weight = embed_tokens.weight`). This saves ~60M parameters with negligible quality impact.

## HuggingFace adapter

The core `Transformer` class knows nothing about HuggingFace. A separate adapter (`unbox_platform/model/hf_adapter.py`) wraps it in `PreTrainedModel` / `PretrainedConfig` subclasses:

```python
class UnboxForCausalLM(PreTrainedModel):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
```

This lets TRL, PEFT, and standard eval harnesses consume the model without any changes to core model code. The adapter handles:

- `save_pretrained` / `from_pretrained` — standard HF checkpoint format
- Gradient checkpointing — required by TRL's `SFTConfig`
- Weight tying — via `_tied_weights_keys`
- Checkpoint interop — `from_unbox_checkpoint()` loads raw `.pt` training checkpoints

## Gradient checkpointing

`TransformerBlock` supports gradient checkpointing via a flag set by the HF adapter:

```python
def forward(self, x, freqs_cis, mask=None):
    if self.gradient_checkpointing and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward, x, freqs_cis, mask, use_reentrant=False
        )
    return self._forward(x, freqs_cis, mask)
```

This trades compute for memory: activations are recomputed during the backward pass rather than stored. TRL's `SFTConfig` enables it by default.
