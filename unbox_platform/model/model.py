"""Decoder-only transformer for causal language modeling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq/xk: (B, T, n_heads, head_dim)
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_c * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
        return x_out.type_as(x)

    return rotate(xq), rotate(xk)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeat = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        xq = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA: expand KV heads to match Q heads
        if self.kv_repeat > 1:
            xk = xk.repeat_interleave(self.kv_repeat, dim=2)
            xv = xv.repeat_interleave(self.kv_repeat, dim=2)

        # (B, n_heads, T, head_dim) for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config)
        self.gradient_checkpointing = False

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, freqs_cis, mask, use_reentrant=False
            )
        return self._forward(x, freqs_cis, mask)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies; register as buffer so they move with .to(device)
        freqs_cis = precompute_freqs_cis(
            config.head_dim, config.max_seq_len * 2, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def to(self, *args, **kwargs) -> "Transformer":
        # freqs_cis is complex64 and cannot survive a dtype cast to bfloat16/float16
        # (PyTorch silently drops the imaginary part with a warning). Remove it before
        # the cast so super().to() never sees it, then recompute on the correct device.
        self._buffers.pop("freqs_cis", None)
        result = super().to(*args, **kwargs)
        device = next(result.parameters()).device
        freqs_cis = precompute_freqs_cis(
            result.config.head_dim, result.config.max_seq_len * 2, result.config.rope_theta
        )
        result.register_buffer("freqs_cis", freqs_cis.to(device), persistent=False)
        return result

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature != 1.0:
                logits = logits / temperature

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p nucleus
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return input_ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
