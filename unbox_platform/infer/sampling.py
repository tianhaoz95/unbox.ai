"""Token sampling strategies — pure tensor ops, no I/O."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .messaging import SamplingParams


def sample(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Sample next tokens from a batch of logits.

    Args:
        logits: (batch, vocab_size) — raw pre-softmax scores
        params: shared sampling hyperparameters for this batch

    Returns:
        token_ids: (batch,) int64
    """
    if params.temperature == 0.0:
        return logits.argmax(dim=-1)

    if params.temperature != 1.0:
        logits = logits / params.temperature

    if params.top_k > 0:
        logits = _top_k_filter(logits, params.top_k)

    if params.top_p < 1.0:
        logits = _top_p_filter(logits, params.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    threshold, _ = torch.topk(logits, k, dim=-1)
    min_threshold = threshold[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # remove tokens once cumulative prob exceeds p (shift by 1 to keep the token that crosses)
    remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
    # scatter back to original ordering
    return torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
