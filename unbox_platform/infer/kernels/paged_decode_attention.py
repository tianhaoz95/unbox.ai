"""Paged decode attention kernel (Triton).

Computes attention for a single decode step: each sequence contributes one
query token (the newly generated token) attending over all its KV blocks.

Why a custom kernel
───────────────────
F.scaled_dot_product_attention requires contiguous K/V tensors. During decode,
K/V lives in paged blocks scattered across the KV pool. This kernel iterates
over the block table to gather blocks into the attention computation without
materialising a contiguous K/V buffer.

Interface
─────────
  paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size)

  q           : (batch, num_heads, head_dim)       — one query per sequence
  k_pool      : (num_blocks, block_size, num_kv_heads, head_dim)
  v_pool      : (num_blocks, block_size, num_kv_heads, head_dim)
  block_tables: (batch, max_blocks_per_seq)         — -1 = padding
  seq_lens    : (batch,)                            — actual KV length per seq
  block_size  : int

  returns     : (batch, num_heads, head_dim)

GQA support: num_heads may be a multiple of num_kv_heads. Each KV head serves
kv_repeat = num_heads // num_kv_heads query heads.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Triton kernel — only compiled when triton is available and tensors are on GPU
# ---------------------------------------------------------------------------

def _build_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        Q_ptr, K_pool_ptr, V_pool_ptr, Block_tables_ptr, Seq_lens_ptr, Out_ptr,
        q_stride_b, q_stride_h, q_stride_d,
        kv_stride_blk, kv_stride_tok, kv_stride_h, kv_stride_d,
        bt_stride_b,
        out_stride_b, out_stride_h, out_stride_d,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_blocks_per_seq: tl.constexpr,
        scale: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        seq_len = tl.load(Seq_lens_ptr + batch_idx)

        q_base = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h
        d_range = tl.arange(0, head_dim)
        q = tl.load(q_base + d_range * q_stride_d)

        m = tl.full([1], float("-inf"), dtype=tl.float32)
        l = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([head_dim], dtype=tl.float32)

        num_full_blocks = (seq_len + block_size - 1) // block_size
        for blk in range(max_blocks_per_seq):
            if blk >= num_full_blocks:
                break
            block_idx = tl.load(Block_tables_ptr + batch_idx * bt_stride_b + blk)
            if block_idx < 0:
                break

            tokens_in_block = tl.minimum(block_size, seq_len - blk * block_size)
            for tok in range(block_size):
                if tok >= tokens_in_block:
                    break
                kv_base = (block_idx * kv_stride_blk
                           + tok * kv_stride_tok
                           + kv_head_idx * kv_stride_h)
                k = tl.load(K_pool_ptr + kv_base + d_range * kv_stride_d)
                v = tl.load(V_pool_ptr + kv_base + d_range * kv_stride_d)
                score = tl.sum(q * k, axis=0) * scale

                m_new = tl.maximum(m, score)
                exp_score = tl.exp(score - m_new)
                l = l * tl.exp(m - m_new) + exp_score
                acc = acc * tl.exp(m - m_new) + exp_score * v
                m = m_new

        acc = acc / (l + 1e-8)
        out_base = Out_ptr + batch_idx * out_stride_b + head_idx * out_stride_h
        tl.store(out_base + d_range * out_stride_d, acc.to(tl.float16))

    return _kernel


_triton_kernel = None


def paged_decode_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton-accelerated paged decode attention (GPU only).

    Args:
        q           : (batch, num_heads, head_dim)
        k_pool      : (num_blocks, block_size, num_kv_heads, head_dim)
        v_pool      : (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch, max_blocks_per_seq)  int32, -1 = unused
        seq_lens    : (batch,)  int32
        block_size  : int

    Returns:
        out: (batch, num_heads, head_dim)
    """
    global _triton_kernel
    if _triton_kernel is None:
        _triton_kernel = _build_triton_kernel()

    batch, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k_pool.shape
    max_blocks_per_seq = block_tables.shape[1]
    scale = head_dim ** -0.5
    out = torch.empty_like(q)

    _triton_kernel[(batch, num_heads)](
        q, k_pool, v_pool, block_tables, seq_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_tables.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        scale=scale,
    )
    return out


# ---------------------------------------------------------------------------
# Pure-PyTorch reference — CPU fallback and correctness testing
# ---------------------------------------------------------------------------

def paged_decode_attention_ref(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for testing and CPU fallback."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[2]
    kv_repeat = num_heads // num_kv_heads
    scale = head_dim ** -0.5
    out = torch.zeros_like(q)

    for b in range(batch):
        seq_len = seq_lens[b].item()
        for h in range(num_heads):
            kv_h = h // kv_repeat
            qi = q[b, h]
            scores: list[torch.Tensor] = []
            vs: list[torch.Tensor] = []
            for pos in range(seq_len):
                blk = pos // block_size
                slot = pos % block_size
                block_idx = block_tables[b, blk].item()
                k = k_pool[block_idx, slot, kv_h]
                v = v_pool[block_idx, slot, kv_h]
                scores.append((qi * k).sum() * scale)
                vs.append(v)
            scores_t = torch.stack(scores)
            weights = torch.softmax(scores_t, dim=0)
            vs_t = torch.stack(vs)
            out[b, h] = (weights.unsqueeze(-1) * vs_t).sum(0)
    return out
