"""Single-worker inference engine.

Loads the model and KV cache, then exposes two forward methods:
  - prefill_step: process a prompt chunk, write KV, return logits for last token
  - decode_step: process a batch of single decode tokens using paged attention

The engine is stateless with respect to sequences — all sequence state lives in
the scheduler and KV cache. Engine methods accept explicit arguments for the
tokens to process and the KV cache slots to write.

Paged decode attention
──────────────────────
During decode we cannot use F.scaled_dot_product_attention because K/V lives
in scattered paged blocks. We use either the Triton kernel (GPU) or the
reference implementation (CPU / no Triton).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.hf_adapter import UnboxConfig, UnboxForCausalLM
from ..model.model import Transformer, apply_rotary_emb
from .config import InferConfig
from .kvcache import KVPool, PagedKVCache, SequenceState
from .sampling import SamplingParams, sample


def _load_dtype(dtype_str: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


class InferenceEngine:
    """Wraps the Transformer for inference with a paged KV cache."""

    def __init__(self, config: InferConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.dtype = _load_dtype(config.dtype)

        self.model = self._load_model()
        self.model_config: ModelConfig = self.model.model.config

        self.kvcache = PagedKVCache(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            num_blocks=config.max_num_blocks,
            block_size=config.block_size,
            device=device,
            dtype=self.dtype,
        )

        # try to import Triton kernel; fall back to reference on CPU
        self._use_triton = device.type == "cuda"
        if self._use_triton:
            try:
                from .kernels.paged_decode_attention import paged_decode_attention
                self._paged_attn = paged_decode_attention
            except Exception:
                self._use_triton = False

        if not self._use_triton:
            from .kernels.paged_decode_attention import paged_decode_attention_ref
            self._paged_attn = paged_decode_attention_ref

    def _load_model(self) -> UnboxForCausalLM:
        import yaml
        with open(self.config.model_config_path) as f:
            raw = yaml.safe_load(f)
        model_cfg = ModelConfig.from_dict(raw.get("model", raw))
        unbox_cfg = UnboxConfig.from_model_config(model_cfg)

        model = UnboxForCausalLM.from_unbox_checkpoint(
            self.config.model_path,
            unbox_cfg,
            device=self.device,
        )
        model = model.to(self.dtype).eval()
        return model

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill_step(
        self,
        seq: SequenceState,
        chunk_token_ids: list[int],
        start_pos: int,
    ) -> Optional[torch.Tensor]:
        """Prefill a chunk of tokens, writing KV into the paged cache.

        Returns logits (vocab_size,) for the last token in the chunk, or None
        if this is not the final chunk.
        """
        m = self.model.model
        T = len(chunk_token_ids)
        ids = torch.tensor([chunk_token_ids], device=self.device, dtype=torch.long)
        x = m.embed_tokens(ids)  # (1, T, hidden)

        freqs_cis = m.freqs_cis[start_pos: start_pos + T]

        for layer_idx, layer in enumerate(m.layers):
            # ensure blocks exist before writing
            self.kvcache.ensure_blocks(seq)

            normed = layer.attn_norm(x)
            B, t, _ = normed.shape

            xq = layer.attn.q_proj(normed).view(B, t, layer.attn.num_heads, layer.attn.head_dim)
            xk = layer.attn.k_proj(normed).view(B, t, layer.attn.num_kv_heads, layer.attn.head_dim)
            xv = layer.attn.v_proj(normed).view(B, t, layer.attn.num_kv_heads, layer.attn.head_dim)

            xq = layer.attn.q_norm(xq)
            xk = layer.attn.k_norm(xk)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # write new KV into paged cache
            for tok in range(t):
                pos = start_pos + tok
                self.kvcache.write_kv(layer_idx, seq, pos, xk[0, tok], xv[0, tok])

            # for prefill, use standard SDPA over the chunk (contiguous)
            if layer.attn.kv_repeat > 1:
                xk_sdpa = xk.repeat_interleave(layer.attn.kv_repeat, dim=2)
                xv_sdpa = xv.repeat_interleave(layer.attn.kv_repeat, dim=2)
            else:
                xk_sdpa, xv_sdpa = xk, xv

            xq_t = xq.transpose(1, 2)
            xk_t = xk_sdpa.transpose(1, 2)
            xv_t = xv_sdpa.transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, t, -1)
            attn_out = layer.attn.o_proj(attn_out)

            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        x = m.norm(x)
        logits = m.lm_head(x[0, -1])  # (vocab_size,)
        return logits

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def decode_step(
        self,
        seqs: list[SequenceState],
        token_ids: list[int],
    ) -> torch.Tensor:
        """Run one decode step for a batch of sequences.

        Args:
            seqs      : list of SequenceState (one per sequence in the batch)
            token_ids : the last generated token for each sequence

        Returns:
            logits: (batch, vocab_size)
        """
        batch = len(seqs)
        m = self.model.model
        device = self.device

        ids = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(1)  # (B, 1)
        x = m.embed_tokens(ids)  # (B, 1, hidden)

        # build block tables and seq_lens for paged attention
        max_blocks = max(len(s.block_table) for s in seqs)
        block_tables = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
        seq_lens = torch.zeros(batch, dtype=torch.int32, device=device)
        for i, seq in enumerate(seqs):
            btlen = len(seq.block_table)
            block_tables[i, :btlen] = torch.tensor(seq.block_table, dtype=torch.int32)
            seq_lens[i] = seq.total_len  # includes the token we're about to generate

        nh = self.model_config.num_heads
        head_dim = self.model_config.head_dim
        kv_repeat = nh // self.model_config.num_kv_heads

        # compute positions for RoPE: the decode token sits at position total_len - 1
        positions = (seq_lens - 1).long()  # (B,)

        for layer_idx, layer in enumerate(m.layers):
            normed = layer.attn_norm(x)  # (B, 1, hidden)

            xq = layer.attn.q_proj(normed).view(batch, 1, nh, head_dim)
            xk = layer.attn.k_proj(normed).view(batch, 1, layer.attn.num_kv_heads, head_dim)
            xv = layer.attn.v_proj(normed).view(batch, 1, layer.attn.num_kv_heads, head_dim)

            xq = layer.attn.q_norm(xq)
            xk = layer.attn.k_norm(xk)

            # per-sequence RoPE at the decode position
            for b in range(batch):
                pos = positions[b].item()
                fc = m.freqs_cis[pos: pos + 1]  # (1, head_dim//2)
                rq, rk = apply_rotary_emb(xq[b:b+1], xk[b:b+1], fc)  # (1, 1, nh, hd)
                xq[b:b+1] = rq
                xk[b:b+1] = rk

            # write new KV token into paged cache
            for b, seq in enumerate(seqs):
                pos = seq.total_len - 1
                self.kvcache.write_kv(layer_idx, seq, pos, xk[b, 0], xv[b, 0])

            k_pool, v_pool = self.kvcache.get_kv_layer(layer_idx)

            # paged decode attention: (B, nh, head_dim)
            q_for_kernel = xq[:, 0, :, :]  # (B, nh, head_dim)
            attn_out = self._paged_attn(
                q_for_kernel, k_pool, v_pool,
                block_tables, seq_lens, self.config.block_size,
            )  # (B, nh, head_dim)

            attn_out = attn_out.contiguous().view(batch, 1, -1)
            attn_out = layer.attn.o_proj(attn_out)

            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        x = m.norm(x)
        logits = m.lm_head(x[:, 0, :])  # (B, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Convenience: sample next tokens for a decode batch
    # ------------------------------------------------------------------

    def sample_batch(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> list[int]:
        token_ids = sample(logits, params)
        return token_ids.tolist()
