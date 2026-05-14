"""HuggingFace-compatible adapter for the Unbox Transformer model.

Wraps the existing Transformer and ModelConfig in HF base classes so the
model can be used with TRL, PEFT, and other HF-ecosystem tooling without
modifying the core model code.

Key differences from a native HF model:
- attention_mask is accepted but ignored; causal masking is always applied
  internally via scaled_dot_product_attention(is_causal=True)
- past_key_values (KV cache) are not yet supported
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .model import Transformer, precompute_freqs_cis


class UnboxConfig(PretrainedConfig):
    """PretrainedConfig wrapper around ModelConfig."""

    model_type = "unbox"

    def __init__(
        self,
        hidden_size: int = 1792,
        num_layers: int = 24,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        ffn_intermediate_size: int = 4864,
        vocab_size: int = 32768,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-6,
        tie_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_intermediate_size = ffn_intermediate_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tie_embeddings = tie_embeddings
        # Map to HF's standard tie_word_embeddings so PreTrainedModel.tie_weights() works
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        # vocab_size is a PretrainedConfig built-in field
        super().__init__(vocab_size=vocab_size, **kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            ffn_intermediate_size=self.ffn_intermediate_size,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_embeddings=self.tie_embeddings,
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "UnboxConfig":
        return cls(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            ffn_intermediate_size=config.ffn_intermediate_size,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            tie_embeddings=config.tie_embeddings,
        )


class UnboxForCausalLM(PreTrainedModel):
    """HF-compatible causal LM wrapper around the Unbox Transformer."""

    config_class = UnboxConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # transformers 5.x: {alias_key: canonical_key} — lm_head is the alias, embed_tokens is canonical
    _tied_weights_keys = {"model.lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: UnboxConfig) -> None:
        super().__init__(config)
        self.model = Transformer(config.to_model_config())
        self.post_init()

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _recompute_rope_buffers(self) -> None:
        """Recompute freqs_cis outside HF's _fast_init context.

        HF's from_pretrained patches torch.ones_like (and friends) during model
        construction to skip costly zero-init. precompute_freqs_cis uses
        torch.ones_like internally, so it produces garbage values under that
        context. Calling this method after from_pretrained returns fixes it.
        """
        cfg = self.config.to_model_config()
        freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta)
        self.model.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "UnboxForCausalLM":
        model = super().from_pretrained(*args, **kwargs)
        model._recompute_rope_buffers()
        return model

    # --- embedding access hooks required by PreTrainedModel ---

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: torch.nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, value: torch.nn.Linear) -> None:
        self.model.lm_head = value

    # --- forward ---

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # attention_mask is accepted for API compatibility but not used;
        # causal masking is handled internally via is_causal=True in SDPA.
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # --- checkpoint interop ---

    @classmethod
    def from_unbox_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: UnboxConfig,
        device: str | torch.device = "cpu",
    ) -> "UnboxForCausalLM":
        """Load from a .pt checkpoint saved by unbox_platform.train.checkpoint.

        The checkpoint stores a flat state dict from the bare Transformer.
        This method adds the required 'model.' prefix before loading.
        """
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        raw_sd = state["model"]
        prefixed_sd = {"model." + k: v for k, v in raw_sd.items()}
        model = cls(config)
        model.load_state_dict(prefixed_sd)
        model._recompute_rope_buffers()
        model.eval()
        return model
