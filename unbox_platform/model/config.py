"""Model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Dimensions
    hidden_size: int = 1792
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 8          # GQA: fewer KV heads than query heads
    ffn_intermediate_size: int = 4864  # ceil(hidden_size * π / 64) * 64

    # Vocabulary / sequence
    vocab_size: int = 32768
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.0

    # RoPE
    rope_theta: float = 500000.0   # high base for long-context extrapolation

    # Misc
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True    # tie input embedding and lm_head weights

    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
