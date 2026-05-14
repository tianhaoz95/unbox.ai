"""Tests for the HuggingFace compatibility adapter."""

import tempfile
from pathlib import Path

import pytest
import torch

from unbox_platform.model import ModelConfig, Transformer
from unbox_platform.model.hf_adapter import UnboxConfig, UnboxForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


# Small config for fast tests
SMALL_CFG = dict(
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    ffn_intermediate_size=128,
    vocab_size=256,
    max_seq_len=32,
)


@pytest.fixture
def unbox_config() -> UnboxConfig:
    return UnboxConfig(**SMALL_CFG)


@pytest.fixture
def model(unbox_config: UnboxConfig) -> UnboxForCausalLM:
    return UnboxForCausalLM(unbox_config)


def test_config_roundtrip(unbox_config: UnboxConfig) -> None:
    model_cfg = unbox_config.to_model_config()
    restored = UnboxConfig.from_model_config(model_cfg)
    assert restored.hidden_size == unbox_config.hidden_size
    assert restored.num_layers == unbox_config.num_layers
    assert restored.num_heads == unbox_config.num_heads
    assert restored.num_kv_heads == unbox_config.num_kv_heads
    assert restored.vocab_size == unbox_config.vocab_size
    assert restored.max_seq_len == unbox_config.max_seq_len


def test_forward_output_type(model: UnboxForCausalLM, unbox_config: UnboxConfig) -> None:
    input_ids = torch.randint(0, unbox_config.vocab_size, (2, 8))
    out = model(input_ids)
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.loss is None
    assert out.logits.shape == (2, 8, unbox_config.vocab_size)


def test_forward_with_labels(model: UnboxForCausalLM, unbox_config: UnboxConfig) -> None:
    input_ids = torch.randint(0, unbox_config.vocab_size, (2, 8))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.loss.item() > 0


def test_attention_mask_accepted(model: UnboxForCausalLM, unbox_config: UnboxConfig) -> None:
    # attention_mask is accepted without error (even though it's not used internally)
    input_ids = torch.randint(0, unbox_config.vocab_size, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    out = model(input_ids, attention_mask=mask)
    assert out.logits.shape == (2, 8, unbox_config.vocab_size)


def test_save_and_load_pretrained(model: UnboxForCausalLM, unbox_config: UnboxConfig) -> None:
    input_ids = torch.randint(0, unbox_config.vocab_size, (1, 8))
    with torch.no_grad():
        logits_before = model(input_ids).logits

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        assert (Path(tmp) / "config.json").exists()
        loaded = UnboxForCausalLM.from_pretrained(tmp)

    with torch.no_grad():
        logits_after = loaded(input_ids).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_from_unbox_checkpoint(unbox_config: UnboxConfig) -> None:
    # Build a bare Transformer, save in unbox checkpoint format, then load via adapter
    model_cfg = unbox_config.to_model_config()
    transformer = Transformer(model_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "latest.pt"
        torch.save({"step": 1, "epoch": 0, "loss": 1.0, "model": transformer.state_dict()}, ckpt_path)

        adapted = UnboxForCausalLM.from_unbox_checkpoint(ckpt_path, unbox_config)

    input_ids = torch.randint(0, unbox_config.vocab_size, (1, 8))
    with torch.no_grad():
        logits_adapted = adapted(input_ids).logits
        logits_raw, _ = transformer(input_ids)

    assert torch.allclose(logits_adapted, logits_raw, atol=1e-5)


def test_tied_embeddings_preserved(unbox_config: UnboxConfig) -> None:
    model = UnboxForCausalLM(unbox_config)
    assert model.model.lm_head.weight is model.model.embed_tokens.weight
