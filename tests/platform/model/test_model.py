"""Tests for platform/model."""

import pytest
import torch

from unbox_platform.model import ModelConfig, Transformer


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_intermediate_size=256,
        vocab_size=512,
        max_seq_len=64,
    )


def test_model_forward_shape(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    model.eval()
    B, T = 2, 32
    input_ids = torch.randint(0, small_config.vocab_size, (B, T))
    logits, loss = model(input_ids)
    assert logits.shape == (B, T, small_config.vocab_size)
    assert loss is None


def test_model_forward_with_labels(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    model.eval()
    B, T = 2, 32
    input_ids = torch.randint(0, small_config.vocab_size, (B, T))
    labels = input_ids.clone()
    logits, loss = model(input_ids, labels)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_model_loss_ignores_padding(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    model.eval()
    B, T = 2, 32
    input_ids = torch.randint(0, small_config.vocab_size, (B, T))
    labels = input_ids.clone()
    labels[:, T // 2:] = -100  # mask second half

    _, loss_partial = model(input_ids, labels)
    labels_full = input_ids.clone()
    _, loss_full = model(input_ids, labels_full)

    # Losses should differ when different tokens are masked
    assert loss_partial.item() != loss_full.item()


def test_model_parameter_count(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    n_params = model.num_parameters()
    assert n_params > 0
    # With tied embeddings, lm_head.weight == embed_tokens.weight — verify count is sane
    assert n_params < 100_000_000  # small test model


def test_tied_embeddings(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_model_generate(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    model.eval()
    input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
    output = model.generate(input_ids, max_new_tokens=10, eos_id=2)
    assert output.shape[0] == 1
    assert output.shape[1] >= 8  # at least the prompt


def test_sequence_length_exceeds_max_raises(small_config: ModelConfig) -> None:
    model = Transformer(small_config)
    too_long = torch.randint(0, small_config.vocab_size, (1, small_config.max_seq_len + 1))
    with pytest.raises(AssertionError):
        model(too_long)


def test_config_validation() -> None:
    with pytest.raises(AssertionError):
        # num_heads not divisible by num_kv_heads
        ModelConfig(hidden_size=128, num_heads=4, num_kv_heads=3)


@pytest.mark.gpu
def test_model_forward_gpu(small_config: ModelConfig) -> None:
    device = torch.device("cuda")
    model = Transformer(small_config).to(device)
    input_ids = torch.randint(0, small_config.vocab_size, (2, 16)).to(device)
    logits, _ = model(input_ids)
    assert logits.device.type == "cuda"
