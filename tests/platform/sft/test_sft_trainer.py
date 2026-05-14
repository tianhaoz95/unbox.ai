"""Tests for the SFT training module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from unbox_platform.model.hf_adapter import UnboxConfig, UnboxForCausalLM
from unbox_platform.sft.config import SFTTrainConfig


SMALL_CFG = dict(
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    ffn_intermediate_size=128,
    vocab_size=256,
    max_seq_len=32,
)

CHAT_MESSAGES = [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
    ],
    [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to weather data."},
    ],
]


@pytest.fixture
def small_model() -> UnboxForCausalLM:
    return UnboxForCausalLM(UnboxConfig(**SMALL_CFG))


@pytest.fixture
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Minimal tokenizer sufficient for SFT smoke tests."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = HFTokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()

    # Train on a tiny corpus so all test strings are representable
    corpus = [
        "Hello how are you I'm doing well thank you What is equals Tell me a joke "
        "Why did the chicken cross the road weather data access",
    ]
    from tokenizers.trainers import BpeTrainer
    trainer = BpeTrainer(
        vocab_size=256,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template=(
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
        ),
    )
    return fast_tok


@pytest.fixture
def chat_dataset() -> Dataset:
    return Dataset.from_dict({"messages": CHAT_MESSAGES})


def test_sft_config_defaults() -> None:
    cfg = SFTTrainConfig()
    assert cfg.max_lr == 2e-5
    assert cfg.num_epochs == 1
    assert cfg.packing is False


def test_sft_config_fields() -> None:
    cfg = SFTTrainConfig(
        checkpoint_path="ckpt.pt",
        max_seq_len=512,
        batch_size=4,
        use_wandb=False,
    )
    assert cfg.checkpoint_path == "ckpt.pt"
    assert cfg.max_seq_len == 512
    assert cfg.batch_size == 4


def test_sft_trainer_one_step(
    small_model: UnboxForCausalLM,
    tiny_tokenizer: PreTrainedTokenizerFast,
    chat_dataset: Dataset,
) -> None:
    """SFTTrainer runs one step without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        training_args = SFTConfig(
            output_dir=tmp,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=not torch.cuda.is_available(),
            max_length=32,
        )
        trainer = SFTTrainer(
            model=small_model,
            args=training_args,
            train_dataset=chat_dataset,
            processing_class=tiny_tokenizer,
        )
        result = trainer.train()
    assert result.training_loss >= 0


def test_sft_trainer_saves_model(
    small_model: UnboxForCausalLM,
    tiny_tokenizer: PreTrainedTokenizerFast,
    chat_dataset: Dataset,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        training_args = SFTConfig(
            output_dir=tmp,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=not torch.cuda.is_available(),
            max_length=32,
        )
        trainer = SFTTrainer(
            model=small_model,
            args=training_args,
            train_dataset=chat_dataset,
            processing_class=tiny_tokenizer,
        )
        trainer.train()
        trainer.save_model(tmp)
        assert (Path(tmp) / "config.json").exists()
        # Verify the saved model can be loaded back
        loaded = UnboxForCausalLM.from_pretrained(tmp)
        assert loaded is not None


def test_build_model_from_unbox_checkpoint(small_model: UnboxForCausalLM) -> None:
    """from_unbox_checkpoint produces a model with identical weights."""
    from unbox_platform.model import Transformer

    model_cfg = UnboxConfig(**SMALL_CFG).to_model_config()
    transformer = Transformer(model_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "latest.pt"
        torch.save({"step": 0, "epoch": 0, "loss": 0.0, "model": transformer.state_dict()}, ckpt)

        adapted = UnboxForCausalLM.from_unbox_checkpoint(ckpt, UnboxConfig(**SMALL_CFG))

    input_ids = torch.randint(0, SMALL_CFG["vocab_size"], (1, 8))
    with torch.no_grad():
        logits_adapted = adapted(input_ids).logits
        logits_raw, _ = transformer(input_ids)

    assert torch.allclose(logits_adapted, logits_raw, atol=1e-5)
