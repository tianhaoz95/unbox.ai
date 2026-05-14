"""Tests for platform/data."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from unbox_platform.data import DataConfig, PretrainDataset


class _MockTokenizer:
    """Minimal tokenizer stub for dataset tests."""
    vocab_size = 256
    bos_id = 1
    eos_id = 2
    pad_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple char-level encoding for testing
        ids = [min(ord(c), 255) for c in text]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if i > 2)


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    docs = [{"text": "abcdef " * 50} for _ in range(200)]
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


@pytest.fixture
def data_config(jsonl_file: Path) -> DataConfig:
    return DataConfig(
        data_path=str(jsonl_file),
        max_seq_len=64,
        num_workers=0,
        eval_fraction=0.1,
    )


def test_dataset_builds(data_config: DataConfig) -> None:
    tok = _MockTokenizer()
    ds = PretrainDataset(tok, data_config, split="train")
    assert len(ds) > 0


def test_item_shape(data_config: DataConfig) -> None:
    tok = _MockTokenizer()
    ds = PretrainDataset(tok, data_config, split="train")
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (data_config.max_seq_len,)
    assert item["labels"].shape == (data_config.max_seq_len,)


def test_item_dtypes(data_config: DataConfig) -> None:
    tok = _MockTokenizer()
    ds = PretrainDataset(tok, data_config, split="train")
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long


def test_labels_mask_padding(data_config: DataConfig) -> None:
    tok = _MockTokenizer()
    ds = PretrainDataset(tok, data_config, split="train")
    item = ds[0]
    # Non-pad positions should not be masked
    non_pad = item["input_ids"] != tok.pad_id
    assert (item["labels"][non_pad] != -100).all()


def test_train_eval_split(data_config: DataConfig) -> None:
    tok = _MockTokenizer()
    ds_train = PretrainDataset(tok, data_config, split="train")
    ds_eval = PretrainDataset(tok, data_config, split="eval")
    # eval split should be smaller (10% fraction)
    assert len(ds_eval) < len(ds_train)
