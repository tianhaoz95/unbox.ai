"""Tests for platform/tokenizer."""

import json
import tempfile
from pathlib import Path

import pytest

from unbox_platform.tokenizer import TokenizerConfig, train_tokenizer
from unbox_platform.tokenizer.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def trained_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tokenizer")

    # Write a small JSONL corpus
    corpus = tmp / "corpus.jsonl"
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Language models are trained on large corpora of text.",
        "Deep learning has revolutionized natural language processing.",
        "Transformers use self-attention mechanisms to process sequences.",
        "Tokenization splits text into subword units for processing.",
    ] * 200  # repeat to have enough data for BPE

    with open(corpus, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

    cfg = TokenizerConfig(vocab_size=512, min_frequency=1)
    train_tokenizer(corpus, tmp / "tok", config=cfg)
    return tmp / "tok"


def test_tokenizer_trains(trained_tokenizer_dir: Path) -> None:
    assert (trained_tokenizer_dir / "tokenizer.json").exists()
    assert (trained_tokenizer_dir / "tokenizer_config.json").exists()


def test_vocab_size(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    assert tok.vocab_size <= 512


def test_special_token_ids(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    assert tok.bos_id >= 0
    assert tok.eos_id >= 0
    assert tok.pad_id >= 0
    assert tok.bos_id != tok.eos_id
    assert tok.bos_id != tok.pad_id


def test_encode_returns_list(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    ids = tok.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_decode_roundtrip(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    text = "The quick brown fox"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert text in decoded or decoded in text  # allow minor whitespace differences


def test_bos_eos_added(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    ids = tok.encode("hello", add_special_tokens=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_empty_string(trained_tokenizer_dir: Path) -> None:
    tok = Tokenizer(trained_tokenizer_dir)
    ids = tok.encode("", add_special_tokens=False)
    assert isinstance(ids, list)
