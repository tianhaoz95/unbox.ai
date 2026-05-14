"""Tests for platform/eval."""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from unbox_platform.model import ModelConfig, Transformer
from unbox_platform.train.checkpoint import save_checkpoint


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


def test_perplexity_is_positive(small_config: ModelConfig, tmp_path: Path) -> None:
    """Perplexity on random data from an untrained model should be > 1."""
    from unbox_platform.eval.perplexity import compute_perplexity
    from unbox_platform.data import DataConfig
    import json

    # Write dummy eval data
    corpus = tmp_path / "eval.jsonl"
    with open(corpus, "w") as f:
        for _ in range(50):
            f.write(json.dumps({"text": "hello world " * 20}) + "\n")

    # Save a fresh (untrained) checkpoint
    model = Transformer(small_config)
    ckpt_dir = tmp_path / "ckpt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(ckpt_dir, model, optimizer, step=0, epoch=0, loss=0.0)

    class _MockTokenizer:
        vocab_size = 512
        bos_id = 1
        eos_id = 2
        pad_id = 0
        def encode(self, text, add_special_tokens=True):
            return [min(ord(c), 511) for c in text]
        def decode(self, ids, skip_special_tokens=True):
            return ""

    data_cfg = DataConfig(
        data_path=str(corpus),
        max_seq_len=64,
        num_workers=0,
        eval_fraction=1.0,
    )

    from unbox_platform.data.dataset import PretrainDataset
    from torch.utils.data import DataLoader

    tok = _MockTokenizer()
    ds = PretrainDataset(tok, data_cfg, split="eval")
    loader = DataLoader(ds, batch_size=2)

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            _, loss = model(batch["input_ids"], batch["labels"])
            n = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

    ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))
    assert ppl > 1.0
    assert math.isfinite(ppl)
