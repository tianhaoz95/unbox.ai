"""Pretraining dataset: tokenize-on-the-fly from JSONL or HuggingFace datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .config import DataConfig
from unbox_platform.tokenizer import Tokenizer


class PretrainDataset(Dataset):
    """
    Streams text from a JSONL file or HuggingFace dataset, tokenizes on-the-fly,
    and returns fixed-length chunks for next-token prediction.

    Each item is a dict with:
      input_ids: LongTensor of shape (max_seq_len,)
      labels:    LongTensor of shape (max_seq_len,) — same as input_ids but with
                 padding positions masked to -100
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: DataConfig,
        split: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.split = split

        self._chunks: list[list[int]] = []
        self._build_chunks()

    def _iter_texts(self) -> Iterator[str]:
        path = Path(self.config.data_path)
        if path.exists():
            yield from self._iter_jsonl(path)
        else:
            yield from self._iter_hf_dataset()

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
        # Count total lines first to determine the eval boundary
        with open(path, encoding="utf-8") as f:
            total_docs = sum(1 for l in f if l.strip())
        n_eval = max(1, int(self.config.eval_fraction * total_docs))

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                is_eval_doc = i < n_eval
                if self.split == "eval" and not is_eval_doc:
                    continue
                if self.split == "train" and is_eval_doc:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(self.config.text_field, "")
                except json.JSONDecodeError:
                    text = line
                if text:
                    yield text

    def _iter_hf_dataset(self) -> Iterator[str]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset(
            self.config.data_path,
            name=self.config.dataset_name,
            split="train",
            streaming=True,
            cache_dir=self.config.cache_dir,
            trust_remote_code=False,
        )
        n_eval = int(self.config.eval_fraction * 9_765_625)  # approx docs in 10BT

        for i, example in enumerate(ds):
            if self.split == "eval":
                if i >= n_eval:
                    return
            else:
                if i < n_eval:
                    continue
            text = example.get(self.config.text_field, "")
            if text:
                yield text

    def _build_chunks(self) -> None:
        """Tokenize all texts and split into max_seq_len chunks."""
        buffer: list[int] = []
        for text in self._iter_texts():
            # encode without auto-adding BOS/EOS since we pack documents end-to-end
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            # document boundary: prepend BOS, append EOS
            ids = [self.tokenizer.bos_id] + ids + [self.tokenizer.eos_id]
            buffer.extend(ids)

            while len(buffer) >= self.max_seq_len:
                self._chunks.append(buffer[: self.max_seq_len])
                buffer = buffer[self.max_seq_len :]

        # Drop the last incomplete chunk to keep all sequences the same length

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self._chunks[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def build_dataloader(
    tokenizer: Tokenizer,
    config: DataConfig,
    split: str = "train",
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = PretrainDataset(tokenizer, config, split=split)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
