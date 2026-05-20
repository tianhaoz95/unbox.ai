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

    NOTE: Eagerly tokenizes the full corpus at construction time.
    Use StreamingPretrainDataset for large corpora (millions of documents).
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
        from datasets import load_dataset

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


class StreamingPretrainDataset(torch.utils.data.IterableDataset):
    """
    Memory-efficient streaming dataset for large corpora.
    Tokenizes documents on-the-fly without loading everything into RAM.
    Use this for production training runs; use PretrainDataset for tests.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: DataConfig,
        split: str = "train",
        skip_chunks: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.split = split
        self.skip_chunks = skip_chunks

    def _iter_texts(self) -> Iterator[str]:
        path = Path(self.config.data_path)
        if path.exists():
            yield from self._iter_jsonl(path)
        else:
            yield from self._iter_hf_dataset()

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
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
        from datasets import load_dataset

        ds = load_dataset(
            self.config.data_path,
            name=self.config.dataset_name,
            split="train",
            streaming=True,
            cache_dir=self.config.cache_dir,
            trust_remote_code=False,
        )
        n_eval = int(self.config.eval_fraction * 9_765_625)

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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        skipped = 0
        for text in self._iter_texts():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = [self.tokenizer.bos_id] + ids + [self.tokenizer.eos_id]
            buffer.extend(ids)
            while len(buffer) >= self.max_seq_len:
                chunk = buffer[: self.max_seq_len]
                buffer = buffer[self.max_seq_len :]
                if skipped < self.skip_chunks:
                    skipped += 1
                    continue
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_id] = -100
                yield {"input_ids": input_ids, "labels": labels}

    def estimate_num_chunks(self) -> int:
        """Estimate chunk count via a fast line-count pass + avg tokens/doc heuristic."""
        path = Path(self.config.data_path)
        if not path.exists():
            return 10_000  # fallback for HF streaming datasets
        with open(path, encoding="utf-8") as f:
            total_docs = sum(1 for l in f if l.strip())
        n_eval = max(1, int(self.config.eval_fraction * total_docs))
        n_docs = n_eval if self.split == "eval" else total_docs - n_eval
        avg_tokens_per_doc = 500  # conservative estimate for FineWeb-Edu
        return max(1, (n_docs * avg_tokens_per_doc) // self.max_seq_len)


def build_dataloader(
    tokenizer: Tokenizer,
    config: DataConfig,
    split: str = "train",
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = StreamingPretrainDataset(tokenizer, config, split=split)

    # IterableDataset does not support DistributedSampler; multi-GPU sharding
    # must be done inside __iter__ (not yet implemented).
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # multiple workers each iterate the full dataset independently
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
