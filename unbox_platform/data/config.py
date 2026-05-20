"""Data pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    # Path to a JSONL file or a HuggingFace dataset name
    data_path: str = "HuggingFaceFW/fineweb-edu"
    dataset_name: str = "sample-10BT"   # HF subset name; ignored for local files
    text_field: str = "text"

    max_seq_len: int = 2048
    num_workers: int = 4

    # Fraction of data held out for evaluation
    eval_fraction: float = 0.001        # 0.1% of 10B tokens ≈ 10M tokens for eval

    # Local cache directory for downloaded datasets
    cache_dir: str = "data/cache"
