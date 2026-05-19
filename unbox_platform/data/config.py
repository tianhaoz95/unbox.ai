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

    # Dataset source: "huggingface" (default) or "modelscope" (for restricted regions).
    # When using modelscope, set ms_dataset_name to the ModelScope repo ID if it differs
    # from data_path (e.g. "AI-ModelScope/fineweb-edu"). Leave empty to try data_path as-is.
    dataset_source: str = "huggingface"
    ms_dataset_name: str = ""
    ms_dataset_subset: str = ""  # ModelScope subset name (equivalent to HF's `name` param)
