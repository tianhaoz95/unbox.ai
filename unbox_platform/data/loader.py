"""Dataset loading utility that supports both HuggingFace and ModelScope sources."""

from __future__ import annotations

from typing import Any


def load_dataset_from_source(
    hf_name: str,
    split: str,
    source: str = "huggingface",
    ms_name: str = "",
    ms_subset: str = "",
    streaming: bool = False,
    cache_dir: str | None = None,
    **kwargs: Any,
):
    """Load a dataset from HuggingFace or ModelScope.

    Args:
        hf_name: HuggingFace dataset name (e.g. "HuggingFaceH4/ultrachat_200k").
        split: Dataset split (e.g. "train_sft").
        source: "huggingface" or "modelscope".
        ms_name: ModelScope dataset name. Falls back to hf_name if empty.
        ms_subset: ModelScope subset name (equivalent to HF's `name` parameter).
        streaming: Whether to load as a streaming / iterable dataset.
        cache_dir: Local cache directory.
        **kwargs: Forwarded to the underlying loader.
    """
    if source == "modelscope":
        from modelscope.msdatasets import MsDataset  # type: ignore[import]

        dataset_name = ms_name or hf_name
        load_kwargs: dict[str, Any] = {"split": split, "use_streaming": streaming}
        if ms_subset:
            load_kwargs["subset_name"] = ms_subset
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        load_kwargs.update(kwargs)
        return MsDataset.load(dataset_name, **load_kwargs)

    # Default: HuggingFace
    from datasets import load_dataset

    hf_kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if cache_dir:
        hf_kwargs["cache_dir"] = cache_dir
    hf_kwargs.update(kwargs)
    return load_dataset(hf_name, **hf_kwargs)
