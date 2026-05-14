"""
Megatron-Core process group initialization and model wrapping.

Megatron-Core manages three parallel dimensions:
  - Tensor Parallel  (TP): splits individual layers across GPUs within a node
  - Pipeline Parallel (PP): assigns different layers to different nodes
  - Data Parallel    (DP): standard data parallelism; inferred from world_size / (TP * PP)

For single-GPU or DP-only runs, set tp=1, pp=1 — Megatron-Core still manages
the process groups and distributed optimizer correctly.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .config import ParallelConfig

if TYPE_CHECKING:
    pass


def init_distributed() -> tuple[int, int]:
    """Initialize torch.distributed. Returns (rank, world_size)."""
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def setup_megatron(config: ParallelConfig, rank: int, world_size: int) -> None:
    """Initialize Megatron-Core parallel state."""
    try:
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=config.tensor_parallel_size,
            pipeline_model_parallel_size=config.pipeline_parallel_size,
        )
    except ImportError:
        # megatron-core not installed — fall back to no-op (single GPU / plain DDP)
        if rank == 0:
            print(
                "Warning: megatron-core not found. "
                "Running without tensor/pipeline parallelism. "
                "Install megatron-core for multi-GPU TP/PP support."
            )


def setup_model(
    model: nn.Module,
    config: ParallelConfig,
    rank: int,
    world_size: int,
) -> nn.Module:
    """
    Wrap model for distributed training.

    With Megatron-Core installed and TP/PP > 1, the model should already
    have been constructed with Megatron's column/row parallel linears before
    calling this function. This function handles the DDP wrapper for the
    data-parallel dimension.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dp_size = world_size // (config.tensor_parallel_size * config.pipeline_parallel_size)

    if dp_size > 1 and torch.distributed.is_initialized():
        try:
            from megatron.core import parallel_state
            dp_group = parallel_state.get_data_parallel_group()
        except ImportError:
            dp_group = None

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            process_group=dp_group,
        )

    return model


def get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")
