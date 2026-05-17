"""NCCL tensor-parallel setup for inference workers.

Each worker pool (prefill or decode) may span multiple GPUs. Within the pool,
weights are sharded Megatron-style:
  - Q/K/V/gate/up projections: column-parallel (each GPU holds a column shard)
  - output/down projections: row-parallel (each GPU holds a row shard + contributes partial sum)

An NCCL all-reduce after each row-parallel layer reconstructs the full activation.

For single-GPU workers (tensor_parallel_size=1) this module is a no-op.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_worker_group(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29600",
    backend: str = "nccl",
) -> None:
    """Initialise a process group for a worker pool.

    Each worker pool forms its own process group. In disaggregated mode,
    prefill workers and decode workers each call this independently.
    """
    if world_size == 1:
        return
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce sum across the worker group (no-op for world_size=1)."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def get_tp_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_tp_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def shard_column_parallel(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return the column shard for this rank.

    Column-parallel: split output dimension (dim 0) across ranks.
    weight shape: (out_features, in_features)
    """
    if world_size == 1:
        return weight
    chunk_size = weight.shape[0] // world_size
    return weight[rank * chunk_size: (rank + 1) * chunk_size].contiguous()


def shard_row_parallel(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return the row shard for this rank.

    Row-parallel: split input dimension (dim 1) across ranks.
    weight shape: (out_features, in_features)
    """
    if world_size == 1:
        return weight
    chunk_size = weight.shape[1] // world_size
    return weight[:, rank * chunk_size: (rank + 1) * chunk_size].contiguous()


def apply_tp_to_model(model: torch.nn.Module, rank: int, world_size: int) -> None:
    """Shard model weights in-place for tensor parallelism.

    This is a simple eager sharding approach: replaces weight tensors in
    attention and FFN layers with rank-local shards.
    """
    if world_size == 1:
        return

    from ..model.model import Attention, FeedForward

    for module in model.modules():
        if isinstance(module, Attention):
            # column-parallel: q/k/v projections
            module.q_proj.weight.data = shard_column_parallel(module.q_proj.weight.data, rank, world_size)
            module.k_proj.weight.data = shard_column_parallel(module.k_proj.weight.data, rank, world_size)
            module.v_proj.weight.data = shard_column_parallel(module.v_proj.weight.data, rank, world_size)
            # row-parallel: output projection (requires all-reduce after)
            module.o_proj.weight.data = shard_row_parallel(module.o_proj.weight.data, rank, world_size)

        elif isinstance(module, FeedForward):
            # column-parallel: gate and up projections
            module.gate_proj.weight.data = shard_column_parallel(module.gate_proj.weight.data, rank, world_size)
            module.up_proj.weight.data = shard_column_parallel(module.up_proj.weight.data, rank, world_size)
            # row-parallel: down projection (requires all-reduce after)
            module.down_proj.weight.data = shard_row_parallel(module.down_proj.weight.data, rank, world_size)
