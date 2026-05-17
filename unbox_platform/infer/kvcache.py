"""Paged KV cache: block allocator, radix-tree prefix cache, LRU eviction.

Design
──────
Memory is divided into fixed-size blocks of `block_size` tokens. Each
sequence gets a block table — a list of block indices — that the paged decode
attention kernel uses to scatter-gather into those blocks.

The RadixCache maps token-id prefixes to block lists, enabling prefix reuse
across requests with shared prompts. When the pool is full, LRU eviction
frees blocks from the least-recently-used leaf of the radix tree.

Block layout (stored in the KVPool tensor):
  shape: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)
    dim 0: transformer layer index
    dim 1: 0 = K, 1 = V
    dim 2: block index
    dim 3: token slot within the block
    dim 4: KV head index
    dim 5: head dimension
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Block allocator
# ---------------------------------------------------------------------------

class BlockAllocator:
    """Free-list allocator over a fixed pool of block indices."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("KV cache OOM: no free blocks")
        return self._free.pop()

    def free(self, block_idx: int) -> None:
        self._free.append(block_idx)

    def free_many(self, block_indices: list[int]) -> None:
        self._free.extend(block_indices)


# ---------------------------------------------------------------------------
# Radix tree node
# ---------------------------------------------------------------------------

@dataclass
class RadixNode:
    token_ids: list[int] = field(default_factory=list)
    block_indices: list[int] = field(default_factory=list)
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: Optional["RadixNode"] = None
    last_access_time: float = field(default_factory=time.monotonic)
    ref_count: int = 0  # number of active sequences using this node

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Radix cache
# ---------------------------------------------------------------------------

class RadixCache:
    """Prefix cache backed by a radix tree.

    Sequences with a shared token prefix share their KV blocks, avoiding
    redundant prefill computation.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        self._allocator = allocator
        self._block_size = block_size
        self._root = RadixNode()
        # LRU order over leaf nodes (node_id → node), most-recent last
        self._lru: OrderedDict[int, RadixNode] = OrderedDict()

    # --- public API ---

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Return (block_table, matched_len) for the longest cached prefix."""
        block_table: list[int] = []
        matched_len = 0
        node = self._root
        pos = 0

        while pos < len(token_ids):
            first_token = token_ids[pos]
            if first_token not in node.children:
                break
            child = node.children[first_token]
            child_len = len(child.token_ids)
            candidate = token_ids[pos: pos + child_len]
            if candidate != child.token_ids:
                break
            block_table.extend(child.block_indices)
            matched_len += child_len
            pos += child_len
            node = child

        if block_table:
            node.last_access_time = time.monotonic()
            self._lru_touch(node)

        return block_table, matched_len

    def insert(self, token_ids: list[int], block_table: list[int]) -> None:
        """Insert a completed prefix into the cache."""
        node = self._root
        pos = 0
        block_pos = 0

        while pos < len(token_ids):
            first_token = token_ids[pos]
            chunk_tokens_per_block = self._block_size

            if first_token not in node.children:
                # insert remaining tokens as a new child chain
                while pos < len(token_ids):
                    end = min(pos + chunk_tokens_per_block, len(token_ids))
                    chunk = token_ids[pos:end]
                    needed_blocks = (len(chunk) + self._block_size - 1) // self._block_size
                    blocks = block_table[block_pos: block_pos + needed_blocks]
                    child = RadixNode(
                        token_ids=chunk,
                        block_indices=blocks,
                        parent=node,
                        last_access_time=time.monotonic(),
                    )
                    node.children[first_token] = child
                    self._lru_add(child)
                    node = child
                    pos = end
                    block_pos += needed_blocks
                    if pos < len(token_ids):
                        first_token = token_ids[pos]
                return

            child = node.children[first_token]
            child_len = len(child.token_ids)
            candidate = token_ids[pos: pos + child_len]

            if candidate == child.token_ids:
                # exact match — traverse
                pos += child_len
                block_pos += len(child.block_indices)
                node = child
            else:
                # partial match — split node (not implemented for simplicity; skip)
                return

    def evict_lru(self) -> int:
        """Evict the LRU leaf node. Returns number of blocks freed."""
        for node_id, node in self._lru.items():
            if node.is_leaf() and node.ref_count == 0:
                blocks = node.block_indices
                if node.parent is not None:
                    first_token = node.token_ids[0]
                    node.parent.children.pop(first_token, None)
                del self._lru[node_id]
                self._allocator.free_many(blocks)
                return len(blocks)
        return 0

    # --- internal ---

    def _lru_add(self, node: RadixNode) -> None:
        self._lru[id(node)] = node

    def _lru_touch(self, node: RadixNode) -> None:
        node_id = id(node)
        if node_id in self._lru:
            self._lru.move_to_end(node_id)


# ---------------------------------------------------------------------------
# Sequence state
# ---------------------------------------------------------------------------

@dataclass
class SequenceState:
    request_id: str
    prompt_token_ids: list[int]
    block_table: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    # position of the next token to prefill (chunked prefill cursor)
    prefill_cursor: int = 0
    # number of tokens whose KV is already written into the cache
    num_cached_tokens: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_len(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    def all_input_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids


# ---------------------------------------------------------------------------
# KV pool (the actual tensor storage)
# ---------------------------------------------------------------------------

class KVPool:
    """Physical KV cache storage tensor.

    shape: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.data = torch.zeros(
            num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        )

    def write_kv(
        self,
        layer_idx: int,
        block_idx: int,
        slot: int,
        k: torch.Tensor,  # (num_kv_heads, head_dim)
        v: torch.Tensor,  # (num_kv_heads, head_dim)
    ) -> None:
        self.data[layer_idx, 0, block_idx, slot] = k
        self.data[layer_idx, 1, block_idx, slot] = v

    def get_kv_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (K, V) tensors for a layer: (num_blocks, block_size, num_kv_heads, head_dim)."""
        return self.data[layer_idx, 0], self.data[layer_idx, 1]

    def serialize_blocks(self, block_indices: list[int]) -> bytes:
        """Serialise a subset of blocks to bytes for ZMQ KV transfer."""
        blocks = self.data[:, :, block_indices]  # (num_layers, 2, len, block_size, nh, hd)
        return blocks.cpu().numpy().tobytes()

    def deserialize_blocks(self, data: bytes, block_indices: list[int], shape_hint: tuple) -> None:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.float32).reshape(shape_hint)
        t = torch.from_numpy(arr.copy())
        self.data[:, :, block_indices] = t.to(self.data.device, dtype=self.data.dtype)


# ---------------------------------------------------------------------------
# PagedKVCache: top-level interface used by engine + scheduler
# ---------------------------------------------------------------------------

class PagedKVCache:
    """Combines BlockAllocator, RadixCache, and KVPool into one interface."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.block_size = block_size
        self._allocator = BlockAllocator(num_blocks)
        self._radix = RadixCache(self._allocator, block_size)
        self._pool = KVPool(num_layers, num_kv_heads, head_dim, num_blocks, block_size, device, dtype)

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free

    def prepare_sequence(self, seq: SequenceState) -> int:
        """Allocate blocks for a new sequence, matching cached prefix if any.

        Returns number of tokens that can be skipped (already cached).
        """
        cached_blocks, matched_len = self._radix.match_prefix(seq.prompt_token_ids)
        seq.block_table = list(cached_blocks)
        seq.num_cached_tokens = matched_len
        seq.prefill_cursor = matched_len
        return matched_len

    def ensure_blocks(self, seq: SequenceState) -> None:
        """Allocate a new block if the next token would overflow the last block."""
        needed = seq.total_len
        allocated = len(seq.block_table) * self.block_size
        while allocated < needed:
            # try to evict if OOM
            if self._allocator.num_free == 0:
                freed = self._radix.evict_lru()
                if freed == 0:
                    raise RuntimeError("KV cache OOM: cannot evict any blocks")
            seq.block_table.append(self._allocator.allocate())
            allocated += self.block_size

    def write_kv(
        self,
        layer_idx: int,
        seq: SequenceState,
        token_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        block_idx = seq.block_table[token_pos // self.block_size]
        slot = token_pos % self.block_size
        self._pool.write_kv(layer_idx, block_idx, slot, k, v)

    def get_kv_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.get_kv_layer(layer_idx)

    def finish_sequence(self, seq: SequenceState) -> None:
        """Insert completed prefix into radix cache; free blocks not shared."""
        self._radix.insert(seq.prompt_token_ids, seq.block_table)

    def free_sequence(self, seq: SequenceState) -> None:
        """Free all blocks owned by a sequence (not in radix cache)."""
        self._allocator.free_many(seq.block_table)

    def serialize_blocks(self, block_indices: list[int]) -> bytes:
        return self._pool.serialize_blocks(block_indices)
