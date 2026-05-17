"""Per-worker request scheduler.

Two scheduler variants share a common interface:

  UnifiedScheduler  — used in unified mode. One loop handles both chunked
                      prefill and continuous decode batching on the same worker.

  PrefillScheduler  — used in disaggregated mode on prefill workers. Focuses on
                      draining prompts in chunks; hands off finished sequences.

  DecodeScheduler   — used in disaggregated mode on decode workers. Batches all
                      in-flight sequences for decode; accepts new sequences from
                      the prefill pool via KV transfer.

All schedulers share a waiting queue (newly arrived requests) and a running
queue (sequences being actively decoded or prefilled).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from .kvcache import PagedKVCache, SequenceState
from .messaging import GenerateRequest, SamplingParams


def _request_to_seq(req: GenerateRequest) -> SequenceState:
    return SequenceState(
        request_id=req.request_id,
        prompt_token_ids=req.prompt_token_ids,
    )


# ---------------------------------------------------------------------------
# Unified scheduler (unified mode)
# ---------------------------------------------------------------------------

class UnifiedScheduler:
    """Handles both prefill and decode in one loop.

    Each step returns either:
      - a prefill batch: (seq, chunk_token_ids, start_pos) to pass to engine.prefill_step
      - a decode batch: list[SequenceState] with their last token to engine.decode_step
    """

    def __init__(
        self,
        kvcache: PagedKVCache,
        max_batch_size: int,
        prefill_chunk_size: int,
    ) -> None:
        self._kvcache = kvcache
        self._max_batch = max_batch_size
        self._chunk_size = prefill_chunk_size

        self._waiting: deque[tuple[SequenceState, SamplingParams]] = deque()
        # sequences whose prefill is complete and are being decoded
        self._decoding: list[tuple[SequenceState, SamplingParams]] = []
        # sequences mid-prefill (chunked)
        self._prefilling: Optional[tuple[SequenceState, SamplingParams]] = None

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._prefilling or self._decoding)

    def add_request(self, req: GenerateRequest) -> None:
        seq = _request_to_seq(req)
        self._kvcache.prepare_sequence(seq)
        params = req.sampling_params
        self._waiting.append((seq, params))

    def step(self) -> Optional[tuple[str, object]]:
        """Return the next unit of work as ("prefill", (seq, chunk, start)) or
        ("decode", [(seq, params, last_token_id), ...]), or None if idle.
        """
        # priority 1: continue an in-progress chunked prefill
        if self._prefilling is not None:
            seq, params = self._prefilling
            chunk, start = self._next_prefill_chunk(seq)
            if chunk:
                return ("prefill", (seq, params, chunk, start))
            # prefill complete
            self._prefilling = None
            self._decoding.append((seq, params))

        # priority 2: start prefill for a waiting sequence
        if self._waiting:
            seq, params = self._waiting.popleft()
            self._kvcache.ensure_blocks(seq)
            chunk, start = self._next_prefill_chunk(seq)
            if chunk:
                # _next_prefill_chunk already advanced seq.prefill_cursor
                if seq.prefill_cursor < seq.num_prompt_tokens:
                    self._prefilling = (seq, params)
                else:
                    self._decoding.append((seq, params))
                return ("prefill", (seq, params, chunk, start))

        # priority 3: decode batch
        if self._decoding:
            batch = self._decoding[: self._max_batch]
            items = []
            for seq, params in batch:
                last_token = (
                    seq.generated_token_ids[-1]
                    if seq.generated_token_ids
                    else seq.prompt_token_ids[-1]
                )
                self._kvcache.ensure_blocks(seq)
                items.append((seq, params, last_token))
            return ("decode", items)

        return None

    def finish_prefill(self, seq: SequenceState) -> None:
        """Called after engine.prefill_step to advance cursor."""
        # cursor already advanced in step(); mark complete if at end
        if seq.prefill_cursor >= seq.num_prompt_tokens:
            self._kvcache.finish_sequence(seq)

    def on_token(self, seq: SequenceState, token_id: int, params: SamplingParams) -> bool:
        """Record generated token; return True if sequence is finished."""
        seq.generated_token_ids.append(token_id)
        stop = (
            token_id in params.stop_token_ids
            or len(seq.generated_token_ids) >= params.max_new_tokens
        )
        if stop:
            reason = "stop" if token_id in params.stop_token_ids else "length"
            seq.finished = True
            seq.finish_reason = reason
            self._decoding = [(s, p) for s, p in self._decoding if s.request_id != seq.request_id]
            self._kvcache.free_sequence(seq)
            return True
        return False

    def _next_prefill_chunk(self, seq: SequenceState) -> tuple[list[int], int]:
        cursor = seq.prefill_cursor
        end = min(cursor + self._chunk_size, seq.num_prompt_tokens)
        chunk = seq.prompt_token_ids[cursor:end]
        seq.prefill_cursor = end
        return chunk, cursor


# ---------------------------------------------------------------------------
# Prefill scheduler (disaggregated mode)
# ---------------------------------------------------------------------------

class PrefillScheduler:
    """Processes prompts in chunks; produces finished-prefill events for handoff."""

    def __init__(
        self,
        kvcache: PagedKVCache,
        prefill_chunk_size: int,
    ) -> None:
        self._kvcache = kvcache
        self._chunk_size = prefill_chunk_size
        self._queue: deque[tuple[SequenceState, SamplingParams]] = deque()

    @property
    def has_work(self) -> bool:
        return bool(self._queue)

    def add_request(self, req: GenerateRequest) -> None:
        seq = _request_to_seq(req)
        self._kvcache.prepare_sequence(seq)
        self._kvcache.ensure_blocks(seq)
        self._queue.append((seq, req.sampling_params))

    def next_chunk(self) -> Optional[tuple[SequenceState, SamplingParams, list[int], int]]:
        """Return (seq, params, chunk, start_pos) or None."""
        if not self._queue:
            return None
        seq, params = self._queue[0]
        cursor = seq.prefill_cursor
        end = min(cursor + self._chunk_size, seq.num_prompt_tokens)
        chunk = seq.prompt_token_ids[cursor:end]
        seq.prefill_cursor = end
        if seq.prefill_cursor >= seq.num_prompt_tokens:
            self._queue.popleft()
            self._kvcache.finish_sequence(seq)
            seq.finished = False  # not finished — handing off to decode worker
        return seq, params, chunk, cursor


# ---------------------------------------------------------------------------
# Decode scheduler (disaggregated mode)
# ---------------------------------------------------------------------------

class DecodeScheduler:
    """Batches all in-flight sequences for continuous decode."""

    def __init__(
        self,
        kvcache: PagedKVCache,
        max_batch_size: int,
    ) -> None:
        self._kvcache = kvcache
        self._max_batch = max_batch_size
        self._seqs: dict[str, tuple[SequenceState, SamplingParams]] = {}

    @property
    def has_work(self) -> bool:
        return bool(self._seqs)

    def add_sequence(self, seq: SequenceState, params: SamplingParams) -> None:
        """Accept a sequence whose KV has been transferred from a prefill worker."""
        self._kvcache.ensure_blocks(seq)
        self._seqs[seq.request_id] = (seq, params)

    def next_batch(self) -> list[tuple[SequenceState, SamplingParams, int]]:
        """Return up to max_batch_size (seq, params, last_token_id) tuples."""
        items = []
        for seq, params in list(self._seqs.values())[: self._max_batch]:
            last_token = (
                seq.generated_token_ids[-1]
                if seq.generated_token_ids
                else seq.prompt_token_ids[-1]
            )
            self._kvcache.ensure_blocks(seq)
            items.append((seq, params, last_token))
        return items

    def on_token(self, seq: SequenceState, token_id: int, params: SamplingParams) -> bool:
        seq.generated_token_ids.append(token_id)
        stop = (
            token_id in params.stop_token_ids
            or len(seq.generated_token_ids) >= params.max_new_tokens
        )
        if stop:
            reason = "stop" if token_id in params.stop_token_ids else "length"
            seq.finished = True
            seq.finish_reason = reason
            del self._seqs[seq.request_id]
            self._kvcache.free_sequence(seq)
            return True
        return False
