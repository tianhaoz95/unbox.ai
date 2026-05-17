"""Worker process entry point.

Each worker is a separate Python process. It:
  1. Initialises distributed (NCCL) if tensor_parallel_size > 1
  2. Loads the model and KV cache via InferenceEngine
  3. Runs a scheduler loop, polling for requests over ZMQ
  4. Sends generated tokens back over ZMQ

Worker types
────────────
  "unified"  — handles both prefill and decode (unified mode)
  "prefill"  — chunked prefill only; transfers KV to decode workers (disaggregated)
  "decode"   — continuous decode; receives KV from prefill workers (disaggregated)

Entry point
───────────
  python -m unbox_platform.infer.worker --config ... --worker-type unified --worker-id 0
"""

from __future__ import annotations

import argparse
import time
from typing import Literal

import torch
import yaml
from .config import InferConfig
from .distributed import apply_tp_to_model, get_tp_rank, get_tp_world_size, init_worker_group
from .engine import InferenceEngine
from .kvcache import SequenceState
from .messaging import (
    GenerateRequest,
    GenerateResponse,
    KVTransferMsg,
    RequestSocket,
    ResultSocket,
    KVTransferSocket,
    SamplingParams,
)
from .scheduler import DecodeScheduler, PrefillScheduler, UnifiedScheduler


WorkerType = Literal["unified", "prefill", "decode"]


def _get_device(worker_id: int) -> torch.device:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return torch.device(f"cuda:{worker_id % num_gpus}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Unified worker
# ---------------------------------------------------------------------------

def run_unified_worker(config: InferConfig, worker_id: int) -> None:
    device = _get_device(worker_id)
    engine = InferenceEngine(config, device)

    tp_rank = get_tp_rank()
    tp_world = get_tp_world_size()
    if tp_world > 1:
        apply_tp_to_model(engine.model.model, tp_rank, tp_world)

    scheduler = UnifiedScheduler(
        engine.kvcache,
        config.max_batch_size,
        config.prefill_chunk_size,
    )

    import zmq
    ctx = zmq.Context()
    req_sock = RequestSocket(ctx, config.worker_request_port + worker_id, bind=False)
    res_sock = ResultSocket(ctx, config.worker_result_port, bind=False)

    print(f"[worker-{worker_id}] unified worker ready on {device}")

    while True:
        # drain incoming requests (non-blocking)
        while True:
            req = req_sock.recv_noblock()
            if req is None:
                break
            scheduler.add_request(req)

        work = scheduler.step()
        if work is None:
            time.sleep(0.001)
            continue

        kind, payload = work

        if kind == "prefill":
            seq, params, chunk, start_pos = payload
            logits = engine.prefill_step(seq, chunk, start_pos)
            scheduler.finish_prefill(seq)
            if logits is not None and seq.prefill_cursor >= seq.num_prompt_tokens:
                # prefill complete — sample first decode token
                token_id = engine.sample_batch(logits.unsqueeze(0), params)[0]
                finished = scheduler.on_token(seq, token_id, params)
                resp = GenerateResponse(
                    request_id=seq.request_id,
                    token_id=token_id,
                    finished=finished,
                    finish_reason=seq.finish_reason,
                )
                res_sock.send(resp)

        elif kind == "decode":
            items = payload
            seqs = [s for s, _, _ in items]
            token_ids = [t for _, _, t in items]
            params_list = [p for _, p, _ in items]

            logits = engine.decode_step(seqs, token_ids)

            for i, (seq, params, _) in enumerate(items):
                next_token = engine.sample_batch(logits[i:i+1], params)[0]
                finished = scheduler.on_token(seq, next_token, params)
                resp = GenerateResponse(
                    request_id=seq.request_id,
                    token_id=next_token,
                    finished=finished,
                    finish_reason=seq.finish_reason,
                )
                res_sock.send(resp)


# ---------------------------------------------------------------------------
# Prefill worker (disaggregated)
# ---------------------------------------------------------------------------

def run_prefill_worker(config: InferConfig, worker_id: int) -> None:
    device = _get_device(worker_id)
    engine = InferenceEngine(config, device)

    scheduler = PrefillScheduler(engine.kvcache, config.prefill_chunk_size)

    import zmq
    ctx = zmq.Context()
    req_sock = RequestSocket(ctx, config.worker_request_port + worker_id, bind=False)
    kv_sock = KVTransferSocket(ctx, config.kv_transfer_port, bind=False)
    res_sock = ResultSocket(ctx, config.worker_result_port, bind=False)

    print(f"[prefill-worker-{worker_id}] ready on {device}")

    while True:
        while True:
            req = req_sock.recv_noblock()
            if req is None:
                break
            scheduler.add_request(req)

        result = scheduler.next_chunk()
        if result is None:
            time.sleep(0.001)
            continue

        seq, params, chunk, start_pos = result
        logits = engine.prefill_step(seq, chunk, start_pos)

        if seq.prefill_cursor >= seq.num_prompt_tokens:
            # prefill complete — sample first token and transfer KV to decode worker
            if logits is not None:
                token_id = engine.sample_batch(logits.unsqueeze(0), params)[0]
                res_sock.send(GenerateResponse(
                    request_id=seq.request_id,
                    token_id=token_id,
                    finished=False,
                ))
                seq.generated_token_ids.append(token_id)

            kv_bytes = engine.kvcache.serialize_blocks(seq.block_table)
            kv_msg = KVTransferMsg(
                request_id=seq.request_id,
                prompt_token_ids=seq.prompt_token_ids,
                sampling_params=params,
                block_table=seq.block_table,
                kv_data=kv_bytes,
            )
            kv_sock.send(kv_msg)


# ---------------------------------------------------------------------------
# Decode worker (disaggregated)
# ---------------------------------------------------------------------------

def run_decode_worker(config: InferConfig, worker_id: int) -> None:
    device = _get_device(worker_id)
    engine = InferenceEngine(config, device)

    scheduler = DecodeScheduler(engine.kvcache, config.max_batch_size)

    import zmq
    ctx = zmq.Context()
    kv_sock = KVTransferSocket(ctx, config.kv_transfer_port, bind=True)
    res_sock = ResultSocket(ctx, config.worker_result_port, bind=False)

    print(f"[decode-worker-{worker_id}] ready on {device}")

    while True:
        # accept KV transfers from prefill workers
        while True:
            msg = kv_sock.recv_noblock()
            if msg is None:
                break
            # reconstruct sequence state from transfer
            seq = SequenceState(
                request_id=msg.request_id,
                prompt_token_ids=msg.prompt_token_ids,
                block_table=list(msg.block_table),
                num_cached_tokens=len(msg.prompt_token_ids),
                prefill_cursor=len(msg.prompt_token_ids),
            )
            # deserialize KV into local pool
            # shape: (num_layers, 2, num_blocks_transferred, block_size, num_kv_heads, head_dim)
            num_blocks = len(msg.block_table)
            mc = engine.model_config
            shape = (mc.num_layers, 2, num_blocks, config.block_size, mc.num_kv_heads, mc.head_dim)
            engine.kvcache._pool.deserialize_blocks(msg.kv_data, msg.block_table, shape)
            scheduler.add_sequence(seq, msg.sampling_params)

        batch = scheduler.next_batch()
        if not batch:
            time.sleep(0.001)
            continue

        seqs = [s for s, _, _ in batch]
        token_ids = [t for _, _, t in batch]
        logits = engine.decode_step(seqs, token_ids)

        for i, (seq, params, _) in enumerate(batch):
            next_token = engine.sample_batch(logits[i:i+1], params)[0]
            finished = scheduler.on_token(seq, next_token, params)
            res_sock.send(GenerateResponse(
                request_id=seq.request_id,
                token_id=next_token,
                finished=finished,
                finish_reason=seq.finish_reason,
            ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-type", choices=["unified", "prefill", "decode"], default="unified")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-world", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    config = InferConfig(**{k: v for k, v in raw.items() if k in InferConfig.__dataclass_fields__})

    if args.tp_world > 1:
        init_worker_group(rank=args.tp_rank, world_size=args.tp_world)

    if args.worker_type == "unified":
        run_unified_worker(config, args.worker_id)
    elif args.worker_type == "prefill":
        run_prefill_worker(config, args.worker_id)
    elif args.worker_type == "decode":
        run_decode_worker(config, args.worker_id)


if __name__ == "__main__":
    main()
