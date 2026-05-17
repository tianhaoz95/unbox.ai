"""Inference engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 512
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class InferConfig:
    # --- paths ---
    model_path: str = "checkpoints/pretrain/760m/latest.pt"
    model_config_path: str = "configs/pretrain/760m.yaml"
    tokenizer_path: str = "checkpoints/tokenizer"

    # --- serving mode ---
    # "unified": one worker pool handles prefill + decode (default, single-GPU friendly)
    # "disaggregated": separate prefill and decode worker pools
    mode: Literal["unified", "disaggregated"] = "unified"

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- worker topology ---
    # unified mode: num_workers workers each handle both phases
    # disaggregated mode: num_prefill_workers + num_decode_workers pools
    num_workers: int = 1
    num_prefill_workers: int = 1
    num_decode_workers: int = 1

    # tensor parallelism degree within each worker pool
    tensor_parallel_size: int = 1

    # --- ZMQ ports ---
    # server → router
    router_frontend_port: int = 5550
    # router → workers (requests)
    worker_request_port: int = 5551
    # workers → router (token stream)
    worker_result_port: int = 5552
    # prefill workers → decode workers (KV transfer, disaggregated only)
    kv_transfer_port: int = 5553

    # --- KV cache ---
    block_size: int = 16          # tokens per KV block
    max_num_blocks: int = 4096    # total blocks in the pool (~GPU VRAM budget)
    max_seq_len: int = 2048

    # --- scheduler ---
    max_batch_size: int = 32
    # chunked prefill: max tokens processed per prefill step
    prefill_chunk_size: int = 512

    # --- sampling defaults (per-request overridable) ---
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # --- precision ---
    dtype: str = "bfloat16"       # "bfloat16" | "float16" | "float32"
