"""ZMQ socket abstractions for intra-engine communication.

Message types
─────────────
  GenerateRequest   server → router → worker
  GenerateResponse  worker → router → server  (one per token, streamed)
  KVTransferMsg     prefill worker → decode worker  (disaggregated mode)

All messages are serialised with pickle over ZMQ PUSH/PULL or PUB/SUB.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import zmq as _zmq


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 512
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class GenerateRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    # disaggregated mode: set by router to direct prefill output to a decode worker
    assigned_decode_worker: Optional[int] = None


@dataclass
class GenerateResponse:
    request_id: str
    token_id: int
    finished: bool = False
    # finish_reason: "stop" | "length" | None
    finish_reason: Optional[str] = None


@dataclass
class KVTransferMsg:
    """Carries serialised KV blocks from a prefill worker to a decode worker."""
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # list of (layer_idx, block_idx_in_cache) for the decode worker to re-insert
    block_table: list[int]
    # raw KV tensors as CPU bytes; shape: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)
    kv_data: bytes


# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------

def _send(socket, obj: object) -> None:
    socket.send(pickle.dumps(obj))


def _recv(socket) -> object:
    return pickle.loads(socket.recv())


def _send_noblock(socket, obj: object) -> bool:
    import zmq
    try:
        socket.send(pickle.dumps(obj), zmq.NOBLOCK)
        return True
    except zmq.Again:
        return False


def _recv_noblock(socket) -> Optional[object]:
    import zmq
    try:
        return pickle.loads(socket.recv(zmq.NOBLOCK))
    except zmq.Again:
        return None


class RequestSocket:
    """PUSH socket: send GenerateRequests to the worker pool."""

    def __init__(self, ctx, port: int, bind: bool = False) -> None:
        import zmq
        self._sock = ctx.socket(zmq.PUSH)
        addr = f"tcp://*:{port}" if bind else f"tcp://localhost:{port}"
        if bind:
            self._sock.bind(addr)
        else:
            self._sock.connect(addr)

    def send(self, req: GenerateRequest) -> None:
        _send(self._sock, req)

    def recv(self) -> GenerateRequest:
        return _recv(self._sock)  # type: ignore[return-value]

    def recv_noblock(self) -> Optional[GenerateRequest]:
        return _recv_noblock(self._sock)  # type: ignore[return-value]

    def close(self) -> None:
        self._sock.close()


class ResultSocket:
    """PUB/SUB for streaming GenerateResponse tokens back to the server."""

    def __init__(self, ctx, port: int, bind: bool = False) -> None:
        import zmq
        socket_type = zmq.PUB if bind else zmq.SUB
        self._sock = ctx.socket(socket_type)
        addr = f"tcp://*:{port}" if bind else f"tcp://localhost:{port}"
        if bind:
            self._sock.bind(addr)
        else:
            self._sock.connect(addr)
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all

    def send(self, resp: GenerateResponse) -> None:
        _send(self._sock, resp)

    def recv(self) -> GenerateResponse:
        return _recv(self._sock)  # type: ignore[return-value]

    def recv_noblock(self) -> Optional[GenerateResponse]:
        return _recv_noblock(self._sock)  # type: ignore[return-value]

    def close(self) -> None:
        self._sock.close()


class KVTransferSocket:
    """PUSH/PULL for KV block transfers in disaggregated mode."""

    def __init__(self, ctx, port: int, bind: bool = False) -> None:
        import zmq
        self._sock = ctx.socket(zmq.PUSH if not bind else zmq.PULL)
        addr = f"tcp://*:{port}" if bind else f"tcp://localhost:{port}"
        if bind:
            self._sock.bind(addr)
        else:
            self._sock.connect(addr)

    def send(self, msg: KVTransferMsg) -> None:
        _send(self._sock, msg)

    def recv(self) -> KVTransferMsg:
        return _recv(self._sock)  # type: ignore[return-value]

    def recv_noblock(self) -> Optional[KVTransferMsg]:
        return _recv_noblock(self._sock)  # type: ignore[return-value]

    def close(self) -> None:
        self._sock.close()
