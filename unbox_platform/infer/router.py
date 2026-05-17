"""Router / dispatcher.

Unified mode
────────────
  Accepts requests from the server and round-robins them across the worker pool.
  No KV transfer tracking needed.

Disaggregated mode
──────────────────
  Routes requests to the prefill pool. After a prefill worker finishes and sends
  the KV transfer, the router assigns the sequence to a decode worker.
  Tracks in-flight state to match token stream responses back to HTTP clients.

The router runs in its own thread within the server process (not a separate
process) to avoid the overhead of cross-process serialisation for the
request-dispatch hot path. It communicates with workers via ZMQ.
"""

from __future__ import annotations

import itertools
import threading
import uuid
from typing import Callable, Optional

from .config import InferConfig
from .messaging import (
    GenerateRequest,
    GenerateResponse,
    RequestSocket,
    ResultSocket,
    SamplingParams,
)


class Router:
    """Dispatches requests to workers and collects token-stream responses.

    token_callback(request_id, token_id, finished, finish_reason) is called
    from the router's background thread for each token produced.
    """

    def __init__(
        self,
        config: InferConfig,
        token_callback: Callable[[str, int, bool, Optional[str]], None],
    ) -> None:
        self.config = config
        self._callback = token_callback
        import zmq
        self._ctx = zmq.Context()

        # round-robin counter for worker assignment
        self._worker_counter = itertools.cycle(range(config.num_workers))

        # one PUSH socket per worker
        self._req_sockets: list[RequestSocket] = [
            RequestSocket(self._ctx, config.worker_request_port + i, bind=True)
            for i in range(config.num_workers)
        ]

        # subscribe to results from all workers
        self._res_socket = ResultSocket(self._ctx, config.worker_result_port, bind=True)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._result_loop, daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        """Dispatch a request. Returns request_id."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        req = GenerateRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        worker_idx = next(self._worker_counter)
        self._req_sockets[worker_idx].send(req)
        return request_id

    def stop(self) -> None:
        self._stop.set()
        for sock in self._req_sockets:
            sock.close()
        self._res_socket.close()
        self._ctx.term()

    def _result_loop(self) -> None:
        """Background thread: drain the result socket and call token_callback."""
        while not self._stop.is_set():
            resp = self._res_socket.recv_noblock()
            if resp is None:
                import time
                time.sleep(0.0005)
                continue
            self._callback(resp.request_id, resp.token_id, resp.finished, resp.finish_reason)


class DisaggregatedRouter(Router):
    """Router for disaggregated mode.

    Extends the base router with prefill-pool routing and KV-transfer tracking.
    Prefill workers are assigned via round-robin; decode workers are assigned
    per-request and embedded in the GenerateRequest so the prefill worker knows
    where to send the KV.
    """

    def __init__(
        self,
        config: InferConfig,
        token_callback: Callable[[str, int, bool, Optional[str]], None],
    ) -> None:
        self.config = config
        self._callback = token_callback
        import zmq
        self._ctx = zmq.Context()

        self._prefill_counter = itertools.cycle(range(config.num_prefill_workers))
        self._decode_counter = itertools.cycle(range(config.num_decode_workers))

        self._prefill_sockets: list[RequestSocket] = [
            RequestSocket(self._ctx, config.worker_request_port + i, bind=True)
            for i in range(config.num_prefill_workers)
        ]
        # decode workers listen on offset ports
        offset = config.num_prefill_workers
        self._decode_sockets: list[RequestSocket] = [
            RequestSocket(self._ctx, config.worker_request_port + offset + i, bind=True)
            for i in range(config.num_decode_workers)
        ]

        self._res_socket = ResultSocket(self._ctx, config.worker_result_port, bind=True)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._result_loop, daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())

        decode_worker = next(self._decode_counter)
        req = GenerateRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            assigned_decode_worker=decode_worker,
        )

        prefill_worker = next(self._prefill_counter)
        self._prefill_sockets[prefill_worker].send(req)
        return request_id

    def stop(self) -> None:
        self._stop.set()
        for sock in self._prefill_sockets + self._decode_sockets:
            sock.close()
        self._res_socket.close()
        self._ctx.term()

    def _result_loop(self) -> None:
        while not self._stop.is_set():
            resp = self._res_socket.recv_noblock()
            if resp is None:
                import time
                time.sleep(0.0005)
                continue
            self._callback(resp.request_id, resp.token_id, resp.finished, resp.finish_reason)


def make_router(
    config: InferConfig,
    token_callback: Callable[[str, int, bool, Optional[str]], None],
) -> Router:
    if config.mode == "disaggregated":
        return DisaggregatedRouter(config, token_callback)
    return Router(config, token_callback)
