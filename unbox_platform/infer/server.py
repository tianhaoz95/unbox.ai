"""FastAPI inference server.

Endpoints
─────────
  POST /generate                   — simple text-in / text-out
  POST /v1/chat/completions        — OpenAI-compatible chat (streaming via SSE)
  GET  /v1/models                  — list available models

The server owns the tokenizer and per-request detokenization buffers.
Workers are launched as subprocesses at startup.

Usage
─────
  .venv/bin/python -m unbox_platform.infer.server --config configs/infer/serve.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import subprocess
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import InferConfig
from .messaging import SamplingParams
from .router import make_router
from .tokenizer import IncrementalDetokenizer, InferTokenizer


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateBody(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionBody(BaseModel):
    model: str = "unbox"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    stream: bool = True


# ---------------------------------------------------------------------------
# Global server state (populated in lifespan)
# ---------------------------------------------------------------------------

_config: Optional[InferConfig] = None
_tokenizer: Optional[InferTokenizer] = None
_detokenizer: Optional[IncrementalDetokenizer] = None
_router = None
# futures waiting for token stream: request_id → asyncio.Queue
_queues: dict[str, asyncio.Queue] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None


def _token_callback(
    request_id: str,
    token_id: int,
    finished: bool,
    finish_reason: Optional[str],
) -> None:
    """Called from the router's background thread on each new token."""
    global _loop
    if _loop is None or request_id not in _queues:
        return
    q = _queues[request_id]
    asyncio.run_coroutine_threadsafe(
        q.put((token_id, finished, finish_reason)), _loop
    )


# ---------------------------------------------------------------------------
# Worker subprocess management
# ---------------------------------------------------------------------------

_worker_procs: list[subprocess.Popen] = []


def _launch_workers(config: InferConfig, config_path: str) -> None:
    if config.mode == "unified":
        for i in range(config.num_workers):
            proc = subprocess.Popen([
                sys.executable, "-m", "unbox_platform.infer.worker",
                "--config", config_path,
                "--worker-type", "unified",
                "--worker-id", str(i),
            ])
            _worker_procs.append(proc)
    else:
        for i in range(config.num_prefill_workers):
            proc = subprocess.Popen([
                sys.executable, "-m", "unbox_platform.infer.worker",
                "--config", config_path,
                "--worker-type", "prefill",
                "--worker-id", str(i),
            ])
            _worker_procs.append(proc)
        for i in range(config.num_decode_workers):
            proc = subprocess.Popen([
                sys.executable, "-m", "unbox_platform.infer.worker",
                "--config", config_path,
                "--worker-type", "decode",
                "--worker-id", str(i),
            ])
            _worker_procs.append(proc)


def _stop_workers() -> None:
    for proc in _worker_procs:
        proc.terminate()
    for proc in _worker_procs:
        proc.wait()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="unbox.ai inference server")


@app.on_event("startup")
async def startup() -> None:
    global _loop
    _loop = asyncio.get_event_loop()


@app.on_event("shutdown")
async def shutdown() -> None:
    if _router is not None:
        _router.stop()
    _stop_workers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sampling_params(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> SamplingParams:
    assert _tokenizer is not None
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_token_ids=[_tokenizer.eos_id],
    )


async def _token_stream(request_id: str) -> AsyncIterator[tuple[int, bool, Optional[str]]]:
    q: asyncio.Queue = asyncio.Queue()
    _queues[request_id] = q
    try:
        while True:
            token_id, finished, finish_reason = await asyncio.wait_for(q.get(), timeout=30.0)
            yield token_id, finished, finish_reason
            if finished:
                break
    finally:
        _queues.pop(request_id, None)


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(body: GenerateBody):
    assert _tokenizer is not None and _detokenizer is not None and _router is not None

    request_id = str(uuid.uuid4())
    prompt_ids = _tokenizer.encode(body.prompt, add_bos=True)
    params = _make_sampling_params(body.max_new_tokens, body.temperature, body.top_k, body.top_p)

    _detokenizer.register(request_id, prompt_ids)
    _router.submit(prompt_ids, params, request_id)

    if body.stream:
        async def event_stream():
            async for token_id, finished, finish_reason in _token_stream(request_id):
                text = _detokenizer.step(request_id, token_id) or ""
                if finished:
                    text += _detokenizer.finish(request_id)
                yield f"data: {text}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # non-streaming: collect all tokens
    tokens: list[int] = []
    async for token_id, finished, _ in _token_stream(request_id):
        tokens.append(token_id)
        if finished:
            break
    _detokenizer.remove(request_id)
    text = _tokenizer.decode(tokens)
    return {"generated_text": text, "token_count": len(tokens)}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionBody):
    assert _tokenizer is not None and _detokenizer is not None and _router is not None

    request_id = str(uuid.uuid4())
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    prompt_text = _tokenizer.apply_chat_template(messages)
    prompt_ids = _tokenizer.encode(prompt_text, add_bos=False)
    params = _make_sampling_params(body.max_tokens, body.temperature, body.top_k, body.top_p)

    _detokenizer.register(request_id, prompt_ids)
    _router.submit(prompt_ids, params, request_id)

    if body.stream:
        async def sse_stream():
            import json
            async for token_id, finished, finish_reason in _token_stream(request_id):
                delta = _detokenizer.step(request_id, token_id) or ""
                if finished:
                    delta += _detokenizer.finish(request_id)
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": body.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    # non-streaming
    tokens: list[int] = []
    finish_reason = None
    async for token_id, finished, fr in _token_stream(request_id):
        tokens.append(token_id)
        if finished:
            finish_reason = fr
            break
    _detokenizer.remove(request_id)
    text = _tokenizer.decode(tokens)

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {"completion_tokens": len(tokens)},
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "unbox", "object": "model", "owned_by": "unbox-ai"}],
    }


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_server(config_path: str) -> None:
    global _config, _tokenizer, _detokenizer, _router

    import uvicorn

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    _config = InferConfig(**{k: v for k, v in raw.items() if k in InferConfig.__dataclass_fields__})

    _tokenizer = InferTokenizer(_config.tokenizer_path)
    _detokenizer = IncrementalDetokenizer(_tokenizer)
    _router = make_router(_config, _token_callback)

    _launch_workers(_config, config_path)

    uvicorn.run(app, host=_config.host, port=_config.port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    launch_server(args.config)


if __name__ == "__main__":
    main()
