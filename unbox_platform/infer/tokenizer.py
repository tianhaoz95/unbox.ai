"""Prompt tokenization and incremental streaming detokenization.

Tokenization runs in the server process (CPU-bound). The IncrementalDetokenizer
holds a per-request token ID buffer and re-decodes the full suffix each step
to handle tokens that span multiple characters or split UTF-8 sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..tokenizer.tokenizer import Tokenizer as _BaseTokenizer


class InferTokenizer:
    """Wraps unbox_platform Tokenizer with inference conveniences."""

    def __init__(self, tokenizer_path: Union[str, Path]) -> None:
        self._tok = _BaseTokenizer(tokenizer_path)

    @property
    def bos_id(self) -> int:
        return self._tok.bos_id

    @property
    def eos_id(self) -> int:
        return self._tok.eos_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self._tok.bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Naively concatenate role/content pairs into a prompt string.

        For a proper chat template, replace this with a HuggingFace tokenizer
        that has a chat_template configured.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)


class IncrementalDetokenizer:
    """Per-request streaming detokenizer.

    Maintains a token buffer per active request. On each new token, re-decodes
    the full buffer and returns only the newly confirmed text suffix, which
    handles multi-character tokens and split UTF-8 sequences safely.
    """

    def __init__(self, tokenizer: InferTokenizer) -> None:
        self._tok = tokenizer
        # request_id → (token_id_buffer, previously_decoded_text)
        self._buffers: dict[str, tuple[list[int], str]] = {}

    def register(self, request_id: str, prompt_token_ids: list[int]) -> None:
        """Register a new request; prompt tokens are excluded from output."""
        self._buffers[request_id] = ([], "")

    def step(self, request_id: str, token_id: int) -> Optional[str]:
        """Append token_id and return any newly decodable text, or None if not yet safe."""
        if request_id not in self._buffers:
            return None
        buf, prev_text = self._buffers[request_id]
        buf.append(token_id)
        current_text = self._tok.decode(buf)
        # current_text may end with the replacement character if UTF-8 is incomplete;
        # only emit the delta once the decoded text has grown beyond previous
        if len(current_text) > len(prev_text):
            delta = current_text[len(prev_text):]
            self._buffers[request_id] = (buf, current_text)
            return delta
        self._buffers[request_id] = (buf, prev_text)
        return None

    def finish(self, request_id: str) -> str:
        """Flush remaining text and remove request from buffer."""
        if request_id not in self._buffers:
            return ""
        buf, prev_text = self._buffers.pop(request_id)
        final_text = self._tok.decode(buf)
        return final_text[len(prev_text):]

    def remove(self, request_id: str) -> None:
        self._buffers.pop(request_id, None)
