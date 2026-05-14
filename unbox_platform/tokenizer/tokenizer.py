"""Lightweight wrapper around a trained tokenizer for use in training/inference."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer


class Tokenizer:
    """Thin wrapper around a saved tokenizers.Tokenizer with convenience methods."""

    def __init__(self, tokenizer_path: Path | str) -> None:
        path = Path(tokenizer_path)
        if path.is_dir():
            path = path / "tokenizer.json"
        self._tok = _HFTokenizer.from_file(str(path))
        self._tok.enable_padding(pad_id=self.pad_id, pad_token="<|pad|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("<|eos|>")

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<|pad|>")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self._tok.id_to_token(id)
