"""Train a 32K ByteLevel BPE tokenizer on a text corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


SPECIAL_TOKENS = [
    "<|pad|>",       # 0 — padding
    "<|bos|>",       # 1 — begin of sequence
    "<|eos|>",       # 2 — end of sequence
    "<|unk|>",       # 3 — unknown (rarely used with ByteLevel BPE)
    "<|sep|>",       # 4 — separator
    # Chat / instruction tokens
    "<|im_start|>",  # 5
    "<|im_end|>",    # 6
    # Reasoning
    "<think>",       # 7
    "</think>",      # 8
    # Tool use
    "<tool_call>",   # 9
    "</tool_call>",  # 10
    "<tool_response>",   # 11
    "</tool_response>",  # 12
    # Reserved buffer for future extensions
    "<|reserved_0|>",  # 13
    "<|reserved_1|>",  # 14
    "<|reserved_2|>",  # 15
    "<|reserved_3|>",  # 16
    "<|reserved_4|>",  # 17
    "<|reserved_5|>",  # 18
    "<|reserved_6|>",  # 19
    "<|reserved_7|>",  # 20
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    min_frequency: int = 2
    # Number of initial merges to use for the BPE trainer
    # (tokenizers library handles this automatically)


def _text_iterator(data_path: Path, max_lines: int | None = None) -> Iterator[str]:
    """Yield lines of text from a JSONL file (field: 'text') or plain text file."""
    count = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", ""))
            except json.JSONDecodeError:
                text = line
            if text:
                yield text
                count += 1
                if max_lines is not None and count >= max_lines:
                    break


def train_tokenizer(
    data_path: Path,
    output_dir: Path,
    config: TokenizerConfig | None = None,
    max_lines: int | None = None,
) -> None:
    if config is None:
        config = TokenizerConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _text_iterator(data_path, max_lines=max_lines),
        trainer=trainer,
    )

    # Post-processor: add BOS on encode, EOS on encode_pair
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|>:0 $A:0 <|eos|>:0",
        pair="<|bos|>:0 $A:0 <|eos|>:0 $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Write tokenizer_config.json for transformers compatibility
    config_dict = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 2048,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": CHAT_TEMPLATE,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Tokenizer saved to {output_dir}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL or text file")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-lines", type=int, default=None, help="Cap lines for quick runs")
    args = parser.parse_args()

    cfg = TokenizerConfig(vocab_size=args.vocab_size)
    train_tokenizer(args.data, args.output, cfg, max_lines=args.max_lines)


if __name__ == "__main__":
    main()
