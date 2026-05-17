# 3 · Tokenizer Training

## What a tokenizer does

A tokenizer converts raw text into a sequence of integer IDs that a neural network can process. The choice of tokenization scheme affects:

- **Vocabulary efficiency** — how many tokens are needed to represent average text
- **Compression ratio** — how many characters map to one token (higher = fewer tokens = faster training)
- **Unknown token handling** — whether out-of-vocabulary characters can be represented
- **Multilingual coverage** — how well the vocabulary covers non-English text

## Why train from scratch

Using a pre-existing tokenizer (e.g., GPT-4's cl100k_base) would tie us to someone else's vocabulary design choices. Training our own gives us:

1. A vocabulary optimised for our specific corpus (FineWeb-Edu, predominantly English)
2. Control over special token layout
3. A complete understanding of every component — no black boxes

## Algorithm: ByteLevel BPE

We use **Byte-Level BPE** (Byte Pair Encoding operating on UTF-8 bytes rather than Unicode characters):

- **Byte-level** — every possible byte value (0–255) is in the base vocabulary. This means any UTF-8 string can be represented without `<unk>` tokens, regardless of language or character set.
- **BPE** — iteratively merge the most frequent adjacent token pairs, building up the vocabulary from single bytes to common subwords and words.

This is the same scheme used by GPT-2, GPT-4, and Llama.

## Vocabulary size: 32,768

| Vocab size | Compression ratio | Tradeoff |
|---|---|---|
| 6,400 (minimind) | ~3–4 chars/token | Fast but poor on English; sequences are longer |
| **32,768 (ours)** | **~4 chars/token** | Good balance for English-heavy corpus |
| 100,000+ (GPT-4) | ~4.5 chars/token | Better multilingual coverage; larger embedding table |

32,768 = 2^15, which keeps the embedding table size manageable (~60M parameters at hidden_size=1792) while achieving good compression on English text.

## Special tokens

| Token | ID | Purpose |
|---|---|---|
| `<unk>` | 0 | Unknown (never used with ByteLevel BPE, but required by the spec) |
| `<pad>` | 1 | Padding (used in batched inference) |
| `<s>` | 2 | Beginning of sequence (BOS) |
| `</s>` | 3 | End of sequence (EOS) |
| `<|im_start|>` | 4 | ChatML turn start |
| `<|im_end|>` | 5 | ChatML turn end |

## Chat template: ChatML

The tokenizer includes a Jinja2 chat template that formats multi-turn conversations for instruction tuning:

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>assistant
```

This is the **ChatML** format, used by Qwen, InternLM, and others. The `<|im_start|>` / `<|im_end|>` tokens delimit each turn, and `add_generation_prompt=True` appends the assistant prefix to prompt generation.

The template is stored in `tokenizer_config.json` and applied automatically by HuggingFace's tokenizer infrastructure — SFTTrainer reads it from there when formatting training examples.

## Training

```bash
.venv/bin/python -m unbox_platform.tokenizer.train \
    --data data/fineweb_edu_10bt.jsonl \
    --output checkpoints/tokenizer
```

The trained tokenizer is saved in HuggingFace format (`tokenizer.json` + `tokenizer_config.json`), making it compatible with `PreTrainedTokenizerFast.from_pretrained()`.

## HuggingFace compatibility

The tokenizer wraps the trained BPE model in `PreTrainedTokenizerFast`, which means:

- TRL's `SFTTrainer` can apply the chat template automatically
- The tokenizer can be saved and loaded with `save_pretrained` / `from_pretrained`
- It integrates with the broader HF ecosystem for evaluation
