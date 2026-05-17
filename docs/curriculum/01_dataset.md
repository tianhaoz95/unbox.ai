# 1 · Dataset Selection

## The core question

A language model is, at its foundation, a compression of its training data. The dataset choice is the single most consequential decision in the entire pipeline — it determines what the model can know, how it reasons, and what biases it carries.

## What makes a good pretraining corpus

A pretraining corpus needs to balance four competing properties:

| Property | Why it matters | Common failure mode |
|---|---|---|
| **Scale** | More tokens → better generalisation up to the Chinchilla-optimal point | Too small → underfitting; too large → diminishing returns |
| **Quality** | Educational, well-written text → better reasoning and language structure | Web crawls contain enormous amounts of spam, SEO content, and duplicates |
| **Diversity** | Broad topics → better transfer to downstream tasks | Domain-specific corpora produce domain-specific models |
| **Licensing** | Legal clarity for research and deployment | Many large corpora have unclear or restrictive terms |

## Our choice: FineWeb-Edu sample-10BT

We use [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) `sample-10BT` from Hugging Face — a 10-billion-token subset of FineWeb filtered for educational quality.

**Why FineWeb-Edu specifically:**

- **Quality filter.** FineWeb-Edu applies a classifier trained to score web pages on educational value (1–5 scale), keeping only high-scoring pages. This dramatically reduces spam, SEO content, and low-effort text compared to raw Common Crawl.
- **Right size for our budget.** At ~33,000 tokens/sec on our hardware, 10B tokens requires ~84 hours — matching our available compute window for the first training run.
- **Chinchilla alignment.** For a 760M parameter model, Chinchilla-optimal training is ~15B tokens. 10B tokens gets us to ~67% of optimal — sufficient to validate the pipeline and produce a capable base model.
- **Open license.** FineWeb-Edu is released under ODC-By, with clear terms for research use.
- **English-heavy.** The corpus is predominantly English, matching our tokenizer and evaluation setup.

## Token budget calculation

```
Hardware:       NVIDIA GB10 (DGX Spark)
Throughput:     ~4,250 tokens/sec (measured)
Training time:  ~84 hours for 10B tokens

760M parameters × 20 tokens/param (Chinchilla) = 15.2B optimal tokens
10B tokens = 66% of Chinchilla-optimal → good enough for pipeline validation
```

## What we chose not to use

| Dataset | Why rejected |
|---|---|
| The Pile | Older; less quality filtering; some subsets have unclear licensing |
| RedPajama | Good quality but 1.2T tokens — far exceeds our token budget |
| C4 | Decent quality but English-only and heavily deduplicated to the point of reduced diversity |
| Raw Common Crawl | Requires significant cleaning work; 300B+ tokens — too large |

## Downloading the data

```bash
.venv/bin/python -m unbox_platform.data.prepare \
    --output data/fineweb_edu_10bt.jsonl
```

The download streams shards from Hugging Face Hub with progress reporting. The output is a JSONL file where each line has a `"text"` field — the format expected by the pretraining data pipeline.

## Data pipeline

Once downloaded, the data flows through:

```
JSONL file
    ↓ StreamingPretrainDataset (lazy iteration, no eager tokenisation)
    ↓ Tokenise on-the-fly: [BOS] + tokens + [EOS]
    ↓ Pack into max_seq_len=2048 chunks
    ↓ DataLoader → training loop
```

Lazy streaming avoids loading 30GB into memory. The `skip_chunks` mechanism allows training to resume from a checkpoint without re-processing already-seen data.
