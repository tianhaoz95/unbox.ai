# 2 · Data Cleaning & Preparation

## Why cleaning matters

Raw web text is noisy. A typical Common Crawl snapshot contains:

- Duplicate pages (exact and near-duplicate)
- Boilerplate: navigation menus, cookie notices, footer text
- Low-quality content: SEO spam, auto-generated text, scraped product listings
- Encoding errors and garbled text
- Personally identifiable information

Training on uncleaned data teaches the model to reproduce these patterns. Quality filtering is one of the highest-leverage interventions available — the FineWeb-Edu paper shows that filtering Common Crawl with an educational quality classifier produces models that score significantly higher on knowledge and reasoning benchmarks, with fewer training tokens.

## What FineWeb-Edu already does for us

By choosing FineWeb-Edu, we inherit a multi-stage cleaning pipeline:

1. **URL filtering** — removes known spam and adult content domains
2. **Language identification** — keeps English pages (using fastText)
3. **Quality heuristics** — removes pages with too many special characters, too-short lines, or repetitive content
4. **Deduplication** — MinHash deduplication at paragraph and document level
5. **Educational quality scoring** — a Llama-3-8B classifier scores each page 1–5 for educational value; only pages scoring ≥ 3 are kept

This means our data pipeline starts from already-clean data — we do not need to implement deduplication or quality filtering ourselves for this training run.

## What our pipeline does

Our data preparation layer (`unbox_platform/data/`) handles:

### Format normalisation

The raw dataset provides a `"text"` field per document. The pipeline:

1. Strips leading/trailing whitespace
2. Prepends `[BOS]` and appends `[EOS]` token IDs
3. Packs tokens into fixed-length `max_seq_len=2048` chunks — no padding wasted

### Streaming vs. eager loading

The original `PretrainDataset` loaded all documents eagerly — tokenising 9.6M documents upfront before training could begin. For a 30GB corpus this took too long and exhausted memory.

`StreamingPretrainDataset` fixes this with lazy iteration:

```python
def __iter__(self):
    buffer = []
    for text in self._iter_texts():           # reads one document at a time
        ids = [bos] + tokenizer.encode(text) + [eos]
        buffer.extend(ids)
        while len(buffer) >= max_seq_len:
            yield chunk(buffer[:max_seq_len]) # emit packed chunk
            buffer = buffer[max_seq_len:]
```

Memory usage stays constant regardless of corpus size.

### Resume correctness

When training resumes from a checkpoint at step `N`, the dataset skips the first `N × batch_size × grad_accumulation_steps` chunks — the exact number already consumed. This ensures the model never trains twice on the same data within an epoch.

```python
skip_chunks = start_step * grad_accumulation_steps * batch_size
train_loader.dataset.skip_chunks = skip_chunks
```

## Future work: custom cleaning

For training runs on raw Common Crawl (rather than FineWeb-Edu), `unbox_platform/data/` will need:

- **Deduplication** — exact hash deduplication + MinHash LSH for near-duplicates
- **Quality filtering** — perplexity-based filtering (high-perplexity text under a small reference LM tends to be low quality), or a trained classifier
- **Language filtering** — fastText language identification
- **PII removal** — regex-based scrubbing of emails, phone numbers, and other identifiers

These are not implemented yet — they will be added when the pipeline is extended to raw web data.
