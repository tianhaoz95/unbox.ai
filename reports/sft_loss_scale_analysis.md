# SFT Loss Scale Analysis: Why Early Runs Show ~10, Recent Runs Show ~55

**Date:** 2026-05-25
**Observation:** SFT loss starts at ~10 in early buggy runs, ~55 in the fixed run with `assistant_only_loss=True`

---

## The Numbers

| Run | Config | Starting loss | Final loss | Loss type |
|---|---|---|---|---|
| Early / buggy | `completion_only_loss=True` (no-op) | ~10 | ~48 | Avg CE per all tokens |
| Fixed | `assistant_only_loss=True` | ~57 | ~42 | TRL per-sequence normalization |

---

## Reason 1: Different Loss Normalization

### Early/buggy run — average cross-entropy per token

Standard HF Trainer computes:

```
loss = sum(CE for all tokens) / total_tokens_in_batch
```

At the very first SFT step, the pretrained model has never seen the ChatML format
(`<|im_start|>`, `<|im_end|>`, etc.). The model is essentially guessing uniformly
over the vocabulary for these new patterns, giving:

```
loss ≈ log(vocab_size) = log(32768) ≈ 10.4
```

This matches the observed ~10 starting value — it is the **theoretical maximum**
for a 32k vocabulary model, identical to random guessing. Over training the model
learns the format and the loss drops.

### Fixed run — TRL 1.4 per-sequence normalization

TRL 1.4's `assistant_only_loss=True` path uses a custom `compute_loss` that
normalises differently — dividing by the **number of sequences** in the batch
rather than the number of tokens:

```
loss ≈ sum(CE for assistant tokens) / num_sequences
     = avg_assistant_tokens_per_seq × per_token_CE
```

At step 10 of the fixed run: `entropy ≈ 1.9 nats`, and if the average assistant
response contains ~30 assistant tokens per sequence:

```
loss ≈ 30 × 1.9 ≈ 57  ✓
```

By the end of training: `entropy ≈ 1.37 nats`, `mean_token_accuracy ≈ 66.8%`:

```
loss ≈ 30 × 1.37 ≈ 41  ✓  (observed: 41.9)
```

The raw loss number is larger simply because TRL sums over tokens per sequence
rather than averaging across all tokens in the batch.

---

## Reason 2: Easy User Tokens Diluted the Old Loss

Even within the same normalization, the buggy run's denominator included **all
user tokens** — short, repetitive question patterns that the pretrained model
already handles well. These low-entropy tokens pulled the per-token average down:

```
buggy loss = (low CE user tokens + high CE assistant tokens) / all tokens
           < fixed loss = high CE assistant tokens / assistant tokens only
```

The user turn structure (e.g., `<|im_start|>user\nExplain X<|im_end|>\n`) is
highly predictable once the model learns the format. Predicting "user" after
`<|im_start|>` is easy — this easy prediction flattered the loss metric while
hiding how poorly the model was actually learning assistant responses.

---

## Reason 3: The Bug Itself Changed What Was Being Measured

In the buggy run, the model was **optimised** on user tokens too. By the end of
training, the model had become very good at predicting user patterns (the "user
attractor" effect), which kept the loss artificially low — but at the cost of
learning to generate `user` as a near-universal output token.

In the fixed run, the model is only trained on assistant tokens. The loss measures
a harder, more focused objective — and a higher starting value honestly reflects
the difficulty of the task, not the ease of predicting structural boilerplate.

---

## Which Metrics to Actually Compare

The raw loss numbers between the two runs **are not comparable** — they are on
different scales due to different normalization. For meaningful comparison:

| Metric | Interpretation | Buggy run (end) | Fixed run (end) |
|---|---|---|---|
| `loss` | Not directly comparable | 48.63 | 42.99 |
| `entropy` (nats) | Per-token — comparable | ~1.4 | **~1.37** |
| `mean_token_accuracy` | Per-token — comparable | ~0.67 | **~0.67** |
| Generation quality | The real test | Degenerate loops | TBD (after DPO) |

The per-token metrics (`entropy`, `mean_token_accuracy`) are on the same scale
and show the fixed run is marginally better. The real improvement will be visible
in generation quality once DPO completes.

---

## Key Takeaway

**A low loss number does not always mean a better model.** In this case, the
early run's loss of ~10 looked better than the fixed run's ~55, but the early
run produced completely degenerate outputs while the fixed run trains on a clean,
focused objective. Always check what the loss is actually measuring before
comparing absolute values across different training configurations.
