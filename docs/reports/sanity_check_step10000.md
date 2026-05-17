# Sanity Check Report — Step 10,000

**Date:** 2026-05-15
**Checkpoint:** `checkpoints/pretrain/760m/step_00010000.pt`
**Commit:** `b9a2024`

---

## Context

### Model

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer (GQA, SwiGLU, RoPE) |
| Parameters | ~760M |
| Hidden size | 1792 |
| Layers | 24 |
| Attention heads | 16 (8 KV heads, GQA 2:1) |
| FFN intermediate | 4864 |
| Vocab size | 32,768 |
| Max seq len | 2048 |
| Positional encoding | RoPE (θ = 500,000) |
| Tied embeddings | Yes |

### Training Setup

| Parameter | Value |
|---|---|
| Dataset | FineWeb-Edu `sample-10BT` |
| Total estimated steps | ~73,718 |
| Steps completed | 10,000 (13.6%) |
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Max LR | 3.0e-4 |
| LR schedule | Cosine with 2,000 warmup steps |
| Dtype | bfloat16 |
| Hardware | NVIDIA GB10 (DGX Spark) |
| Tokens/sec | ~4,245 |

---

## Training Metrics at Step 10,000

### Perplexity Progression (eval set)

| Step | Eval Loss | Perplexity |
|---|---|---|
| 2,500 | 3.8927 | 49.04 |
| 3,000 | 3.7343 | 41.86 |
| 4,000 | 3.5897 | 36.22 |
| 5,000 | 3.4752 | 32.30 |
| 6,000 | 3.3854 | 29.53 |
| 7,000 | 3.3376 | 28.15 |
| 8,000 | 3.2888 | 26.81 |
| 9,000 | 3.2426 | 25.60 |
| 10,000 | **3.2218** | **25.07** |

Perplexity has dropped 49% since the first eval checkpoint (step 2,500) and continues declining steadily with no signs of plateauing.

### Training Loss (near step 10,000)

Training loss is oscillating between 3.0–3.4, consistent with the eval loss. No divergence or anomalous spikes observed.

---

## Qualitative Sampling Results

All samples generated with `temperature=0.8`, `top_k=50`, `top_p=0.9`, `max_new_tokens=120`.

---

### Prompt 1: Science / factual

**Prompt:** `The theory of relativity states that`

**Output:**
> I have a question about the origin of the term "universe." It seems that the idea is just that. It's a bit similar to what I have for sale in the "gray" or "red" versions of the Bible.
> I've been reading a book called "Messiah" by the late John Sennett and "The Mockingbird" by the late John Sennett. I read it and thought, "It's a wonderful thing." It's a very short, easy, and interesting piece of writing.

---

### Prompt 2: Narrative / creative

**Prompt:** `Once upon a time in a small village`

**Output:**
> You are not alone.
> It is not uncommon to hear the term "gay" used when discussing the subject of gay rights. It is also sometimes used to refer to those who have not been invited to a gay or lesbian relationship, but still have not been invited to a gay or lesbian relationship.
> When referring to gay people, it is important to remember that there is not a one-size-fits-all definition of a "gay," but that may not always be the case.

---

### Prompt 3: Instructional / how-to

**Prompt:** `The best way to learn programming is`

**Output:**
> The Science of Teaching
> by Christine R. Noyes, Ph.D.
> The Science of Teaching, by Christine R. Noyes, Ph.D., is one of the best scientific curricula. Students who are interested in the science of teaching and learning, and who can use the resources and techniques of this new approach, will be well prepared to pursue their studies in this highly interactive and effective environment.
> This course is aimed at science educators, and the science of teaching, that will provide an engaging and engaging classroom.

---

## Assessment

### What is working

- **Grammatical coherence:** all three outputs form correctly structured English sentences with proper subject-verb agreement, punctuation, and paragraph breaks.
- **Consistent style:** the model has internalized the expository prose style of FineWeb-Edu (educational web text), producing outputs that read like excerpts from articles or course descriptions.
- **No degenerate behaviour:** no repetition loops, no token-level gibberish, no empty output.

### What is expected at this stage

- **Factual hallucination:** the model invents plausible-sounding but incorrect facts (e.g., attributing books to fabricated authors, conflating unrelated topics). This is normal at 13.6% of the training budget — the model is learning language structure before factual associations.
- **Prompt drift:** outputs do not stay strictly on-topic (relativity → Bible → books). Instruction-following and topic grounding require more training tokens and, ultimately, SFT.

### Verdict

**Pass.** The model is learning to generate coherent English text as expected at this stage of pretraining. Perplexity is declining smoothly. Training resumed after this check.

---

## Bug Found During This Check

**`freqs_cis` corruption on dtype cast** (`model.to(bfloat16)`)

The RoPE frequency buffer (`freqs_cis`) is stored as `complex64`. When `model.to(bfloat16)` is called in the sampling script, PyTorch silently discards the imaginary parts, destroying all positional information and producing degenerate output (repeated `v` characters).

**Fix:** `Transformer.to()` now removes `freqs_cis` from the buffer dict before calling `super().to()`, then recomputes it on the correct device afterwards. This covers all callers (sampling, evaluation, fine-tuning). The same root cause — HF's `_fast_init` patching `torch.ones_like` — had previously been fixed in the HF adapter; this is the identical problem at the bare-model level.

**Commit:** `b9a2024`
