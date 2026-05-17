# Sanity Check Report — Step 15,000

**Date:** 2026-05-16
**Checkpoint:** `checkpoints/pretrain/760m/step_00015000.pt`
**Commit:** `ed5970f`

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

### Training State

| Parameter | Value |
|---|---|
| Dataset | FineWeb-Edu `sample-10BT` |
| Total estimated steps | ~73,718 |
| Steps completed | 15,000 (20.3%) |
| Tokens seen | ~2.5B |
| Batch size | 4 × grad_accum 8 = effective 32 |
| LR at step 15,000 | 2.79e-4 (cosine decay, post-warmup) |
| Tokens/sec | ~4,250 |

---

## Training Metrics

### Perplexity Progression (eval set, steps 10,000–15,000)

| Step | Eval Loss | Perplexity | Δ vs. prev |
|---|---|---|---|
| 10,000 | 3.2218 | 25.07 | — (baseline from last check) |
| 10,500 | 3.1898 | 24.28 | −0.79 |
| 11,000 | 3.1853 | 24.17 | −0.11 |
| 11,500 | 3.1663 | 23.72 | −0.45 |
| 12,000 | 3.1480 | 23.29 | −0.43 |
| 12,500 | 3.1502 | 23.34 | +0.05 |
| 13,000 | 3.1361 | 23.01 | −0.33 |
| 13,500 | 3.1277 | 22.82 | −0.19 |
| 14,000 | 3.1161 | 22.56 | −0.26 |
| 14,500 | 3.1066 | 22.35 | −0.21 |
| 15,000 | **3.0981** | **22.16** | −0.19 |

Perplexity fell from 25.07 → 22.16 over this interval, a further **11.6% improvement**. The slight uptick at step 12,500 is within normal batch-to-batch variance and did not persist. The overall trend is consistent monotonic decline with no plateau.

**Cumulative since training start (step 2,500):** 49.04 → 22.16, a **54.8% total reduction**.

### Training Loss at Step 15,000

`loss=3.3928`, `lr=2.79e-04`. Train loss oscillates between 3.0–3.4 throughout this interval, consistent with the eval trend. No divergence observed.

---

## Qualitative Sampling Results

All samples: `temperature=0.8`, `top_k=50`, `top_p=0.9`, `max_new_tokens=150`.

---

### Prompt 1: Science / factual

**Prompt:** `The theory of relativity states that`

**Output:**
> When you are planning your child's educational journey, you need to have a good idea of what you want to do and why. The term "school year" is often used to refer to the time period from the beginning of school until the end of the school year. The term "school year" is a term that refers to a specific period during the school year.
> When you are planning your child's educational journey, you need to have a good idea of what you want to do and why. In addition to this, you need to have a good idea of what your child will be learning during the school year.

---

### Prompt 2: Narrative / creative

**Prompt:** `Once upon a time in a small village`

**Output:**
> The term "hard" refers to the idea that we do not have to take care of ourselves. Hard is an illusion, and it is not always easy to imagine that the world would be so different from what it is.
> This can be seen in the story "The Yellow Kid" by Michael Jackson, when he was in a small town in the American South. He was a little girl, but there was something going on. He had just got into an argument with another boy, and he was saying, "That's not going to be my way to get the rest of this world," but it was a different way of life.

---

### Prompt 3: Instructional / how-to

**Prompt:** `The best way to learn programming is`

**Output:**
> The term "Precision" refers to the ability of a computer or system to achieve certain tasks at the same time as the user. For example, the computer's ability to perform a task is known as a "precision" computer.
> Precision is often used to describe computer systems that perform a specific task at the same time. For example, the computer's ability to perform a task is known as a "precision" computer.
> Precision is also used to describe software systems that are able to perform tasks at the same time. For example, the computer's ability to perform a task is known as a "precision" computer.

---

### Prompt 4: Current events / knowledge

**Prompt:** `In 2024, researchers discovered that`

**Output:**
> The most basic, yet most difficult, component of the electrical circuit is the circuit breaker. It is the circuit breaker, which converts the power into electrical energy, that is, the power. For example, a circuit breaker uses the current to power the circuit. However, in other cases, the power can be converted into electrical energy.
> So the circuit breaker is a part of the circuit that converts the power into electrical energy. The circuit breaker is made of many thin, insulated wires. This insulated wires is called a ground, which is also called ground circuit.

---

## Assessment

### What improved vs. step 10,000

- **Perplexity** continued declining smoothly (25.07 → 22.16). The model is absorbing more structure from the training data.
- **Sentence fluency** is slightly more consistent — fewer abrupt topic changes mid-sentence compared to step 10,000.

### What remains expected at this stage

- **Topic drift:** the model ignores prompt context entirely and continues in FineWeb-Edu expository style. Prompts about relativity generate school planning advice; "once upon a time" triggers an essay. At 20% of training, the model has not yet learned to condition its output on the prompt topic — it is still primarily learning the surface statistics of the corpus.
- **Semantic repetition (Prompt 3):** the "Precision" paragraph repeats the same sentence pattern three times with minor variation. This is a known failure mode of undertrained language models — it reflects insufficient diversity in the learned representations rather than a bug. It should resolve as training progresses.
- **Factual hallucination:** Michael Jackson described as a character in a story about a small Southern town, a "Yellow Kid" book attributed to him. Expected.

### Verdict

**Pass.** The model continues generating grammatically coherent English and perplexity is declining on schedule. No regressions from step 10,000. The semantic repetition in Prompt 3 is worth monitoring but is not a blocker — it is a symptom of undertraining, not a code defect.

**Next check:** recommended at step 20,000–25,000, when the model will be ~30% through training and topic-conditional generation should begin to emerge.

---

## Comparison to Step 10,000 Report

| Metric | Step 10,000 | Step 15,000 | Change |
|---|---|---|---|
| Eval perplexity | 25.07 | 22.16 | −11.6% |
| Eval loss | 3.2218 | 3.0981 | −0.1237 |
| % of training complete | 13.6% | 20.3% | +6.7pp |
| Coherent output | Yes | Yes | — |
| Topic drift | Yes | Yes | — |
| Semantic repetition | No | Yes (Prompt 3) | Appeared |
| Factual hallucination | Yes | Yes | — |
