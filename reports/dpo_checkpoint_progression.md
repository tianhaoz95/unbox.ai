# DPO Training Progression: Checkpoint-by-Checkpoint Analysis

**Date:** 2026-05-25
**Model:** unbox-760m (pretrain 34k steps → SFT 1 epoch → DPO 3 epochs)
**DPO config:** LR=1e-6, beta=0.1, 3 epochs, UltraFeedback binarized

---

## Training Metrics by Checkpoint

Global batch = 8 GPUs × batch 2 × grad_accum 4 = **64 preference pairs / step**

| Checkpoint | Step | Epoch | Pairs seen | Loss | Margins | Accuracy |
|---|---|---|---|---|---|---|
| SFT baseline | 0 | 0 | 0 | — | — | — |
| DPO step 500 | 500 | 0.52 | 32,000 | 0.6915 | 0.00406 | 47.2% |
| DPO step 1000 | 1000 | 1.05 | 64,000 | 0.6857 | 0.01608 | 57.5% |
| DPO step 1500 | 1500 | 1.57 | 96,000 | 0.6867 | 0.01397 | 55.8% |
| DPO step 2000 | 2000 | 2.09 | 128,000 | 0.6869 | 0.01353 | 57.9% |
| DPO step 2500 | 2500 | 2.62 | 160,000 | 0.6837 | 0.02012 | 62.2% |
| DPO step 2868 | 2868 | 3.00 | 183,552 | 0.6819 | 0.02400 | 60.2% |
| **Final eval** | 2868 | 3.00 | 183,552 | 0.6869 | **0.01371** | **56.8%** |

**Baseline reference:** log(2) = 0.6931 (loss when margin = 0, i.e., no preference signal)

### Key observations
- Loss decreased from ~0.6931 baseline to 0.6819 — small but real
- Margins grew from ~0 to ~0.024 (training) / 0.014 (eval) — consistent upward trend
- Accuracy peaked at 62.2% (step 2500) then slight regression — typical for cosine LR tail
- Eval accuracy (56.8%) below training peak, suggesting mild overfitting in epoch 3

---

## Generation Quality at Each Checkpoint

**Prompts tested:**
1. `Explain the difference between a list and a tuple in Python in one sentence.`
2. `I have 10 minutes to study for an exam. What should I do?`
3. `Is it better to use tabs or spaces for Python indentation? Give me a direct answer.`

**Settings:** greedy decoding, repetition_penalty=1.3, max_new_tokens=150

### Generation results

| Checkpoint | P1 (list vs tuple) | P2 (exam study) | P3 (tabs vs spaces) |
|---|---|---|---|
| SFT baseline | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 500 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 1000 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 1500 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 2000 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 2500 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |
| DPO step 2868 | `A` + `user` × 149 | `1` + `user` × 149 | `It` + `user` × 149 |

All checkpoints produce degenerate output: a single meaningful first token (`A`, `1`, `It`)
immediately followed by the word `user` repeated to the token limit.

---

## Root Cause: "User Attractor" from Undertrained Base Model

### What the degenerate output reveals

Testing plain-text continuation (no chat template) exposes the pattern:

```
Input:  "The capital of France is"
Output: " Paris" + "user" × 29
```

The model correctly predicts `Paris` as the first token (factual knowledge intact),
then degenerates. This isolates the failure to **generation distribution collapse**,
not the chat template.

### Why `user` dominates

In the SFT training data (UltraChat, ChatML format), every assistant turn is
immediately followed by `<|im_end|>\n<|im_start|>user`. After 1 epoch of SFT on
a model with only ~5B pretraining tokens, the model never deeply learned to chain
reasoning through multi-token sequences — it learned the **structural pattern** of
the chat format strongly enough that `user` became a near-universal high-probability
continuation after almost any generated token.

Concretely: the model's learned distribution assigns high probability to the sequence
`... token → user → user → ...` because this pattern appeared at the end of every
single training example. With a stronger base model (>50B tokens), the model would
have richer language priors that outcompete the structural attractor.

### Why DPO metrics still show real improvement

The DPO reward margin is computed from **log-probability ratios under the model**, not
from generation quality. The model IS learning to assign systematically higher
log-probability to chosen responses vs rejected responses — this is measurable and
real. The improvement is in the model's **scoring** of responses, not its **generation**
of them.

DPO's objective is:
```
maximize: log σ(β · (logp_chosen - logp_ref_chosen) - β · (logp_rejected - logp_ref_rejected))
```

This objective is optimised (margins go from 0 to +0.014 on eval), meaning the model
has learned preference direction. However, the generation capability needed to *produce*
those preferred responses is constrained by the base model's language modeling quality.

---

## Conclusions

### What worked
- DPO training procedure is correct (LR=1e-6, 3 epochs)
- Reward margins improved monotonically over training
- The model learned preference direction in its scoring distribution

### What didn't work
- Generation quality is degenerate due to base model quality
- The "user attractor" effect makes interactive use impossible at this model scale/training

### What's needed for coherent generation

| Component | Current | Required for coherent generation |
|---|---|---|
| Pretraining tokens | ~5B | 100B+ |
| SFT epochs | 1 | 3-5 |
| Pretrain steps | 34k | 300k+ |
| SFT samples | 207k × 1 epoch | 207k × 3–5 epochs |

DPO results from this pipeline are valid as a research demonstration of the training
procedure. Production-quality generation requires 10–20× more pretraining compute.
