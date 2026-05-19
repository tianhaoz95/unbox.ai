# DPO Failure Analysis: SFT Model Too Weak

## Summary

DPO training on `unbox-760m-sft` (checkpoint-3000) produced zero improvement after
2,500 steps. This report documents the diagnosis, the evidence that confirms the SFT
model is the root cause, the lessons learned, and the recommended next step.

---

## Hypothesis

**The SFT checkpoint is too weak a foundation for DPO.**

DPO works by amplifying an existing preference signal already latent in the policy
model. Specifically, the DPO gradient is proportional to the difference between the
policy's log-probability ratios for chosen vs. rejected responses. If the model assigns
nearly identical probabilities to both, the gradient is near zero and DPO cannot learn.

---

## Evidence

### 1. DPO loss stuck at ln(2)

The DPO loss for a model with no preference signal is:

```
L = -log(σ(0)) = ln(2) ≈ 0.6931
```

This is the value when the policy is identical to the reference model and assigns
equal probability to chosen and rejected responses.

| Statistic | Value |
|---|---|
| Mean train loss over 2,500 steps | **0.6929** |
| ln(2) | **0.6931** |
| Loss at step 10 | 0.6914 |
| Loss at step 2,500 | 0.6955 |
| Net change | **+0.0040** (no improvement) |

The loss never meaningfully departed from ln(2) across the entire run.

### 2. Rewards/margins are indistinguishable from noise

`rewards/margins` is the most important DPO metric — it measures the per-token
log-probability gap between chosen and rejected, relative to the reference model.
A working DPO run shows this metric growing steadily positive.

| Statistic | Value | Interpretation |
|---|---|---|
| Mean margin (steps 10–500, SFT init) | **+0.00012** | Essentially zero |
| Mean margin (full 2,500 steps) | **+0.00092** | Still essentially zero |
| Stdev of margins | 0.00462 | All variation is noise |
| Trend (slope per step) | +0.00000041 | +0.001 over 2,500 steps — negligible |

The SFT model's initial margins at step 10–500 (before DPO gradients have moved
the model) confirm the root cause: the SFT model starts with near-zero preference
discrimination, so there is no signal for DPO to amplify.

### 3. Accuracy stuck at 50%

`rewards/accuracies` measures what fraction of pairs the policy correctly ranks
(chosen > rejected). Random is 50%.

| Period | Mean accuracy |
|---|---|
| Steps 10–500 | **0.488** |
| Full 2,500 steps | **0.497** |

2,500 steps of DPO training moved this metric by less than 1 percentage point.

### 4. Eval loss flat across all checkpoints

| Step | Eval loss |
|---|---|
| 200 | 0.69279 |
| 800 | 0.69214 |
| 1600 | 0.69258 |
| 2400 | 0.69176 |

Variation is within ±0.001 — consistent with measurement noise, not learning.

### 5. Corroborating evidence from prior SFT baseline eval

Before DPO training, `dpo_eval.py` measured the SFT model's intrinsic preference
discrimination on 493 held-out pairs:

| Metric | Value |
|---|---|
| Win rate | **49.1%** (random = 50%) |
| Mean per-token margin | **+0.020** |

Win rate of 49.1% means the SFT model slightly *prefers* rejected responses to
chosen ones on average — it has no useful preference signal at all.

---

## Why the SFT Model Is Weak

The SFT checkpoint was trained for only **3,000 steps** (11.6% of one epoch on
UltraChat 200k). This is enough to make the model chatty and produce structurally
coherent responses, but not enough for it to internalize what makes a response
*good* vs. *bad*. Key failure modes observed in qualitative eval:

- Conflates "transformer model" (ML) with electrical transformer
- Repetition loops with no stopping criterion
- Ignores explicit format instructions ("use bullet points")
- Generates the wrong number of items

DPO can shift the model's preferences between two responses it already generates
fluently — it cannot inject knowledge the model doesn't have. With a model this
undertrained, the chosen and rejected responses look statistically identical to the
model: same token distribution, same log-probability magnitude, same structure.

---

## What Was Ruled Out

| Alternative hypothesis | Evidence against |
|---|---|
| Learning rate too low | Loss flat at exactly ln(2), not slowly converging. A low-LR run shows gradual improvement; this shows none. |
| Dataset quality issue | ultrafeedback_binarized is a standard, widely-used DPO dataset with clear quality gaps between chosen/rejected. The data is not the problem. |
| Implementation bug | DPO trainer was verified to start at ln(2) (expected) and the loss shape matches a model with zero gradient signal — not a numerical error pattern. |
| Too few steps | Margin trend of +0.00000041/step predicts +0.003 after 7,642 full steps. This is not a "needs more time" situation. |

---

## Lessons Learned

1. **DPO requires a capable SFT baseline.** The SFT model must already be able to
   generate the kinds of responses present in the preference dataset. 3,000 steps
   (11.6% of one epoch) is insufficient. Industry practice is 1–3 full epochs of SFT
   before DPO.

2. **Check initial margins before committing to a long DPO run.** If `rewards/margins`
   is near zero at step 10–50, abort. The signal is not there.

3. **ln(2) is a red flag, not a starting point.** A loss of 0.693 at step 10 is
   expected. A loss of 0.693 at step 500 means the gradients are zero.

4. **Per-token normalization matters.** The raw `logps/chosen` was consistently lower
   than `logps/rejected` (e.g., -605 vs. -469 at step 10) — but this is a length
   artifact (chosen responses tend to be longer). The per-token `rewards/margins`
   metric correctly normalizes for this.

---

## Next Steps

### Option A: Continue SFT, then retry DPO

Train SFT for the full epoch (~25,800 steps) or at least to ~15,000 steps.
Then re-run `dpo_eval.py` baseline to verify initial margins > 0.05 before starting DPO.

**Pros:** Fixes the root cause directly. A stronger SFT model will benefit all
downstream stages (DPO, GRPO, inference quality).  
**Cons:** ~20 more hours of SFT training before we can restart DPO.

### Option B: Switch to GRPO (recommended)

GRPO generates its own rollouts and scores them with a verifiable reward function.
It does not depend on the model already distinguishing chosen from rejected — it
creates its own supervision signal. This makes it more robust to a weak SFT baseline.

**Pros:** Can make progress without retraining SFT. GRPO is also closer to the
state-of-the-art RLVR pipeline (used in DeepSeek-R1, etc.).  
**Cons:** Requires a verifiable reward function (e.g., format checking, math answer
verification) — not applicable to open-ended chat quality without a judge model.

### Recommended path

Use **GRPO with a format/instruction-following reward** (e.g., reward for producing
bullet points when asked, correct list length, structured output adherence). This
is verifiable without a judge model and directly targets the instruction-following
failures observed in the SFT qualitative eval.

---

## Appendix: Key Metrics at a Glance

| Metric | Step 10 | Step 500 | Step 1500 | Step 2500 | Expected (working DPO) |
|---|---|---|---|---|---|
| Loss | 0.6914 | 0.6913 | 0.6914 | 0.6955 | Declining from 0.693 |
| rewards/margins | +0.0037 | +0.0041 | +0.0042 | varies ≈0 | Steadily +growing |
| rewards/accuracies | 0.400 | 0.600 | 0.487 | varies ≈0.5 | Approaching 1.0 |
| eval_loss | — | 0.6928 | 0.6926 | 0.6918 | Declining |
