# DPO Failure: Learning Rate Too Small

**Date:** 2026-05-24
**Run:** `wandb/offline-run-20260524_044025-0yg78upn`
**Config:** `configs/rl/dpo.yaml`, `learning_rate: 5.0e-7`

---

## Symptom

DPO training ran for 956 steps but produced no measurable preference learning:

| Step | loss | rewards/margins | rewards/accuracies |
|---|---|---|---|
| 10 | 0.6922 | 0.00236 | 41.7% |
| 50 | 0.6938 | -0.00069 | 49.1% |
| 920 | 0.6935 | -0.0000377 | 49.4% |
| 950 | 0.6927 | 0.00141 | 51.6% |
| eval (final) | 0.694 | -0.001003 | 49.5% |

- **Loss stuck at ~0.6931** = log(2), the theoretical value when reward margin = 0
- **Accuracy ~50%** = random (model cannot distinguish chosen from rejected)
- **Margins oscillate near zero** with no upward trend

---

## Root Cause 1: Learning Rate Too Small

`learning_rate: 5.0e-7` is at the bottom of the workable range for DPO (typical: `1e-6` to `5e-6`).

With only 956 steps and a very small LR, the policy barely moved from the reference.
Evidence: `rewards/chosen` and `rewards/rejected` both hovered near zero throughout,
meaning `logp_policy ≈ logp_reference` for both sequences — the KL constraint dominated
and the model never escaped the reference distribution.

---

## Root Cause 2: Reference Model Assigns Higher Probability to Rejected Responses

Raw log-probabilities under the reference (SFT checkpoint-3000):

```
logps/chosen   ≈ -450 to -480   (lower probability)
logps/rejected ≈ -380 to -420   (higher probability)
```

The reference already "prefers" rejected over chosen throughout training.
This is partly a **length effect** (chosen responses in UltraFeedback are typically
longer, so total sequence log-prob is lower), but also indicates the SFT model was
not trained well enough to assign high probability to high-quality assistant responses.

DPO's KL penalty (`beta × KL(policy || reference)`) penalises deviating from a reference
that already has the ordering wrong. The gradient must overcome this handicap before
it can produce a positive margin, requiring a higher LR than was set.

---

## Contributing Factor: Undertrained Base Model

The pretraining run completed only ~34k steps (~4.8B tokens on a sharded dataset),
and SFT ran for one epoch on 207k examples. The model's instruction-following capability
is limited, which reduces the quality of the starting point for DPO.

---

## Fix Applied

Increased `learning_rate` from `5.0e-7` to `1.0e-6` in `configs/rl/dpo.yaml`.

```yaml
# Before
learning_rate: 5.0e-7

# After
learning_rate: 1.0e-6
```

---

## Longer-Term Recommendations

1. **Validate SFT quality first** — run qualitative sampling before DPO; if the SFT model
   generates incoherent responses, DPO cannot fix it.
2. **Check `logps/chosen` vs `logps/rejected` at step 0** — if `logps/chosen` < `logps/rejected`
   from the start, the reference model has inverted preferences and a higher LR or more epochs
   are needed to overcome the KL penalty.
3. **Typical DPO LR range**: `1e-6` to `5e-6` for a 1B-scale model with beta=0.1.
4. **Monitor rewards/accuracies** — should exceed 60% by the end of training for DPO to
   be considered successful; below 55% indicates marginal or no learning.
