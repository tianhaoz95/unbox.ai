# DPO Underfitting: Margins Still Rising at End of Training

**Date:** 2026-05-25
**Preceding issue:** `reports/dpo_failure_lr_too_small.md`
**Run:** `wandb/offline-run-20260524_132838-pzukz105` (LR=1e-6, 1 epoch)

---

## Incident

After fixing the learning rate from `5e-7` to `1e-6`, DPO training produced real
learning (positive margins, accuracy above random). However, inspecting the metric
trajectory revealed the model had not converged when training stopped:

**Reward margins throughout the 1-epoch run:**

| Step | rewards/margins | rewards/accuracies | LR |
|---|---|---|---|
| 10 | 0.003576 | 42.5% | 9e-8 |
| 100 | ~0.003 | ~50% | ~1e-6 (peak) |
| 500 | ~0.005 | ~52% | ~5e-7 |
| 940 | 0.00855 | 54.4% | 9.73e-10 |
| 950 | 0.007223 | 55.0% | 1.65e-10 |
| eval (final) | 0.005833 | 54.2% | — |

The margins were **still trending upward** at step 940–950 rather than plateauing,
indicating active learning at the end of training.

---

## Findings

1. **Margins did not plateau.** A converged DPO run shows margins stabilising or
   slowly declining as the model saturates. The upward slope at the end of epoch 1
   signals the model had more capacity to learn.

2. **LR decayed to near zero before convergence.** The cosine schedule over 956
   steps brought LR from `1e-6` to `~1e-10` by the final steps. The near-zero LR
   suppressed gradient updates in the last ~150 steps, effectively halting learning
   before the model found a stable optimum.

3. **Final eval slightly below training.** `eval_margins=0.005833` vs training
   `margins≈0.0085` at the same epoch — a small gap consistent with normal
   generalisation, not overfitting.

---

## Hypothesis

One epoch (956 gradient steps) is insufficient for DPO on this model. The gradient
signal from preference pairs is weaker than SFT (no absolute right answer, only
relative ranking), so more passes over the data are needed. The cosine schedule
running out before convergence compounds this: the model spends the last ~15% of
training at near-zero LR, wasting potential learning.

With 3 epochs (2868 steps), the cosine schedule decays over a longer horizon
(LR stays meaningful for ~3× as many steps), and the model sees each preference
pair 3 times.

---

## Solution

1. **Delete previous checkpoints** to prevent HF Trainer from resuming with a
   stale scheduler state (saved at LR≈0, would make extra epochs train at zero LR).

2. **Set `num_epochs: 3`** in `configs/rl/dpo.yaml`:

```yaml
# Before
num_epochs: 1

# After
num_epochs: 3
```

3. **Restart training from scratch** — total steps: 956 × 3 = **2868**.

---

## Result

Run in progress: `wandb/offline-run-20260525_*` (task `b2porebqb`), ETA ~2 hours.

Early metrics at step 10: margins=0.002018, accuracy=40.2% — consistent with a
fresh start. Expected outcome: margins of 0.02–0.05 and accuracy 60–65% by epoch 3
if the upward trend from epoch 1 continues across all 3 epochs.

*Update this section after training completes.*

---

## Key Takeaways

- **Always inspect margin trajectory, not just final value.** A still-rising margin
  at epoch end means underfitting regardless of whether the absolute value looks OK.
- **Cosine LR schedule horizon must match actual convergence time.** If margins
  haven't plateaued by the time LR hits zero, more epochs are needed.
- **Do not resume from a 1-epoch DPO checkpoint to add more epochs.** The scheduler
  state saved at LR≈0 makes the extra epochs train at near-zero LR. Always start
  fresh with the target epoch count.
