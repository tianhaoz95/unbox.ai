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

Run: `wandb/offline-run-20260525_002816-5x4gw4yx`, completed in ~2 hours.

### Comparison across all three DPO runs

| Run | LR | Epochs | eval margins | eval accuracy |
|---|---|---|---|---|
| Run 1 (failed) | 5e-7 | 1 | -0.00100 | 49.5% (random) |
| Run 2 (LR fix) | 1e-6 | 1 | +0.00583 | 54.2% |
| Run 3 (this fix) | 1e-6 | 3 | **+0.01371** | **56.8%** |

Margins improved **2.4×** over the 1-epoch run. Training accuracy briefly reached
60–61% (steps 2830, 2860) before the eval settled at 56.8%, indicating genuine
preference learning with some train/eval gap.

The margins were still showing upward movement at the end of epoch 3 (training
accuracy 60% at step 2830–2860 vs eval 56.8%), suggesting the model's preference
learning capacity isn't fully saturated. Further gains are likely possible with
more epochs or a better-trained SFT base model.

### Interpretation

The 3-epoch fix confirmed the hypothesis: the model needed more gradient steps
and a longer LR horizon to develop stable preference representations. Results
remain modest (target: 60–70%) primarily because the underlying SFT model quality
limits how clearly the model can distinguish chosen from rejected responses.

---

## Key Takeaways

- **Always inspect margin trajectory, not just final value.** A still-rising margin
  at epoch end means underfitting regardless of whether the absolute value looks OK.
- **Cosine LR schedule horizon must match actual convergence time.** If margins
  haven't plateaued by the time LR hits zero, more epochs are needed.
- **Do not resume from a 1-epoch DPO checkpoint to add more epochs.** The scheduler
  state saved at LR≈0 makes the extra epochs train at near-zero LR. Always start
  fresh with the target epoch count.
