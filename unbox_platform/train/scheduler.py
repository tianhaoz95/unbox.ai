"""Learning rate schedule: linear warmup followed by cosine decay with floor."""

from __future__ import annotations

import math


def get_lr(
    step: int,
    total_steps: int,
    max_lr: float,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 2000,
) -> float:
    min_lr = max_lr * min_lr_ratio

    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine
