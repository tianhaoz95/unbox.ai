"""Tests for LR scheduler."""

import math

import pytest

from unbox_platform.train.scheduler import get_lr


def test_warmup_starts_at_zero() -> None:
    lr = get_lr(step=0, total_steps=1000, max_lr=1e-3, warmup_steps=100)
    assert lr == 0.0


def test_warmup_reaches_max() -> None:
    lr = get_lr(step=100, total_steps=1000, max_lr=1e-3, warmup_steps=100)
    assert abs(lr - 1e-3) < 1e-8


def test_lr_decays_after_warmup() -> None:
    lr_mid = get_lr(step=500, total_steps=1000, max_lr=1e-3, warmup_steps=100)
    lr_end = get_lr(step=999, total_steps=1000, max_lr=1e-3, warmup_steps=100)
    assert lr_mid > lr_end


def test_min_lr_floor() -> None:
    lr_end = get_lr(step=1000, total_steps=1000, max_lr=1e-3, min_lr_ratio=0.1, warmup_steps=100)
    assert lr_end >= 1e-3 * 0.1 - 1e-10


def test_lr_monotone_during_warmup() -> None:
    lrs = [get_lr(step=i, total_steps=1000, max_lr=1e-3, warmup_steps=100) for i in range(101)]
    for a, b in zip(lrs, lrs[1:]):
        assert b >= a
