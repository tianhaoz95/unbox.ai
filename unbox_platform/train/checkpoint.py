"""Checkpoint save and resume utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    output_dir: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    config_dict: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model

    state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if config_dict is not None:
        state["config"] = config_dict

    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(state, ckpt_path)

    # Keep a "latest" pointer for easy resume
    latest_path = output_dir / "latest.pt"
    torch.save(state, latest_path)

    # Write a small JSON manifest for human inspection
    manifest = {"step": step, "epoch": epoch, "loss": loss, "checkpoint": str(ckpt_path)}
    with open(output_dir / "latest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "latest.pt"

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    return state


def find_latest_checkpoint(output_dir: Path | str) -> Path | None:
    output_dir = Path(output_dir)
    latest = output_dir / "latest.pt"
    if latest.exists():
        return latest
    return None
