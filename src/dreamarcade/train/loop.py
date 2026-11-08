"""Shared training mechanics: cosine LR, AMP on CUDA, infinite batches."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def cycle_batches(loader) -> Iterator:
    while True:
        if len(loader) == 0:
            raise RuntimeError("dataloader is empty; collect a longer dataset")
        yield from loader


def autocast_context(device: torch.device, enabled: bool):
    use = enabled and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use)
