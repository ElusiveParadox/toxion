
from __future__ import annotations

import math


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int,
                         min_factor: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def step_decay_factor(step: int, step_size: int, gamma: float = 0.1) -> float:
    return gamma ** (step // step_size)


def make_scheduler(optimizer, factor_fn):
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer, lr_lambda=factor_fn)
