
from __future__ import annotations

import torch

from toxion.training.advanced_losses import masked_weighted_bce


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return masked_weighted_bce(logits, targets, pos_weight=None)
