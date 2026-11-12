
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weights(y: np.ndarray, clip: tuple[float, float] = (0.1, 100.0)) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    pos = np.nansum(y == 1, axis=0).astype(np.float64)
    neg = np.nansum(y == 0, axis=0).astype(np.float64)
    weights = np.ones(y.shape[1])
    ok = (pos > 0) & (neg > 0)
    weights[ok] = np.clip(neg[ok] / pos[ok], *clip)
    return weights.astype(np.float32)


def _masked_parts(logits: torch.Tensor, targets: torch.Tensor):
    mask = ~torch.isnan(targets)
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    return mask, safe


def masked_weighted_bce(logits: torch.Tensor, targets: torch.Tensor,
                        pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    mask, safe = _masked_parts(logits, targets)
    per = F.binary_cross_entropy_with_logits(logits, safe, reduction="none", pos_weight=pos_weight)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def masked_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                      gamma: float = 2.0, alpha: float | None = None) -> torch.Tensor:
    mask, safe = _masked_parts(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    p_t = torch.exp(-bce)
    per = (1.0 - p_t).clamp(min=0.0) ** gamma * bce
    if alpha is not None:
        per = per * (alpha * safe + (1.0 - alpha) * (1.0 - safe))
    return (per * mask).sum() / mask.sum().clamp(min=1)


class UncertaintyWeightedLoss(nn.Module):

    def __init__(self, n_tasks: int = 12):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def task_losses(self, logits: torch.Tensor, targets: torch.Tensor):
        mask, safe = _masked_parts(logits, targets)
        per = F.binary_cross_entropy_with_logits(logits, safe, reduction="none") * mask
        counts = mask.sum(dim=0)
        present = counts > 0
        return per.sum(dim=0) / counts.clamp(min=1), present

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses, present = self.task_losses(logits, targets)
        terms = torch.exp(-self.log_var) * losses + 0.5 * self.log_var
        return (terms * present).sum() / present.sum().clamp(min=1)

    def task_weights(self) -> torch.Tensor:
        return torch.exp(-self.log_var.detach())
