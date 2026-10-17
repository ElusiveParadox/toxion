"""Masked multitask losses for imbalanced Tox21 labels. Targets contain NaN
where a molecule was not tested for an endpoint; those entries contribute
nothing and every loss is normalized by the number of labeled entries.

Per-entry binary cross-entropy with logit z, target y, p = sigmoid(z):
    bce = -[y log p + (1 - y) log(1 - p)]
and p_t = p if y = 1 else 1 - p, so bce = -log p_t.

Class weighting: positives are weighted by w_t = n_neg / n_pos (per endpoint)
    loss = -[w_t y log p + (1 - y) log(1 - p)]

Focal loss (Lin et al. 2017):
    FL = alpha_t * (1 - p_t)^gamma * bce,   alpha_t = alpha if y = 1 else 1 - alpha
    gamma = 0 and alpha = None gives plain BCE. Use gamma = 0 or gamma >= 1.

Uncertainty weighting (Kendall et al. 2018), s_t = log sigma_t^2 learned per task:
    L = sum_t [ exp(-s_t) * L_t + 0.5 * s_t ]
    minimized over s_t at s_t* = log(2 L_t), i.e. weight exp(-s_t*) = 1 / (2 L_t):
    tasks with a larger loss get a smaller weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weights(y: np.ndarray, clip: tuple[float, float] = (0.1, 100.0)) -> np.ndarray:
    """Per-task n_neg / n_pos from a [n, T] label array with NaN for missing,
    clipped to `clip`. Tasks with no positives or no negatives get 1.0."""
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
    """pos_weight: [n_tasks] or None. With None this equals plain masked BCE."""
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
    """Learned per-task loss weights. Add `log_var` to the optimizer. Tasks with
    no labels in a batch are skipped (neither loss nor penalty). Returns the mean
    over tasks present in the batch."""

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
