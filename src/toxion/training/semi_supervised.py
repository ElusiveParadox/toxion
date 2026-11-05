
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from toxion.training.losses import masked_bce_loss


@dataclass
class SemiSupervisedConfig:
    lambda_cons: float = 0.5
    lambda_pl: float = 0.5
    hi: float = 0.9
    lo: float = 0.05
    ema_decay: float = 0.99
    ramp_epochs: int = 3
    unlabeled_ratio: int = 8


class EMATeacher:
    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        for t, s in zip(self.model.parameters(), student.parameters()):
            t.mul_(self.decay).add_(s.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def probs(self, *inputs) -> torch.Tensor:
        return torch.sigmoid(self.model(*inputs))


def consistency_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    return ((torch.sigmoid(student_logits) - teacher_probs) ** 2).mean()


def pseudo_targets(teacher_probs: torch.Tensor, hi: float, lo: float) -> torch.Tensor:
    t = torch.full_like(teacher_probs, float("nan"))
    t[teacher_probs >= hi] = 1.0
    t[teacher_probs <= lo] = 0.0
    return t


def unlabeled_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
                   cfg: SemiSupervisedConfig, ramp: float) -> torch.Tensor:
    cons = consistency_loss(student_logits, teacher_probs)
    pseudo = masked_bce_loss(student_logits, pseudo_targets(teacher_probs, cfg.hi, cfg.lo))
    return ramp * (cfg.lambda_cons * cons + cfg.lambda_pl * pseudo)


def ramp_factor(epoch: int, ramp_epochs: int) -> float:
    return 1.0 if ramp_epochs <= 0 else min(1.0, (epoch + 1) / ramp_epochs)
