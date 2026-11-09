
from __future__ import annotations

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    @property
    def encoder(self):
        return self.members[0].encoder

    def member_logits(self, *inputs) -> torch.Tensor:
        return torch.stack([m(*inputs) for m in self.members], dim=0)

    def forward(self, *inputs) -> torch.Tensor:
        return self.member_logits(*inputs).mean(dim=0)
