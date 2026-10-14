"""Custom PyTorch-compatible optimizers written from scratch."""

from __future__ import annotations
import math
import torch
from torch.optim import Optimizer

def _draft_clip_grad(grad, max_norm=1.0):
    norm = torch.norm(grad)
    if norm > max_norm:
        grad.mul_(max_norm / (norm + 1e-6))

class _ScratchOptimizer(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["step"] = 0

class SGDMomentum(_ScratchOptimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(g).detach()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(g)
                p.add_(buf, alpha=-lr)
                state["step"] += 1
        return loss
