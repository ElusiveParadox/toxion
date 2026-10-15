"""First-order optimizers written from the update equations, as
torch.optim.Optimizer subclasses. Conventions match torch.optim (default
hyperparameters, where eps is added, how weight decay enters) so each class
can be checked numerically against its torch counterpart.

Notation: theta = parameter, g = gradient, lr = learning rate, t = step count.

SGD (momentum mu, optional Nesterov, L2 weight decay lambda):
    g <- g + lambda * theta
    buf <- g                          (t = 1)
    buf <- mu * buf + g               (t > 1)
    d <- g + mu * buf if nesterov else buf
    theta <- theta - lr * d

RMSprop (decay alpha):
    g <- g + lambda * theta
    v <- alpha * v + (1 - alpha) * g^2
    theta <- theta - lr * g / (sqrt(v) + eps)

Adam (beta1, beta2):
    g <- g + lambda * theta                       (L2, coupled)
    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * g^2
    m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

AdamW: same, but weight decay is decoupled from the gradient:
    theta <- theta * (1 - lr * lambda)            (before the Adam step)
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def _check_hparam(name: str, value: float, low: float = 0.0, high: float | None = None) -> None:
    if value < low or (high is not None and value >= high):
        raise ValueError(f"invalid {name}: {value}")


class _ScratchOptimizer(Optimizer):
    def _params_with_grad(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError(f"{type(self).__name__} does not support sparse gradients")
            yield p, p.grad

    def _closure_loss(self, closure):
        if closure is None:
            return None
        with torch.enable_grad():
            return closure()


class SGD(_ScratchOptimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        _check_hparam("lr", lr)
        _check_hparam("momentum", momentum)
        _check_hparam("weight_decay", weight_decay)
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._closure_loss(closure)
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p, g in self._params_with_grad(group):
                if wd != 0:
                    g = g + wd * p
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = g.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(mu).add_(g)
                    g = g + mu * buf if group["nesterov"] else buf
                p.add_(g, alpha=-lr)
        return loss


class RMSprop(_ScratchOptimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        _check_hparam("lr", lr)
        _check_hparam("alpha", alpha, 0.0, 1.0)
        _check_hparam("eps", eps)
        _check_hparam("weight_decay", weight_decay)
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._closure_loss(closure)
        for group in self.param_groups:
            for p, g in self._params_with_grad(group):
                if group["weight_decay"] != 0:
                    g = g + group["weight_decay"] * p
                state = self.state[p]
                if not state:
                    state["square_avg"] = torch.zeros_like(p)
                v = state["square_avg"]
                v.mul_(group["alpha"]).addcmul_(g, g, value=1 - group["alpha"])
                p.addcdiv_(g, v.sqrt().add_(group["eps"]), value=-group["lr"])
        return loss


class Adam(_ScratchOptimizer):
    _decoupled_weight_decay = False

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        _check_hparam("lr", lr)
        _check_hparam("beta1", betas[0], 0.0, 1.0)
        _check_hparam("beta2", betas[1], 0.0, 1.0)
        _check_hparam("eps", eps)
        _check_hparam("weight_decay", weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._closure_loss(closure)
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p, g in self._params_with_grad(group):
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    if self._decoupled_weight_decay:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g + wd * p

                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                bias1 = 1 - b1**t
                bias2 = 1 - b2**t
                denom = (v / bias2).sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)
        return loss


class AdamW(Adam):
    _decoupled_weight_decay = True

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
