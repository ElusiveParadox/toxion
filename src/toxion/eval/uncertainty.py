
from __future__ import annotations

import torch


def mc_dropout_predict(model: torch.nn.Module, forward_fn, n_samples: int = 30):
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(forward_fn()) for _ in range(n_samples)], dim=0)
    finally:
        model.train(was_training)
    return probs.mean(dim=0), probs.std(dim=0)
