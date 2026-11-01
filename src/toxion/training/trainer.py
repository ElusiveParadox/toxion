
from __future__ import annotations

import logging
import os
import time
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import DataLoader

from toxion.data.augment import Augmenter
from toxion.training.losses import masked_bce_loss
from toxion.training.semi_supervised import EMATeacher, SemiSupervisedConfig, ramp_factor, unlabeled_loss

logger = logging.getLogger(__name__)


class EarlyStopper:

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter = float("-inf"), 0

    def is_improvement(self, score: float) -> bool:
        return score > self.best + self.min_delta

    def step(self, score: float) -> bool:
        if self.is_improvement(score):
            self.best, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _atomic_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict_logits(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        logits.append(model(*batch.model_inputs()).cpu().numpy())
        targets.append(batch.targets.cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


@torch.no_grad()
def eval_split(model, loader: DataLoader, device: str) -> tuple[float, float]:
    from toxion.data.loader import TOX21_TASKS
    from toxion.eval.metrics import masked_roc_auc

    model.eval()
    total, n, logits, targets = 0.0, 0, [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(*batch.model_inputs())
        total += masked_bce_loss(out, batch.targets).item()
        n += 1
        logits.append(out.cpu().numpy())
        targets.append(batch.targets.cpu().numpy())
    if not logits:
        return 0.0, float("nan")
    roc = masked_roc_auc(np.concatenate(targets), np.concatenate(logits), TOX21_TASKS[: logits[0].shape[1]])
    return total / max(n, 1), (float(np.mean(list(roc.values()))) if roc else float("nan"))


def eval_loss(model, loader: DataLoader, device: str) -> float:
    return eval_split(model, loader, device)[0]


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, device: str,
                epochs: int = 100, lr: float = 1e-3, patience: int = 15,
                augmenter: Augmenter | None = None,
                unlabeled_loader: DataLoader | None = None,
                ssl_cfg: SemiSupervisedConfig | None = None,
                state_path: str | None = None, resume: bool = False,
                grad_clip: float = 5.0, weight_decay: float = 0.0, schedule: str = "none",
                monitor: str = "val_loss"):
    model.to(device)
    if monitor not in {"val_loss", "val_auc"}:
        raise ValueError("monitor must be 'val_loss' or 'val_auc'")
    optimizer = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if weight_decay > 0
                 else torch.optim.Adam(model.parameters(), lr=lr))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
                 if schedule == "cosine" else None)
    stopper = EarlyStopper(patience=patience)
    best_state, start_epoch = None, 0

    if resume and state_path and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(st["model_state"])
        optimizer.load_state_dict(st["optimizer_state"])
        best_state, start_epoch = st["best_state"], st["epoch"] + 1
        stopper.best, stopper.counter = st["stopper_best"], st["stopper_counter"]
        if scheduler is not None and st.get("scheduler_state"):
            scheduler.load_state_dict(st["scheduler_state"])
        logger.info("resumed from %s at epoch %d", state_path, start_epoch)

    use_ssl = unlabeled_loader is not None and ssl_cfg is not None and len(unlabeled_loader.dataset) > 0
    teacher = EMATeacher(model, ssl_cfg.ema_decay) if use_ssl else None
    unl_iter = cycle(unlabeled_loader) if use_ssl else None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum, n_batches = 0.0, 0
        ramp = ramp_factor(epoch, ssl_cfg.ramp_epochs) if use_ssl else 0.0
        for batch in train_loader:
            batch = batch.to(device)
            view = augmenter(batch) if augmenter else batch
            optimizer.zero_grad()
            loss = masked_bce_loss(model(*view.model_inputs()), view.targets)
            if use_ssl:
                ub = next(unl_iter).to(device)
                t_probs = teacher.probs(*ub.model_inputs())
                s_view = augmenter(ub) if augmenter else ub
                loss = loss + unlabeled_loss(model(*s_view.model_inputs()), t_probs, ssl_cfg, ramp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if teacher is not None:
                teacher.update(model)
            loss_sum += loss.item()
            n_batches += 1

        val_loss, val_auc = eval_split(model, val_loader, device)
        score = val_auc if monitor == "val_auc" else -val_loss
        if score != score:
            score = -val_loss
        improved = stopper.is_improvement(score)
        if scheduler is not None:
            scheduler.step()
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stop = stopper.step(score)
        logger.info("epoch %d train_loss=%.4f val_loss=%.4f val_auc=%.4f lr=%.2e (%.1fs)%s", epoch,
                    loss_sum / max(n_batches, 1), val_loss, val_auc, lr_used,
                    time.time() - t0, " *" if improved else "")
        if state_path:
            _atomic_save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_state": best_state,
                "stopper_best": stopper.best, "stopper_counter": stopper.counter,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            }, state_path)
        if stop:
            logger.info("early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
