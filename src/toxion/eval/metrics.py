
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _per_task(fn, y_true, y_pred, task_names, mask=None, warn=True):
    if mask is None:
        mask = ~np.isnan(y_true)
    out = {}
    for t, name in enumerate(task_names):
        m = mask[:, t]
        yt, yp = y_true[m, t], y_pred[m, t]
        if len(np.unique(yt)) < 2:
            if warn:
                logger.warning("Task %s: one class only in this split, skipping", name)
            continue
        out[name] = float(fn(yt, yp))
    return out


def masked_roc_auc(y_true, y_pred, task_names, mask=None) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    return _per_task(roc_auc_score, y_true, y_pred, task_names, mask)


def masked_pr_auc(y_true, y_pred, task_names, mask=None) -> dict[str, float]:
    from sklearn.metrics import average_precision_score
    return _per_task(average_precision_score, y_true, y_pred, task_names, mask, warn=False)


def aggregate_across_seeds(results_per_seed: list[dict[str, float]]) -> dict:
    tasks = sorted(set().union(*[r.keys() for r in results_per_seed]))
    out = {t: (float(np.mean([r[t] for r in results_per_seed if t in r])),
               float(np.std([r[t] for r in results_per_seed if t in r]))) for t in tasks}
    means = [float(np.mean(list(r.values()))) for r in results_per_seed if r]
    out["__overall__"] = (float(np.mean(means)), float(np.std(means)))
    return out


def youden_thresholds(y_true: np.ndarray, probs: np.ndarray, default: float = 0.5) -> np.ndarray:
    from sklearn.metrics import roc_curve

    out = np.full(y_true.shape[1], default, dtype=np.float64)
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        yt, yp = y_true[m, t], probs[m, t]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, thr = roc_curve(yt, yp)
        best = int(np.argmax(tpr - fpr))
        out[t] = float(min(max(thr[best], 1e-6), 1.0))
    return out


def fpr_capped_thresholds(y_true: np.ndarray, probs: np.ndarray, fpr_cap: float, default: float = 0.5) -> np.ndarray:
    from sklearn.metrics import roc_curve

    out = np.full(y_true.shape[1], default, dtype=np.float64)
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        yt, yp = y_true[m, t], probs[m, t]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, thr = roc_curve(yt, yp)
        ok = fpr <= fpr_cap
        out[t] = float(min(max(thr[ok][np.argmax(tpr[ok])], 1e-6), 1.0)) if ok.any() else 1.0
    return out
