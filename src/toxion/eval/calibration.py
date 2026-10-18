"""Calibration of binary probabilities (NumPy + SciPy only).

Expected calibration error, equal-width bins b over [0, 1] on the
positive-class probability p:
    ECE = sum_b (n_b / N) * | mean(y in b) - mean(p in b) |
Brier score = mean (p - y)^2.

Temperature scaling (Guo et al. 2017): p = sigmoid(z / T), T > 0 fit on a
validation set by minimizing the negative log-likelihood
    NLL(beta) = sum_i [ softplus(beta z_i) - y_i beta z_i ],  beta = 1 / T.
NLL is convex in beta (it is a logistic loss in the single feature z), so the
minimum is unique. Dividing logits by a positive constant does not change their
order, so ROC-AUC and PR-AUC are unchanged; only probability values move.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def _select(y_true, p, mask):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    keep = ~np.isnan(y_true) if mask is None else np.asarray(mask, dtype=bool) & ~np.isnan(y_true)
    return y_true[keep], p[keep]


def reliability_curve(y_true, p, n_bins: int = 10, mask=None):
    """Returns (mean_confidence, observed_frequency, counts), each [n_bins];
    empty bins have NaN confidence/frequency and count 0."""
    y, p = _select(y_true, p, mask)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = bin_id == b
        counts[b] = sel.sum()
        if counts[b]:
            conf[b] = p[sel].mean()
            freq[b] = y[sel].mean()
    return conf, freq, counts


def expected_calibration_error(y_true, p, n_bins: int = 10, mask=None) -> float:
    conf, freq, counts = reliability_curve(y_true, p, n_bins, mask)
    total = counts.sum()
    if total == 0:
        return float("nan")
    used = counts > 0
    return float((counts[used] / total * np.abs(freq[used] - conf[used])).sum())


def brier_score(y_true, p, mask=None) -> float:
    y, p = _select(y_true, p, mask)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def negative_log_likelihood(y_true, logits, temperature: float = 1.0, mask=None) -> float:
    y, z = _select(y_true, logits, mask)
    z = z / temperature
    return float(np.mean(np.logaddexp(0.0, z) - y * z)) if len(y) else float("nan")


def fit_temperature(logits, y_true, mask=None, bounds: tuple[float, float] = (-6.0, 6.0)) -> float:
    """Single temperature T minimizing validation NLL. Searches u = log(beta)
    in `bounds` (so T in roughly [0.0025, 403]). Returns 1.0 if only one class
    is present."""
    from scipy.optimize import minimize_scalar

    y, z = _select(y_true, logits, mask)
    if len(np.unique(y)) < 2:
        return 1.0

    def nll(u: float) -> float:
        zz = np.exp(u) * z
        return float(np.mean(np.logaddexp(0.0, zz) - y * zz))

    res = minimize_scalar(nll, bounds=bounds, method="bounded", options={"xatol": 1e-8})
    return float(np.exp(-res.x))


def fit_temperature_per_task(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """[n, T] logits and labels (NaN = missing) -> [T] temperatures."""
    return np.array([fit_temperature(logits[:, t], y_true[:, t]) for t in range(logits.shape[1])])


def apply_temperature(logits: np.ndarray, temperature) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / np.asarray(temperature, dtype=np.float64)
