import numpy as np

from toxion.eval.metrics import aggregate_across_seeds, masked_pr_auc, masked_roc_auc, youden_thresholds
from toxion.eval.uncertainty import mc_dropout_predict


def test_masked_auc_skips_nan_and_single_class_tasks():
    y = np.array([[0, 1, np.nan], [1, 1, np.nan], [0, 0, np.nan], [1, 0, 1.0]], dtype=float)
    p = np.array([[.1, .9, .5], [.8, .7, .5], [.2, .1, .5], [.9, .2, .5]])
    roc = masked_roc_auc(y, p, ["a", "b", "c"])
    assert set(roc) == {"a", "b"} and roc["a"] == 1.0
    assert set(masked_pr_auc(y, p, ["a", "b", "c"])) == {"a", "b"}


def test_aggregate_across_seeds():
    agg = aggregate_across_seeds([{"a": 0.6, "b": 0.8}, {"a": 0.8, "b": 0.8}])
    assert agg["a"][0] == 0.7 and abs(agg["__overall__"][0] - 0.75) < 1e-9


def test_youden_threshold_separates_classes():
    y = np.array([[0.], [0.], [0.], [1.], [1.], [np.nan]])
    p = np.array([[.1], [.2], [.3], [.8], [.9], [.99]])
    thr = youden_thresholds(y, p)
    assert 0.3 < thr[0] <= 0.8
    assert youden_thresholds(np.zeros((4, 1)), np.full((4, 1), .5))[0] == 0.5


def test_mc_dropout_has_spread_and_restores_mode():
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 2)).eval()
    x = torch.randn(3, 4)
    mean, std = mc_dropout_predict(model, lambda: model(x), n_samples=20)
    assert mean.shape == std.shape == (3, 2) and std.sum() > 0 and not model.training


def test_fpr_capped_threshold_is_stricter_than_youden():
    from toxion.eval.metrics import fpr_capped_thresholds

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)[:, None]
    p = np.clip(0.35 * y + rng.normal(0.3, 0.2, y.shape), 0.001, 0.999)
    neg = y[:, 0] == 0
    thr = fpr_capped_thresholds(y, p, 0.05)[0]
    assert (p[neg, 0] >= thr).mean() <= 0.05
    assert thr >= youden_thresholds(y, p)[0]
    assert fpr_capped_thresholds(np.zeros((4, 1)), np.full((4, 1), .5), 0.1)[0] == 0.5
