import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toxion.training.advanced_losses import (
    UncertaintyWeightedLoss,
    compute_pos_weights,
    masked_focal_loss,
    masked_weighted_bce,
)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(16, 5, generator=g, dtype=torch.float64)
    targets = (torch.rand(16, 5, generator=g, dtype=torch.float64) < 0.3).double()
    targets[torch.rand(16, 5, generator=g, dtype=torch.float64) < 0.3] = float("nan")
    return logits, targets


def _reference(logits, targets, weight_fn=None):
    z, y = logits.numpy(), targets.numpy()
    keep = ~np.isnan(y)
    p = 1 / (1 + np.exp(-z))
    pt = np.where(y == 1, p, 1 - p)
    bce = -np.log(np.where(keep, pt, 1.0))
    if weight_fn is not None:
        bce = weight_fn(z, y, p, pt, bce)
    return bce[keep].sum() / keep.sum()


def test_weighted_bce_without_weights_equals_plain_masked_bce(batch):
    logits, targets = batch
    assert masked_weighted_bce(logits, targets).item() == pytest.approx(_reference(logits, targets))


def test_pos_weight_formula(batch):
    logits, targets = batch
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

    def weighted(z, y, p, pt, bce):
        wy = np.where(y == 1, w.numpy()[None, :], 1.0)
        return wy * bce

    got = masked_weighted_bce(logits, targets, pos_weight=w).item()
    assert got == pytest.approx(_reference(logits, targets, weighted))


def test_focal_gamma_zero_is_bce_and_gamma_two_matches_formula(batch):
    logits, targets = batch
    assert masked_focal_loss(logits, targets, gamma=0.0).item() == pytest.approx(_reference(logits, targets))
    got = masked_focal_loss(logits, targets, gamma=2.0, alpha=0.25).item()
    alpha_t = lambda y: np.where(y == 1, 0.25, 0.75)
    ref = _reference(logits, targets, lambda z, y, p, pt, bce: alpha_t(y) * (1 - pt) ** 2 * bce)
    assert got == pytest.approx(ref)


def test_focal_downweights_easy_examples_relative_to_hard():
    z = torch.tensor([[4.0], [-4.0]], dtype=torch.float64)
    y = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    easy = masked_focal_loss(z[:1], y[:1], gamma=2.0).item() / masked_focal_loss(z[:1], y[:1], gamma=0.0).item()
    hard = masked_focal_loss(z[1:], y[1:], gamma=2.0).item() / masked_focal_loss(z[1:], y[1:], gamma=0.0).item()
    assert easy < 1e-3 < 0.9 < hard


@pytest.mark.parametrize("loss_fn", [
    lambda z, y: masked_weighted_bce(z, y),
    lambda z, y: masked_focal_loss(z, y, gamma=2.0),
    lambda z, y: UncertaintyWeightedLoss(5)(z, y),
])
def test_missing_entries_get_zero_gradient_and_all_nan_batch_is_finite(loss_fn, batch):
    logits, targets = batch
    logits = logits.clone().requires_grad_(True)
    loss_fn(logits, targets).backward()
    assert torch.all(logits.grad[torch.isnan(targets)] == 0)

    logits2 = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    loss = loss_fn(logits2, torch.full((4, 5), float("nan"), dtype=torch.float64))
    loss.backward()
    assert loss.item() == 0.0 and torch.all(torch.isfinite(logits2.grad))


def test_compute_pos_weights():
    y = np.array([[1, 0, 0], [0, 0, 1], [0, 0, np.nan], [0, 0, 1], [np.nan, 0, 1]], dtype=float)
    w = compute_pos_weights(y)
    assert w[0] == pytest.approx(3.0)
    assert w[1] == 1.0
    assert w[2] == pytest.approx(1 / 3)
    assert compute_pos_weights(np.array([[1.0], [0.0]] + [[0.0]] * 998), clip=(0.1, 100.0))[0] == 100.0


def test_uncertainty_loss_starts_as_mean_task_bce_and_skips_absent_tasks(batch):
    logits, targets = batch
    module = UncertaintyWeightedLoss(5).double()
    task_losses, present = module.task_losses(logits, targets)
    assert module(logits, targets).item() == pytest.approx(task_losses[present].mean().item())

    targets_absent = targets.clone()
    targets_absent[:, 2] = float("nan")
    losses, present = module.task_losses(logits, targets_absent)
    assert present.tolist() == [True, True, False, True, True]
    out = module(logits, targets_absent)
    out.backward()
    assert module.log_var.grad[2] == 0


def test_uncertainty_optimum_is_log_two_times_task_loss(batch):
    logits, targets = batch
    module = UncertaintyWeightedLoss(5).double()
    losses, present = module.task_losses(logits, targets)
    with torch.no_grad():
        module.log_var.copy_(torch.log(2 * losses))
    module(logits, targets).backward()
    assert torch.allclose(module.log_var.grad, torch.zeros(5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(module.task_weights(), 1 / (2 * losses), rtol=1e-10)
