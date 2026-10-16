import math

import pytest

torch = pytest.importorskip("torch")

from toxion.optim.optimizers import SGD, Adam, AdamW, RMSprop
from toxion.optim.schedules import make_scheduler, step_decay_factor, warmup_cosine_factor


def _problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(32, 4, generator=g, dtype=torch.float64)
    y = torch.randn(32, 2, generator=g, dtype=torch.float64)
    w = torch.randn(4, 2, generator=g, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    return x, y, [w, b]


def _run(opt_cls, kwargs, steps=40):
    x, y, params = _problem()
    opt = opt_cls(params, **kwargs)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((x @ params[0] + params[1] - y) ** 2).mean() + 0.05 * (params[0] ** 4).sum()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in params]


CASES = [
    (SGD, torch.optim.SGD, dict(lr=0.05)),
    (SGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9)),
    (SGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9, nesterov=True)),
    (SGD, torch.optim.SGD, dict(lr=0.05, momentum=0.8, weight_decay=0.01)),
    (RMSprop, torch.optim.RMSprop, dict(lr=0.01)),
    (RMSprop, torch.optim.RMSprop, dict(lr=0.01, alpha=0.9, weight_decay=0.01)),
    (Adam, torch.optim.Adam, dict(lr=0.01)),
    (Adam, torch.optim.Adam, dict(lr=0.01, betas=(0.8, 0.99), weight_decay=0.01)),
    (AdamW, torch.optim.AdamW, dict(lr=0.01)),
    (AdamW, torch.optim.AdamW, dict(lr=0.01, weight_decay=0.1)),
]


@pytest.mark.parametrize("ours,ref,kwargs", CASES, ids=[f"{c[0].__name__}-{i}" for i, c in enumerate(CASES)])
def test_matches_torch_optim(ours, ref, kwargs):
    for a, b in zip(_run(ours, kwargs), _run(ref, kwargs)):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-10)


def test_adam_first_step_is_lr_times_sign_of_gradient():
    p = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    p.grad = torch.tensor([0.5, -3.0], dtype=torch.float64)
    Adam([p], lr=0.1, eps=1e-8).step()
    expected = torch.tensor([1.0 - 0.1, -2.0 + 0.1], dtype=torch.float64)
    torch.testing.assert_close(p.detach(), expected, rtol=1e-6, atol=1e-6)


def test_adamw_decays_weights_even_with_zero_gradient():
    p = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    p.grad = torch.zeros(1, dtype=torch.float64)
    AdamW([p], lr=0.1, weight_decay=0.5).step()
    assert p.item() == pytest.approx(2.0 * (1 - 0.1 * 0.5))


def test_l2_decay_is_swamped_by_large_gradient_in_adam_but_not_in_adamw():
    def after_one_step(cls, wd):
        p = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        p.grad = torch.tensor([100.0], dtype=torch.float64)
        cls([p], lr=0.1, weight_decay=wd).step()
        return p.item()

    assert abs(after_one_step(Adam, 0.5) - after_one_step(Adam, 0.0)) < 1e-6
    assert after_one_step(AdamW, 0.0) - after_one_step(AdamW, 0.5) == pytest.approx(0.1)


def test_adam_converges_on_quadratic():
    p = torch.tensor([5.0, -3.0], dtype=torch.float64, requires_grad=True)
    opt = Adam([p], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        ((p - torch.tensor([1.0, 2.0], dtype=torch.float64)) ** 2).sum().backward()
        opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([1.0, 2.0], dtype=torch.float64), atol=1e-2, rtol=0)


@pytest.mark.filterwarnings("ignore:Sparse invariant checks")
def test_invalid_hyperparameters_and_sparse_grads_rejected():
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        SGD([p], lr=-1)
    with pytest.raises(ValueError):
        SGD([p], lr=0.1, momentum=0.0, nesterov=True)
    with pytest.raises(ValueError):
        Adam([p], betas=(1.0, 0.999))
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,))
    with pytest.raises(RuntimeError, match="sparse"):
        SGD([p], lr=0.1).step()


def test_warmup_cosine_values():
    f = lambda s: warmup_cosine_factor(s, warmup_steps=10, total_steps=110, min_factor=0.1)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.1 + 0.9 * 0.5)
    assert f(110) == pytest.approx(0.1)
    assert f(500) == pytest.approx(0.1)
    values = [f(s) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_warmup_cosine_without_warmup_starts_at_one():
    assert warmup_cosine_factor(0, 0, 100) == pytest.approx(1.0)
    assert warmup_cosine_factor(100, 0, 100) == pytest.approx(0.0, abs=1e-12)


def test_step_decay_values():
    assert [step_decay_factor(s, 3, 0.5) for s in (0, 2, 3, 6)] == [1.0, 1.0, 0.5, 0.25]


def test_scheduler_drives_optimizer_lr():
    p = torch.zeros(1, requires_grad=True)
    opt = SGD([p], lr=0.4)
    sched = make_scheduler(opt, lambda s: warmup_cosine_factor(s, 4, 20))
    lrs = []
    for _ in range(6):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    expected = [0.4 * warmup_cosine_factor(s, 4, 20) for s in range(6)]
    assert lrs == pytest.approx(expected)
    assert math.isclose(lrs[0], 0.1)
