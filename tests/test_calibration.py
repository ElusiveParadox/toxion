import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from toxion.eval.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_per_task,
    negative_log_likelihood,
    reliability_curve,
    sigmoid,
)


def test_sigmoid_is_stable_and_correct():
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    out = sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 1 / (1 + np.exp(-np.clip(z, -50, 50))), atol=1e-12)


def test_ece_hand_computed():
    p = np.array([0.95, 0.95, 0.05, 0.05])
    y = np.array([1, 0, 0, 0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.25)
    conf, freq, counts = reliability_curve(y, p, n_bins=10)
    assert counts.tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert np.isnan(conf[3])


def test_brier_and_nan_labels_are_ignored():
    y = np.array([1.0, 0.0, np.nan])
    p = np.array([0.8, 0.3, 0.99])
    assert brier_score(y, p) == pytest.approx((0.04 + 0.09) / 2)
    assert expected_calibration_error(y, p) == expected_calibration_error(y[:2], p[:2])


def test_calibrated_predictions_have_small_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.random(p.size) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.01


@pytest.fixture(scope="module")
def overconfident():
    rng = np.random.default_rng(1)
    true_logit = rng.normal(0, 1.5, 100_000)
    y = (rng.random(true_logit.size) < sigmoid(true_logit)).astype(float)
    return 3.0 * true_logit, y


def test_temperature_recovers_known_scale_and_lowers_ece_and_nll(overconfident):
    logits, y = overconfident
    T = fit_temperature(logits, y)
    assert T == pytest.approx(3.0, abs=0.1)
    before = expected_calibration_error(y, sigmoid(logits))
    after = expected_calibration_error(y, sigmoid(apply_temperature(logits, T)))
    assert after < before / 3
    assert negative_log_likelihood(y, logits, T) < negative_log_likelihood(y, logits, 1.0)


def test_temperature_scaling_leaves_auc_unchanged(overconfident):
    logits, y = overconfident
    T = fit_temperature(logits, y)
    assert roc_auc_score(y, logits) == roc_auc_score(y, apply_temperature(logits, T))


def test_fit_temperature_single_class_and_per_task(overconfident):
    logits, y = overconfident
    assert fit_temperature(logits, np.zeros_like(y)) == 1.0
    mat_logits = np.column_stack([logits, logits / 3.0])
    mat_y = np.column_stack([y, y])
    T = fit_temperature_per_task(mat_logits, mat_y)
    assert T[0] == pytest.approx(3.0, abs=0.1) and T[1] == pytest.approx(1.0, abs=0.05)
