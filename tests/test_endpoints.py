import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score

from toxion.analysis.endpoints import (
    endpoint_dependence,
    labels_to_array,
    mutual_information_matrix,
    phi_matrix,
)
from toxion.data.loader import TOX21_TASKS


def test_phi_matches_pandas_pairwise_correlation(synthetic_df):
    ours = phi_matrix(labels_to_array(synthetic_df))
    ref = synthetic_df[TOX21_TASKS].corr().to_numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-9, equal_nan=True)


def test_mutual_information_matches_sklearn(synthetic_df):
    y = labels_to_array(synthetic_df)
    mi = mutual_information_matrix(y)
    for i, j in [(0, 0), (0, 3), (2, 7), (5, 11), (9, 10)]:
        both = ~np.isnan(y[:, i]) & ~np.isnan(y[:, j])
        ref = mutual_info_score(y[both, i], y[both, j]) / np.log(2)
        assert mi[i, j] == pytest.approx(ref, abs=1e-9)


def test_perfect_anti_and_independent_pairs():
    a = np.array([0, 0, 1, 1] * 25, dtype=float)
    same, flipped = a.copy(), 1 - a
    independent = np.array([0, 1, 0, 1] * 25, dtype=float)
    y = np.column_stack([a, same, flipped, independent])
    phi, mi = phi_matrix(y), mutual_information_matrix(y)
    assert phi[0, 1] == pytest.approx(1.0)
    assert phi[0, 2] == pytest.approx(-1.0)
    assert phi[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert mi[0, 1] == pytest.approx(1.0)
    assert mi[0, 2] == pytest.approx(1.0)
    assert mi[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert mi[0, 0] == pytest.approx(1.0)


def test_constant_endpoint_gives_nan_phi_not_error():
    y = np.column_stack([np.zeros(10), np.array([0, 1] * 5, dtype=float)])
    assert np.isnan(phi_matrix(y)[0, 1])


def test_endpoint_dependence_frames_and_colabeled_counts(synthetic_df):
    out = endpoint_dependence(synthetic_df)
    assert set(out) == {"phi", "mi_bits", "colabeled"}
    assert list(out["phi"].index) == TOX21_TASKS
    labeled = synthetic_df[TOX21_TASKS].notna().to_numpy().astype(int)
    np.testing.assert_array_equal(out["colabeled"].to_numpy(), labeled.T @ labeled)
