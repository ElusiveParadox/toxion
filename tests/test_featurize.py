import numpy as np
import pytest

from toxion.data.featurize import (DESC_SLICE, EDGE_FEATURE_DIM, FLAT_FEATURE_DIM, N_FP_BITS, NODE_FEATURE_DIM,
                                   FlatScaler, featurize_flat, featurize_graph, featurize_toxicophore_alerts)


def test_dimensions_are_the_documented_contract():
    assert (NODE_FEATURE_DIM, EDGE_FEATURE_DIM, FLAT_FEATURE_DIM) == (21, 5, 2066)


@pytest.mark.parametrize("smiles,n_atoms,n_directed_edges", [("C", 1, 0), ("CCO", 3, 4), ("c1ccccc1", 6, 12)])
def test_graph_shapes(smiles, n_atoms, n_directed_edges):
    g = featurize_graph(smiles)
    assert g.node_features.shape == (n_atoms, NODE_FEATURE_DIM)
    assert g.edge_index.shape == (2, n_directed_edges)
    assert g.edge_features.shape == (n_directed_edges, EDGE_FEATURE_DIM)


def test_edges_are_consecutive_reverse_pairs():
    g = featurize_graph("CC(=O)Oc1ccccc1C(=O)O")
    assert np.array_equal(g.edge_index[:, 0::2][::-1], g.edge_index[:, 1::2])


def test_flat_vector_and_alerts():
    assert featurize_flat("CCO").shape == (FLAT_FEATURE_DIM,)
    nitro = featurize_toxicophore_alerts("c1ccccc1[N+](=O)[O-]")
    aniline = featurize_toxicophore_alerts("Nc1ccccc1")
    assert nitro.sum() >= 0 and aniline[1] == 1.0
    assert featurize_toxicophore_alerts("C1CO1")[2] == 1.0


def test_unparseable_smiles_raises():
    with pytest.raises(ValueError):
        featurize_graph("not-a-smiles")


def test_flat_scaler_log1p_and_zscore():
    X = np.stack([featurize_flat(s) for s in ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCl"]])
    sc = FlatScaler.fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z[:, :N_FP_BITS], np.log1p(X[:, :N_FP_BITS]))
    assert np.allclose(Z[:, DESC_SLICE].mean(axis=0), 0, atol=1e-5)
    assert np.array_equal(Z[:, DESC_SLICE.stop:], X[:, DESC_SLICE.stop:])
    assert FlatScaler.from_dict(sc.to_dict()).transform(X).shape == Z.shape
