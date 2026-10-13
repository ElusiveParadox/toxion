import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

from toxion.analysis.chemspace import (
    internal_diversity,
    morgan_bits,
    nearest_neighbor_similarity,
    pca_svd,
    scaffold_stats,
    tanimoto_matrix,
)


@pytest.fixture(scope="module")
def bits(synthetic_df):
    return morgan_bits(synthetic_df["smiles_canonical"].tolist(), n_bits=256)


def test_morgan_bits_shape_and_binary(bits):
    assert bits.shape == (150, 256) and bits.dtype == np.uint8
    assert set(np.unique(bits)) <= {0, 1}


def test_morgan_bits_rejects_bad_smiles():
    with pytest.raises(ValueError, match="Unparseable"):
        morgan_bits(["CCO", "not_a_smiles"])


def test_pca_svd_matches_sklearn(bits):
    ours = pca_svd(bits, n_components=5)
    ref = PCA(n_components=5, svd_solver="full").fit(bits.astype(np.float64))
    np.testing.assert_allclose(ours.explained_variance_ratio, ref.explained_variance_ratio_, rtol=1e-8)
    np.testing.assert_allclose(ours.explained_variance, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(np.abs(ours.scores), np.abs(ref.transform(bits.astype(np.float64))), atol=1e-8)
    np.testing.assert_allclose(np.abs(ours.components), np.abs(ref.components_), atol=1e-8)


def test_pca_scores_are_centered_and_uncorrelated(bits):
    r = pca_svd(bits, n_components=4)
    np.testing.assert_allclose(r.scores.mean(axis=0), 0, atol=1e-9)
    cov = np.cov(r.scores.T)
    np.testing.assert_allclose(cov - np.diag(np.diag(cov)), 0, atol=1e-8)
    assert r.full_variance_ratio.sum() == pytest.approx(1.0)


def test_tanimoto_matches_rdkit(synthetic_df, bits):
    smiles = synthetic_df["smiles_canonical"].tolist()[:20]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    ref = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    np.testing.assert_allclose(tanimoto_matrix(bits[:20]), ref, atol=1e-6)


def test_tanimoto_edge_cases():
    a = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    sim = tanimoto_matrix(a)
    assert sim[0, 0] == 1.0
    assert sim[1, 1] == 0.0 and sim[0, 1] == 0.0


def test_nearest_neighbor_excludes_self_and_handles_chunks(bits):
    full = tanimoto_matrix(bits)
    np.fill_diagonal(full, -1.0)
    sim, idx = nearest_neighbor_similarity(bits, chunk=37)
    np.testing.assert_allclose(sim, full.max(axis=1), atol=1e-6)
    assert not np.any(idx == np.arange(len(bits)))


def test_internal_diversity_extremes():
    same = np.tile(np.array([1, 1, 0, 0], dtype=np.uint8), (5, 1))
    orthogonal = np.eye(5, dtype=np.uint8)
    assert internal_diversity(same) == pytest.approx(0.0)
    assert internal_diversity(orthogonal) == pytest.approx(1.0)


def test_scaffold_stats_known_case():
    stats = scaffold_stats(["c1ccccc1C", "c1ccccc1Cl", "CCO", "c1ccncc1"])
    assert stats["n_scaffolds"] == 3
    assert stats["n_acyclic"] == 1
    assert stats["singleton_scaffolds"] == 2
    assert stats["largest_group_fraction"] == 0.5
    assert stats["entropy_bits"] == pytest.approx(1.5)
    assert stats["effective_n_scaffolds"] == pytest.approx(2**1.5)
