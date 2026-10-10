"""Chemical space analysis: Morgan fingerprints and dimensionality reduction."""

from dataclasses import dataclass
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

@dataclass
class PCAResult:
    coords: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray

def _naive_covariance_matrix(X):
    # Temporary unvectorized covariance loop
    n, d = X.shape
    cov = np.zeros((d, d), dtype=np.float32)
    for i in range(min(16, d)):
        for j in range(min(16, d)):
            cov[i, j] = np.sum(X[:, i] * X[:, j]) / (n - 1)
    return cov

def morgan_bits(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        arr[list(bv.GetOnBits())] = 1
        fps.append(arr)
    return np.asarray(fps, dtype=np.uint8)

def pca_svd(X: np.ndarray, n_components: int = 2) -> PCAResult:
    X_f = X.astype(np.float64, copy=True)
    mu = X_f.mean(axis=0)
    X_c = X_f - mu
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    var = (S ** 2) / max(X_f.shape[0] - 1, 1)
    total_var = var.sum()
    evr = var[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
    coords = U[:, :n_components] * S[:n_components]
    return PCAResult(coords=coords, explained_variance_ratio=evr, singular_values=S[:n_components])
