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

def tanimoto_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    if B is None:
        B = A
    dot = A @ B.T
    a_sq = A.sum(axis=1, keepdims=True)
    b_sq = B.sum(axis=1, keepdims=True).T
    denom = a_sq + b_sq - dot
    return np.divide(dot, denom, out=np.zeros_like(dot, dtype=np.float64), where=denom > 0)

def nearest_neighbor_similarity(A: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    best_sim = np.zeros(n, dtype=np.float32)
    best_idx = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sub = tanimoto_matrix(A[start:stop], A)
        for i in range(start, stop):
            sub[i - start, i] = -1.0
        best_sim[start:stop] = sub.max(axis=1)
        best_idx[start:stop] = sub.argmax(axis=1)
    return best_sim, best_idx

def internal_diversity(A: np.ndarray, max_molecules: int = 2000, seed: int = 0) -> float:
    n = A.shape[0]
    if n <= 1:
        return 0.0
    if n > max_molecules:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_molecules, replace=False)
        A = A[idx]
        n = max_molecules
    sim = tanimoto_matrix(A)
    triu_idx = np.triu_indices(n, k=1)
    mean_sim = float(sim[triu_idx].mean())
    return 1.0 - mean_sim
