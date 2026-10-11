"""Chemical-space analysis of the cleaned molecules.

PCA via SVD of the centered fingerprint matrix Xc = U S V^T:
    explained variance of component k = S_k^2 / (n - 1)
    scores = Xc V_k = U_k S_k
Tanimoto similarity of binary fingerprints a, b:
    T(a, b) = |a & b| / (|a| + |b| - |a & b|)
Scaffold diversity: Shannon entropy H = -sum_s p_s log2 p_s over Bemis-Murcko
scaffold frequencies, and the effective number of scaffolds 2**H.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


def morgan_bits(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """[n, n_bits] uint8 Morgan bit fingerprints."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Unparseable SMILES at row {i}: {smi}")
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


@dataclass
class PCAResult:
    mean: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    full_variance_ratio: np.ndarray


def pca_svd(X: np.ndarray, n_components: int = 2) -> PCAResult:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError("PCA needs at least 2 samples")
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(axis=1)])
    signs[signs == 0] = 1.0
    Vt = Vt * signs[:, None]
    U = U * signs[None, :]
    var = S**2 / (n - 1)
    total = var.sum()
    k = min(n_components, len(S))
    ratio = var / total if total > 0 else np.zeros_like(var)
    return PCAResult(
        mean=mean,
        components=Vt[:k],
        scores=U[:, :k] * S[:k],
        explained_variance=var[:k],
        explained_variance_ratio=ratio[:k],
        full_variance_ratio=ratio,
    )


def tanimoto_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Tanimoto similarity between rows of binary matrices.
    Two all-zero fingerprints get similarity 0."""
    A = np.asarray(A, dtype=np.float32)
    B = A if B is None else np.asarray(B, dtype=np.float32)
    inter = A @ B.T
    union = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nearest_neighbor_similarity(A: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the highest Tanimoto similarity to any *other* row and
    that row's index. Chunked so memory stays O(chunk * n)."""
    n = A.shape[0]
    if n < 2:
        raise ValueError("need at least 2 molecules")
    best = np.empty(n, dtype=np.float32)
    idx = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sim = tanimoto_matrix(A[start:stop], A)
        sim[np.arange(stop - start), np.arange(start, stop)] = -1.0
        idx[start:stop] = sim.argmax(axis=1)
        best[start:stop] = sim.max(axis=1)
    return best, idx


def internal_diversity(A: np.ndarray, max_molecules: int = 2000, seed: int = 0) -> float:
    """Mean pairwise (1 - Tanimoto) over distinct pairs, on a random subsample
    of at most max_molecules rows."""
    n = A.shape[0]
    if n > max_molecules:
        rows = np.random.default_rng(seed).choice(n, size=max_molecules, replace=False)
        A, n = A[rows], max_molecules
    if n < 2:
        raise ValueError("need at least 2 molecules")
    sim = tanimoto_matrix(A).astype(np.float64)
    off_diag_sum = sim.sum() - np.trace(sim)
    return float(1.0 - off_diag_sum / (n * (n - 1)))


def scaffold_stats(smiles: list[str], top_k: int = 10) -> dict:
    """Bemis-Murcko scaffold distribution. The empty scaffold '' is acyclic
    molecules. Matters for scaffold splitting: a very large group lands whole
    in one split."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        scaffolds.append("" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    counts = Counter(scaffolds)
    n = len(scaffolds)
    freqs = np.array(list(counts.values()), dtype=np.float64) / n
    entropy = float(-(freqs * np.log2(freqs)).sum())
    ranked = counts.most_common(top_k)
    return {
        "n_molecules": n,
        "n_scaffolds": len(counts),
        "n_acyclic": counts.get("", 0),
        "singleton_scaffolds": sum(1 for c in counts.values() if c == 1),
        "largest_group_fraction": ranked[0][1] / n if ranked else float("nan"),
        "top_k_fraction": sum(c for _, c in ranked) / n,
        "entropy_bits": entropy,
        "effective_n_scaffolds": float(2.0**entropy),
        "top_scaffolds": [(s, c) for s, c in ranked],
    }
