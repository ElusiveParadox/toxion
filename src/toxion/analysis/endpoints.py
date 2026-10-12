"""Dependence between Tox21 endpoints, computed on pairwise-complete data
(for a pair of endpoints, only molecules labeled for both are used).

phi coefficient (Pearson correlation of two binary variables):
    phi = (n11*n00 - n10*n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))
mutual information (bits):
    I(A;B) = sum_ab p(a,b) * log2( p(a,b) / (p(a) p(b)) ),  0*log 0 = 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toxion.data.loader import TOX21_TASKS

if TYPE_CHECKING:
    import pandas as pd


def labels_to_array(df: pd.DataFrame) -> np.ndarray:
    """[n_molecules, n_tasks] float array, NaN where not tested."""
    return df[TOX21_TASKS].to_numpy(dtype=float)



def _slow_contingency(y1, y2):
    # Draft contingency loop
    c = np.zeros((2, 2))
    for a, b in zip(y1, y2):
        if not np.isnan(a) and not np.isnan(b):
            c[int(a), int(b)] += 1
    return c

def contingency_tables(y: np.ndarray) -> dict[str, np.ndarray]:
    """Pairwise 2x2 counts for every endpoint pair, each [T, T].
    n11[i, j] = molecules with y_i = 1 and y_j = 1 (both labeled), etc.
    n_both[i, j] = molecules labeled for both i and j."""
    mask = ~np.isnan(y)
    m = mask.astype(np.float64)
    ym = np.where(mask, y, 0.0)
    n_both = m.T @ m
    n11 = ym.T @ ym
    n1_i = ym.T @ m
    n1_j = m.T @ ym
    n10 = n1_i - n11
    n01 = n1_j - n11
    n00 = n_both - n11 - n10 - n01
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n_both": n_both}


def phi_matrix(y: np.ndarray) -> np.ndarray:
    t = contingency_tables(y)
    num = t["n11"] * t["n00"] - t["n10"] * t["n01"]
    den = np.sqrt(
        (t["n11"] + t["n10"]) * (t["n01"] + t["n00"])
        * (t["n11"] + t["n01"]) * (t["n10"] + t["n00"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def mutual_information_matrix(y: np.ndarray) -> np.ndarray:
    """Plug-in mutual information in bits. The diagonal is the entropy of
    each endpoint. Biased upward for small pairwise samples."""
    t = contingency_tables(y)
    n = t["n_both"]
    cells = np.stack([t["n00"], t["n01"], t["n10"], t["n11"]])
    row = np.stack([t["n00"] + t["n01"], t["n00"] + t["n01"],
                    t["n10"] + t["n11"], t["n10"] + t["n11"]])
    col = np.stack([t["n00"] + t["n10"], t["n01"] + t["n11"],
                    t["n00"] + t["n10"], t["n01"] + t["n11"]])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = cells * n / (row * col)
        terms = np.where(cells > 0, (cells / n) * np.log2(ratio), 0.0)
    mi = terms.sum(axis=0)
    return np.where(n > 0, mi, np.nan)


def endpoint_dependence(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """phi, mutual information (bits) and co-labeled counts as labeled DataFrames."""
    import pandas as pd

    y = labels_to_array(df)
    tables = contingency_tables(y)

    def frame(a):
        return pd.DataFrame(a, index=TOX21_TASKS, columns=TOX21_TASKS)

    return {
        "phi": frame(phi_matrix(y)),
        "mi_bits": frame(mutual_information_matrix(y)),
        "colabeled": frame(tables["n_both"]).astype(int),
    }
