"""Synthetic Tox21-shaped data for testing the pipeline before (or without)
the real dataset. Molecules are random substituted rings; labels are drawn from
a made-up rule with a shared latent factor so endpoints are correlated.

NEVER report anything computed from this data as a result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toxion.data.loader import TOX21_TASKS

if TYPE_CHECKING:
    import pandas as pd

TEMPLATES = [
    "c1{a}c{b}c{c}ccc1",
    "c1{a}c{b}c{c}cnc1",
    "c1{a}c{b}c{c}cs1",
    "c1{a}c{b}c{c}co1",
    "C1{a}C{b}C{c}CCC1",
    "C1{a}C{b}C{c}CNC1",
]
SUBSTITUENTS = ["", "", "Cl", "Br", "F", "N", "O", "C", "CC", "C(=O)O",
                "[N+](=O)[O-]", "C#N", "OC", "S", "C(F)(F)F"]
HALOGENS = {"Cl", "Br", "F"}


def _branch(s: str) -> str:
    return f"({s})" if s else ""


def make_synthetic_clean(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """DataFrame with smiles_canonical + the 12 Tox21 columns (NaN = missing),
    n unique molecules (fewer if the generator runs out of distinct ones)."""
    import pandas as pd
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    n_tasks = len(TOX21_TASKS)
    base = rng.uniform(-2.5, -0.5, n_tasks)
    loading = rng.uniform(0.4, 1.5, n_tasks)
    flag_names = ["amine", "aromatic", "sulfur", "ether"]
    extra_flag = rng.integers(0, len(flag_names), n_tasks)
    extra_w = rng.uniform(-1.0, 1.0, n_tasks)
    p_missing = rng.uniform(0.1, 0.45, n_tasks)

    seen, rows = set(), []
    for _ in range(40 * n):
        if len(rows) >= n:
            break
        template = TEMPLATES[rng.integers(len(TEMPLATES))]
        subs = [SUBSTITUENTS[rng.integers(len(SUBSTITUENTS))] for _ in range(3)]
        mol = Chem.MolFromSmiles(template.format(a=_branch(subs[0]), b=_branch(subs[1]), c=_branch(subs[2])))
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        seen.add(canonical)

        z = (1.2 * ("[N+](=O)[O-]" in subs) + 0.8 * any(s in HALOGENS for s in subs)
             + 0.7 * ("C(F)(F)F" in subs) - 0.6 * ("C(=O)O" in subs)
             + 0.5 * template.startswith("c") + rng.normal(0, 0.3))
        flags = {"amine": "N" in subs, "aromatic": template.startswith("c"),
                 "sulfur": "S" in subs, "ether": "OC" in subs}
        flag_vec = np.array([float(flags[flag_names[k]]) for k in extra_flag])
        prob = 1.0 / (1.0 + np.exp(-(base + loading * z + extra_w * flag_vec)))
        labels = (rng.random(n_tasks) < prob).astype(float)
        labels[rng.random(n_tasks) < p_missing] = np.nan
        rows.append([canonical, *labels])

    return pd.DataFrame(rows, columns=["smiles_canonical", *TOX21_TASKS])


def make_synthetic_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Same data in the raw Tox21 CSV layout (12 task columns + mol_id + smiles),
    readable by `toxion prepare-data`."""
    df = make_synthetic_clean(n, seed)
    raw = df[TOX21_TASKS].copy()
    raw["mol_id"] = [f"SYN{i:05d}" for i in range(len(df))]
    raw["smiles"] = df["smiles_canonical"].values
    return raw
