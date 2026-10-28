
from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd


def get_scaffold(smiles: str, include_chirality: bool = False) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(df: pd.DataFrame, seed: int, frac_train: float = 0.8, frac_val: float = 0.1,
                   frac_test: float = 0.1, smiles_col: str = "smiles_canonical"):
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    groups_by_scaffold = defaultdict(list)
    for pos, smiles in enumerate(df[smiles_col]):
        groups_by_scaffold[get_scaffold(smiles)].append(pos)

    n_total = len(df)
    n_train, n_val = int(frac_train * n_total), int(frac_val * n_total)
    n_test = n_total - n_train - n_val

    big_limit = min(n_val, n_test) / 2
    rng = random.Random(seed)
    big = [g for g in groups_by_scaffold.values() if len(g) > big_limit]
    small = [g for g in groups_by_scaffold.values() if len(g) <= big_limit]
    rng.shuffle(big)
    rng.shuffle(small)
    groups = big + small
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            test.extend(group)
    as_arr = lambda x: np.array(x, dtype=np.int64)
    return as_arr(train), as_arr(val), as_arr(test)


def assert_no_scaffold_overlap(df: pd.DataFrame, train_idx, val_idx, test_idx,
                               smiles_col: str = "smiles_canonical") -> None:
    scaffolds = df[smiles_col].apply(get_scaffold)
    tr, va, te = (set(scaffolds.iloc[i]) for i in (train_idx, val_idx, test_idx))
    assert not (tr & va), "scaffold leak: train/val"
    assert not (tr & te), "scaffold leak: train/test"
    assert not (va & te), "scaffold leak: val/test"


def make_and_cache_splits(df: pd.DataFrame, seeds: list[int], out_dir: str,
                          smiles_col: str = "smiles_canonical") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        tr, va, te = scaffold_split(df, seed, smiles_col=smiles_col)
        assert_no_scaffold_overlap(df, tr, va, te, smiles_col)
        np.savez(os.path.join(out_dir, f"split_seed{seed}.npz"), train_idx=tr, val_idx=va, test_idx=te)


def load_split(out_dir: str, seed: int):
    data = np.load(os.path.join(out_dir, f"split_seed{seed}.npz"))
    return data["train_idx"], data["val_idx"], data["test_idx"]
