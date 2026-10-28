import numpy as np

from toxion.data.loader import TOX21_TASKS
from toxion.data.splits import assert_no_scaffold_overlap, load_split, make_and_cache_splits, scaffold_split


def test_split_partitions_and_has_no_scaffold_leak(synthetic_df):
    tr, va, te = scaffold_split(synthetic_df, seed=0)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(len(synthetic_df)))
    assert_no_scaffold_overlap(synthetic_df, tr, va, te)


def test_split_is_deterministic_per_seed(synthetic_df):
    a = scaffold_split(synthetic_df, seed=3)
    b = scaffold_split(synthetic_df, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_big_scaffold_groups_cannot_empty_validation():
    import pandas as pd

    acyclic = [f"{'C' * k}O" for k in range(1, 41)]
    rings = [f"C1CC1{'C' * k}" if k else "C1CC1" for k in range(0, 0)]
    singles = ["C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
               "C1CCOC1", "C1COC1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "C1CCNCC1", "C1CCNC1", "c1cncnc1",
               "c1ccc2ncccc2c1", "C1=CCCCC1", "c1ccc(cc1)-c1ccccc1", "C1CCC2CCCCC2C1", "c1ccc2occc2c1",
               "c1cnccn1", "c1ccnnc1", "C1CCSC1", "c1cscn1", "c1cocn1", "c1c[nH]cn1", "c1ccc2[nH]cnc2c1",
               "C1CCOCC1", "C1COCCO1", "C1CNCCN1"]
    df = pd.DataFrame({"smiles_canonical": acyclic + rings + singles})
    for seed in range(5):
        tr, va, te = scaffold_split(df, seed=seed)
        assert len(tr) > 0 and len(va) > 0 and len(te) > 0
        assert set(range(40)) <= set(tr.tolist())
        assert_no_scaffold_overlap(df, tr, va, te)


def test_cache_roundtrip(synthetic_df, tmp_path):
    make_and_cache_splits(synthetic_df, [0, 1], str(tmp_path))
    tr, va, te = load_split(str(tmp_path), 1)
    assert len(tr) + len(va) + len(te) == len(synthetic_df)
    assert set(TOX21_TASKS) <= set(synthetic_df.columns)
