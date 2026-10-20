import numpy as np
import pandas as pd

from toxion.data.loader import TOX21_TASKS, load_and_clean
from toxion.data.synthetic import make_synthetic_clean, make_synthetic_raw


def test_synthetic_is_unique_deterministic_and_has_missing_labels(synthetic_df):
    assert len(synthetic_df) == 150
    assert synthetic_df["smiles_canonical"].is_unique
    again = make_synthetic_clean(n=150, seed=0)
    pd.testing.assert_frame_equal(again, synthetic_df)
    missing = synthetic_df[TOX21_TASKS].isna().to_numpy().mean()
    assert 0.05 < missing < 0.5
    assert set(np.unique(synthetic_df[TOX21_TASKS].to_numpy()[~synthetic_df[TOX21_TASKS].isna().to_numpy()])) <= {0.0, 1.0}


def test_synthetic_raw_round_trips_through_loader(tmp_path):
    path = tmp_path / "raw.csv.gz"
    make_synthetic_raw(60, seed=1).to_csv(path, index=False)
    df = load_and_clean(str(path))
    assert len(df) == 60
