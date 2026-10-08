"""Load raw Tox21 data, canonicalize SMILES, deduplicate, keep missing
labels as NaN (never zero-filled: zero would mean 'tested inactive', NaN
means 'not tested', and the masked loss/metrics downstream depend on that
distinction being preserved).

pandas and RDKit are imported inside the functions that use them, so
importing this module (and running `toxion --help`) stays cheap.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if the input doesn't parse
    (including non-string input such as a NaN cell)."""
    from rdkit import Chem

    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _count_label_conflicts(df: pd.DataFrame) -> int:
    """Number of canonical SMILES that appear more than once with
    disagreeing non-missing labels on at least one task."""
    nunique = df.groupby("smiles_canonical")[TOX21_TASKS].nunique(dropna=True)
    return int((nunique > 1).any(axis=1).sum())


def load_and_clean(raw_path: str, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    raw_path: path to the raw Tox21 CSV (as distributed by DeepChem/MoleculeNet;
    .csv.gz is read directly).
    Returns a DataFrame with columns: smiles_canonical + TOX21_TASKS (floats,
    NaN for missing labels), deduplicated on canonical SMILES (first
    occurrence kept).
    """
    import pandas as pd

    df = pd.read_csv(raw_path)
    missing_cols = [c for c in [smiles_col, *TOX21_TASKS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df["smiles_canonical"] = df[smiles_col].apply(canonicalize_smiles)

    n_unparseable = int(df["smiles_canonical"].isna().sum())
    if n_unparseable:
        logger.warning("Dropping %d rows with unparseable SMILES", n_unparseable)
    df = df.dropna(subset=["smiles_canonical"]).copy()

    for task in TOX21_TASKS:
        df[task] = pd.to_numeric(df[task], errors="coerce")

    n_conflicts = _count_label_conflicts(df)
    if n_conflicts:
        logger.warning(
            "%d duplicated molecules have conflicting labels; keeping the first "
            "occurrence of each", n_conflicts,
        )

    n_before = len(df)
    df = df.drop_duplicates(subset="smiles_canonical", keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d duplicate molecules (same canonical SMILES)", n_dropped)

    out_cols = ["smiles_canonical", *TOX21_TASKS]
    df = df[out_cols].reset_index(drop=True)
    logger.info("Clean dataset: %d molecules, %d tasks", len(df), len(TOX21_TASKS))
    return df


def save_clean(df: pd.DataFrame, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_path, index=False)


def load_clean(path: str) -> pd.DataFrame:
    import pandas as pd

    return pd.read_parquet(path)
