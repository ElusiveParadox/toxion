"""Label statistics for the cleaned Tox21 table: per-endpoint class balance
and overall missingness. Used by `toxion summary` and the data-check notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from toxion.data.loader import TOX21_TASKS

if TYPE_CHECKING:
    import pandas as pd


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per endpoint: labeled / missing counts, positives, negatives,
    positive_rate (positives / labeled). NaN labels are 'not tested' and are
    never counted as negatives."""
    import pandas as pd

    n = len(df)
    rows = {}
    for task in TOX21_TASKS:
        col = df[task]
        labeled = int(col.notna().sum())
        positives = int((col == 1).sum())
        negatives = int((col == 0).sum())
        rows[task] = {
            "labeled": labeled,
            "missing": n - labeled,
            "missing_pct": round(100 * (n - labeled) / n, 1) if n else float("nan"),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": round(positives / labeled, 3) if labeled else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def overall_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    total = n * len(TOX21_TASKS)
    present = df[TOX21_TASKS].notna()
    labeled = int(present.to_numpy().sum())
    return {
        "molecules": n,
        "labeled_entries": labeled,
        "total_entries": total,
        "missing_pct": round(100 * (total - labeled) / total, 1) if total else float("nan"),
        "molecules_without_labels": int((~present.any(axis=1)).sum()),
        "mean_labels_per_molecule": round(labeled / n, 2) if n else float("nan"),
    }
