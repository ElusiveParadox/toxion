"""Run the advanced EDA on a cleaned table and write results to a folder:
CSV tables plus a JSON summary. Plots live in the notebook, not here."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

from toxion.analysis.chemspace import (
    internal_diversity,
    morgan_bits,
    nearest_neighbor_similarity,
    pca_svd,
    scaffold_stats,
)
from toxion.analysis.endpoints import endpoint_dependence

if TYPE_CHECKING:
    import pandas as pd


def run_analysis(df: pd.DataFrame, out_dir: str, n_bits: int = 2048, n_components: int = 10,
                 max_molecules: int | None = None, seed: int = 0) -> dict:
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    if max_molecules and len(df) > max_molecules:
        df = df.sample(max_molecules, random_state=seed).reset_index(drop=True)

    for name, frame in endpoint_dependence(df).items():
        frame.to_csv(os.path.join(out_dir, f"endpoints_{name}.csv"))

    smiles = df["smiles_canonical"].tolist()
    bits = morgan_bits(smiles, n_bits=n_bits)

    pca = pca_svd(bits, n_components)
    k = len(pca.explained_variance_ratio)
    pd.DataFrame({
        "component": np.arange(1, k + 1),
        "explained_variance_ratio": pca.explained_variance_ratio,
        "cumulative": np.cumsum(pca.explained_variance_ratio),
    }).to_csv(os.path.join(out_dir, "pca_variance.csv"), index=False)
    if k >= 2:
        pd.DataFrame({"smiles_canonical": smiles, "pc1": pca.scores[:, 0], "pc2": pca.scores[:, 1]}
                     ).to_csv(os.path.join(out_dir, "pca_scores_top2.csv"), index=False)

    nn_sim, _ = nearest_neighbor_similarity(bits)
    scaf = scaffold_stats(smiles)
    pd.DataFrame(scaf["top_scaffolds"], columns=["scaffold", "count"]).to_csv(
        os.path.join(out_dir, "scaffold_top.csv"), index=False)

    cumulative_full = np.cumsum(pca.full_variance_ratio)
    summary = {
        "n_molecules": len(df),
        "fingerprint": {"type": "morgan_bits", "radius": 2, "n_bits": n_bits},
        "pca": {
            "variance_ratio_top": [float(v) for v in pca.explained_variance_ratio],
            "components_for_90pct_variance": int(np.searchsorted(cumulative_full, 0.9) + 1),
        },
        "nearest_neighbor_tanimoto": {
            "mean": float(nn_sim.mean()),
            "median": float(np.median(nn_sim)),
            "p10": float(np.percentile(nn_sim, 10)),
            "p90": float(np.percentile(nn_sim, 90)),
            "fraction_above_0.7": float((nn_sim > 0.7).mean()),
        },
        "internal_diversity": internal_diversity(bits),
        "scaffolds": {k_: v for k_, v in scaf.items() if k_ != "top_scaffolds"},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
