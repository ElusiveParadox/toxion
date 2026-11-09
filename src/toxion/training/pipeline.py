
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from toxion import config
from toxion.data.augment import AugmentConfig, Augmenter
from toxion.data.featurize import FlatScaler, featurize_flat, featurize_graph
from toxion.data.graph_dataset import MolDataset, collate_batch
from toxion.data.loader import TOX21_TASKS, load_clean
from toxion.data.splits import scaffold_split
from toxion.eval.calibration import apply_temperature, expected_calibration_error, fit_temperature_per_task, sigmoid
from toxion.eval.metrics import fpr_capped_thresholds, masked_pr_auc, masked_roc_auc, youden_thresholds

logger = logging.getLogger(__name__)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(clean_path: str) -> str:
    return clean_path + ".manifest.json"


def write_manifest(clean_path: str, raw_path: str, n_rows: int) -> dict:
    manifest = {"raw_path": raw_path, "raw_sha256": file_sha256(raw_path), "n_rows": int(n_rows)}
    with open(manifest_path(clean_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def read_manifest(clean_path: str) -> dict:
    p = manifest_path(clean_path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Re-run `toxion prepare-data` so the clean file records which raw file it came from."
        )
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class TrainConfig:
    model_type: str = "fusion"
    hidden_dim: int = 128
    T: int = 4
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    seed: int = 0
    augment: bool = True
    max_molecules: int | None = None
    feature_version: int = 1
    ensemble_size: int = 1
    pretrained: str | None = None
    split_mode: str = "scaffold"
    weight_decay: float = 0.0
    schedule: str = "none"
    monitor: str = "val_loss"
    dropout: float | None = None
    node_dropout: float = 0.0
    node_mask_p: float = 0.10
    edge_drop_p: float = 0.10
    bit_drop_p: float = 0.10


@dataclass
class TrainingData:
    df: pd.DataFrame
    graphs: list
    flat_raw: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def featurize_dataframe(df: pd.DataFrame, cache_path: str | None = None, feature_version: int = 1):
    smiles = df["smiles_canonical"].tolist()
    key = hashlib.sha1(("\n".join(smiles) + f"|v{feature_version}").encode()).hexdigest()
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            return cached["graphs"], cached["flat"]
    graphs = [featurize_graph(s, feature_version) for s in smiles]
    flat = np.stack([featurize_flat(s) for s in smiles]).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": key, "graphs": graphs, "flat": flat}, fh)
    return graphs, flat


def feature_cache_path(clean_path: str, feature_version: int = 1) -> str:
    return clean_path + (".features.pkl" if feature_version == 1 else f".features.v{feature_version}.pkl")


def random_split(n: int, seed: int, frac_train: float = 0.8, frac_val: float = 0.1):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(frac_train * n), int((frac_train + frac_val) * n)
    return perm[:a].astype(np.int64), perm[a:b].astype(np.int64), perm[b:].astype(np.int64)


def load_training_data(clean_path: str, split_seed: int, max_molecules: int | None = None,
                       split_mode: str = "scaffold", feature_version: int = 1) -> TrainingData:
    df = load_clean(clean_path)
    cache_path = feature_cache_path(clean_path, feature_version)
    if max_molecules:
        df = df.iloc[:max_molecules].reset_index(drop=True)
        cache_path = None
    graphs, flat = featurize_dataframe(df, cache_path, feature_version)
    labels = df[TOX21_TASKS].to_numpy(dtype=np.float32)
    tr, va, te = scaffold_split(df, split_seed) if split_mode == "scaffold" else random_split(len(df), split_seed)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"split has an empty part (train {len(tr)}, val {len(va)}, test {len(te)}); too few distinct scaffolds?")
    logger.info("split seed %d: train %d / val %d / test %d", split_seed, len(tr), len(va), len(te))
    return TrainingData(df, graphs, flat, labels, tr, va, te)


def make_dataset(data: TrainingData, idx: np.ndarray, scaler: FlatScaler) -> MolDataset:
    return MolDataset([data.graphs[i] for i in idx], scaler.transform(data.flat_raw[idx]), data.labels[idx])


def make_loader(ds, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                      num_workers=0, generator=gen if shuffle else None)


def make_augmenter(enabled: bool, seed: int = 0, cfg: AugmentConfig | None = None) -> Augmenter | None:
    if not enabled:
        return None
    return Augmenter(cfg or AugmentConfig(enabled=True), generator=torch.Generator().manual_seed(seed))


def calibrate(logits_val: np.ndarray, y_val: np.ndarray):
    temps = fit_temperature_per_task(logits_val, y_val)
    probs = sigmoid(apply_temperature(logits_val, temps))
    cap = config.fpr_cap()
    thresholds = fpr_capped_thresholds(y_val, probs, cap) if cap else youden_thresholds(y_val, probs)
    return temps, thresholds


def mean_ece(y: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for t in range(y.shape[1]):
        if len(np.unique(y[~np.isnan(y[:, t]), t])) < 2:
            continue
        vals.append(expected_calibration_error(y[:, t], probs[:, t]))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(logits: np.ndarray, y: np.ndarray, temps: np.ndarray) -> dict:
    probs = sigmoid(apply_temperature(logits, temps))
    roc = masked_roc_auc(y, probs, TOX21_TASKS)
    pr = masked_pr_auc(y, probs, TOX21_TASKS)
    return {
        "roc_auc_mean": float(np.mean(list(roc.values()))) if roc else float("nan"),
        "pr_auc_mean": float(np.mean(list(pr.values()))) if pr else float("nan"),
        "ece_mean": mean_ece(y, probs),
        "roc_auc": roc,
    }


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
