
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from toxion import __version__, config
from toxion.data.featurize import EDGE_FEATURE_DIM, FLAT_FEATURE_DIM, NODE_FEATURE_DIM, FlatScaler
from toxion.data.loader import TOX21_TASKS

FORMAT_VERSION = 1


class RealDataGateError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: torch.nn.Module
    meta: dict
    scaler: FlatScaler
    temperatures: np.ndarray
    thresholds: np.ndarray
    ad_bits: np.ndarray
    known_smiles: frozenset
    device: str = "cpu"


def build_model(model_type: str, hidden_dim: int, T: int, dropout: float | None = None,
                node_dropout: float = 0.0, node_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                ensemble_size: int = 1) -> torch.nn.Module:
    if ensemble_size > 1:
        from toxion.models.ensemble import EnsembleModel

        return EnsembleModel([build_model(model_type, hidden_dim, T, dropout, node_dropout, node_dim, edge_dim)
                              for _ in range(ensemble_size)])
    from toxion.models.fusion import FusionModel
    from toxion.models.mpnn import MPNNClassifier

    if model_type == "fusion":
        kw = {} if dropout is None else {"dropout": dropout}
        return FusionModel(node_dim, edge_dim, FLAT_FEATURE_DIM, hidden_dim=hidden_dim, T=T,
                           n_tasks=len(TOX21_TASKS), node_dropout=node_dropout, **kw)
    if model_type == "mpnn":
        kw = {} if dropout is None else {"dropout": dropout}
        return MPNNClassifier(node_dim, edge_dim, hidden_dim=hidden_dim, T=T,
                              n_tasks=len(TOX21_TASKS), node_dropout=node_dropout, **kw)
    if model_type == "gin":
        from toxion.models.gnn_baseline import GINBaseline

        return GINBaseline(node_dim, hidden_dim=hidden_dim, n_layers=max(T - 1, 1),
                           n_tasks=len(TOX21_TASKS))
    raise ValueError(f"unknown model_type: {model_type}")


def make_meta(*, model_type, hidden_dim, T, scaler: FlatScaler, temperatures, thresholds,
              data_sha256, n_molecules, split_seed, metrics, extras=None) -> dict:
    import datetime

    meta = {
        "format": FORMAT_VERSION, "toxion_version": __version__,
        "model_type": model_type, "hidden_dim": int(hidden_dim), "T": int(T),
        "node_dim": NODE_FEATURE_DIM, "edge_dim": EDGE_FEATURE_DIM, "flat_dim": FLAT_FEATURE_DIM,
        "tasks": list(TOX21_TASKS), "scaler": scaler.to_dict(),
        "temperatures": [float(x) for x in temperatures],
        "thresholds": [float(x) for x in thresholds],
        "data_sha256": data_sha256, "n_molecules": int(n_molecules),
        "split_seed": int(split_seed), "synthetic": False, "metrics": metrics,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if extras:
        meta.update(extras)
    return meta


def make_checkpoint(model: torch.nn.Module, meta: dict, ad_bits: np.ndarray, known_smiles) -> dict:
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        "ad_bits": torch.from_numpy(np.packbits(ad_bits.astype(np.uint8), axis=1)),
        "known_smiles": sorted(known_smiles),
    }


def save_checkpoint(path: str, ckpt: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def check_gate(meta: dict) -> None:
    problems = []
    if meta.get("synthetic"):
        problems.append("checkpoint is marked synthetic")
    if meta.get("data_sha256") != config.real_sha256():
        problems.append("training data hash does not match the pinned real Tox21 file")
    if int(meta.get("n_molecules", 0)) < config.MIN_REAL_MOLECULES:
        problems.append(f"only {meta.get('n_molecules', 0)} training molecules (< {config.MIN_REAL_MOLECULES})")
    if problems:
        raise RealDataGateError(
            "checkpoint was not trained on the real Tox21 data: " + "; ".join(problems)
            + ". Train with: toxion prepare-data (on the real file) then toxion train-fusion."
        )


def load_checkpoint(path: str, gate: bool = True, device: str = "cpu") -> LoadedModel:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Train one on the real data first: toxion train-fusion"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    if gate:
        check_gate(meta)
    model = build_model(meta["model_type"], meta["hidden_dim"], meta["T"], meta.get("dropout"),
                        meta.get("node_dropout", 0.0), meta.get("node_dim", NODE_FEATURE_DIM),
                        meta.get("edge_dim", EDGE_FEATURE_DIM), meta.get("ensemble_size", 1))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    bits = np.unpackbits(ckpt["ad_bits"].numpy(), axis=1)[:, :2048]
    return LoadedModel(
        model=model, meta=meta, scaler=FlatScaler.from_dict(meta["scaler"]),
        temperatures=np.array(meta["temperatures"]), thresholds=np.array(meta["thresholds"]),
        ad_bits=bits, known_smiles=frozenset(ckpt["known_smiles"]), device=device,
    )
