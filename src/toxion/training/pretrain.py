
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from toxion.data.featurize import PRETRAIN_ELEMENTS, atom_element_classes, edge_dim, featurize_graph, node_dim
from toxion.data.graph_dataset import collate_batch
from toxion.data.loader import canonicalize_smiles
from toxion.models.mpnn import MPNNEncoder

logger = logging.getLogger(__name__)
SMILES_COLUMNS = ("smiles", "canonical_smiles", "smiles_canonical", "mol", "structure")
MAX_ATOMS = 120


class NodePretrainNet(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int, T: int):
        super().__init__()
        self.encoder = MPNNEncoder(node_feat_dim, edge_feat_dim, hidden_dim, T)
        self.head = nn.Linear(hidden_dim, len(PRETRAIN_ELEMENTS) + 1)

    def forward(self, node_features, edge_index, edge_features):
        return self.head(self.encoder.node_states(node_features, edge_index, edge_features))


def read_smiles(paths: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for path in paths:
        df = pd.read_csv(path)
        col = next((c for c in df.columns if c.strip().lower() in SMILES_COLUMNS), None)
        if col is None:
            raise ValueError(f"{path}: no SMILES column (looked for {', '.join(SMILES_COLUMNS)}); columns: {list(df.columns)[:8]}")
        for raw in df[col].dropna().astype(str):
            can = canonicalize_smiles(raw.strip())
            if can is not None:
                seen.setdefault(can)
    return list(seen)


def build_pretrain_set(smiles: list[str], feature_version: int, exclude: set[str] | None = None,
                       max_molecules: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    pool = [s for s in smiles if not exclude or s not in exclude]
    rng.shuffle(pool)
    graphs, classes, used = [], [], []
    for s in pool:
        g = featurize_graph(s, feature_version)
        if g.node_features.shape[0] > MAX_ATOMS:
            continue
        graphs.append(g)
        classes.append(atom_element_classes(s))
        used.append(s)
        if max_molecules and len(used) >= max_molecules:
            break
    return graphs, classes, used


def _collate(batch):
    graphs, classes = zip(*batch)
    b = collate_batch([(g, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)) for g in graphs])
    return b, torch.from_numpy(np.concatenate(classes))


def _batches(graphs, classes, idx, batch_size, shuffle, rng):
    order = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(order), batch_size):
        yield _collate([(graphs[j], classes[j]) for j in order[i:i + batch_size]])


def _masked_step(net, batch, cls, mask_rate, device, gen):
    n = batch.node_features.shape[0]
    mask = torch.rand(n, generator=gen) < mask_rate
    if not mask.any():
        mask[torch.randint(n, (1,), generator=gen)] = True
    mask = mask.to(device)
    x = batch.node_features.to(device).clone()
    x[mask] = 0.0
    logits = net(x, batch.edge_index.to(device), batch.edge_features.to(device))[mask]
    target = cls.to(device)[mask]
    return nn.functional.cross_entropy(logits, target), (logits.argmax(-1) == target).float().mean().item(), target


def run_pretraining(paths: list[str], out_path: str, hidden_dim: int = 128, T: int = 4, feature_version: int = 2,
                    epochs: int = 3, batch_size: int = 32, lr: float = 1e-3, mask_rate: float = 0.15,
                    max_molecules: int | None = 30000, exclude: set[str] | None = None, seed: int = 0,
                    device: str = "cpu") -> dict:
    smiles = read_smiles(paths)
    logger.info("%d unique parseable molecules in %d file(s)", len(smiles), len(paths))
    graphs, classes, used = build_pretrain_set(smiles, feature_version, exclude, max_molecules, seed)
    if len(used) < 100:
        raise ValueError(f"only {len(used)} usable molecules after filtering; need at least 100")
    n_val = max(20, len(used) // 20)
    val_idx, train_idx = np.arange(n_val), np.arange(n_val, len(used))
    torch.manual_seed(seed)
    gen, rng = torch.Generator().manual_seed(seed), np.random.default_rng(seed)
    net = NodePretrainNet(node_dim(feature_version), edge_dim(feature_version), hidden_dim, T).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    all_classes = np.concatenate([classes[i] for i in val_idx])
    majority = float(np.bincount(all_classes).max() / len(all_classes))
    best_loss, best = float("inf"), None
    for epoch in range(epochs):
        t0, tr = time.time(), []
        net.train()
        for batch, cls in _batches(graphs, classes, train_idx, batch_size, True, rng):
            loss, _, _ = _masked_step(net, batch, cls, mask_rate, device, gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tr.append(loss.item())
        net.eval()
        vl, va = [], []
        with torch.no_grad():
            vgen = torch.Generator().manual_seed(123)
            for batch, cls in _batches(graphs, classes, val_idx, batch_size, False, rng):
                l, a, _ = _masked_step(net, batch, cls, mask_rate, device, vgen)
                vl.append(l.item())
                va.append(a)
        v_loss, v_acc = float(np.mean(vl)), float(np.mean(va))
        logger.info("pretrain epoch %d train_loss=%.4f val_loss=%.4f val_masked_atom_acc=%.3f (majority-class %.3f) (%.0fs)",
                    epoch, float(np.mean(tr)), v_loss, v_acc, majority, time.time() - t0)
        if v_loss < best_loss:
            best_loss = v_loss
            best = {k: v.detach().cpu().clone() for k, v in net.encoder.state_dict().items()}
            best_acc = v_acc
    meta = {"hidden_dim": hidden_dim, "T": T, "feature_version": feature_version,
            "node_dim": node_dim(feature_version), "edge_dim": edge_dim(feature_version),
            "n_molecules": len(used), "val_masked_atom_acc": best_acc, "majority_class_acc": majority,
            "epochs": epochs, "sources": [os.path.basename(p) for p in paths]}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"encoder_state": best, "meta": meta}, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    return meta


def load_pretrained_encoder(model: nn.Module, path: str, hidden_dim: int, T: int, feature_version: int) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    want = {"hidden_dim": hidden_dim, "feature_version": feature_version}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError("pretrained encoder does not match this training run (pretrained vs requested): "
                         + ", ".join(f"{k}: {a} vs {b}" for k, (a, b) in bad.items()))
    state = {k: v for k, v in ckpt["encoder_state"].items() if not k.startswith("readout.")}
    members = list(model.members) if hasattr(model, "members") else [model]
    for m in members:
        if not hasattr(m, "encoder"):
            raise ValueError("pretrained weights only apply to the MPNN and fusion models")
        missing, unexpected = m.encoder.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("readout.") for k in missing):
            raise ValueError(f"pretrained encoder keys do not fit the model: missing {missing}, unexpected {unexpected}")
    return meta
