
from __future__ import annotations

import logging
import os

import numpy as np
import torch

from toxion import device as device_mod
from toxion.analysis.chemspace import morgan_bits
from toxion.data.augment import AugmentConfig
from toxion.data.featurize import FlatScaler, edge_dim, node_dim
from toxion.eval.calibration import sigmoid
from toxion.eval.metrics import masked_roc_auc
from toxion.data.loader import TOX21_TASKS
from toxion.inference.registry import build_model, make_checkpoint, make_meta, save_checkpoint
from toxion.training import pipeline as pl
from toxion.training.trainer import predict_logits, train_model

logger = logging.getLogger(__name__)


def _auc(logits, y) -> float:
    r = masked_roc_auc(y, sigmoid(logits), TOX21_TASKS)
    return float(np.mean(list(r.values()))) if r else float("nan")


def run_training(clean_path: str, cfg: pl.TrainConfig, out_path: str, device: str | None = None,
                 resume: bool = False) -> dict:
    manifest = pl.read_manifest(clean_path)
    device = device or device_mod.pick_device()
    batch_size = device_mod.choose_batch_size(cfg.batch_size)
    logger.info("device=%s batch_size=%d (one per-edge matrix tensor ~%.0f MB, estimate)", device, batch_size,
                device_mod.estimate_edge_memory_mb(batch_size, cfg.hidden_dim))

    data = pl.load_training_data(clean_path, cfg.seed, cfg.max_molecules, cfg.split_mode, cfg.feature_version)
    scaler = FlatScaler.fit(data.flat_raw[data.train_idx])
    train_ds = pl.make_dataset(data, data.train_idx, scaler)
    val_loader = pl.make_loader(pl.make_dataset(data, data.val_idx, scaler), batch_size, False)
    test_loader = pl.make_loader(pl.make_dataset(data, data.test_idx, scaler), batch_size, False)
    aug_cfg = AugmentConfig(node_mask_p=cfg.node_mask_p, edge_drop_p=cfg.edge_drop_p, bit_drop_p=cfg.bit_drop_p)

    members, val_logits_all, test_logits_all = [], [], []
    for k in range(cfg.ensemble_size):
        seed_k = cfg.seed + k
        pl.seed_everything(seed_k)
        model = build_model(cfg.model_type, cfg.hidden_dim, cfg.T, cfg.dropout, cfg.node_dropout,
                            node_dim(cfg.feature_version), edge_dim(cfg.feature_version))
        if cfg.pretrained:
            from toxion.training.pretrain import load_pretrained_encoder

            info = load_pretrained_encoder(model, cfg.pretrained, cfg.hidden_dim, cfg.T, cfg.feature_version)
            if k == 0:
                logger.info("loaded pretrained encoder (%d molecules, masked-atom acc %.3f)", info["n_molecules"],
                            info["val_masked_atom_acc"])
        member_file = out_path + f".member{k}.pt"
        if resume and cfg.ensemble_size > 1 and os.path.exists(member_file):
            model.load_state_dict(torch.load(member_file, map_location="cpu", weights_only=True))
            logger.info("member %d restored from %s", k, member_file)
        else:
            logger.info("training member %d/%d (seed %d)", k + 1, cfg.ensemble_size, seed_k)
            state = out_path + (".state" if cfg.ensemble_size == 1 else f".state{k}")
            model = train_model(model, pl.make_loader(train_ds, batch_size, True, seed_k), val_loader, device,
                                epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience, state_path=state, resume=resume,
                                augmenter=pl.make_augmenter(cfg.augment, seed_k, aug_cfg),
                                weight_decay=cfg.weight_decay, schedule=cfg.schedule, monitor=cfg.monitor)
            if cfg.ensemble_size > 1:
                torch.save({k_: v.cpu() for k_, v in model.state_dict().items()}, member_file)
            if os.path.exists(state):
                os.remove(state)
        model.to(device)
        v, vy = predict_logits(model, val_loader, device)
        t, ty = predict_logits(model, test_loader, device)
        val_logits_all.append(v)
        test_logits_all.append(t)
        members.append(model.cpu())

    val_logits, test_logits = np.mean(val_logits_all, axis=0), np.mean(test_logits_all, axis=0)
    temps, thresholds = pl.calibrate(val_logits, vy)
    metrics = {"val": pl.evaluate(val_logits, vy, temps), "test": pl.evaluate(test_logits, ty, temps),
               "augment": cfg.augment, "epochs_cap": cfg.epochs, "ensemble_size": cfg.ensemble_size}
    if cfg.ensemble_size > 1:
        metrics["members"] = {"val_auc": [_auc(v, vy) for v in val_logits_all],
                              "test_auc": [_auc(t, ty) for t in test_logits_all]}

    if cfg.ensemble_size > 1:
        from toxion.models.ensemble import EnsembleModel

        final = EnsembleModel(members)
    else:
        final = members[0]
    sha = manifest["raw_sha256"] if not cfg.max_molecules else f"{manifest['raw_sha256']}:debug{cfg.max_molecules}"
    meta = make_meta(model_type=cfg.model_type, hidden_dim=cfg.hidden_dim, T=cfg.T, scaler=scaler,
                     temperatures=temps, thresholds=thresholds, data_sha256=sha,
                     n_molecules=len(data.df), split_seed=cfg.seed, metrics=metrics,
                     extras={"split_mode": cfg.split_mode, "dropout": cfg.dropout,
                             "node_dropout": cfg.node_dropout, "weight_decay": cfg.weight_decay,
                             "feature_version": cfg.feature_version, "node_dim": node_dim(cfg.feature_version),
                             "edge_dim": edge_dim(cfg.feature_version), "ensemble_size": cfg.ensemble_size,
                             "pretrained": os.path.basename(cfg.pretrained) if cfg.pretrained else None})
    train_smiles = data.df["smiles_canonical"].iloc[data.train_idx].tolist()
    save_checkpoint(out_path, make_checkpoint(final, meta, morgan_bits(train_smiles), set(data.df["smiles_canonical"])))
    for k in range(cfg.ensemble_size):
        f = out_path + f".member{k}.pt"
        if os.path.exists(f):
            os.remove(f)
    return metrics
