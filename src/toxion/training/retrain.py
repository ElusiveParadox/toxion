
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np

from toxion import config, device as device_mod
from toxion.data.featurize import featurize_flat, featurize_graph
from toxion.data.graph_dataset import MolDataset
from toxion.db.models import STATUS_REJECTED, STATUS_TRAINED
from toxion.inference.registry import RealDataGateError, load_checkpoint, make_checkpoint, save_checkpoint
from toxion.training import pipeline as pl
from toxion.training.semi_supervised import SemiSupervisedConfig
from toxion.training.trainer import predict_logits, train_model

logger = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    decision: str
    message: str
    n_molecules: int = 0
    auc_before: float | None = None
    auc_after: float | None = None
    ece_before: float | None = None
    ece_after: float | None = None


def should_promote(auc_before: float, auc_after: float, ece_before: float, ece_after: float,
                   auc_tol: float = config.PROMOTE_AUC_TOLERANCE,
                   ece_tol: float = config.PROMOTE_ECE_TOLERANCE) -> bool:
    return bool(auc_after >= auc_before - auc_tol and ece_after <= ece_before + ece_tol)


def retrain(store, ckpt_path: str | None = None, clean_path: str | None = None, device: str | None = None,
            epochs: int = 15, lr: float = 2e-4, patience: int = 5, batch_n: int | None = None,
            ssl_cfg: SemiSupervisedConfig | None = None, augment: bool = True, seed: int = 0) -> RetrainOutcome:
    ckpt_path = ckpt_path or config.checkpoint_path()
    clean_path = clean_path or config.clean_path()
    batch_n = batch_n or config.retrain_batch_size()
    ssl_cfg = ssl_cfg or SemiSupervisedConfig()
    device = device or device_mod.pick_device()

    loaded = load_checkpoint(ckpt_path, gate=True, device="cpu")
    claimed = store.claim_batch(batch_n)
    if not claimed:
        return RetrainOutcome("skipped", f"fewer than {batch_n} new molecules in the database")
    ids = [m.id for m in claimed]
    auc_before = auc_after = ece_before = ece_after = None
    try:
        manifest = pl.read_manifest(clean_path)
        if manifest["raw_sha256"] != loaded.meta["data_sha256"]:
            raise RealDataGateError("clean data on disk does not match the data the active model was trained on")
        pl.seed_everything(seed)
        batch_size = device_mod.choose_batch_size(32)
        data = pl.load_training_data(clean_path, loaded.meta["split_seed"], split_mode=loaded.meta.get("split_mode", "scaffold"),
                                  feature_version=loaded.meta.get("feature_version", 1))
        scaler = loaded.scaler
        train_loader = pl.make_loader(pl.make_dataset(data, data.train_idx, scaler), batch_size, True, seed)
        val_loader = pl.make_loader(pl.make_dataset(data, data.val_idx, scaler), batch_size, False)
        test_loader = pl.make_loader(pl.make_dataset(data, data.test_idx, scaler), batch_size, False)

        pool = [m.canonical_smiles for m in claimed]
        pool += store.trained_smiles(config.UNLABELED_POOL_CAP - len(pool))
        flat = scaler.transform(np.stack([featurize_flat(s) for s in pool]))
        unl_ds = MolDataset([featurize_graph(s, loaded.meta.get("feature_version", 1)) for s in pool], flat,
                            np.full((len(pool), len(loaded.thresholds)), np.nan, dtype=np.float32))
        unl_loader = pl.make_loader(unl_ds, max(1, batch_size // ssl_cfg.unlabeled_ratio), True, seed)

        val_logits, val_y = predict_logits(loaded.model.to(device), val_loader, device)
        before = pl.evaluate(val_logits, val_y, loaded.temperatures)
        auc_before, ece_before = before["roc_auc_mean"], before["ece_mean"]

        candidate = copy.deepcopy(loaded.model)
        candidate = train_model(candidate, train_loader, val_loader, device, epochs=epochs, lr=lr,
                                patience=patience, augmenter=pl.make_augmenter(augment, seed), monitor="val_auc",
                                weight_decay=float(loaded.meta.get("weight_decay") or 0.0),
                                unlabeled_loader=unl_loader, ssl_cfg=ssl_cfg)
        val_logits, val_y = predict_logits(candidate, val_loader, device)
        temps, thresholds = pl.calibrate(val_logits, val_y)
        after = pl.evaluate(val_logits, val_y, temps)
        auc_after, ece_after = after["roc_auc_mean"], after["ece_mean"]

        if should_promote(auc_before, auc_after, ece_before, ece_after):
            test_logits, test_y = predict_logits(candidate, test_loader, device)
            meta = copy.deepcopy(loaded.meta)
            meta["temperatures"] = [float(x) for x in temps]
            meta["thresholds"] = [float(x) for x in thresholds]
            meta["metrics"] = {"val": after, "test": pl.evaluate(test_logits, test_y, temps),
                               "augment": augment, "retrain": True}
            meta["retrain_count"] = int(meta.get("retrain_count", 0)) + 1
            meta["unlabeled_used"] = len(pool)
            ckpt = make_checkpoint(candidate.cpu(), meta, loaded.ad_bits, loaded.known_smiles)
            save_checkpoint(ckpt_path, ckpt)
            store.set_status(ids, STATUS_TRAINED)
            decision = "promoted"
            msg = f"val ROC-AUC {auc_before:.4f} -> {auc_after:.4f}, ECE {ece_before:.4f} -> {ece_after:.4f}"
        else:
            store.set_status(ids, STATUS_REJECTED)
            decision = "rejected"
            msg = (f"kept the current model: val ROC-AUC {auc_before:.4f} -> {auc_after:.4f}, "
                   f"ECE {ece_before:.4f} -> {ece_after:.4f}")
        store.record_run(n_molecules=len(ids), decision=decision, val_auc_before=auc_before,
                         val_auc_after=auc_after, ece_before=ece_before, ece_after=ece_after)
        return RetrainOutcome(decision, msg, len(ids), auc_before, auc_after, ece_before, ece_after)
    except BaseException:
        store.release(ids)
        try:
            store.record_run(n_molecules=len(ids), decision="aborted", val_auc_before=auc_before,
                             val_auc_after=auc_after, ece_before=ece_before, ece_after=ece_after)
        except Exception:
            pass
        raise
