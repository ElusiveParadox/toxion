
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from toxion import config
from toxion.analysis.chemspace import morgan_bits, tanimoto_matrix
from toxion.chem.resolver import ResolveError
from toxion.data.featurize import N_FP_BITS, featurize_flat, featurize_graph, morgan_on_bits
from toxion.data.graph_dataset import collate_batch
from toxion.eval.calibration import apply_temperature, sigmoid
from toxion.eval.uncertainty import mc_dropout_predict
from toxion.inference.registry import LoadedModel

logger = logging.getLogger(__name__)

DISCLAIMER = ("Screening output from a model trained on 12 Tox21 in-vitro assays. "
              "It is not a safety determination.")


@dataclass
class Prediction:
    canonical_smiles: str
    probs: np.ndarray
    stds: np.ndarray
    flagged: list[tuple[str, float, float, float]]
    toxic: bool
    nn_similarity: float
    in_domain: bool


@dataclass
class ProbeResult:
    name: str
    cid: int | None
    iupac_name: str | None
    from_cache: bool
    prediction: Prediction
    status: str
    fingerprint: dict = field(default_factory=dict)
    new_count: int | None = None
    batch_size: int | None = None
    notes: list[str] = field(default_factory=list)


def _batch_for(loaded: LoadedModel, smiles: str, device: str):
    graph = featurize_graph(smiles, loaded.meta.get("feature_version", 1))
    flat = loaded.scaler.transform(featurize_flat(smiles)[None, :])[0]
    batch = collate_batch([(graph, flat, np.full(len(loaded.thresholds), np.nan, dtype=np.float32))])
    return batch.to(device)


def predict_smiles(loaded: LoadedModel, smiles: str, mc_samples: int | None = None) -> Prediction:
    mc_samples = config.MC_DROPOUT_SAMPLES if mc_samples is None else mc_samples
    device = loaded.device
    batch = _batch_for(loaded, smiles, device)
    model = loaded.model
    model.eval()
    with torch.no_grad():
        logits = model(*batch.model_inputs()).cpu().numpy()[0]
    probs = sigmoid(apply_temperature(logits, loaded.temperatures))

    if mc_samples > 1:
        temps = torch.tensor(loaded.temperatures, dtype=torch.float32, device=device)
        mean_p, std_p = mc_dropout_predict(
            model, lambda: model(*batch.model_inputs()) / temps, n_samples=mc_samples)
        stds = std_p.cpu().numpy()[0]
    else:
        stds = np.zeros_like(probs)

    flagged = [(loaded.meta["tasks"][t], float(probs[t]), float(stds[t]), float(loaded.thresholds[t]))
               for t in range(len(probs)) if probs[t] >= loaded.thresholds[t]]
    flagged.sort(key=lambda x: x[1] / max(x[3], 1e-6), reverse=True)

    sim = float(tanimoto_matrix(morgan_bits([smiles]), loaded.ad_bits).max()) if len(loaded.ad_bits) else 0.0
    return Prediction(smiles, probs, stds, flagged, bool(flagged), sim, sim >= config.AD_SIMILARITY_CUTOFF)


class ProbeSession:

    def __init__(self, loaded: LoadedModel, resolver, store=None,
                 retrain_fn: Callable | None = None, batch_size: int | None = None,
                 auto_retrain: bool | None = None, mc_samples: int | None = None):
        self.loaded, self.resolver, self.store = loaded, resolver, store
        self.retrain_fn = retrain_fn
        self.batch_size = batch_size or config.retrain_batch_size()
        self.auto_retrain = config.auto_retrain_enabled() if auto_retrain is None else auto_retrain
        self.mc_samples = mc_samples

    def handle(self, name: str, out: Callable[[str], None] = print) -> ProbeResult:
        notes: list[str] = []
        cid = iupac = None
        cached = None
        if self.store is not None:
            try:
                cached = self.store.lookup_alias(name)
            except Exception as exc:
                notes.append(f"database unavailable ({type(exc).__name__}); not storing")
                self.store = None
        if cached is not None:
            smiles, cid, from_cache = cached.canonical_smiles, cached.pubchem_cid, True
        else:
            res = self.resolver.resolve(name)
            smiles, cid, iupac, from_cache = res.canonical_smiles, res.cid, res.iupac_name, False

        pred = predict_smiles(self.loaded, smiles, self.mc_samples)
        result = ProbeResult(name, cid, iupac, from_cache, pred, "not_stored", fingerprint=morgan_on_bits(smiles),
                             notes=notes)

        if smiles in self.loaded.known_smiles:
            result.status = "known"
            return result
        if self.store is None:
            return result
        try:
            if not from_cache:
                self.store.add_alias(name, smiles, cid)
            mol, created = self.store.add_molecule(
                smiles, name, cid, pred.nn_similarity,
                json.dumps({t: round(float(p), 4) for t, p in zip(self.loaded.meta["tasks"], pred.probs)}))
            result.status = "stored" if created else "already_stored"
            result.new_count, result.batch_size = self.store.count_new(), self.batch_size
        except Exception as exc:
            result.notes.append(f"database error ({type(exc).__name__}); not stored")
            return result

        if (created and self.auto_retrain and self.retrain_fn is not None
                and result.new_count >= self.batch_size):
            out(f"{result.new_count} new molecules collected: retraining (Ctrl-C aborts, the current model stays active)...")
            try:
                outcome = self.retrain_fn(self.store)
                result.notes.append(f"retrain: {outcome.decision}. {outcome.message}")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                result.notes.append(f"retrain failed: {exc}")
        return result


def format_result(r: ProbeResult) -> str:
    p = r.prediction
    lines = []
    src = f"PubChem CID {r.cid}" if r.cid and not r.from_cache else ("cached lookup" if r.from_cache else "")
    lines.append(f"SMILES : {p.canonical_smiles}" + (f"   ({src})" if src else ""))
    if r.iupac_name:
        lines.append(f"Matched: {r.iupac_name}")
    if r.fingerprint:
        head = f"Morgan FP: radius 2, {N_FP_BITS} bits, {len(r.fingerprint)} set (bit:count)"
        lines.append(head)
        row = "  "
        for bit, count in sorted(r.fingerprint.items()):
            item = f"{bit}:{count} "
            if len(row) + len(item) > 100:
                lines.append(row.rstrip())
                row = "  "
            row += item
        lines.append(row.rstrip())
    n_tasks = len(p.probs)
    if p.toxic:
        lines.append(f"Verdict: TOXIC on {len(p.flagged)} of {n_tasks} endpoints")
        for task, prob, std, thr in p.flagged:
            lines.append(f"  {task:<14} p={prob:.2f} (+/- {std:.2f})  threshold {thr:.2f}")
    else:
        lines.append(f"Verdict: NOT TOXIC on any of the {n_tasks} endpoints")
    conf = "" if p.in_domain else "   [outside training chemistry, low confidence]"
    lines.append(f"Nearest training molecule similarity: {p.nn_similarity:.2f}{conf}")
    if r.status == "known":
        lines.append("Note: this molecule is in the Tox21 training data.")
    elif r.status == "stored":
        lines.append(f"New molecule stored ({r.new_count}/{r.batch_size} until retrain).")
    elif r.status == "already_stored":
        lines.append("Already stored in the database from an earlier lookup.")
    lines.extend(r.notes)
    return "\n".join(lines)


def interactive_loop(session: ProbeSession, input_fn: Callable[[str], str] | None = None,
                     out: Callable[[str], None] = print) -> int:
    input_fn = input_fn or input
    out(DISCLAIMER)
    out("Type an IUPAC name (or 'exit' to quit).")
    while True:
        try:
            name = input_fn("IUPAC name> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return 0
        if not name:
            continue
        if name.lower() in {"exit", "quit"}:
            return 0
        try:
            result = session.handle(name, out)
        except ResolveError as exc:
            out(f"error: {exc}")
            continue
        except KeyboardInterrupt:
            out("")
            return 0
        except Exception as exc:
            logger.exception("probe failed for %r", name)
            out(f"error: {type(exc).__name__}: {exc}")
            continue
        out(format_result(result))
