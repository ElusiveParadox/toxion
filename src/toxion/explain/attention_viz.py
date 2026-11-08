
from __future__ import annotations

import numpy as np
import torch


def get_atom_attention(encoder, node_features, edge_index, edge_features, batch_index,
                       num_graphs: int = 1) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        _, alpha = encoder(node_features, edge_index, edge_features, batch_index, num_graphs,
                           return_attention=True)
    return alpha.cpu().numpy()


def render_atom_scores(smiles: str, scores: np.ndarray, out_path: str, signed: bool = False,
                       size: tuple[int, int] = (480, 400)) -> None:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable SMILES: {smiles}")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != mol.GetNumAtoms():
        raise ValueError(f"{len(scores)} scores for {mol.GetNumAtoms()} atoms")
    scale = float(np.abs(scores).max()) or 1.0
    colors = {}
    for i, s in enumerate(scores):
        w = min(abs(s) / scale, 1.0)
        if signed and s < 0:
            colors[i] = (1.0 - 0.8 * w, 1.0 - 0.6 * w, 1.0)
        else:
            colors[i] = (1.0, 1.0 - 0.85 * w, 1.0 - 0.85 * w)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                                       highlightAtomColors=colors)
    drawer.FinishDrawing()
    with open(out_path, "wb") as fh:
        fh.write(drawer.GetDrawingText())


def attention_for_smiles(loaded, smiles: str) -> np.ndarray:
    from toxion.inference.probe import _batch_for

    if not hasattr(loaded.model, "encoder"):
        raise ValueError("attention explanations need the MPNN or fusion model (the GIN baseline has no Set2Set readout)")
    b = _batch_for(loaded, smiles, loaded.device)
    return get_atom_attention(loaded.model.encoder, b.node_features, b.edge_index, b.edge_features,
                              b.batch_index, 1)
