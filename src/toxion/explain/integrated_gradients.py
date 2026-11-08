
from __future__ import annotations

import numpy as np
import torch


def atom_importance_ig(model, node_features, edge_index, edge_features, batch_index, num_graphs: int,
                       flat_features, task_index: int, n_steps: int = 50) -> tuple[np.ndarray, float]:
    from captum.attr import IntegratedGradients

    model.eval()

    def forward(nf_steps):
        outs = [model(nf, edge_index, edge_features, batch_index, num_graphs, flat_features)[:, task_index].sum()
                for nf in nf_steps]
        return torch.stack(outs)

    ig = IntegratedGradients(forward)
    x = node_features.unsqueeze(0)
    attr, delta = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=n_steps, return_convergence_delta=True)
    return attr[0].sum(dim=-1).detach().cpu().numpy(), float(delta.abs().max())


def ig_for_smiles(loaded, smiles: str, task: str, n_steps: int = 50) -> tuple[np.ndarray, float]:
    from toxion.inference.probe import _batch_for

    tasks = loaded.meta["tasks"]
    if task not in tasks:
        raise ValueError(f"unknown endpoint {task!r}; choose from {', '.join(tasks)}")
    b = _batch_for(loaded, smiles, loaded.device)
    return atom_importance_ig(loaded.model, b.node_features, b.edge_index, b.edge_features, b.batch_index,
                              1, b.flat_features, tasks.index(task), n_steps)
