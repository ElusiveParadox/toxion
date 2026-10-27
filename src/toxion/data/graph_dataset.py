
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from toxion.data.featurize import EDGE_FEATURE_DIM, MolGraph


class GraphBatch(NamedTuple):
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch_index: torch.Tensor
    num_graphs: int
    flat_features: torch.Tensor
    targets: torch.Tensor

    def to(self, device) -> "GraphBatch":
        return GraphBatch(
            self.node_features.to(device), self.edge_index.to(device),
            self.edge_features.to(device), self.batch_index.to(device),
            self.num_graphs, self.flat_features.to(device), self.targets.to(device),
        )

    def model_inputs(self) -> tuple:
        return (self.node_features, self.edge_index, self.edge_features,
                self.batch_index, self.num_graphs, self.flat_features)


class MolDataset(Dataset):

    def __init__(self, graphs: list[MolGraph], flat_features: np.ndarray, labels: np.ndarray):
        assert len(graphs) == len(flat_features) == len(labels)
        self.graphs = graphs
        self.flat_features = flat_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.flat_features[idx], self.labels[idx]


def collate_batch(batch) -> GraphBatch:
    graphs, flats, labels = zip(*batch)
    nodes, edge_parts, edge_feats, batch_index = [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        n = g.node_features.shape[0]
        nodes.append(g.node_features)
        batch_index.append(np.full(n, gi, dtype=np.int64))
        if g.edge_index.shape[1] > 0:
            edge_parts.append(g.edge_index + offset)
            edge_feats.append(g.edge_features)
        offset += n

    if edge_parts:
        edge_index = torch.from_numpy(np.concatenate(edge_parts, axis=1)).long()
        edge_features = torch.from_numpy(np.concatenate(edge_feats, axis=0)).float()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, graphs[0].edge_features.shape[1]), dtype=torch.float32)

    return GraphBatch(
        torch.from_numpy(np.concatenate(nodes, axis=0)).float(),
        edge_index, edge_features,
        torch.from_numpy(np.concatenate(batch_index)).long(),
        len(graphs),
        torch.from_numpy(np.stack(flats)).float(),
        torch.from_numpy(np.stack(labels)).float(),
    )
