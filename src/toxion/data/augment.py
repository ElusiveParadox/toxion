
from __future__ import annotations

from dataclasses import dataclass

import torch

from toxion.data.featurize import N_FP_BITS
from toxion.data.graph_dataset import GraphBatch


@dataclass
class AugmentConfig:
    enabled: bool = True
    node_mask_p: float = 0.10
    edge_drop_p: float = 0.10
    bit_drop_p: float = 0.10


class Augmenter:
    def __init__(self, cfg: AugmentConfig | None = None, generator: torch.Generator | None = None):
        self.cfg = cfg or AugmentConfig()
        self.generator = generator

    def _rand(self, *shape, device):
        return torch.rand(*shape, generator=self.generator, device="cpu").to(device)

    def __call__(self, batch: GraphBatch) -> GraphBatch:
        c = self.cfg
        if not c.enabled:
            return batch
        node_features = batch.node_features
        edge_index, edge_features = batch.edge_index, batch.edge_features
        flat = batch.flat_features
        dev = node_features.device

        if c.node_mask_p > 0 and node_features.shape[0] > 0:
            keep = (self._rand(node_features.shape[0], device=dev) >= c.node_mask_p).float().unsqueeze(-1)
            node_features = node_features * keep

        n_edges = edge_index.shape[1]
        if c.edge_drop_p > 0 and n_edges >= 2 and n_edges % 2 == 0:
            keep_pair = self._rand(n_edges // 2, device=dev) >= c.edge_drop_p
            keep = keep_pair.repeat_interleave(2)
            edge_index, edge_features = edge_index[:, keep], edge_features[keep]

        if c.bit_drop_p > 0:
            keep = (self._rand(flat.shape[0], N_FP_BITS, device=dev) >= c.bit_drop_p).float()
            flat = flat.clone()
            flat[:, :N_FP_BITS] = flat[:, :N_FP_BITS] * keep

        return batch._replace(node_features=node_features, edge_index=edge_index,
                              edge_features=edge_features, flat_features=flat)
