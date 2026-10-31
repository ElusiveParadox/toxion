
from __future__ import annotations

import torch
import torch.nn as nn


class GINBaseline(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int = 128, n_layers: int = 3,
                 n_tasks: int = 12, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import GINConv

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, node_features, edge_index, edge_features, batch_index, num_graphs, flat_features=None):
        from torch_geometric.nn import global_mean_pool

        h = self.node_encoder(node_features)
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, edge_index)))
        return self.head(global_mean_pool(h, batch_index, size=num_graphs))
