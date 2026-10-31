
from __future__ import annotations

import torch
import torch.nn as nn

from toxion.models.mpnn import MPNNEncoder


class FusionHead(nn.Module):
    def __init__(self, mpnn_embed_dim: int, flat_feat_dim: int, hidden_dim: int = 256,
                 n_tasks: int = 12, dropout: float = 0.3):
        super().__init__()
        self.flat_encoder = nn.Sequential(nn.Linear(flat_feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(mpnn_embed_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, mpnn_embedding: torch.Tensor, flat_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([mpnn_embedding, self.flat_encoder(flat_features)], dim=-1)
        return self.classifier(combined)


class FusionModel(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, flat_feat_dim: int,
                 hidden_dim: int = 128, T: int = 4, n_tasks: int = 12, dropout: float = 0.3,
                 node_dropout: float = 0.0):
        super().__init__()
        self.encoder = MPNNEncoder(node_feat_dim, edge_feat_dim, hidden_dim, T, node_dropout=node_dropout)
        self.head = FusionHead(self.encoder.embedding_dim, flat_feat_dim,
                               hidden_dim=hidden_dim * 2, n_tasks=n_tasks, dropout=dropout)

    def forward(self, node_features, edge_index, edge_features, batch_index, num_graphs,
                flat_features):
        emb = self.encoder(node_features, edge_index, edge_features, batch_index, num_graphs)
        return self.head(emb, flat_features)
