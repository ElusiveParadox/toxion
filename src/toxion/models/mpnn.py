
from __future__ import annotations

import torch
import torch.nn as nn


def _segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.zeros(num_segments, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _segment_max(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.full((num_segments,), float("-inf"), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    return out


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    seg_max = _segment_max(scores, index, num_segments)
    exp_scores = (scores - seg_max[index]).exp()
    seg_sum = torch.zeros(num_segments, dtype=scores.dtype, device=scores.device)
    seg_sum.index_add_(0, index, exp_scores)
    return exp_scores / seg_sum[index].clamp(min=1e-12)


class EdgeNetworkMessage(nn.Module):

    def __init__(self, edge_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * hidden_dim),
        )

    def edge_matrices(self, e_vw: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(e_vw).view(e_vw.shape[0], self.hidden_dim, self.hidden_dim)

    def apply_matrices(self, A: torch.Tensor, h_w: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)

    def forward(self, h_w: torch.Tensor, e_vw: torch.Tensor) -> torch.Tensor:
        return self.apply_matrices(self.edge_matrices(e_vw), h_w)


class Set2Set(nn.Module):

    def __init__(self, hidden_dim: int, n_steps: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim)

    def forward(self, node_states, batch_index, num_graphs, return_attention: bool = False):
        dev = node_states.device
        h = torch.zeros(num_graphs, self.hidden_dim, device=dev)
        c = torch.zeros(num_graphs, self.hidden_dim, device=dev)
        q_star = torch.zeros(num_graphs, 2 * self.hidden_dim, device=dev)
        alpha = None
        for _ in range(self.n_steps):
            h, c = self.lstm(q_star, (h, c))
            scores = (h[batch_index] * node_states).sum(dim=-1)
            alpha = _segment_softmax(scores, batch_index, num_graphs)
            r = _segment_sum(node_states * alpha.unsqueeze(-1), batch_index, num_graphs)
            q_star = torch.cat([h, r], dim=-1)
        return (q_star, alpha) if return_attention else q_star


class MPNNEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 128,
                 T: int = 4, set2set_steps: int = 6, node_dropout: float = 0.0):
        super().__init__()
        self.T = T
        self.node_drop = nn.Dropout(node_dropout)
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.message_fn = EdgeNetworkMessage(edge_feat_dim, hidden_dim)
        self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = Set2Set(hidden_dim, n_steps=set2set_steps)
        self.embedding_dim = hidden_dim * 2

    def node_states(self, node_features, edge_index, edge_features):
        h = self.node_encoder(node_features)
        src, dst = edge_index[0], edge_index[1]
        A = self.message_fn.edge_matrices(edge_features)
        for _ in range(self.T):
            messages = self.message_fn.apply_matrices(A, h[src])
            agg = _segment_sum(messages, dst, h.shape[0])
            h = self.node_drop(self.update_fn(agg, h))
        return h

    def forward(self, node_features, edge_index, edge_features, batch_index, num_graphs,
                return_attention: bool = False):
        h = self.node_states(node_features, edge_index, edge_features)
        return self.readout(h, batch_index, num_graphs, return_attention=return_attention)


class MPNNClassifier(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 128,
                 T: int = 4, n_tasks: int = 12, dropout: float = 0.2, node_dropout: float = 0.0):
        super().__init__()
        self.encoder = MPNNEncoder(node_feat_dim, edge_feat_dim, hidden_dim, T, node_dropout=node_dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, node_features, edge_index, edge_features, batch_index, num_graphs,
                flat_features=None):
        emb = self.encoder(node_features, edge_index, edge_features, batch_index, num_graphs)
        return self.head(emb)
