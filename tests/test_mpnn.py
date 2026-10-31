import numpy as np
import torch

from toxion.data.featurize import EDGE_FEATURE_DIM, FLAT_FEATURE_DIM, NODE_FEATURE_DIM, featurize_flat, featurize_graph
from toxion.data.graph_dataset import collate_batch
from toxion.models.fusion import FusionModel
from toxion.models.mpnn import MPNNClassifier, MPNNEncoder, _segment_sum

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C", "CCN(CC)CC", "O=C=O", "ClC(Cl)Cl"]


def _batch(smiles=SMILES):
    rows = [(featurize_graph(s), featurize_flat(s), np.full(12, np.nan, dtype=np.float32)) for s in smiles]
    return collate_batch(rows)


def test_collate_offsets_and_shapes():
    b = _batch()
    n_nodes = b.node_features.shape[0]
    assert b.num_graphs == len(SMILES) and b.batch_index.shape == (n_nodes,)
    assert b.edge_index.max().item() < n_nodes and b.edge_features.shape[0] == b.edge_index.shape[1]
    assert b.flat_features.shape == (len(SMILES), FLAT_FEATURE_DIM)


def test_collate_graph_without_edges_only():
    b = _batch(["C", "O"])
    assert b.edge_index.shape == (2, 0) and b.edge_features.shape == (0, EDGE_FEATURE_DIM)
    model = MPNNClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=8, T=2)
    assert model(*b.model_inputs()).shape == (2, 12)


def test_gradients_reach_edge_network_and_gru():
    b = _batch()
    model = MPNNClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=16, T=3)
    model(*b.model_inputs()).sum().backward()
    assert model.encoder.message_fn.edge_mlp[0].weight.grad.abs().sum() > 0
    assert model.encoder.update_fn.weight_ih.grad.abs().sum() > 0


def test_set2set_attention_sums_to_one_per_graph():
    b = _batch()
    enc = MPNNEncoder(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=16, T=2)
    emb, alpha = enc(b.node_features, b.edge_index, b.edge_features, b.batch_index, b.num_graphs, return_attention=True)
    assert emb.shape == (len(SMILES), 32)
    sums = _segment_sum(alpha.unsqueeze(-1), b.batch_index, b.num_graphs).squeeze(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_edge_matrices_computed_once_match_per_round_reference():
    torch.manual_seed(0)
    b = _batch()
    enc = MPNNEncoder(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=16, T=3)

    def reference(enc, batch):
        h = enc.node_encoder(batch.node_features)
        src, dst = batch.edge_index
        for _ in range(enc.T):
            msg = enc.message_fn(h[src], batch.edge_features)
            h = enc.update_fn(_segment_sum(msg, dst, h.shape[0]), h)
        return enc.readout(h, batch.batch_index, batch.num_graphs)

    out_new = enc(b.node_features, b.edge_index, b.edge_features, b.batch_index, b.num_graphs)
    out_ref = reference(enc, b)
    assert torch.allclose(out_new, out_ref, atol=1e-6)
    g_new = torch.autograd.grad(out_new.sum(), enc.message_fn.edge_mlp[0].weight, retain_graph=True)[0]
    g_ref = torch.autograd.grad(out_ref.sum(), enc.message_fn.edge_mlp[0].weight)[0]
    assert torch.allclose(g_new, g_ref, atol=1e-5)


def test_fusion_forward_backward_and_permutation_invariance():
    b = _batch()
    model = FusionModel(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, FLAT_FEATURE_DIM, hidden_dim=16, T=2).eval()
    logits = model(*b.model_inputs())
    assert logits.shape == (len(SMILES), 12)
    logits.sum().backward()
    perm = [3, 0, 6, 1, 5, 2, 4]
    logits_perm = model(*_batch([SMILES[i] for i in perm]).model_inputs())
    assert torch.allclose(logits[perm], logits_perm, atol=1e-5)
