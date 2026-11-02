import numpy as np
import torch

from toxion.data.augment import AugmentConfig, Augmenter
from toxion.data.featurize import N_FP_BITS, featurize_flat, featurize_graph
from toxion.data.graph_dataset import collate_batch

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1CCN", "CCOC(=O)C=C"] * 4


def _batch():
    return collate_batch([(featurize_graph(s), featurize_flat(s), np.full(12, np.nan, dtype=np.float32)) for s in SMILES])


def _aug(**kw):
    return Augmenter(AugmentConfig(**kw), generator=torch.Generator().manual_seed(0))


def test_disabled_and_zero_probability_are_identity():
    b = _batch()
    assert _aug(enabled=False)(b) is b
    out = _aug(node_mask_p=0, edge_drop_p=0, bit_drop_p=0)(b)
    assert torch.equal(out.node_features, b.node_features) and torch.equal(out.edge_index, b.edge_index)
    assert torch.equal(out.flat_features, b.flat_features)


def test_input_batch_is_not_modified():
    b = _batch()
    before = (b.node_features.clone(), b.edge_index.clone(), b.flat_features.clone())
    _aug(node_mask_p=0.5, edge_drop_p=0.5, bit_drop_p=0.5)(b)
    assert torch.equal(b.node_features, before[0]) and torch.equal(b.edge_index, before[1])
    assert torch.equal(b.flat_features, before[2])


def test_edge_dropout_removes_whole_bonds_symmetrically():
    out = _aug(node_mask_p=0, bit_drop_p=0, edge_drop_p=0.4)(_batch())
    ei = out.edge_index
    assert ei.shape[1] % 2 == 0 and out.edge_features.shape[0] == ei.shape[1]
    pairs = {(int(a), int(b)) for a, b in ei.t()}
    assert all((b, a) in pairs for a, b in pairs)
    assert ei.shape[1] < _batch().edge_index.shape[1]


def test_bit_dropout_only_touches_fingerprint_block():
    b = _batch()
    out = _aug(node_mask_p=0, edge_drop_p=0, bit_drop_p=0.5)(b)
    assert torch.equal(out.flat_features[:, N_FP_BITS:], b.flat_features[:, N_FP_BITS:])
    assert (out.flat_features[:, :N_FP_BITS] == 0).sum() > (b.flat_features[:, :N_FP_BITS] == 0).sum()


def test_node_masking_zeros_whole_rows_and_is_seeded():
    out = _aug(edge_drop_p=0, bit_drop_p=0, node_mask_p=0.5)(_batch())
    zero_rows = (out.node_features.abs().sum(dim=1) == 0).sum().item()
    assert 0 < zero_rows < out.node_features.shape[0]
    again = _aug(edge_drop_p=0, bit_drop_p=0, node_mask_p=0.5)(_batch())
    assert torch.equal(out.node_features, again.node_features)
