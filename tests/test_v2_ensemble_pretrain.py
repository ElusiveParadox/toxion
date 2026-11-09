import numpy as np
import pytest
import torch

from toxion.data.featurize import (EDGE_FEATURE_DIM_V2, NODE_FEATURE_DIM_V2, PRETRAIN_ELEMENTS, atom_element_classes,
                                   edge_dim, featurize_flat, featurize_graph, node_dim)
from toxion.data.graph_dataset import collate_batch
from toxion.inference.registry import build_model, load_checkpoint

MOLS = ["C", "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", "C/C=C/C", "[Cu+2].[O-]S(=O)(=O)[O-]", "C[Hg]Cl", "O=[N+]([O-])c1ccccc1", "[U]"]


@pytest.mark.parametrize("smiles", MOLS)
def test_v2_extends_v1_exactly_and_is_finite(smiles):
    g1, g2 = featurize_graph(smiles, 1), featurize_graph(smiles, 2)
    assert g2.node_features.shape[1] == NODE_FEATURE_DIM_V2 == node_dim(2) == 50
    assert g2.edge_features.shape[1] == EDGE_FEATURE_DIM_V2 == edge_dim(2) == 10
    assert np.array_equal(g1.node_features, g2.node_features[:, :21]) and np.array_equal(g1.edge_features, g2.edge_features[:, :5])
    assert np.isfinite(g2.node_features).all() and np.abs(g2.node_features).max() < 20


def test_v2_encodes_charge_ring_chirality_and_element_properties():
    nitro = featurize_graph("O=[N+]([O-])c1ccccc1", 2).node_features
    assert (nitro[:, 21 + 7 + 1 + 0] == 1).sum() == 1 and nitro[:, 21 + 7 + 3].sum() == 1
    ring_flag = 21 + 7 + 6
    assert nitro[:, ring_flag].sum() == 6
    ala = featurize_graph("C[C@H](N)C(=O)O", 2).node_features
    chir = 21 + 7 + 6 + 6
    assert ala[:, chir + 1:chir + 3].sum() == 1
    props = lambda smi: featurize_graph(smi, 2).node_features[0, -5:]
    assert props("[Hg](C)C")[0] > props("C")[0]
    assert not np.array_equal(props("[Zn](C)C"), props("[Cd](C)C"))


def test_v2_bond_features_and_edgeless_collate():
    assert featurize_graph("C/C=C/C", 2).edge_features[:, 8].max() == 1
    assert featurize_graph("CC=CC", 2).edge_features[:, 8].max() == 0 and featurize_graph("CC=CC", 2).edge_features[:, 7].min() == 1
    assert featurize_graph("C=CC=C", 2).edge_features[:, 5].max() == 1
    assert featurize_graph("CC=CC", 2).edge_features[:, 5].max() == 0
    assert featurize_graph("C1CC1", 2).edge_features[:, 6].min() == 1
    b = collate_batch([(featurize_graph(s, 2), np.zeros(3, np.float32), np.zeros(2, np.float32)) for s in ("C", "O")])
    assert b.edge_features.shape == (0, 10)


def test_atom_element_classes():
    c = atom_element_classes("C[Hg]Cl[U]" if False else "C[Hg]Cl")
    assert list(c) == [PRETRAIN_ELEMENTS.index("C"), PRETRAIN_ELEMENTS.index("Hg"), PRETRAIN_ELEMENTS.index("Cl")]
    assert atom_element_classes("[U]")[0] == len(PRETRAIN_ELEMENTS)


def test_unknown_feature_version_rejected():
    with pytest.raises(ValueError):
        featurize_graph("CCO", 3)


def test_ensemble_forward_is_the_mean_and_backprops():
    from toxion.models.ensemble import EnsembleModel

    rows = [(featurize_graph(s, 2), featurize_flat(s), np.full(12, np.nan, np.float32)) for s in MOLS[:4]]
    b = collate_batch(rows)
    ens = build_model("fusion", 8, 2, node_dim=50, edge_dim=10, ensemble_size=3)
    assert isinstance(ens, EnsembleModel) and len(ens.members) == 3
    ens.eval()
    stacked = ens.member_logits(*b.model_inputs())
    assert torch.allclose(ens(*b.model_inputs()), stacked.mean(0), atol=1e-6) and stacked.std(0).sum() > 0
    ens(*b.model_inputs()).sum().backward()
    assert all(p.grad is not None for p in ens.members[2].parameters() if p.requires_grad)
    assert hasattr(ens, "encoder")


def test_trained_v2_ensemble_checkpoint_roundtrip_and_probe(real_gate_v2):
    from toxion.inference.probe import predict_smiles

    m = real_gate_v2["metrics"]
    assert m["ensemble_size"] == 2 and len(m["members"]["test_auc"]) == 2
    loaded = load_checkpoint(real_gate_v2["ckpt"])
    assert loaded.meta["feature_version"] == 2 and loaded.meta["node_dim"] == 50 and loaded.meta["ensemble_size"] == 2
    pred = predict_smiles(loaded, "CC(=O)Oc1ccccc1C(=O)O", mc_samples=3)
    assert np.isfinite(pred.probs).all() and pred.probs.shape == (12,)
    assert not any(f.endswith(".pt") and ".member" in f for f in map(str, real_gate_v2["root"].iterdir()))


def test_v1_checkpoints_still_load_and_predict(real_gate):
    loaded = load_checkpoint(real_gate["ckpt"])
    assert loaded.meta.get("feature_version", 1) == 1 and loaded.meta.get("ensemble_size", 1) == 1


def test_explain_and_retrain_work_on_an_ensemble_with_v2_features(real_gate_v2, tmp_path):
    import shutil

    from toxion.db.store import MoleculeStore
    from toxion.explain.attention_viz import attention_for_smiles
    from toxion.explain.integrated_gradients import ig_for_smiles
    from toxion.training import retrain as rt

    loaded = load_checkpoint(real_gate_v2["ckpt"])
    assert abs(attention_for_smiles(loaded, "CCO").sum() - 1) < 1e-5
    scores, delta = ig_for_smiles(loaded, "CCO", "NR-AhR", 20)
    assert scores.shape == (3,) and np.isfinite(scores).all()

    ckpt = str(tmp_path / "a.pt")
    shutil.copy(real_gate_v2["ckpt"], ckpt)
    store = MoleculeStore(f"sqlite:///{tmp_path / 'r.sqlite'}")
    for i, smi in enumerate(["CCOC(C)=O", "CCNCC", "CCCC=O", "CC(C)=O"]):
        store.add_molecule(smi, f"n{i}")
    orig = rt.should_promote
    rt.should_promote = lambda *a, **k: True
    try:
        out = rt.retrain(store, ckpt_path=ckpt, clean_path=real_gate_v2["clean"], device="cpu", batch_n=4, epochs=1, patience=1)
    finally:
        rt.should_promote = orig
    assert out.decision == "promoted"
    after = load_checkpoint(ckpt)
    assert after.meta["ensemble_size"] == 2 and after.meta["feature_version"] == 2 and after.meta["retrain_count"] == 1



def _write_smiles(path, smiles, col="SMILES", quote_newline=False):
    import pandas as pd

    pd.DataFrame({col: [s + ("\n" if quote_newline else "") for s in smiles], "other": range(len(smiles))}).to_csv(path, index=False)


def test_read_smiles_detects_column_dedupes_and_skips_bad(tmp_path):
    from toxion.training.pretrain import read_smiles

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_smiles(a, ["OCC", "CCO", "not smiles", "c1ccccc1"], quote_newline=True)
    _write_smiles(b, ["c1ccccc1", "CC(=O)O"], col="smiles")
    assert sorted(read_smiles([str(a), str(b)])) == sorted(["CCO", "c1ccccc1", "CC(=O)O"])
    bad = tmp_path / "c.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="no SMILES column"):
        read_smiles([str(bad)])


def test_pretraining_learns_masked_atoms_and_transfers(tmp_path, synthetic_df):
    from toxion.training.pretrain import load_pretrained_encoder, run_pretraining

    f = tmp_path / "mols.csv"
    smiles = synthetic_df.smiles_canonical.tolist()
    _write_smiles(f, smiles)
    out = str(tmp_path / "enc.pt")
    meta = run_pretraining([str(f)], out, hidden_dim=16, T=2, feature_version=2, epochs=3, batch_size=16,
                           max_molecules=300, exclude=set(smiles[:5]), device="cpu")
    assert meta["n_molecules"] >= 100 and 0 <= meta["val_masked_atom_acc"] <= 1
    assert meta["val_masked_atom_acc"] >= meta["majority_class_acc"] - 0.05

    model = build_model("fusion", 16, 2, node_dim=50, edge_dim=10)
    before = model.encoder.node_encoder.weight.clone()
    readout_before = model.encoder.readout.lstm.weight_ih.clone()
    load_pretrained_encoder(model, out, 16, 2, 2)
    assert not torch.equal(model.encoder.node_encoder.weight, before)
    assert torch.equal(model.encoder.readout.lstm.weight_ih, readout_before)
    ens = build_model("fusion", 16, 2, node_dim=50, edge_dim=10, ensemble_size=2)
    load_pretrained_encoder(ens, out, 16, 2, 2)
    assert torch.equal(ens.members[0].encoder.node_encoder.weight, ens.members[1].encoder.node_encoder.weight)


def test_pretrained_encoder_mismatch_is_refused(tmp_path, synthetic_df):
    from toxion.training.pretrain import load_pretrained_encoder, run_pretraining

    f = tmp_path / "m.csv"
    _write_smiles(f, synthetic_df.smiles_canonical.tolist())
    out = str(tmp_path / "enc.pt")
    run_pretraining([str(f)], out, hidden_dim=16, T=2, feature_version=2, epochs=1, max_molecules=200, device="cpu")
    with pytest.raises(ValueError, match="hidden_dim: 16 vs 32"):
        load_pretrained_encoder(build_model("fusion", 32, 2, node_dim=50, edge_dim=10), out, 32, 2, 2)
    with pytest.raises(ValueError, match="feature_version"):
        load_pretrained_encoder(build_model("fusion", 16, 2), out, 16, 2, 1)
    with pytest.raises(ValueError, match="at least 100"):
        run_pretraining([str(f)], out, hidden_dim=8, T=1, feature_version=2, epochs=1, max_molecules=50, device="cpu")


def test_training_can_start_from_a_pretrained_encoder(tmp_path, synthetic_df):
    from toxion.data.loader import save_clean
    from toxion.training import pipeline as pl
    from toxion.training.pretrain import run_pretraining
    from toxion.training.runner import run_training

    f = tmp_path / "m.csv"
    _write_smiles(f, synthetic_df.smiles_canonical.tolist())
    enc = str(tmp_path / "enc.pt")
    run_pretraining([str(f)], enc, hidden_dim=16, T=2, feature_version=2, epochs=1, max_molecules=250, device="cpu")
    clean = str(tmp_path / "c.parquet")
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"x")
    save_clean(synthetic_df, clean)
    pl.write_manifest(clean, str(raw), len(synthetic_df))
    cfg = pl.TrainConfig(hidden_dim=16, T=2, batch_size=16, epochs=1, patience=1, split_mode="random",
                         feature_version=2, pretrained=enc)
    run_training(clean, cfg, str(tmp_path / "m.pt"), device="cpu")
    assert torch.load(str(tmp_path / "m.pt"), weights_only=True)["meta"]["pretrained"] == "enc.pt"
