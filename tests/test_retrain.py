import shutil

import pytest
import torch

from toxion.db.store import MoleculeStore
from toxion.inference.registry import RealDataGateError, load_checkpoint
from toxion.training import retrain as rt

POOL = ["CCOC(C)=O", "CCNCC", "CCCC=O", "CC(C)=O"]


@pytest.fixture
def setup(real_gate, tmp_path):
    ckpt = str(tmp_path / "active.pt")
    shutil.copy(real_gate["ckpt"], ckpt)
    store = MoleculeStore(f"sqlite:///{tmp_path / 'r.sqlite'}")
    for i, smi in enumerate(POOL):
        store.add_molecule(smi, f"name{i}")
    return ckpt, store, real_gate["clean"]


def run(setup, **kw):
    ckpt, store, clean = setup
    return rt.retrain(store, ckpt_path=ckpt, clean_path=clean, device="cpu", batch_n=4, epochs=1, patience=1, **kw)


def test_should_promote_rules():
    assert rt.should_promote(0.70, 0.70, 0.05, 0.05)
    assert rt.should_promote(0.70, 0.696, 0.05, 0.059)
    assert not rt.should_promote(0.70, 0.69, 0.05, 0.05)
    assert not rt.should_promote(0.70, 0.71, 0.05, 0.07)
    assert not rt.should_promote(0.70, float("nan"), 0.05, 0.05)


def test_not_enough_molecules_changes_nothing(setup):
    ckpt, store, clean = setup
    out = rt.retrain(store, ckpt_path=ckpt, clean_path=clean, device="cpu", batch_n=5, epochs=1)
    assert out.decision == "skipped" and store.status_counts() == {"new": 4}


def test_promotion_replaces_checkpoint_and_marks_trained(setup, monkeypatch):
    ckpt, store, _ = setup
    monkeypatch.setattr(rt, "should_promote", lambda *a, **k: True)
    out = run(setup)
    assert out.decision == "promoted" and store.status_counts() == {"trained": 4}
    meta = load_checkpoint(ckpt).meta
    assert meta["retrain_count"] == 1 and meta["unlabeled_used"] == 4 and meta["metrics"]["retrain"]


def test_rejection_keeps_the_old_checkpoint_byte_for_byte(setup, monkeypatch):
    ckpt, store, _ = setup
    before = open(ckpt, "rb").read()
    monkeypatch.setattr(rt, "should_promote", lambda *a, **k: False)
    out = run(setup)
    assert out.decision == "rejected" and store.status_counts() == {"rejected": 4}
    assert open(ckpt, "rb").read() == before


def test_interrupt_releases_the_batch_and_keeps_the_model(setup, monkeypatch):
    ckpt, store, _ = setup
    before = open(ckpt, "rb").read()

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(rt, "train_model", boom)
    with pytest.raises(KeyboardInterrupt):
        run(setup)
    assert store.status_counts() == {"new": 4} and open(ckpt, "rb").read() == before


def test_data_hash_mismatch_is_refused_and_batch_released(setup, tmp_path):
    ckpt, store, clean = setup
    import json

    p = clean + ".manifest.json"
    original = open(p).read()
    try:
        m = json.loads(original)
        m["raw_sha256"] = "0" * 64
        open(p, "w").write(json.dumps(m))
        with pytest.raises(RealDataGateError):
            run(setup)
    finally:
        open(p, "w").write(original)
    assert store.status_counts() == {"new": 4}


def test_retrained_model_still_produces_finite_predictions(setup, monkeypatch):
    ckpt, _, _ = setup
    monkeypatch.setattr(rt, "should_promote", lambda *a, **k: True)
    run(setup)
    from toxion.inference.probe import predict_smiles

    pred = predict_smiles(load_checkpoint(ckpt), "CCO", mc_samples=2)
    assert torch.isfinite(torch.tensor(pred.probs)).all()
