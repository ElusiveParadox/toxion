import numpy as np
import pytest
import torch

from toxion.data.augment import AugmentConfig, Augmenter
from toxion.data.loader import TOX21_TASKS
from toxion.training import pipeline as pl
from toxion.training.trainer import predict_logits, train_model
from toxion.inference.registry import build_model


def _loaders(df, batch=16):
    graphs, flat = pl.featurize_dataframe(df)
    labels = df[TOX21_TASKS].to_numpy(dtype=np.float32)
    from toxion.data.featurize import FlatScaler
    from toxion.data.graph_dataset import MolDataset

    sc = FlatScaler.fit(flat)
    ds = MolDataset(graphs, sc.transform(flat), labels)
    return pl.make_loader(ds, batch, True, 0), pl.make_loader(ds, batch, False)


def test_training_reduces_loss_and_returns_best_weights(synthetic_df):
    torch.manual_seed(0)
    train, val = _loaders(synthetic_df)
    model = build_model("fusion", 16, 2)
    from toxion.training.trainer import eval_loss

    before = eval_loss(model, val, "cpu")
    model = train_model(model, train, val, "cpu", epochs=6, lr=3e-3, patience=6)
    assert eval_loss(model, val, "cpu") < before


def test_training_with_augmentation_runs(synthetic_df):
    train, val = _loaders(synthetic_df)
    aug = Augmenter(AugmentConfig(), generator=torch.Generator().manual_seed(0))
    model = train_model(build_model("mpnn", 16, 2), train, val, "cpu", epochs=2, augmenter=aug)
    logits, y = predict_logits(model, val, "cpu")
    assert logits.shape == y.shape == (len(synthetic_df), 12) and np.isfinite(logits).all()


def test_checkpoint_resume_continues_instead_of_restarting(synthetic_df, tmp_path):
    train, val = _loaders(synthetic_df)
    state = str(tmp_path / "state.pt")
    torch.manual_seed(0)
    train_model(build_model("fusion", 16, 2), train, val, "cpu", epochs=2, state_path=state)
    st = torch.load(state, map_location="cpu")
    assert st["epoch"] == 1
    model = build_model("fusion", 16, 2)
    train_model(model, train, val, "cpu", epochs=4, state_path=state, resume=True)
    assert torch.load(state, map_location="cpu")["epoch"] == 3


def _run_scripted(monkeypatch, synthetic_df, scores, **kw):
    import toxion.training.trainer as tr

    it = iter(scores)
    snapshots = []
    model = build_model("mpnn", 8, 1)

    def fake_eval(m, loader, device):
        snapshots.append({k: v.clone() for k, v in m.state_dict().items()})
        return next(it)

    monkeypatch.setattr(tr, "eval_split", fake_eval)
    train, val = _loaders(synthetic_df)
    out = tr.train_model(model, train, val, "cpu", **kw)
    return out, snapshots


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("monitor,best_epoch", [("val_loss", 1), ("val_auc", 3)])
def test_best_epoch_is_selected_by_the_monitored_metric(monkeypatch, synthetic_df, monitor, best_epoch):
    scores = [(0.50, 0.60), (0.30, 0.65), (0.40, 0.70), (0.45, 0.80), (0.60, 0.75)]
    model, snaps = _run_scripted(monkeypatch, synthetic_df, scores, epochs=5, patience=10, monitor=monitor)
    assert len(snaps) == 5 and _same(model.state_dict(), snaps[best_epoch])


def test_early_stopping_stops_after_patience_without_improvement(monkeypatch, synthetic_df):
    scores = [(0.5, 0.6), (0.4, 0.6), (0.45, 0.6), (0.46, 0.6), (0.47, 0.6), (0.48, 0.6), (0.1, 0.9)]
    model, snaps = _run_scripted(monkeypatch, synthetic_df, scores, epochs=7, patience=3)
    assert len(snaps) == 5
    assert _same(model.state_dict(), snaps[1])


def test_stopper_semantics_higher_is_better():
    from toxion.training.trainer import EarlyStopper

    s = EarlyStopper(patience=2, min_delta=0.0)
    assert [s.step(x) for x in (0.1, 0.2, 0.15, 0.15)] == [False, False, False, True]


def test_weight_decay_switches_to_adamw_and_cosine_decays_lr(monkeypatch, synthetic_df, caplog):
    import logging

    caplog.set_level(logging.INFO)
    _run_scripted(monkeypatch, synthetic_df, [(0.5, 0.6)] * 3, epochs=3, patience=9, lr=1e-2, weight_decay=1e-2, schedule="cosine")
    lrs = [float(line.split("lr=")[1].split()[0]) for line in caplog.messages if "lr=" in line]
    assert lrs[0] == pytest.approx(1e-2) and lrs[0] > lrs[1] > lrs[2] >= 0


def test_invalid_monitor_is_rejected(synthetic_df):
    train, val = _loaders(synthetic_df)
    with pytest.raises(ValueError, match="monitor"):
        train_model(build_model("mpnn", 8, 1), train, val, "cpu", epochs=1, monitor="accuracy")
