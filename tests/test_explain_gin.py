import numpy as np
import pytest
import torch

from toxion.data.loader import TOX21_TASKS
from toxion.explain.attention_viz import attention_for_smiles, render_atom_scores
from toxion.explain.integrated_gradients import ig_for_smiles
from toxion.inference.registry import build_model, load_checkpoint

SMILES = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def loaded(real_gate):
    return load_checkpoint(real_gate["ckpt"])


def test_attention_is_a_distribution_over_atoms(loaded):
    a = attention_for_smiles(loaded, SMILES)
    assert a.shape == (13,) and abs(a.sum() - 1.0) < 1e-5 and (a >= 0).all()


def test_integrated_gradients_shape_and_completeness(loaded):
    scores, delta = ig_for_smiles(loaded, SMILES, TOX21_TASKS[0], n_steps=100)
    assert scores.shape == (13,) and np.isfinite(scores).all()
    assert delta < 0.05 * max(1.0, np.abs(scores).sum())


def test_ig_is_deterministic_and_rejects_unknown_endpoint(loaded):
    a, _ = ig_for_smiles(loaded, "CCO", TOX21_TASKS[1])
    b, _ = ig_for_smiles(loaded, "CCO", TOX21_TASKS[1])
    assert np.allclose(a, b)
    with pytest.raises(ValueError, match="unknown endpoint"):
        ig_for_smiles(loaded, "CCO", "NOT-A-TASK")


def test_render_writes_png_and_validates_input(tmp_path):
    out = tmp_path / "x.png"
    render_atom_scores("CCO", np.array([0.1, -0.5, 0.9]), str(out), signed=True)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="scores for"):
        render_atom_scores("CCO", np.array([0.1]), str(out))
    with pytest.raises(ValueError, match="Unparseable"):
        render_atom_scores("not smiles", np.array([0.1]), str(out))


def test_gin_baseline_trains_through_the_shared_trainer(synthetic_df):
    pytest.importorskip("torch_geometric")
    from test_trainer import _loaders
    from toxion.training.trainer import eval_loss, train_model

    train, val = _loaders(synthetic_df)
    torch.manual_seed(0)
    model = build_model("gin", 16, 3)
    before = eval_loss(model, val, "cpu")
    model = train_model(model, train, val, "cpu", epochs=5, lr=3e-3, patience=5)
    assert eval_loss(model, val, "cpu") < before


def test_gin_handles_graphs_without_edges():
    pytest.importorskip("torch_geometric")
    from test_mpnn import _batch

    b = _batch(["C", "O"])
    assert build_model("gin", 8, 2)(*b.model_inputs()).shape == (2, 12)


def test_attention_refuses_gin():
    pytest.importorskip("torch_geometric")
    from toxion.inference.registry import LoadedModel
    from toxion.data.featurize import FlatScaler

    fake = LoadedModel(build_model("gin", 8, 2), {"tasks": TOX21_TASKS}, FlatScaler(np.zeros(10), np.ones(10)),
                       np.ones(12), np.full(12, .5), np.zeros((1, 2048)), frozenset())
    with pytest.raises(ValueError, match="MPNN or fusion"):
        attention_for_smiles(fake, "CCO")
