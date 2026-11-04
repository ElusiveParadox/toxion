import copy

import pytest
import torch

from toxion.inference.registry import RealDataGateError, check_gate, load_checkpoint


def test_checkpoint_without_the_pinned_real_hash_is_refused(tiny_env):
    with pytest.raises(RealDataGateError, match="hash"):
        load_checkpoint(tiny_env["ckpt"])


def test_checkpoint_loads_with_weights_only_when_hash_matches(real_gate):
    loaded = load_checkpoint(real_gate["ckpt"])
    assert loaded.meta["model_type"] == "fusion" and not loaded.meta["synthetic"]
    assert loaded.temperatures.shape == loaded.thresholds.shape == (12,)
    assert loaded.ad_bits.shape[1] == 2048 and len(loaded.known_smiles) == len(real_gate["df"])
    assert not loaded.model.training


def test_gate_rejects_synthetic_small_and_debug_checkpoints(real_gate):
    meta = copy.deepcopy(torch.load(real_gate["ckpt"], weights_only=True)["meta"])
    check_gate(meta)
    for change in ({"synthetic": True}, {"n_molecules": 10}, {"data_sha256": meta["data_sha256"] + ":debug300"}):
        with pytest.raises(RealDataGateError):
            check_gate({**meta, **change})


def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-fusion"):
        load_checkpoint(str(tmp_path / "none.pt"))
