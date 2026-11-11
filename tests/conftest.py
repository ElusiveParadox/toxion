import pytest

from toxion.data.synthetic import make_synthetic_clean


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_clean(n=150, seed=0)


import json
import os

from toxion import config


@pytest.fixture(scope="session")
def tiny_env(tmp_path_factory):
    from toxion.data.loader import save_clean
    from toxion.training import pipeline as pl
    from toxion.training.runner import run_training

    root = tmp_path_factory.mktemp("tiny")
    with pytest.MonkeyPatch.context() as mp:
        for var in [v for v in os.environ if v.startswith("TOXION_")]:
            mp.delenv(var)
        return _build_tiny_env(root)


def _build_tiny_env(root):
    from toxion.data.loader import save_clean
    from toxion.training import pipeline as pl
    from toxion.training.runner import run_training

    clean = str(root / "clean.parquet")
    raw = root / "raw.bin"
    raw.write_bytes(b"pretend this is the real tox21 file")
    df = make_synthetic_clean(n=300, seed=1)
    save_clean(df, clean)
    manifest = pl.write_manifest(clean, str(raw), len(df))
    cfg = pl.TrainConfig(hidden_dim=16, T=2, batch_size=16, epochs=2, patience=2, seed=0, split_mode="random")
    ckpt = str(root / "active.pt")
    run_training(clean, cfg, ckpt, device="cpu")
    return {"clean": clean, "ckpt": ckpt, "sha": manifest["raw_sha256"], "root": root, "df": df}


@pytest.fixture
def real_gate(tiny_env, monkeypatch):
    monkeypatch.setenv("TOXION_REAL_SHA256", tiny_env["sha"])
    monkeypatch.setattr(config, "MIN_REAL_MOLECULES", 50)
    return tiny_env


@pytest.fixture(autouse=True)
def _isolate_from_developer_environment(monkeypatch, tmp_path_factory):
    for var in [v for v in os.environ if v.startswith("TOXION_")] + ["DATABASE_URL"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TOXION_ENV_FILE", str(tmp_path_factory.getbasetemp() / "no-such.env"))


@pytest.fixture(scope="session")
def tiny_env_v2(tmp_path_factory):
    from toxion.data.loader import save_clean
    from toxion.training import pipeline as pl
    from toxion.training.runner import run_training

    root = tmp_path_factory.mktemp("tiny_v2")
    clean = str(root / "clean.parquet")
    raw = root / "raw.bin"
    raw.write_bytes(b"another pretend tox21 file")
    df = make_synthetic_clean(n=300, seed=2)
    save_clean(df, clean)
    manifest = pl.write_manifest(clean, str(raw), len(df))
    cfg = pl.TrainConfig(hidden_dim=16, T=2, batch_size=16, epochs=2, patience=2, seed=0, split_mode="random",
                         feature_version=2, ensemble_size=2)
    ckpt = str(root / "active.pt")
    with pytest.MonkeyPatch.context() as mp:
        for var in [v for v in os.environ if v.startswith("TOXION_")]:
            mp.delenv(var)
        metrics = run_training(clean, cfg, ckpt, device="cpu")
    return {"clean": clean, "ckpt": ckpt, "sha": manifest["raw_sha256"], "root": root, "df": df, "metrics": metrics}


@pytest.fixture
def real_gate_v2(tiny_env_v2, monkeypatch):
    monkeypatch.setenv("TOXION_REAL_SHA256", tiny_env_v2["sha"])
    monkeypatch.setattr(config, "MIN_REAL_MOLECULES", 50)
    return tiny_env_v2
