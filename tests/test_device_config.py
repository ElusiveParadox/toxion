import pytest

from toxion import config, device


def test_batch_size_guard_thresholds():
    assert device.choose_batch_size(32, available_gb=6.0) == 32
    assert device.choose_batch_size(32, available_gb=2.0) == 16
    assert device.choose_batch_size(10, available_gb=2.0) == device.MIN_BATCH
    with pytest.raises(MemoryError):
        device.choose_batch_size(32, available_gb=0.5)
    assert device.choose_batch_size(32, available_gb=None) in (32, 16)


def test_edge_memory_estimate_matches_hand_calculation():
    assert round(device.estimate_edge_memory_mb(32, 128, 54)) == 113


def test_device_override_and_cpu_default_without_mps(monkeypatch):
    monkeypatch.setenv("TOXION_DEVICE", "cpu")
    assert device.pick_device() == "cpu"
    monkeypatch.delenv("TOXION_DEVICE")
    assert device.pick_device(prefer="cpu") == "cpu"


def test_dotenv_loader_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nDATABASE_URL="postgresql://u:p@h/db"\nTOXION_AUTO_RETRAIN=0\nKEEP=from_file\n')
    for var in ("DATABASE_URL", "TOXION_AUTO_RETRAIN"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.setenv("KEEP", "from_env")
    config.load_dotenv(str(env))
    assert config.database_url() == "postgresql://u:p@h/db"
    assert config.auto_retrain_enabled() is False
    import os

    assert os.environ["KEEP"] == "from_env"
    monkeypatch.delenv("DATABASE_URL")
    assert config.database_url() is None


def test_pinned_hash_matches_the_real_file_when_present():
    import os

    path = "data/raw/tox21.csv.gz"
    if not os.path.exists(path):
        pytest.skip("real Tox21 file not present")
    from toxion.training.pipeline import file_sha256

    assert file_sha256(path) == config.REAL_TOX21_SHA256


def test_dotenv_inline_comments_quotes_and_export(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TOXION_FPR_CAP=0.10          # at TRAIN time: cap FPR\n"
                   "export A_VAR='quoted # not a comment'\n"
                   "B_VAR=postgresql://u:p@h/db?sslmode=require#frag\n"
                   "C_VAR=   \n")
    for var in ("TOXION_FPR_CAP", "A_VAR", "B_VAR", "C_VAR"):
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    config.load_dotenv(str(env))
    assert config.fpr_cap() == 0.10
    import os

    assert os.environ["A_VAR"] == "quoted # not a comment"
    assert os.environ["B_VAR"].endswith("#frag")
    assert os.environ["C_VAR"] == ""


def test_bad_fpr_cap_names_the_variable(monkeypatch):
    monkeypatch.setenv("TOXION_FPR_CAP", "ten percent")
    with pytest.raises(ValueError, match="TOXION_FPR_CAP"):
        config.fpr_cap()
