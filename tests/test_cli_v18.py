import pytest

from toxion import cli


def test_probe_without_a_trained_model_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TOXION_CHECKPOINT", str(tmp_path / "missing.pt"))
    assert cli.main(["probe", "aspirin"]) == 2
    assert "train-fusion" in capsys.readouterr().err


def test_probe_refuses_a_checkpoint_not_trained_on_real_data(tiny_env, monkeypatch, capsys):
    monkeypatch.setenv("TOXION_CHECKPOINT", tiny_env["ckpt"])
    assert cli.main(["probe", "aspirin"]) == 2
    assert "real Tox21" in capsys.readouterr().err


def test_probe_one_shot_exit_codes(real_gate, monkeypatch, capsys, tmp_path):
    from toxion.chem import resolver as res

    monkeypatch.setenv("TOXION_CHECKPOINT", real_gate["ckpt"])
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    def fake_resolve(self, name):
        if name == "ethanol":
            return res.Resolved(name, 702, "CCO", "ethanol")
        raise res.ResolveError("PubChem has no compound named 'zzz'")

    monkeypatch.setattr(res.PubChemResolver, "resolve", fake_resolve)
    assert cli.main(["probe", "ethanol"]) == 0
    out = capsys.readouterr().out
    assert "SMILES : CCO" in out and "Verdict:" in out and "not a safety determination" in out
    assert cli.main(["probe", "zzz"]) == 1


def test_probe_interactive_mode_reads_names_and_quits(real_gate, monkeypatch, capsys, tmp_path):
    from toxion.chem import resolver as res

    monkeypatch.setenv("TOXION_CHECKPOINT", real_gate["ckpt"])
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(res.PubChemResolver, "resolve", lambda self, n: res.Resolved(n, 1, "CCO", None))
    answers = iter(["ethanol", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["probe"]) == 0
    assert "SMILES : CCO" in capsys.readouterr().out


def test_make_splits_and_prepare_data_manifest(tmp_path, capsys):
    from toxion.data.synthetic import make_synthetic_raw

    raw = tmp_path / "raw.csv.gz"
    make_synthetic_raw(120, 0).to_csv(raw, index=False)
    clean = tmp_path / "clean.parquet"
    assert cli.main(["prepare-data", "--raw-path", str(raw), "--out-path", str(clean)]) == 0
    import json

    manifest = json.load(open(str(clean) + ".manifest.json"))
    assert manifest["n_rows"] > 0 and len(manifest["raw_sha256"]) == 64
    assert cli.main(["make-splits", "--clean-path", str(clean), "--out-dir", str(tmp_path / "sp"), "--seeds", "0,1"]) == 0
    assert (tmp_path / "sp" / "split_seed1.npz").exists()


def test_explain_cli_writes_a_png(real_gate, monkeypatch, tmp_path, capsys):
    from toxion.chem import resolver as res

    monkeypatch.setenv("TOXION_CHECKPOINT", real_gate["ckpt"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(res.PubChemResolver, "resolve", lambda self, n: res.Resolved(n, 1, "CC(=O)Oc1ccccc1C(=O)O", None))
    out = tmp_path / "e.png"
    assert cli.main(["explain", "aspirin", "--task", "NR-AhR", "--out", str(out)]) == 0
    assert out.read_bytes()[:4] == b"\x89PNG" and "top atoms" in capsys.readouterr().out
    assert cli.main(["explain", "aspirin", "--method", "attention", "--out", str(out)]) == 0
    assert cli.main(["explain", "aspirin", "--task", "BAD"]) == 1
