import json

from toxion.cli import main


def test_make_synthetic_prepare_summary_analyze_end_to_end(tmp_path, capsys):
    raw = tmp_path / "syn.csv.gz"
    clean = tmp_path / "clean.parquet"
    out = tmp_path / "analysis"

    assert main(["make-synthetic", "--n", "120", "--out", str(raw)]) == 0
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed
    assert f"toxion prepare-data --raw-path {raw}" in printed
    assert "synthetic_clean.parquet" in printed and "results/analysis_synthetic" in printed
    assert main(["prepare-data", "--raw-path", str(raw), "--out-path", str(clean)]) == 0
    assert main(["summary", "--clean-path", str(clean)]) == 0
    capsys.readouterr()
    assert main(["analyze", "--clean-path", str(clean), "--out-dir", str(out), "--n-bits", "256"]) == 0
    analyze_out = capsys.readouterr().out
    assert "nearest-neighbor Tanimoto" in analyze_out and "scaffolds:" in analyze_out

    for name in ["endpoints_phi.csv", "endpoints_mi_bits.csv", "endpoints_colabeled.csv",
                 "pca_variance.csv", "pca_scores_top2.csv", "scaffold_top.csv", "summary.json"]:
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_molecules"] == 120
    assert 0 < summary["internal_diversity"] < 1
    assert summary["scaffolds"]["n_scaffolds"] >= 2


def test_analyze_missing_clean_file_is_one_line_error(tmp_path, capsys):
    assert main(["analyze", "--clean-path", str(tmp_path / "none.parquet")]) == 1
    err = capsys.readouterr().err
    assert "prepare-data" in err and "Traceback" not in err
