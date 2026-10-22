import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toxion.cli import main
from toxion.data.loader import TOX21_TASKS, save_clean
from toxion.data.summary import overall_stats, summarize_labels


def _clean_df():
    rows = [
        {"smiles_canonical": "CCO", "NR-AR": 1.0, "NR-AhR": 0.0},
        {"smiles_canonical": "CCN", "NR-AR": 0.0},
        {"smiles_canonical": "c1ccccc1"},
        {"smiles_canonical": "CCC", "NR-AR": 0.0, "NR-AhR": 1.0},
    ]
    df = pd.DataFrame(rows)
    for task in TOX21_TASKS:
        if task not in df.columns:
            df[task] = np.nan
    return df[["smiles_canonical", *TOX21_TASKS]]


def test_summarize_labels_counts():
    s = summarize_labels(_clean_df())
    assert s.loc["NR-AR", ["labeled", "missing", "positives", "negatives"]].tolist() == [3, 1, 1, 2]
    assert s.loc["NR-AR", "positive_rate"] == 0.333
    assert s.loc["NR-AhR", "positive_rate"] == 0.5
    assert s.loc["SR-p53", "labeled"] == 0
    assert np.isnan(s.loc["SR-p53", "positive_rate"])


def test_overall_stats():
    stats = overall_stats(_clean_df())
    assert stats["molecules"] == 4
    assert stats["labeled_entries"] == 5
    assert stats["total_entries"] == 48
    assert stats["molecules_without_labels"] == 1


def test_summary_handles_empty_frame():
    empty = _clean_df().iloc[0:0]
    assert overall_stats(empty)["molecules"] == 0
    assert len(summarize_labels(empty)) == len(TOX21_TASKS)


def test_cli_summary_prints_table(tmp_path, capsys):
    path = tmp_path / "clean.parquet"
    save_clean(_clean_df(), str(path))
    assert main(["summary", "--clean-path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "molecules: 4" in out
    assert "positive_rate" in out
    assert "NR-AR" in out


def test_cli_missing_raw_file_is_one_line_error(tmp_path, capsys):
    code = main(["prepare-data", "--raw-path", str(tmp_path / "nope.csv.gz")])
    err = capsys.readouterr().err
    assert code == 1
    assert "Raw file not found" in err and "download_data.py" in err
    assert "Traceback" not in err


def test_cli_missing_clean_file_is_one_line_error(tmp_path, capsys):
    code = main(["summary", "--clean-path", str(tmp_path / "nope.parquet")])
    err = capsys.readouterr().err
    assert code == 1
    assert "prepare-data" in err and "Traceback" not in err


def test_cli_bad_columns_error_and_traceback_flag(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(bad, index=False)
    assert main(["prepare-data", "--raw-path", str(bad)]) == 1
    assert "Missing expected columns" in capsys.readouterr().err
    try:
        main(["--traceback", "prepare-data", "--raw-path", str(bad)])
    except ValueError:
        pass
    else:
        raise AssertionError("--traceback should re-raise")


@pytest.mark.parametrize("name", ["01_data_check.ipynb", "02_advanced_analysis.ipynb"])
def test_notebook_is_valid_json_with_clean_code_cells(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in nb["cells"])
    assert all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
