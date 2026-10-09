import logging

import numpy as np
import pandas as pd
import pytest

from toxion.cli import main
from toxion.data.loader import (
    TOX21_TASKS,
    canonicalize_smiles,
    load_and_clean,
    load_clean,
    save_clean,
)


def _write_raw(path, rows):
    """rows: list of (smiles, {task: value}); unspecified tasks are NaN."""
    records = []
    for i, (smiles, labels) in enumerate(rows):
        record = {task: labels.get(task, np.nan) for task in TOX21_TASKS}
        record["mol_id"] = f"TOX{i}"
        record["smiles"] = smiles
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def test_canonicalize_equivalent_smiles_match():
    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO") == "CCO"


@pytest.mark.parametrize("bad", ["not_a_smiles", float("nan"), None, 42])
def test_canonicalize_returns_none_for_unparseable(bad):
    assert canonicalize_smiles(bad) is None


def test_unparseable_rows_dropped_and_duplicates_collapsed(tmp_path):
    raw = _write_raw(tmp_path / "raw.csv", [
        ("CCO", {"NR-AR": 0.0}),
        ("OCC", {"NR-AR": 0.0}),
        ("not_a_smiles", {"NR-AR": 1.0}),
        ("c1ccccc1", {"SR-p53": 1.0}),
    ])
    df = load_and_clean(raw)
    assert list(df.columns) == ["smiles_canonical", *TOX21_TASKS]
    assert sorted(df["smiles_canonical"]) == ["CCO", "c1ccccc1"]
    assert df.index.tolist() == [0, 1]


def test_missing_labels_stay_nan_not_zero(tmp_path):
    raw = _write_raw(tmp_path / "raw.csv", [("CCO", {"NR-AR": 1.0})])
    df = load_and_clean(raw)
    assert df.loc[0, "NR-AR"] == 1.0
    assert df.loc[0, TOX21_TASKS[1:]].isna().all()


def test_non_numeric_labels_coerced_to_nan(tmp_path):
    raw = _write_raw(tmp_path / "raw.csv", [("CCO", {"NR-AR": "oops"})])
    df = load_and_clean(raw)
    assert np.isnan(df.loc[0, "NR-AR"])


def test_duplicate_keeps_first_and_logs_conflict(tmp_path, caplog):
    raw = _write_raw(tmp_path / "raw.csv", [
        ("CCO", {"NR-AR": 0.0}),
        ("OCC", {"NR-AR": 1.0}),
    ])
    with caplog.at_level(logging.WARNING, logger="toxion.data.loader"):
        df = load_and_clean(raw)
    assert len(df) == 1
    assert df.loc[0, "NR-AR"] == 0.0
    assert "conflicting labels" in caplog.text


def test_missing_task_column_raises(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"smiles": ["CCO"], "NR-AR": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_and_clean(str(path))


def test_gzip_input_and_parquet_roundtrip(tmp_path):
    raw = _write_raw(tmp_path / "raw.csv", [("CCO", {"NR-AR": 1.0}), ("CCN", {})])
    gz = tmp_path / "raw.csv.gz"
    pd.read_csv(raw).to_csv(gz, index=False, compression="gzip")

    df = load_and_clean(str(gz))
    out = tmp_path / "nested" / "clean.parquet"
    save_clean(df, str(out))
    pd.testing.assert_frame_equal(load_clean(str(out)), df)


def test_cli_prepare_data_end_to_end(tmp_path, capsys):
    raw = _write_raw(tmp_path / "raw.csv", [("CCO", {"NR-AR": 1.0}), ("CCN", {})])
    out = tmp_path / "clean.parquet"
    code = main(["prepare-data", "--raw-path", raw, "--out-path", str(out)])
    assert code == 0
    assert "Wrote 2 clean rows" in capsys.readouterr().out
    assert len(load_clean(str(out))) == 2


def test_cli_check_env_reports_core_installed(capsys):
    assert main(["check-env"]) == 0
    out = capsys.readouterr().out
    assert "[core   ] rdkit" in out
    assert "MISSING" not in out.split("[ml")[0]
