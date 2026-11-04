import pytest

from toxion.db.models import STATUS_CLAIMED, STATUS_NEW, STATUS_REJECTED, STATUS_TRAINED
from toxion.db.store import MoleculeStore, normalize_name, normalize_url


@pytest.fixture
def store(tmp_path):
    return MoleculeStore(f"sqlite:///{tmp_path / 'db.sqlite'}")


def test_url_normalisation_picks_psycopg3():
    assert normalize_url("postgresql://u:p@h/db?sslmode=require") == "postgresql+psycopg://u:p@h/db?sslmode=require"
    assert normalize_url("postgres://u@h/db").startswith("postgresql+psycopg://")
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def test_name_normalisation():
    assert normalize_name("  Ethyl   ALCOHOL ") == "ethyl alcohol"


def test_same_molecule_under_two_names_counts_once(store):
    m1, created1 = store.add_molecule("CCO", "ethanol", 702)
    m2, created2 = store.add_molecule("CCO", "ethyl alcohol", 702)
    assert created1 and not created2 and m1.id == m2.id
    assert store.count_new() == 1
    store.add_alias("Ethanol", "CCO", 702)
    store.add_alias("ethyl alcohol", "CCO", 702)
    assert store.lookup_alias("  ETHANOL ").canonical_smiles == "CCO" and store.lookup_alias("nope") is None


def test_claim_needs_a_full_batch_and_is_exclusive(store):
    for i in range(5):
        store.add_molecule("C" * (i + 1), f"n{i}")
    assert store.claim_batch(6) == [] and store.count_new() == 5
    first = store.claim_batch(3)
    assert len(first) == 3 and store.count_new() == 2
    assert store.claim_batch(3) == []
    store.release([m.id for m in first])
    assert store.count_new() == 5


def test_status_transitions_and_pool(store):
    ids = [store.add_molecule("C" * (i + 1), f"n{i}")[0].id for i in range(4)]
    store.set_status(ids[:2], STATUS_TRAINED)
    store.set_status(ids[2:3], STATUS_REJECTED)
    assert store.trained_smiles(10) == ["CC", "C"] and store.trained_smiles(0) == []
    assert store.status_counts() == {STATUS_TRAINED: 2, STATUS_REJECTED: 1, STATUS_NEW: 1}
    assert STATUS_CLAIMED == "claimed"


def test_record_run(store):
    store.record_run(n_molecules=25, decision="promoted", val_auc_before=0.7, val_auc_after=0.71)
