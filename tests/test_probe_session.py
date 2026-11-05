import pytest

from toxion.chem.resolver import ResolveError, Resolved
from toxion.db.store import MoleculeStore
from toxion.inference.probe import ProbeSession, format_result, interactive_loop, predict_smiles
from toxion.inference.registry import load_checkpoint

NEW_MOLECULES = {"ethyl acetate": "CCOC(C)=O", "diethylamine": "CCNCC", "butanal": "CCCC=O", "acetone": "CC(C)=O"}


class FakeResolver:
    def __init__(self, table):
        self.table, self.calls = table, []

    def resolve(self, name):
        self.calls.append(name)
        if name not in self.table:
            raise ResolveError(f"PubChem has no compound named {name!r}")
        return Resolved(name, 100 + len(self.calls), self.table[name], f"IUPAC {name}")


@pytest.fixture
def loaded(real_gate):
    return load_checkpoint(real_gate["ckpt"])


@pytest.fixture
def store(tmp_path):
    return MoleculeStore(f"sqlite:///{tmp_path / 'p.sqlite'}")


def scripted(*lines):
    it = iter(lines)

    def input_fn(prompt):
        item = next(it)
        if isinstance(item, BaseException) or (isinstance(item, type) and issubclass(item, BaseException)):
            raise item
        return item

    return input_fn


def test_prediction_is_well_formed(loaded):
    pred = predict_smiles(loaded, "CCOC(C)=O", mc_samples=5)
    assert pred.probs.shape == pred.stds.shape == (12,) and ((0 <= pred.probs) & (pred.probs <= 1)).all()
    assert pred.toxic == bool(pred.flagged) and 0.0 <= pred.nn_similarity <= 1.0
    assert all(p >= thr for _, p, _, thr in pred.flagged)


def test_known_tox21_molecule_is_not_stored(loaded, store, real_gate):
    known = real_gate["df"]["smiles_canonical"].iloc[0]
    s = ProbeSession(loaded, FakeResolver({"known": known}), store, mc_samples=2)
    r = s.handle("known")
    assert r.status == "known" and store.count_new() == 0
    assert "training data" in format_result(r)


def test_new_molecule_is_stored_once_and_cached_by_name(loaded, store):
    resolver = FakeResolver(NEW_MOLECULES)
    s = ProbeSession(loaded, resolver, store, mc_samples=2, auto_retrain=False)
    assert s.handle("ethyl acetate").status == "stored"
    again = s.handle("Ethyl  Acetate")
    assert again.status == "already_stored" and again.from_cache and resolver.calls == ["ethyl acetate"]
    assert store.count_new() == 1


def test_retrain_triggers_exactly_at_the_batch_size(loaded, store):
    calls = []

    class Outcome:
        decision, message = "rejected", "kept the current model"

    s = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), store, lambda st: calls.append(st.count_new()) or Outcome(),
                     batch_size=3, mc_samples=2)
    results = [s.handle(n, out=lambda m: None) for n in ["ethyl acetate", "diethylamine", "butanal"]]
    assert calls == [3] and not any("retrain" in n for n in results[0].notes + results[1].notes)
    assert any("retrain: rejected" in n for n in results[2].notes)
    s2 = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), store, lambda st: 1 / 0, batch_size=1, mc_samples=2)
    assert any("retrain failed" in n for n in s2.handle("acetone", out=lambda m: None).notes)


def test_auto_retrain_can_be_switched_off(loaded, store):
    s = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), store, lambda st: 1 / 0, batch_size=1, auto_retrain=False, mc_samples=2)
    assert s.handle("acetone").notes == []


def test_database_failure_never_blocks_a_prediction(loaded):
    class Broken:
        def lookup_alias(self, name):
            raise RuntimeError("connection refused")

    r = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), Broken(), mc_samples=2).handle("acetone")
    assert r.status == "not_stored" and any("database unavailable" in n for n in r.notes)


def test_no_store_mode(loaded):
    assert ProbeSession(loaded, FakeResolver(NEW_MOLECULES), None, mc_samples=2).handle("acetone").status == "not_stored"


@pytest.mark.parametrize("ender", ["exit", "QUIT", KeyboardInterrupt, EOFError])
def test_interactive_loop_quits_cleanly(loaded, ender):
    out = []
    s = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), None, mc_samples=2)
    code = interactive_loop(s, scripted("", "acetone", "unknown thing", ender), out.append)
    text = "\n".join(out)
    assert code == 0 and "SMILES : CC(C)=O" in text and "Verdict:" in text
    assert "error: PubChem has no compound named 'unknown thing'" in text


def test_ctrl_c_during_handling_exits_without_traceback(loaded):
    class Interrupting:
        def resolve(self, name):
            raise KeyboardInterrupt

    s = ProbeSession(loaded, Interrupting(), None, mc_samples=2)
    assert interactive_loop(s, scripted("acetone", "never reached"), lambda m: None) == 0


def test_result_carries_and_prints_the_morgan_fingerprint(loaded):
    from toxion.data.featurize import featurize_fingerprint, morgan_on_bits

    r = ProbeSession(loaded, FakeResolver(NEW_MOLECULES), None, mc_samples=2).handle("acetone")
    assert r.fingerprint == morgan_on_bits("CC(C)=O") and sum(r.fingerprint.values()) == featurize_fingerprint("CC(C)=O").sum()
    text = format_result(r)
    assert f"{len(r.fingerprint)} set (bit:count)" in text
    assert all(f"{b}:{c}" in text for b, c in r.fingerprint.items())
    assert max(len(line) for line in text.splitlines() if line.startswith("  ") and ":" in line and "p=" not in line) <= 104
