import json

import pytest

from toxion.chem.resolver import PubChemResolver, ResolveError


class FakePubChem:

    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        for fragment, response in self.routes:
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        return 404, "{}"


def _json(obj):
    return 200, json.dumps(obj)


ASPIRIN_CID = _json({"IdentifierList": {"CID": [2244]}})


def resolver(routes, **kw):
    fake = FakePubChem(routes)
    return PubChemResolver(fetch=fake, sleep=lambda s: None, **kw), fake


def test_resolves_and_canonicalizes():
    r, fake = resolver([
        ("/cids/JSON", ASPIRIN_CID),
        ("/property/SMILES/JSON", _json({"PropertyTable": {"Properties": [{"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}]}})),
        ("/property/IUPACName/JSON", _json({"PropertyTable": {"Properties": [{"CID": 2244, "IUPACName": "2-acetyloxybenzoic acid"}]}})),
    ])
    res = r.resolve("2-acetyloxybenzoic acid")
    assert res.cid == 2244 and res.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert res.iupac_name == "2-acetyloxybenzoic acid"


def test_name_is_url_encoded():
    r, fake = resolver([("/cids/JSON", ASPIRIN_CID), ("/property/SMILES/JSON", _json({"PropertyTable": {"Properties": [{"SMILES": "CCO"}]}}))])
    r.resolve("propan-2-yl 2-(1,3)acetate/x")
    assert "propan-2-yl%202-%281%2C3%29acetate%2Fx" in fake.urls[0] or "%2F" in fake.urls[0]
    assert " " not in fake.urls[0]


def test_falls_back_when_property_name_is_rejected_and_reads_any_smiles_key():
    r, fake = resolver([
        ("/cids/JSON", ASPIRIN_CID),
        ("/property/SMILES/JSON", (400, "bad property")),
        ("/property/IsomericSMILES/JSON", (400, "bad property")),
        ("/property/ConnectivitySMILES/JSON", _json({"PropertyTable": {"Properties": [{"CID": 2244, "ConnectivitySMILES": "CCO"}]}})),
    ])
    assert r.resolve("ethanol").canonical_smiles == "CCO"
    assert any("ConnectivitySMILES" in u for u in fake.urls)


def test_unknown_name_is_a_clean_error():
    r, _ = resolver([("/cids/JSON", (404, '{"Fault": {}}'))])
    with pytest.raises(ResolveError, match="no compound named"):
        r.resolve("zzzznotachemical")


def test_retries_then_reports_unreachable_and_unavailable():
    import urllib.error

    r, fake = resolver([("/cids/JSON", urllib.error.URLError("down"))], retries=2)
    with pytest.raises(ResolveError, match="could not reach PubChem"):
        r.resolve("ethanol")
    assert len(fake.urls) == 3
    r2, _ = resolver([("/cids/JSON", (503, "busy"))], retries=1)
    with pytest.raises(ResolveError, match="unavailable"):
        r2.resolve("ethanol")


def test_retry_succeeds_after_transient_503():
    calls = {"n": 0}

    def flaky(url, timeout):
        if "/cids/" in url:
            calls["n"] += 1
            return (503, "") if calls["n"] == 1 else ASPIRIN_CID
        return _json({"PropertyTable": {"Properties": [{"SMILES": "CCO"}]}})

    assert PubChemResolver(fetch=flaky, sleep=lambda s: None).resolve("x").cid == 2244


def test_rejects_empty_long_and_unparseable():
    r, _ = resolver([("/cids/JSON", ASPIRIN_CID), ("/property/SMILES/JSON", _json({"PropertyTable": {"Properties": [{"SMILES": "not smiles"}]}}))])
    with pytest.raises(ResolveError):
        r.resolve("   ")
    with pytest.raises(ResolveError):
        r.resolve("x" * 600)
    with pytest.raises(ResolveError, match="could not parse"):
        r.resolve("weird")


def test_certificate_failure_gets_an_actionable_hint():
    import urllib.error

    r, _ = resolver([("/cids/JSON", urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain"))], retries=0)
    with pytest.raises(ResolveError, match="SSL_CERT_FILE"):
        r.resolve("ethanol")
