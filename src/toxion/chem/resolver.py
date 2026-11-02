
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from toxion.data.loader import canonicalize_smiles

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SMILES_PROPERTIES = ["SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"]
MAX_NAME_LENGTH = 500


class ResolveError(Exception):
    pass


@dataclass
class Resolved:
    name: str
    cid: int
    canonical_smiles: str
    iupac_name: str | None = None


def _ssl_context():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _default_fetch(url: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "toxion/0.1 (research CLI)", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


class PubChemResolver:
    def __init__(self, fetch: Callable[[str, float], tuple[int, str]] | None = None,
                 timeout: float = 15.0, retries: int = 2, backoff: float = 1.0, sleep=time.sleep):
        self._fetch = fetch or _default_fetch
        self.timeout, self.retries, self.backoff, self._sleep = timeout, retries, backoff, sleep

    def _get(self, url: str) -> tuple[int, str]:
        last: tuple[int, str] = (0, "")
        for attempt in range(self.retries + 1):
            try:
                status, body = self._fetch(url, self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = (0, str(exc))
            else:
                if status != 429 and status < 500:
                    return status, body
                last = (status, body)
            if attempt < self.retries:
                self._sleep(self.backoff * (2 ** attempt))
        if last[0] == 0:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in last[1]:
                hint = (" TLS certificate check failed: a proxy, VPN or antivirus may be intercepting HTTPS. Point "
                        "SSL_CERT_FILE at a CA bundle that includes it (or `pip install certifi`). Toxion never "
                        "disables verification.")
            raise ResolveError(f"could not reach PubChem ({last[1] or 'network error'}).{hint}")
        raise ResolveError(f"PubChem is unavailable right now (HTTP {last[0]})")

    def _cid(self, name: str) -> int:
        url = f"{BASE_URL}/compound/name/{urllib.parse.quote(name, safe='')}/cids/JSON"
        status, body = self._get(url)
        if status == 404:
            raise ResolveError(f"PubChem has no compound named {name!r}")
        if status != 200:
            raise ResolveError(f"PubChem lookup failed for {name!r} (HTTP {status})")
        try:
            cids = json.loads(body)["IdentifierList"]["CID"]
            cid = int(cids[0])
        except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
            raise ResolveError(f"unexpected PubChem response for {name!r}") from None
        if cid <= 0:
            raise ResolveError(f"PubChem has no compound named {name!r}")
        return cid

    def _property(self, cid: int, prop: str) -> dict | None:
        status, body = self._get(f"{BASE_URL}/compound/cid/{cid}/property/{prop}/JSON")
        if status != 200:
            return None
        try:
            return json.loads(body)["PropertyTable"]["Properties"][0]
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            return None

    def resolve(self, name: str) -> Resolved:
        name = (name or "").strip()
        if not name:
            raise ResolveError("empty name")
        if len(name) > MAX_NAME_LENGTH:
            raise ResolveError("name is too long")
        cid = self._cid(name)

        smiles = None
        for prop in SMILES_PROPERTIES:
            record = self._property(cid, prop)
            if record:
                smiles = next((v for k, v in record.items() if "SMILES" in k.upper() and isinstance(v, str) and v), None)
                if smiles:
                    break
        if not smiles:
            raise ResolveError(f"PubChem CID {cid} returned no SMILES")
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ResolveError(f"RDKit could not parse the SMILES PubChem returned for CID {cid}")

        iupac = None
        record = self._property(cid, "IUPACName")
        if record and isinstance(record.get("IUPACName"), str):
            iupac = record["IUPACName"]
        return Resolved(name=name, cid=cid, canonical_smiles=canonical, iupac_name=iupac)
