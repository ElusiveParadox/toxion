
import os
import ssl
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/"
       "250k_rndm_zinc_drugs_clean_3.csv")
OUT = "data/raw/zinc250k.csv"


def fetch(url: str) -> bytes:
    contexts = [ssl.create_default_context()]
    try:
        import certifi

        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    last = None
    for ctx in contexts:
        try:
            with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
                return r.read()
        except (ssl.SSLError, urllib.error.URLError) as exc:
            last = exc
    raise SystemExit(f"download failed: {last}")


def main() -> int:
    data = fetch(URL)
    head = data[:200].decode("utf-8", "replace").lower()
    if "smiles" not in head or len(data) < 1_000_000:
        raise SystemExit(f"unexpected content ({len(data)} bytes); not saving")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(data)
    print(f"saved {OUT} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
