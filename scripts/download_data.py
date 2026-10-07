"""Download the Tox21 CSV (DeepChem/MoleculeNet distribution) into data/raw/.

Usage: python scripts/download_data.py [--url URL] [--out data/raw/tox21.csv.gz]
Standard library only, so it runs before the project is installed.

If Python's TLS certificate check fails (common with the python.org macOS
installer before 'Install Certificates.command' has been run), it retries with
the certifi bundle if available, then with curl, which uses the system
certificates.
"""

import argparse
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
GZIP_MAGIC = b"\x1f\x8b"


class DownloadError(Exception):
    pass


def _is_ssl_error(exc: urllib.error.URLError) -> bool:
    return isinstance(getattr(exc, "reason", None), ssl.SSLError)


def _certifi_context():
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_urllib(url: str, out: Path, context=None) -> None:
    with urllib.request.urlopen(url, context=context, timeout=60) as response, open(out, "wb") as f:
        shutil.copyfileobj(response, f)


def _fetch_curl(url: str, out: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise DownloadError("curl not found")
    result = subprocess.run([curl, "-fsSL", "-o", str(out), url], capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise DownloadError(f"curl failed ({lines[0] if lines else 'exit ' + str(result.returncode)})")


def _fallback_after_ssl_error(url: str, out: Path) -> None:
    print("Python's SSL certificate check failed; trying fallbacks.")
    errors = []
    context = _certifi_context()
    if context is not None:
        try:
            _fetch_urllib(url, out, context)
            print("Downloaded using the certifi certificate bundle.")
            return
        except urllib.error.URLError as exc:
            errors.append(f"certifi: {exc.reason}")
    try:
        _fetch_curl(url, out)
        print("Downloaded using curl.")
        return
    except DownloadError as exc:
        errors.append(str(exc))
    raise DownloadError(
        "SSL certificate verification failed and the fallbacks did not work "
        f"({'; '.join(errors)}). On the python.org macOS installer, run "
        "'Install Certificates.command' from the Python folder in /Applications, "
        f"or download manually: curl -L -o {out} {url}"
    )


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    try:
        try:
            _fetch_urllib(url, out)
        except urllib.error.HTTPError as exc:
            raise DownloadError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
        except urllib.error.URLError as exc:
            if not _is_ssl_error(exc):
                raise DownloadError(f"could not reach {url}: {exc.reason}") from exc
            _fallback_after_ssl_error(url, out)
        with open(out, "rb") as f:
            if f.read(2) != GZIP_MAGIC:
                raise DownloadError("downloaded file is not gzip data (bad URL or an error page)")
    except (DownloadError, OSError) as exc:
        out.unlink(missing_ok=True)
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError(f"download failed: {exc}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/raw/tox21.csv.gz")
    args = parser.parse_args(argv)

    out = Path(args.out)
    try:
        download(args.url, out)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
