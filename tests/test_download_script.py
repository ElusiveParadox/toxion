import gzip
import importlib.util
import ssl
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ssl_error():
    return urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed"))


def _write_gz(path: Path):
    path.write_bytes(gzip.compress(b"smiles,NR-AR\nCCO,0\n"))


def test_gzip_file_accepted(mod, tmp_path):
    src, out = tmp_path / "src.gz", tmp_path / "out" / "tox21.csv.gz"
    _write_gz(src)
    assert mod.main(["--url", src.as_uri(), "--out", str(out)]) == 0
    assert out.exists()


def test_non_gzip_rejected_and_removed(mod, tmp_path, capsys):
    src, out = tmp_path / "src.bin", tmp_path / "tox21.csv.gz"
    src.write_bytes(b"<html>error page</html>")
    assert mod.main(["--url", src.as_uri(), "--out", str(out)]) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "not gzip" in err and "Traceback" not in err


def test_http_error_is_one_line(mod, tmp_path, monkeypatch, capsys):
    def boom(url, out, context=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(mod, "_fetch_urllib", boom)
    assert mod.main(["--url", "https://example.invalid/x.gz", "--out", str(tmp_path / "o.gz")]) == 1
    err = capsys.readouterr().err
    assert "HTTP 404" in err and "Traceback" not in err


def test_unreachable_host_is_one_line(mod, tmp_path, monkeypatch, capsys):
    def boom(url, out, context=None):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(mod, "_fetch_urllib", boom)
    assert mod.main(["--url", "https://example.invalid/x.gz", "--out", str(tmp_path / "o.gz")]) == 1
    assert "could not reach" in capsys.readouterr().err


def test_ssl_error_falls_back_to_curl(mod, tmp_path, monkeypatch, capsys):
    out = tmp_path / "o.gz"
    monkeypatch.setattr(mod, "_fetch_urllib", lambda url, o, context=None: (_ for _ in ()).throw(_ssl_error()))
    monkeypatch.setattr(mod, "_certifi_context", lambda: None)
    monkeypatch.setattr(mod, "_fetch_curl", lambda url, o: _write_gz(o))
    assert mod.main(["--url", "https://example.invalid/x.gz", "--out", str(out)]) == 0
    assert "curl" in capsys.readouterr().out and out.exists()


def test_ssl_error_falls_back_to_certifi_first(mod, tmp_path, monkeypatch, capsys):
    out, calls = tmp_path / "o.gz", []

    def fetch(url, o, context=None):
        calls.append(context)
        if context is None:
            raise _ssl_error()
        _write_gz(o)

    monkeypatch.setattr(mod, "_fetch_urllib", fetch)
    monkeypatch.setattr(mod, "_certifi_context", lambda: object())
    monkeypatch.setattr(mod, "_fetch_curl", lambda url, o: pytest.fail("curl should not be needed"))
    assert mod.main(["--url", "https://example.invalid/x.gz", "--out", str(out)]) == 0
    assert calls[0] is None and calls[1] is not None
    assert "certifi" in capsys.readouterr().out


def test_ssl_error_without_fallbacks_gives_actionable_message(mod, tmp_path, monkeypatch, capsys):
    out = tmp_path / "o.gz"
    monkeypatch.setattr(mod, "_fetch_urllib", lambda url, o, context=None: (_ for _ in ()).throw(_ssl_error()))
    monkeypatch.setattr(mod, "_certifi_context", lambda: None)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    assert mod.main(["--url", "https://example.invalid/x.gz", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "Install Certificates.command" in err and "curl -L -o" in err and "Traceback" not in err
    assert not out.exists()
