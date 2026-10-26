
from __future__ import annotations

import os

DEFAULT_RAW_PATH = "data/raw/tox21.csv.gz"
DEFAULT_CLEAN_PATH = "data/processed/tox21_clean.parquet"
DEFAULT_SPLIT_DIR = "data/processed/splits"
DEFAULT_CHECKPOINT = "results/checkpoints/active.pt"
DEFAULT_TABLE_DIR = "results/tables"

REAL_TOX21_SHA256 = "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360"
MIN_REAL_MOLECULES = 6000

RETRAIN_BATCH_SIZE = 25
UNLABELED_POOL_CAP = 500
AD_SIMILARITY_CUTOFF = 0.4
MC_DROPOUT_SAMPLES = 30
PROMOTE_AUC_TOLERANCE = 0.005
PROMOTE_ECE_TOLERANCE = 0.01


def _parse_env_value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in {'"', "'"}:
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
    for i, ch in enumerate(raw):
        if ch == "#" and (i == 0 or raw[i - 1] in " \t"):
            return raw[:i].strip()
    return raw


def load_dotenv(path: str | None = None) -> None:
    path = path or os.environ.get("TOXION_ENV_FILE", ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            if key and key not in os.environ:
                os.environ[key] = _parse_env_value(value)


def real_sha256() -> str:
    return os.environ.get("TOXION_REAL_SHA256", REAL_TOX21_SHA256).strip().lower()


def checkpoint_path() -> str:
    return os.environ.get("TOXION_CHECKPOINT", DEFAULT_CHECKPOINT)


def clean_path() -> str:
    return os.environ.get("TOXION_CLEAN_PATH", DEFAULT_CLEAN_PATH)


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def auto_retrain_enabled() -> bool:
    return os.environ.get("TOXION_AUTO_RETRAIN", "1").strip().lower() not in {"0", "false", "no", "off"}


def retrain_batch_size() -> int:
    return int(os.environ.get("TOXION_RETRAIN_BATCH", RETRAIN_BATCH_SIZE))


def fpr_cap() -> float | None:
    raw = os.environ.get("TOXION_FPR_CAP", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"TOXION_FPR_CAP must be a number such as 0.10, got {raw!r}") from None
