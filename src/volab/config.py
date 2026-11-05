"""Configuration loading.

One YAML file drives everything. Relative paths in it are resolved against the
project root (the folder that contains `config/`), so the tool behaves the same
whether you run it from the project root, from cron, or from the tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config/config.yaml"


def project_root() -> Path:
    """Project root = two levels above this file (src/volab/config.py -> root)."""
    return Path(__file__).resolve().parents[2]


def load(path: str | None = None) -> dict:
    """Load a config file and resolve paths.

    Order of precedence for the file: explicit `path` > env VOLAB_CONFIG > default.
    The database URL can be overridden with the env var DATABASE_URL, so secrets
    never need to live in a file that is committed to git.
    """
    root = project_root()
    chosen = path or os.environ.get("VOLAB_CONFIG", DEFAULT_CONFIG)
    cfg_path = Path(chosen)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    cfg["_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)
    cfg["database_url"] = os.environ.get("DATABASE_URL", cfg["database_url"])
    for key in ("results_dir", "models_dir", "raw_dir"):
        cfg[key] = str((root / cfg[key]).resolve())

    # Fail early on a misconfigured protocol instead of producing silent nonsense.
    if cfg["horizon"] < 1:
        raise ValueError("horizon must be >= 1")
    if cfg["refit_every"] < 1:
        raise ValueError("refit_every must be >= 1")
    if cfg["index_ticker"] in cfg["tickers"]:
        raise ValueError("index_ticker must not also be listed in tickers")
    return cfg
