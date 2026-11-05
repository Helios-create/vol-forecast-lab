"""Shared test fixtures.

DB tests need a PostgreSQL database named `volab_test` (see docs/EXECUTION.md). If it is not
reachable they are SKIPPED (not failed), so the pure-Python tests still run anywhere.
Override the URL with the TEST_DATABASE_URL environment variable.
"""
import copy
import os

import pytest

from volab import config, db, features, synth

TABLES = "raw_prices, clean_prices, rejects, data_flags, pipeline_runs, model_runs, forecasts, scores"


@pytest.fixture(scope="session")
def base_cfg(tmp_path_factory):
    c = config.load("config/demo.yaml")
    c["database_url"] = os.environ.get("TEST_DATABASE_URL", "postgresql://volab:volab@localhost:5432/volab_test")
    c["start_date"] = "2017-01-01"
    c["synthetic"] = {"end_date": "2020-12-31", "seed": 11}
    c["test_start"] = "2019-07-01"
    c["refit_every"] = 126
    c["models"] = {"garch": False}
    root = tmp_path_factory.mktemp("volab")
    c["raw_dir"], c["results_dir"], c["models_dir"] = str(root / "raw"), str(root / "results"), str(root / "models")
    return c


@pytest.fixture
def cfg(base_cfg):
    return copy.deepcopy(base_cfg)


@pytest.fixture(scope="session")
def prices(base_cfg):
    """Synthetic clean-ish prices (no faults) for pure-Python tests."""
    return synth.generate(base_cfg["tickers"], base_cfg["index_ticker"], base_cfg["start_date"],
                          base_cfg["synthetic"]["end_date"], seed=base_cfg["synthetic"]["seed"])


@pytest.fixture(scope="session")
def panel(prices, base_cfg):
    from volab.load import adjust_ohlc
    p = adjust_ohlc(prices)
    return features.build_panel(p, base_cfg["index_ticker"], base_cfg["tickers"], base_cfg["horizon"], base_cfg["ewma_lambda"])


@pytest.fixture
def db_cfg(cfg):
    try:
        db.init_db(cfg)
        with db.connect(cfg) as conn:
            conn.execute(f"TRUNCATE {TABLES}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL test database not reachable: {exc}")
    return cfg
