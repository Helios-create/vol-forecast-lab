"""The live forward test.

A backtest can be fooled (leakage, tuning on the test set, lucky choices). A LIVE test
cannot: the forecast is written down BEFORE the outcome exists, then scored when it
arrives. This module does three things:

  freeze    fit the final HAR and GBM on all labelled data and save them to disk
            with a version (git commit + training end date). Frozen models are NOT retrained
            until you deliberately freeze a new version.
  forecast  for the latest trading day, write each model's forecast into `forecasts`.
            Rows are immutable (ON CONFLICT DO NOTHING).
  score     for every forecast whose 5-day outcome is now known, write the error into `scores`.

Run order each night (scripts/run_nightly.sh):  ingest -> forecast -> score
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import models
from .data import load_prices
from .db import query_df
from .features import FEATURES_ALL, REQUIRED, build_panel, usable
from .report import git_commit

PARAM_FREE = {"naive": models.predict_naive, "ewma": models.predict_ewma}


def _panel(conn, cfg, upto=None):
    prices = load_prices(conn, cfg, upto)
    return build_panel(prices, cfg["index_ticker"], cfg["tickers"], cfg["horizon"], cfg["ewma_lambda"])


def _saved_gbm_params(cfg: dict) -> dict:
    p = Path(cfg["results_dir"]) / "run_summary.json"
    if p.exists():
        return json.loads(p.read_text())["gbm_params"]
    return models.DEFAULT_GBM


def freeze(cfg: dict, conn, as_of=None) -> str:
    """Fit HAR + GBM on all rows whose label is known, save them, register the version.

    `as_of` (demo/testing) pretends today is an earlier date: only data up to then is used.
    The version records `data_end`, the last date of data the models were allowed to see.
    A frozen model may only forecast dates >= data_end (see forecast()), otherwise it would be
    forecasting days it has already seen.
    """
    panel = _panel(conn, cfg, upto=as_of)
    train = usable(panel)
    train_end = pd.Timestamp(train["trade_date"].max())
    data_end = pd.Timestamp(panel["trade_date"].max())
    version = f"{git_commit(cfg['_root'])}-{data_end:%Y%m%d}"
    params = _saved_gbm_params(cfg)

    out = Path(cfg["models_dir"]) / version
    out.mkdir(parents=True, exist_ok=True)
    coefs = models.fit_har(train)
    (out / "har.json").write_text(json.dumps({k: v.tolist() for k, v in coefs.items()}))
    joblib.dump(models.fit_gbm(train, params, FEATURES_ALL), out / "gbm.joblib")
    (out / "meta.json").write_text(json.dumps({"version": version, "train_end": str(train_end.date()),
                                                "data_end": str(data_end.date()), "n_train": len(train), "gbm_params": params}, indent=2))
    (Path(cfg["models_dir"]) / "LATEST").write_text(version)

    with conn.cursor() as cur:
        for name, prm in (("har", None), ("gbm", params)):
            cur.execute(
                "INSERT INTO model_runs (model_name, model_version, params_json, train_end_date, git_commit, data_source) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (name, version, json.dumps(prm) if prm else None, train_end.date(), git_commit(cfg["_root"]), cfg["data_source"]),
            )
    conn.commit()
    return version


def _latest_version(cfg: dict) -> str | None:
    p = Path(cfg["models_dir"]) / "LATEST"
    return p.read_text().strip() if p.exists() else None


def forecast(cfg: dict, conn, as_of=None) -> dict:
    """Write forecasts for one date (default: the newest date in the data)."""
    started = datetime.now(timezone.utc)
    newest = query_df(conn, "SELECT max(trade_date) AS d FROM clean_prices")["d"].iloc[0]
    panel = _panel(conn, cfg, upto=as_of)
    stocks = panel[panel["ticker"].isin(cfg["tickers"])]
    as_of_date = pd.Timestamp(stocks["trade_date"].max())
    cur_rows = stocks[stocks["trade_date"] == as_of_date]
    cur_rows = cur_rows[cur_rows[REQUIRED].notna().all(axis=1)]
    if cur_rows.empty:
        return {"as_of": str(as_of_date.date()), "written": 0, "note": "no usable rows"}
    is_backfill = bool(as_of_date.date() < pd.Timestamp(newest).date())

    preds: dict[tuple[str, str], np.ndarray] = {(n, "v1"): f(cur_rows) for n, f in PARAM_FREE.items()}
    version = _latest_version(cfg)
    skipped = None
    if version:
        d = Path(cfg["models_dir"]) / version
        data_end = pd.Timestamp(json.loads((d / "meta.json").read_text())["data_end"])
        if as_of_date < data_end:
            # This model already saw data after as_of_date: forecasting it would be leakage.
            version, skipped = None, f"frozen models skipped: as_of {as_of_date.date()} is before their data_end {data_end.date()}"
    if version:
        coefs = {k: np.array(v) for k, v in json.loads((d / "har.json").read_text()).items()}
        preds[("har", version)] = models.predict_har(coefs, cur_rows)
        preds[("gbm", version)] = joblib.load(d / "gbm.joblib").predict(cur_rows[FEATURES_ALL])

    written = 0
    with conn.cursor() as cur:
        for (name, ver), values in preds.items():
            for ticker, v in zip(cur_rows["ticker"], values):
                if not np.isfinite(v):
                    continue
                cur.execute(
                    "INSERT INTO forecasts (model_name, model_version, ticker, as_of_date, horizon_days, predicted_log_var, is_backfill) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (name, ver, ticker, as_of_date.date(), cfg["horizon"], float(v), is_backfill),
                )
                written += cur.rowcount
    conn.commit()
    return {"as_of": str(as_of_date.date()), "written": written, "backfill": is_backfill, "frozen_version": version, "note": skipped}


def score(cfg: dict, conn) -> int:
    """Score every forecast whose outcome is now known. Returns the number of new scores."""
    unscored = query_df(conn, """
        SELECT f.model_name, f.model_version, f.ticker, f.as_of_date, f.horizon_days, f.predicted_log_var, f.is_backfill
        FROM forecasts f
        LEFT JOIN scores s ON s.model_name = f.model_name AND s.model_version = f.model_version
                          AND s.ticker = f.ticker AND s.as_of_date = f.as_of_date AND s.horizon_days = f.horizon_days
        WHERE s.ticker IS NULL""")
    if unscored.empty:
        return 0
    panel = _panel(conn, cfg)
    labels = panel[["ticker", "trade_date", "y"]].dropna()
    unscored["as_of_date"] = pd.to_datetime(unscored["as_of_date"])
    m = unscored.merge(labels, left_on=["ticker", "as_of_date"], right_on=["ticker", "trade_date"])
    n = 0
    with conn.cursor() as cur:
        for r in m.itertuples(index=False):
            cur.execute(
                "INSERT INTO scores (model_name, model_version, ticker, as_of_date, horizon_days, predicted_log_var, "
                "actual_log_var, error, is_backfill) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (r.model_name, r.model_version, r.ticker, r.as_of_date.date(), int(r.horizon_days),
                 float(r.predicted_log_var), float(r.y), float(r.y - r.predicted_log_var), bool(r.is_backfill)),
            )
            n += cur.rowcount
    conn.commit()
    return n


def backfill(cfg: dict, conn, days: int) -> int:
    """Demo/recovery helper: forecast for each of the last `days` dates (flagged is_backfill=TRUE).

    Use it to recover after the nightly job missed a few days. NEVER present backfilled forecasts as
    live evidence: the dashboard separates them.
    """
    dates = query_df(conn, "SELECT DISTINCT trade_date FROM clean_prices WHERE ticker = %s ORDER BY trade_date DESC LIMIT %s",
                     (cfg["index_ticker"], days))["trade_date"].tolist()
    total = 0
    for d in sorted(dates)[:-1]:     # the newest date is the normal live run, not a backfill
        total += forecast(cfg, conn, as_of=d)["written"]
    return total
