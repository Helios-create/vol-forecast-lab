"""Volatility proxy, target and features.

THE VOLATILITY PROXY (Parkinson, 1980)
    pv_t = ( ln(High_t / Low_t) )^2 / ( 4 ln 2 )
  A day's high-low range tells you how much the price moved around that day, using
  only data every price table already has. It is a far more efficient estimate of
  that day's variance than the squared close-to-close return (which uses 1 number
  where the range effectively uses the whole path).

THE TARGET
    y_t = log( mean(pv_{t+1}, ..., pv_{t+h}) )          (h = 5 trading days)
  "How volatile will the NEXT week be?" We forecast the LOG because variance is
  positive and heavily right-skewed; logs make errors more symmetric and make
  the HAR regression well behaved.

THE FEATURES (all use data up to and including day t, never later)
    lpv      log of today's pv
    lpv5     log of the mean pv of the last 5 days          (HAR "weekly" term)
    lpv22    log of the mean pv of the last 22 days         (HAR "monthly" term)
    lewma    log of an exponentially weighted pv (RiskMetrics decay lambda)
    abs_ret  |today's log return|
    ret5     sum of the last 5 log returns
    vol_ratio  today's volume / its 22-day average
    idx_lpv5, idx_lpv22   the same 5/22-day terms for the market index
    ticker_code  integer id of the stock (for the pooled ML model)

LEAKAGE RULE (the most important sentence in this file)
    Any feature at row t may only use information available at the close of day t.
    tests/test_features_leakage.py PROVES this by changing every price after day t
    and checking the features at day t do not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLOOR = 1e-8                       # avoids log(0) when a day's high == low (about a 0.02% range)
FEATURES_HAR = ["lpv", "lpv5", "lpv22"]
FEATURES_ALL = ["lpv", "lpv5", "lpv22", "lewma", "abs_ret", "ret5", "vol_ratio", "idx_lpv5", "idx_lpv22", "ticker_code"]
# Rows must have all of these to be usable by every model (vol_ratio may be NaN: the GBM handles it).
REQUIRED = ["lpv", "lpv5", "lpv22", "lewma", "abs_ret", "ret5", "idx_lpv5", "idx_lpv22"]


def parkinson_var(high, low):
    return np.log(high / low) ** 2 / (4 * np.log(2))


def _log(x):
    return np.log(np.maximum(x, FLOOR))


def build_panel(prices: pd.DataFrame, index_ticker: str, tickers: list[str], horizon: int = 5, ewma_lambda: float = 0.94) -> pd.DataFrame:
    """Turn clean prices (long format) into a modelling panel: one row per (ticker, date).

    Rows near the start (not enough history) have NaN features; rows near the end
    have NaN target (the future has not happened yet). Keep both kinds: the first
    are skipped when training, the second are exactly the rows we FORECAST live.
    """
    p = prices.sort_values(["ticker", "trade_date"]).copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p["pv"] = parkinson_var(p["high"], p["low"])

    idx = p[p.ticker == index_ticker].drop_duplicates("trade_date").set_index("trade_date")["pv"].sort_index()
    if idx.empty:
        raise ValueError(f"index ticker {index_ticker} has no data; run `volab ingest` first")
    idx_feat = pd.DataFrame({"idx_lpv5": _log(idx.rolling(5).mean()), "idx_lpv22": _log(idx.rolling(22).mean())})

    parts = []
    for ticker in tickers:
        g = p[p.ticker == ticker].reset_index(drop=True)
        if g.empty:
            continue
        pv = g["pv"]
        ret = np.log(g["close"]).diff()
        vmean = g["volume"].rolling(22).mean()
        out = pd.DataFrame({
            "ticker": ticker,
            "trade_date": g["trade_date"],
            "pv": pv,
            "ret": ret,
            "lpv": _log(pv),
            "lpv5": _log(pv.rolling(5).mean()),
            "lpv22": _log(pv.rolling(22).mean()),
            "lewma": _log(pv.ewm(alpha=1 - ewma_lambda, adjust=False).mean()),
            "abs_ret": ret.abs(),
            "ret5": ret.rolling(5).sum(),
            "vol_ratio": g["volume"] / vmean.where(vmean > 0),
            # mean(pv[t+1..t+h]) : rolling(h) at position t+h covers t+1..t+h, shift(-h) moves it back to t
            "y": _log(pv.rolling(horizon).mean().shift(-horizon)),
        })
        # The first 22 days of EWMA are a warm-up from a single observation: do not trust them.
        out.loc[out.index < 22, "lewma"] = np.nan
        parts.append(out)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(idx_feat, left_on="trade_date", right_index=True, how="left")
    codes = {t: i for i, t in enumerate(tickers)}
    panel["ticker_code"] = panel["ticker"].map(codes).astype(int)
    return panel


def usable(panel: pd.DataFrame, need_target: bool = True) -> pd.DataFrame:
    """Rows with every required feature (and the target, unless need_target=False)."""
    cols = REQUIRED + (["y"] if need_target else [])
    return panel[panel[cols].notna().all(axis=1)]
