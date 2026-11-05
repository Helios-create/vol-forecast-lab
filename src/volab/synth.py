"""Synthetic market data, for offline development, tests and the demo.

WHY THIS EXISTS
  The real source (Yahoo Finance) needs internet and can fail or change. A synthetic
  generator lets you (1) run and test the entire system anywhere, (2) inject known
  data faults and check the validator catches exactly those, (3) know the TRUE
  answer (volatility here is persistent by construction, so a good model SHOULD
  beat naive persistence) which is a powerful correctness check for the pipeline.

WHAT IT MUST NEVER BE USED FOR
  Results on synthetic data say nothing about real markets. The CLI prints a
  banner and every row is stored with source='synthetic'. Never quote these numbers.

HOW THE DATA IS MADE
  1. A market-wide log-volatility state x_t follows an AR(1) process (mean reverting,
     persistent) with rare upward jumps (crisis episodes). This creates volatility
     clustering: calm periods and turbulent periods that last.
  2. Each stock has its own state z_t and loads on the market state.
     daily sigma_t = exp(mu_stock + 0.8 * x_t + z_t)
  3. Each day is simulated as an overnight gap plus an intraday random-walk path of
     `steps` increments. High/low are the path extremes, so the high-low range
     carries real information about sigma_t (that is what the Parkinson estimator uses).
  4. One stock gets a 1-for-2 stock split in the raw prices (adj_close stays
     continuous) so the adjustment code in load.py is exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHI_MARKET = 0.985   # persistence of the market volatility state (half-life ~ 46 days)
PHI_STOCK = 0.97


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float, jump_p: float = 0.0, jump: float = 0.0):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sd * rng.standard_normal()
        if jump_p and rng.random() < jump_p:
            x[t] += jump
    return x


def _ohlc_from_sigma(rng, sigma: np.ndarray, p0: float, steps: int):
    """Turn a daily volatility path into open/high/low/close (see module docstring, point 3)."""
    n = len(sigma)
    gap = rng.standard_normal(n) * 0.4 * sigma
    inc = rng.standard_normal((n, steps)) * (0.9 * sigma / np.sqrt(steps))[:, None]
    path = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    hi, lo, intraday = path.max(axis=1), path.min(axis=1), path[:, -1]
    log_close = np.log(p0) + np.cumsum(gap + intraday)
    log_open = log_close - intraday                       # open = close - intraday move
    o, h, l, c = np.exp(log_open), np.exp(log_open + hi), np.exp(log_open + lo), np.exp(log_close)
    # Floating-point rounding can leave close a hair outside [low, high] when the close IS the
    # extreme of the day (common in real data too). Enforce the bar invariant exactly.
    return o, np.maximum.reduce([h, o, c]), np.minimum.reduce([l, o, c]), c


def generate(tickers: list[str], index_ticker: str, start: str, end: str, seed: int = 7, steps: int = 40) -> pd.DataFrame:
    """Return a long DataFrame: ticker, trade_date, open, high, low, close, adj_close, volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    market = _ar1(rng, n, PHI_MARKET, 0.06, jump_p=0.002, jump=0.9)

    frames = []
    for i, ticker in enumerate([index_ticker] + list(tickers)):
        is_index = ticker == index_ticker
        z = _ar1(rng, n, PHI_STOCK, 0.04 if not is_index else 0.02)
        mu = np.log(0.008) if is_index else np.log(0.011 + 0.006 * rng.random())
        load = 0.9 if is_index else 0.8
        sigma = np.clip(np.exp(mu + load * market + z), 0.003, 0.15)
        o, h, l, c = _ohlc_from_sigma(rng, sigma, p0=100 + 1900 * rng.random(), steps=steps)
        base_vol = np.exp(rng.normal(14.0, 0.8))
        volume = np.round(base_vol * np.exp(0.6 * (np.log(sigma) - np.log(sigma).mean()) + 0.3 * rng.standard_normal(n)))
        df = pd.DataFrame({
            "ticker": ticker, "trade_date": dates,
            "open": o, "high": h, "low": l, "close": c, "adj_close": c.copy(), "volume": volume,
        })
        if is_index:
            df["volume"] = 0.0  # indices have no volume
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Simulate a 1-for-2 split for the 3rd stock, 60% of the way through the sample.
    if len(tickers) >= 3:
        split_ticker = tickers[2]
        m = out.ticker == split_ticker
        split_date = dates[int(n * 0.6)]
        after = m & (out.trade_date >= split_date)
        cols = ["open", "high", "low", "close"]
        out.loc[m, "adj_close"] = out.loc[m, "close"] * 0.5          # adjusted series: continuous, halved level
        out.loc[after, cols] = out.loc[after, cols] * 0.5            # raw prices halve on the split date
        out.loc[after, "adj_close"] = out.loc[after, "close"]        # after the split raw == adjusted
        out.loc[after, "volume"] = out.loc[after, "volume"] * 2
    return out


def inject_faults(df: pd.DataFrame, seed: int = 1):
    """Corrupt a COPY of the data in known ways. Returns (bad_df, expected).

    `expected` maps rule_name -> how many rows the validator should REJECT for it, or
    how many flags it should raise. Tests assert the validator finds exactly these.
    """
    rng = np.random.default_rng(seed)
    d = df.copy().reset_index(drop=True)
    stocks = [t for t in d.ticker.unique() if not t.startswith("^")]
    a, b, c, e, f = stocks[0], stocks[1], stocks[2 % len(stocks)], stocks[-1], stocks[3 % len(stocks)]

    def pick(ticker, k=1):
        idx = d.index[(d.ticker == ticker) & (d.trade_date > "2018-01-01")]
        return list(rng.choice(idx, size=k, replace=False))

    # 1. negative price -> non_positive_price (1 row)
    i = pick(a)[0]
    d.loc[i, "open"] = -5.0
    # 2. high below low (swap them) -> high_below_others and low_above_others (1 row each)
    i = pick(b)[0]
    hi_v, lo_v = d.loc[i, "high"], d.loc[i, "low"]
    d.loc[i, "high"], d.loc[i, "low"] = lo_v, hi_v
    # 3. null close -> null_values (1 row)
    i = pick(c)[0]
    d.loc[i, "close"] = np.nan
    # 4. duplicate key with different data -> duplicate_key (1 row: the earlier copy is rejected)
    i = pick(e)[0]
    dup = d.loc[[i]].copy()
    dup["close"] = dup["close"] * 1.0001
    dup["high"] = np.maximum(dup["high"], dup["close"])
    # 5. weekend row -> weekend_date (1 row)
    wk = d.loc[[pick(f)[0]]].copy()
    wk["trade_date"] = pd.Timestamp("2019-06-15")  # a Saturday
    wk["close"] = wk["close"] * 1.0
    # 6. stale run of 4 identical bars -> stale_ohlc (3 rows: all but the first)
    idx = pick(a)[0]
    j = d.index[(d.ticker == a)]
    pos = list(j).index(idx)
    run = list(j)[pos:pos + 4]
    for k in run[1:]:
        d.loc[k, ["open", "high", "low", "close", "adj_close"]] = d.loc[run[0], ["open", "high", "low", "close", "adj_close"]].values
    # 7. an extreme move (+40% for one day) -> extreme_move flag (2 flags: up, then back down)
    i = pick(b)[0]
    for col in ("open", "high", "low", "close", "adj_close"):
        d.loc[i, col] = d.loc[i, col] * 1.4
    # 8. a missing trading day -> missing_trading_day flag (1 flag)
    i = pick(e)[0]
    missing_date = d.loc[i, "trade_date"]
    d = d.drop(index=i)

    d = pd.concat([d, dup, wk], ignore_index=True)
    expected = {
        "non_positive_price": 1, "high_below_others": 1, "low_above_others": 2,  # the swapped row and the negative-price row
        "null_values": 1, "duplicate_key": 1, "weekend_date": 1, "stale_ohlc": 3,
        "extreme_move": 2, "missing_trading_day": 1,
    }
    return d, expected, {"missing_date": missing_date}
