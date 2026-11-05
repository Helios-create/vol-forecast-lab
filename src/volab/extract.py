"""Data extraction: get raw daily bars from a source.

Two sources, same output format (long DataFrame):
    ticker, trade_date, open, high, low, close, adj_close, volume

  * yfinance  : real NSE data from Yahoo Finance (needs internet).
  * synthetic : generated offline by synth.py (development, tests, demo).

Failure handling (what "production-minded" means here)
  * Every network call is retried with exponential backoff (1s, 2s, 4s...).
  * One ticker failing never kills the whole run: it is reported in `failed`.
  * Every download is saved to disk BEFORE anything else happens, so if the
    database step fails you still have the data and can replay it.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import synth

COLUMNS = ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]


def _normalize_yf(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert one yfinance history() frame into our standard columns."""
    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)       # drop the timezone, keep the LOCAL (exchange) date
    adj = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
    return pd.DataFrame({
        "ticker": ticker,
        "trade_date": idx.normalize(),
        "open": hist["Open"].to_numpy(), "high": hist["High"].to_numpy(),
        "low": hist["Low"].to_numpy(), "close": hist["Close"].to_numpy(),
        "adj_close": adj.to_numpy(), "volume": hist["Volume"].to_numpy(),
    })


def fetch_yfinance(starts: dict[str, str], end: str | None = None, retries: int = 3, pause: float = 0.5):
    """Download daily bars per ticker. Returns (DataFrame, failed_tickers)."""
    import yfinance as yf  # imported here so synthetic runs and tests do not need it

    frames, failed = [], []
    for ticker, start in starts.items():
        for attempt in range(retries):
            try:
                hist = yf.Ticker(ticker).history(
                    start=start, end=end, auto_adjust=False, actions=False, raise_errors=False
                )
                if hist is None or hist.empty:
                    raise RuntimeError("empty response")
                frames.append(_normalize_yf(hist, ticker))
                break
            except Exception as exc:  # noqa: BLE001 - any failure -> retry, then report
                if attempt == retries - 1:
                    failed.append((ticker, str(exc)))
                else:
                    time.sleep(2 ** attempt)            # exponential backoff
        time.sleep(pause)                               # be polite to the free API
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return df, failed


def fetch_synthetic(cfg: dict, starts: dict[str, str], faults: bool = False):
    """Generate the (deterministic) synthetic history and cut it to the requested starts."""
    s = cfg.get("synthetic", {})
    full = synth.generate(cfg["tickers"], cfg["index_ticker"], cfg["start_date"], s.get("end_date", "2025-12-31"), seed=s.get("seed", 7))
    if faults:
        full, _, _ = synth.inject_faults(full)
    keep = [full[(full.ticker == t) & (full.trade_date >= pd.Timestamp(st))] for t, st in starts.items()]
    return pd.concat(keep, ignore_index=True), []


def save_raw_csv(df: pd.DataFrame, raw_dir: str, source: str, batch_id: str) -> Path:
    path = Path(raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{source}_{batch_id}.csv"
    df.to_csv(out, index=False)
    return out


def new_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
