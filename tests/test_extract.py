"""Extraction tests with a FAKE yfinance (no internet needed): formats, retries, failure isolation."""
import sys
import types

import pandas as pd
import pytest

from volab import extract


def fake_history(n=5, tz="Asia/Kolkata", adj=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz=tz)
    df = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Volume": 100.0}, index=idx)
    if adj:
        df["Adj Close"] = 10.0
    return df


def test_normalize_drops_timezone_and_keeps_local_date():
    out = extract._normalize_yf(fake_history(), "X.NS")
    assert list(out.columns) == extract.COLUMNS
    assert out["trade_date"].dt.tz is None and out["trade_date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert (out["adj_close"] == 10.0).all()


def test_normalize_without_adj_close_falls_back_to_close():
    out = extract._normalize_yf(fake_history(adj=False), "X.NS")
    assert (out["adj_close"] == out["close"]).all()


def _install_fake_yf(monkeypatch, behaviour):
    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **kw):
            return behaviour(self.sym)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def behaviour(sym):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary")
        return fake_history()

    _install_fake_yf(monkeypatch, behaviour)
    df, failed = extract.fetch_yfinance({"A.NS": "2024-01-01"}, retries=3, pause=0)
    assert failed == [] and len(df) == 5 and calls["n"] == 3


def test_one_bad_ticker_does_not_kill_the_run(monkeypatch):
    def behaviour(sym):
        return pd.DataFrame() if sym == "BAD.NS" else fake_history()

    _install_fake_yf(monkeypatch, behaviour)
    df, failed = extract.fetch_yfinance({"A.NS": "2024-01-01", "BAD.NS": "2024-01-01", "B.NS": "2024-01-01"}, retries=2, pause=0)
    assert [t for t, _ in failed] == ["BAD.NS"]
    assert set(df["ticker"]) == {"A.NS", "B.NS"}
