"""Read clean prices back from PostgreSQL."""
from __future__ import annotations

import pandas as pd

from .db import query_df


def load_prices(conn, cfg: dict, upto=None) -> pd.DataFrame:
    """All clean bars for the configured tickers plus the index, optionally only up to a date."""
    tickers = list(cfg["tickers"]) + [cfg["index_ticker"]]
    sql = ("SELECT ticker, trade_date, open, high, low, close, volume FROM clean_prices "
           "WHERE ticker = ANY(%s)")
    params: list = [tickers]
    if upto is not None:
        sql += " AND trade_date <= %s"
        params.append(pd.Timestamp(upto).date())
    sql += " ORDER BY ticker, trade_date"
    df = query_df(conn, sql, params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df
