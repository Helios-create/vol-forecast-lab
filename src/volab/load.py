"""Loading data into PostgreSQL.

KEY IDEAS
  * raw_prices is append-only: a permanent record of exactly what the source said.
  * clean_prices is loaded with an UPSERT, which makes the load IDEMPOTENT:
    running the same ingest twice leaves the table identical (no duplicates).
    This is what makes the nightly job safe to retry after a failure.
  * Bulk loading uses COPY into a temporary staging table, then one INSERT ... SELECT.
    COPY is the fastest way into PostgreSQL (see `volab benchmark` for measured numbers).
"""
from __future__ import annotations

import pandas as pd

from .db import copy_df

CLEAN_COLS = ["ticker", "trade_date", "open", "high", "low", "close", "volume", "source"]


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Apply split/dividend adjustment to open/high/low/close.

    Yahoo's `adj_close` is the close adjusted for splits and dividends. The ratio
    adj_close / close is the adjustment factor for that day. Multiplying ALL FOUR
    prices by the same factor keeps each bar internally consistent (high stays
    >= low) and makes returns continuous across a split. Volume is left raw.
    If adj_close is missing or non-positive, the factor is 1 (no adjustment).
    """
    out = df.copy()
    factor = (out["adj_close"] / out["close"]).where(out["adj_close"] > 0, 1.0).fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col] * factor
    return out.drop(columns=["adj_close"])


def store_raw(conn, df: pd.DataFrame, batch_id: str, source: str) -> int:
    d = df.copy()
    d["batch_id"], d["source"] = batch_id, source
    cols = ["batch_id", "ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    return copy_df(conn, "raw_prices", cols, d)


def upsert_clean(conn, df: pd.DataFrame, source: str) -> dict:
    """Insert new rows and update changed ones. Returns {'inserted': n, 'updated': n}."""
    if df.empty:
        return {"inserted": 0, "updated": 0}
    d = df.copy()
    d["source"] = source
    with conn.cursor() as cur:
        # TEMP table: private to this connection, dropped automatically at commit.
        cur.execute("CREATE TEMP TABLE stg (LIKE clean_prices INCLUDING DEFAULTS) ON COMMIT DROP")
    copy_df(conn, "stg", CLEAN_COLS, d)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO clean_prices (ticker, trade_date, open, high, low, close, volume, source)
            SELECT ticker, trade_date, open, high, low, close, volume, source FROM stg
            ON CONFLICT (ticker, trade_date) DO UPDATE
               SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                   close = EXCLUDED.close, volume = EXCLUDED.volume,
                   source = EXCLUDED.source, updated_at = now()
            RETURNING (xmax = 0) AS inserted
            """
        )
        # `xmax = 0` is a PostgreSQL trick: it is true for a freshly INSERTED row and
        # false for a row that was UPDATED by the ON CONFLICT branch.
        flags = [r[0] for r in cur.fetchall()]
    return {"inserted": sum(flags), "updated": len(flags) - sum(flags)}


def store_rejects(conn, rejects: pd.DataFrame, batch_id: str) -> int:
    if rejects.empty:
        return 0
    d = rejects.copy()
    d["batch_id"] = batch_id
    return copy_df(conn, "rejects", ["batch_id", "ticker", "trade_date", "rule_name", "reason", "row_json"], d)


def store_flags(conn, flags: pd.DataFrame, batch_id: str) -> int:
    if flags.empty:
        return 0
    d = flags.copy()
    d["batch_id"] = batch_id
    return copy_df(conn, "data_flags", ["batch_id", "ticker", "trade_date", "rule_name", "detail"], d)
