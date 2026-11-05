"""The ingest pipeline: fetch -> save raw -> validate -> adjust -> load.

    source (yfinance / synthetic)
        |  extract.py       retries, per-ticker failure isolation
        v
    data/raw/<source>_<batch>.csv     (replayable copy on disk)
        |  load.store_raw   append-only raw_prices table
        v
    validate.py             reject rules + flags
        |  load.adjust_ohlc split/dividend adjustment
        v
    clean_prices (UPSERT)   + rejects + data_flags + pipeline_runs

Everything database-related runs in ONE transaction: either the whole batch lands
or none of it does, so a crash halfway never leaves half a batch behind.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from . import db, extract, load, validate

OVERLAP_DAYS = 7  # re-fetch the last week on every run to pick up vendor revisions


def _starts(conn, cfg: dict, full: bool) -> dict[str, str]:
    """Per-ticker start date: full history for new tickers, last stored date minus overlap otherwise."""
    tickers = list(cfg["tickers"]) + [cfg["index_ticker"]]
    if full:
        return {t: cfg["start_date"] for t in tickers}
    last = db.query_df(conn, "SELECT ticker, max(trade_date) AS last FROM clean_prices GROUP BY ticker")
    last = dict(zip(last["ticker"], pd.to_datetime(last["last"])))
    out = {}
    for t in tickers:
        if t in last:
            out[t] = (last[t] - timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
        else:
            out[t] = cfg["start_date"]
    return out


def ingest(cfg: dict, full: bool = False, faults: bool = False) -> dict:
    """Run one ingest. Returns a summary dict (also printed by the CLI)."""
    started = datetime.now(timezone.utc)
    batch_id = extract.new_batch_id()
    source = cfg["data_source"]

    with db.connect(cfg) as conn:
        starts = _starts(conn, cfg, full)

        if source == "yfinance":
            raw, failed = extract.fetch_yfinance(starts, retries=cfg["fetch"]["retries"], pause=cfg["fetch"]["pause_seconds"])
        elif source == "synthetic":
            raw, failed = extract.fetch_synthetic(cfg, starts, faults=faults)
        else:
            raise ValueError(f"unknown data_source: {source}")

        if raw.empty:
            db.log_run(conn, "ingest", "failed", started, 0, 0, f"no data; failed={failed}")
            conn.commit()
            raise RuntimeError(f"No data fetched. Failures: {failed}")

        csv_path = extract.save_raw_csv(raw, cfg["raw_dir"], source, batch_id)  # replayable copy
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])

        load.store_raw(conn, raw, batch_id, source)
        result = validate.validate(
            raw, index_ticker=cfg["index_ticker"],
            extreme_move=cfg["validation"]["extreme_move"], stale_run=cfg["validation"]["stale_run"],
        )
        clean = load.adjust_ohlc(result.clean)
        counts = load.upsert_clean(conn, clean, source)
        n_rej = load.store_rejects(conn, result.rejects, batch_id)
        n_flag = load.store_flags(conn, result.flags, batch_id)

        status = "partial" if failed else "ok"
        msg = f"batch={batch_id} inserted={counts['inserted']} updated={counts['updated']} rejects={n_rej} flags={n_flag}"
        if failed:
            msg += f" failed_tickers={[t for t, _ in failed]}"
        db.log_run(conn, "ingest", status, started, len(raw), len(clean), msg)
        conn.commit()

    return {"batch_id": batch_id, "status": status, "rows_fetched": len(raw), "rows_clean": len(clean),
            **counts, "rejects": n_rej, "flags": n_flag, "failed": failed, "csv": str(csv_path),
            "validation": result.summary}
