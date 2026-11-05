"""Scale benchmark: how do loading methods and indexes behave as the table grows?

This answers the interview question "your data is small, how would this handle 230 million rows?"
with MEASURED numbers from YOUR machine instead of guesses. All data here is SYNTHETIC
(random numbers) and lives in a scratch table `bench_prices` that is dropped afterwards.

What is measured
  1. Loading speed (rows/second) for three ways of getting rows into PostgreSQL:
       a. one INSERT statement per row        (the naive way)
       b. batched inserts via executemany     (psycopg pipelines many statements)
       c. COPY                                (PostgreSQL's bulk path)
     a and b run on a smaller sample (they are slow); all three are compared on the SAME sample size,
     then COPY is also timed on the full size.
  2. Query time for "average close per ticker in a date range" with and without an index,
     taken from EXPLAIN (ANALYZE), which asks PostgreSQL to actually run the query and time it.
"""
from __future__ import annotations

import io
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import db

COLS = "ticker_id, trade_date, open, high, low, close, volume"


def _synthetic_chunk(n: int, seed: int, start_id: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2000-01-03")
    close = 100 + rng.standard_normal(n).cumsum() * 0.1
    return pd.DataFrame({
        "ticker_id": rng.integers(0, 500, n),
        "trade_date": base + pd.to_timedelta(rng.integers(0, 9000, n), unit="D"),
        "open": close * 0.99, "high": close * 1.02, "low": close * 0.98, "close": close,
        "volume": rng.integers(1_000, 5_000_000, n).astype(float),
    })


def _timeit(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _explain_ms(conn, sql: str) -> float:
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, TIMING OFF) " + sql)
        text = "\n".join(r[0] for r in cur.fetchall())
    m = re.search(r"Execution Time: ([\d.]+) ms", text)
    return float(m.group(1)) if m else float("nan")


def run(cfg: dict, sizes: list[int], small: int = 20_000, log=print) -> pd.DataFrame:
    rows = []
    with db.connect(cfg) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_prices")
        conn.execute("CREATE TABLE bench_prices (ticker_id INT, trade_date DATE, open DOUBLE PRECISION, "
                     "high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION)")
        conn.commit()

        # ---- 1. loading speed on a small sample, three methods
        sample = _synthetic_chunk(small, seed=1)
        tuples = list(db._to_python_rows(sample, COLS.split(", ")))
        ins = f"INSERT INTO bench_prices ({COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s)"

        def row_by_row():
            with conn.cursor() as cur:
                for t in tuples:
                    cur.execute(ins, t)

        def batched():
            with conn.cursor() as cur:
                cur.executemany(ins, tuples)

        def copy_small():
            db.copy_df(conn, "bench_prices", COLS.split(", "), sample)

        for name, fn in (("one INSERT per row", row_by_row), ("executemany (batched)", batched), ("COPY", copy_small)):
            conn.execute("TRUNCATE bench_prices")
            secs = _timeit(fn)
            conn.commit()
            rows.append({"test": "load", "method": name, "rows": small, "seconds": secs, "rows_per_sec": small / secs})
            log(f"  load {name:<24} {small:>9,} rows  {secs:7.2f}s  {small / secs:>12,.0f} rows/s")

        # ---- 2. COPY at full sizes + queries with/without index
        for n in sizes:
            conn.execute("DROP INDEX IF EXISTS bench_idx")
            conn.execute("TRUNCATE bench_prices")
            conn.commit()
            t0 = time.perf_counter()
            done, chunk = 0, 1_000_000
            while done < n:
                part = _synthetic_chunk(min(chunk, n - done), seed=100 + done)
                db.copy_df(conn, "bench_prices", COLS.split(", "), part)
                done += len(part)
            conn.commit()
            secs = time.perf_counter() - t0
            rows.append({"test": "load", "method": "COPY (full size)", "rows": n, "seconds": secs, "rows_per_sec": n / secs})
            log(f"  load COPY (full size){'':<4} {n:>9,} rows  {secs:7.2f}s  {n / secs:>12,.0f} rows/s")

            q = ("SELECT ticker_id, avg(close) FROM bench_prices "
                 "WHERE ticker_id = 42 AND trade_date BETWEEN '2010-01-01' AND '2012-12-31' GROUP BY ticker_id")
            conn.execute("ANALYZE bench_prices")
            ms_no = _explain_ms(conn, q)
            conn.execute("CREATE INDEX bench_idx ON bench_prices (ticker_id, trade_date)")
            conn.execute("ANALYZE bench_prices")
            ms_idx = _explain_ms(conn, q)
            conn.commit()
            rows.append({"test": "query", "method": "no index", "rows": n, "seconds": ms_no / 1000, "rows_per_sec": np.nan})
            rows.append({"test": "query", "method": "index (ticker_id, trade_date)", "rows": n, "seconds": ms_idx / 1000, "rows_per_sec": np.nan})
            log(f"  query at {n:>9,} rows: no index {ms_no:8.1f} ms | with index {ms_idx:8.1f} ms")

        conn.execute("DROP TABLE IF EXISTS bench_prices")
        conn.commit()

    df = pd.DataFrame(rows)
    df["milliseconds"] = df["seconds"] * 1000
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    (out / "benchmark_machine.json").write_text(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                                                            "note": "SYNTHETIC random data; timings are machine-specific"}, indent=2))
    return df
