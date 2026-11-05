"""Thin PostgreSQL helpers (psycopg 3).

Why so thin? SQL is the interesting part of this project, so the code stays close
to it: plain SQL strings, parameters passed separately (never f-strings, which
would invite SQL injection), and COPY for bulk loading.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg


def connect(cfg: dict) -> psycopg.Connection:
    """Open a connection. Use as a context manager: commit on success, rollback on error."""
    return psycopg.connect(cfg["database_url"])


def init_db(cfg: dict) -> None:
    """Create every table (idempotent: safe to run again)."""
    sql = Path(cfg["_root"], "sql", "schema.sql").read_text()
    with connect(cfg) as conn:
        conn.execute(sql)  # no parameters -> psycopg allows several statements in one call


def query_df(conn: psycopg.Connection, sql: str, params=None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _to_python_rows(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Convert a DataFrame slice to plain Python tuples that psycopg can send.

    - pandas Timestamps become datetime.date (the DB column is DATE)
    - NaN/NaT/None become None (NULL in SQL)
    - numpy numbers become Python numbers
    """
    out = df[cols].copy()
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    out = out.astype(object).where(out.notna(), None)
    return list(out.itertuples(index=False, name=None))


def copy_df(conn: psycopg.Connection, table: str, cols: list[str], df: pd.DataFrame) -> int:
    """Bulk-load a DataFrame with COPY (much faster than row-by-row INSERT; see benchmark)."""
    if df.empty:
        return 0
    rows = _to_python_rows(df, cols)
    stmt = f"COPY {table} ({', '.join(cols)}) FROM STDIN"
    with conn.cursor() as cur, cur.copy(stmt) as cp:
        for row in rows:
            cp.write_row(row)
    return len(rows)


def log_run(conn, step: str, status: str, started_at, rows_in=None, rows_out=None, message=None) -> None:
    """Record one pipeline step in pipeline_runs (shown on the dashboard's Data health page)."""
    conn.execute(
        "INSERT INTO pipeline_runs (step, status, rows_in, rows_out, message, started_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (step, status, rows_in, rows_out, message, started_at),
    )
