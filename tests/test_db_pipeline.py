"""Database tests: idempotent loads, rejects table, CHECK constraints, live forward test."""
import psycopg
import pytest

from volab import data, db, live, pipeline


def count(cfg, table):
    with db.connect(cfg) as conn:
        return db.query_df(conn, f"SELECT count(*) AS n FROM {table}")["n"].iloc[0]


def test_ingest_twice_is_idempotent(db_cfg):
    r1 = pipeline.ingest(db_cfg, full=True)
    n_clean, n_raw = count(db_cfg, "clean_prices"), count(db_cfg, "raw_prices")
    r2 = pipeline.ingest(db_cfg, full=True)
    assert count(db_cfg, "clean_prices") == n_clean == r1["inserted"]       # no duplicates
    assert r2["inserted"] == 0 and r2["updated"] == n_clean                  # same rows, updated in place
    assert count(db_cfg, "raw_prices") == 2 * n_raw                          # raw is append-only


def test_incremental_ingest_only_refetches_the_overlap(db_cfg):
    pipeline.ingest(db_cfg, full=True)
    r = pipeline.ingest(db_cfg, full=False)
    assert r["rows_fetched"] < 0.05 * count(db_cfg, "clean_prices")


def test_faults_land_in_rejects_with_reasons(db_cfg):
    r = pipeline.ingest(db_cfg, full=True, faults=True)
    assert r["rejects"] == 10 and r["flags"] == 3
    with db.connect(db_cfg) as conn:
        rej = db.query_df(conn, "SELECT rule_name, reason, row_json FROM rejects")
    assert rej["reason"].str.len().min() > 10 and rej["row_json"].notna().all()


def test_database_refuses_an_impossible_bar(db_cfg):
    with db.connect(db_cfg) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO clean_prices (ticker, trade_date, open, high, low, close, volume, source) "
                         "VALUES ('X', '2020-01-06', 10, 9, 11, 10, 1, 't')")


def test_live_forecasts_are_immutable_gated_and_scored(db_cfg):
    pipeline.ingest(db_cfg, full=True)
    with db.connect(db_cfg) as conn:
        dates = db.query_df(conn, "SELECT DISTINCT trade_date FROM clean_prices WHERE ticker='^NSEI' ORDER BY 1 DESC LIMIT 12")["trade_date"].tolist()
        freeze_day = sorted(dates)[0]
        version = live.freeze(db_cfg, conn, as_of=freeze_day)
        assert version.endswith(freeze_day.strftime("%Y%m%d"))

        # 1. before the freeze date: frozen models must be skipped (they have seen those days)
        early = live.forecast(db_cfg, conn, as_of=sorted(dates)[0] - (sorted(dates)[1] - sorted(dates)[0]) * 10)
        assert early["note"] and "skipped" in early["note"]

        # 2. on/after the freeze date they forecast; a second identical call writes nothing (immutable)
        first = live.forecast(db_cfg, conn, as_of=sorted(dates)[3])
        again = live.forecast(db_cfg, conn, as_of=sorted(dates)[3])
        assert first["written"] > 0 and again["written"] == 0

        # 3. scoring needs the 5-day outcome: the error must equal actual - predicted
        for d in sorted(dates)[:-1]:
            live.forecast(db_cfg, conn, as_of=d)
        n = live.score(db_cfg, conn)
        assert n > 0
        s = db.query_df(conn, "SELECT * FROM scores")
        assert ((s["actual_log_var"] - s["predicted_log_var"]) - s["error"]).abs().max() < 1e-9
        assert (s["is_backfill"]).all()                  # everything here was written after the fact
