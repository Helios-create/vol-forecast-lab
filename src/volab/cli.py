"""Command line interface:  volab <command> [options]

    volab init-db           create the PostgreSQL tables
    volab check-yfinance    quick test that the real data source works
    volab ingest            fetch -> validate -> load (the nightly data step)
    volab status            what is in the database right now
    volab evaluate          the modelling study (walk-forward, tests, regimes)
    volab report            write results/MODEL_CARD.md from the saved results
    volab freeze            fit and save the models used for LIVE forecasting
    volab live-run          write today's forecasts and score the ones that matured
    volab live-backfill     (recovery/demo) forecast for the last N past dates
    volab benchmark         scale benchmark on synthetic rows
    volab export            CSV snapshots for the dashboard / sharing
    volab demo              run everything on SYNTHETIC data (no internet needed)

Use  --config config/demo.yaml  (before the command) to switch to the demo setup.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import benchmark, config, data, db, evaluate, features, live, pipeline, report


def _banner(cfg):
    if cfg["data_source"] == "synthetic":
        print("*" * 70 + "\n*  SYNTHETIC DATA. Results are NOT real. Never put them on a resume.  *\n" + "*" * 70)


def cmd_init_db(cfg, a):
    db.init_db(cfg)
    print("Database initialised:", cfg["database_url"].split("@")[-1])


def cmd_check_yfinance(cfg, a):
    from . import extract
    ticker = a.ticker or cfg["tickers"][0]
    start = (pd.Timestamp.today() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    df, failed = extract.fetch_yfinance({ticker: start}, retries=2, pause=0)
    if failed or df.empty:
        print("FAILED:", failed or "empty result", "\nCheck your internet connection and the ticker symbol.")
        sys.exit(1)
    print(f"OK: {ticker}: {len(df)} rows, last date {df.trade_date.max().date()}")
    print(df.tail(3).to_string(index=False))


def cmd_ingest(cfg, a):
    _banner(cfg)
    r = pipeline.ingest(cfg, full=a.full, faults=a.faults)
    v = r["validation"]
    print(f"batch {r['batch_id']}  status={r['status']}")
    print(f"  fetched {r['rows_fetched']:,} rows  ->  clean {r['rows_clean']:,}  (inserted {r['inserted']:,}, updated {r['updated']:,})")
    print(f"  rejected rows logged: {r['rejects']}   flags logged: {r['flags']}")
    rules = {k: n for k, n in v.items() if k not in ("rows_in", "rows_clean", "flags") and n}
    if rules:
        print("  rejects by rule:", rules)
    if v["flags"]:
        print("  flags:", v["flags"])
    if r["failed"]:
        print("  FAILED TICKERS:", r["failed"])
    print("  raw copy saved:", r["csv"])


def cmd_status(cfg, a):
    _banner(cfg)
    with db.connect(cfg) as conn:
        for t in ("raw_prices", "clean_prices", "rejects", "data_flags", "forecasts", "scores", "model_runs"):
            n = db.query_df(conn, f"SELECT count(*) AS n FROM {t}")["n"].iloc[0]
            print(f"  {t:<14} {n:>10,} rows")
        rng = db.query_df(conn, "SELECT min(trade_date) a, max(trade_date) b, count(DISTINCT ticker) t FROM clean_prices")
        print("  clean data:", rng.iloc[0].to_dict())
        print(db.query_df(conn, "SELECT rule_name, count(*) AS n FROM rejects GROUP BY 1 ORDER BY 2 DESC").to_string(index=False))
        print(db.query_df(conn, "SELECT step, status, rows_in, rows_out, finished_at FROM pipeline_runs ORDER BY id DESC LIMIT 5").to_string(index=False))


def _panel(cfg):
    with db.connect(cfg) as conn:
        prices = data.load_prices(conn, cfg)
    return features.build_panel(prices, cfg["index_ticker"], cfg["tickers"], cfg["horizon"], cfg["ewma_lambda"])


def cmd_evaluate(cfg, a):
    _banner(cfg)
    if a.fast:
        cfg["refit_every"] = max(cfg["refit_every"], 63)
        cfg["models"]["garch"] = False
        print("FAST mode: refit every >=63 days, no GARCH. For the final numbers run without --fast.")
    t0 = datetime.now()
    panel = _panel(cfg)
    res = evaluate.evaluate_panel(panel, cfg, ablation=a.ablation)
    out = report.save_results(res, cfg, extra={"runtime_seconds": round((datetime.now() - t0).total_seconds(), 1), "ablation": a.ablation})
    pd.set_option("display.width", 160)
    print("\nOverall (lower RMSE is better):\n", res["overall"].round(4).to_string(index=False))
    print("\nSignificant improvements over", cfg["benchmark_model"], "(Holm-corrected counts):\n", res["dm_summary"].to_string(index=False))
    print("\nBy regime:\n", res["by_regime"].round(4).to_string(index=False))
    print("\nSaved to", out)


def cmd_report(cfg, a):
    _banner(cfg)
    print("Wrote", report.write_model_card(cfg))


def cmd_freeze(cfg, a):
    with db.connect(cfg) as conn:
        v = live.freeze(cfg, conn, as_of=getattr(a, "as_of", None))
    print("Frozen model version:", v)


def cmd_live_run(cfg, a):
    _banner(cfg)
    with db.connect(cfg) as conn:
        started = datetime.now(timezone.utc)
        f = live.forecast(cfg, conn)
        s = live.score(cfg, conn)
        db.log_run(conn, "forecast+score", "ok", started, f["written"], s, str(f))
        conn.commit()
    print("Forecasts:", f, "| newly scored:", s)


def cmd_live_backfill(cfg, a):
    _banner(cfg)
    with db.connect(cfg) as conn:
        n = live.backfill(cfg, conn, a.days)
        s = live.score(cfg, conn)
    print(f"Backfilled {n} forecasts (flagged is_backfill), newly scored: {s}")


def cmd_benchmark(cfg, a):
    sizes = [int(x) for x in a.sizes.split(",")]
    print("Scale benchmark on SYNTHETIC rows:", sizes)
    df = benchmark.run(cfg, sizes)
    print("\nSaved:", Path(cfg["results_dir"]) / "benchmark.csv")
    print(df.round(3).to_string(index=False))


def cmd_export(cfg, a):
    out = Path(cfg["results_dir"]) / "snapshot"
    out.mkdir(parents=True, exist_ok=True)
    with db.connect(cfg) as conn:
        queries = {
            "pipeline_runs": "SELECT * FROM pipeline_runs ORDER BY id",
            "rejects_by_rule": "SELECT rule_name, count(*) AS n FROM rejects GROUP BY 1 ORDER BY 2 DESC",
            "flags_by_rule": "SELECT rule_name, count(*) AS n FROM data_flags GROUP BY 1 ORDER BY 2 DESC",
            "forecasts": "SELECT * FROM forecasts ORDER BY as_of_date, model_name, ticker",
            "scores": "SELECT * FROM scores ORDER BY as_of_date, model_name, ticker",
            "clean_prices": "SELECT ticker, trade_date, open, high, low, close, volume FROM clean_prices ORDER BY ticker, trade_date",
        }
        for name, sql in queries.items():
            df = db.query_df(conn, sql)
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"  {name}.csv  {len(df):,} rows")
    print("Snapshot in", out)


def cmd_demo(cfg, a):
    if cfg["data_source"] != "synthetic":
        print("The demo only runs with --config config/demo.yaml (synthetic data). Refusing to touch real data.")
        sys.exit(1)
    step = lambda s: print(f"\n=== {s} " + "=" * (60 - len(s)))  # noqa: E731
    step("1. init database"); cmd_init_db(cfg, a)
    with db.connect(cfg) as conn:   # start the demo from a clean slate
        conn.execute("TRUNCATE raw_prices, clean_prices, rejects, data_flags, pipeline_runs, model_runs, forecasts, scores")
    step("2. ingest with injected faults (watch the validator work)"); cmd_ingest(cfg, argparse.Namespace(full=True, faults=True))
    step("3. ingest again, no faults (idempotent: same rows, no duplicates)"); cmd_ingest(cfg, argparse.Namespace(full=True, faults=False))
    step("4. evaluate (fast)"); cmd_evaluate(cfg, argparse.Namespace(fast=True, ablation=True))
    step("5. model card"); cmd_report(cfg, a)
    with db.connect(cfg) as conn:
        past = db.query_df(conn, "SELECT DISTINCT trade_date FROM clean_prices WHERE ticker = %s ORDER BY trade_date DESC LIMIT 16",
                           (cfg["index_ticker"],))["trade_date"].tolist()
    step(f"6. freeze models as if today were {min(past)} (simulates having deployed them 15 days ago)")
    cmd_freeze(cfg, argparse.Namespace(as_of=str(min(past))))
    step("7. simulate 15 live days (flagged as backfill), forecast today, then score")
    cmd_live_backfill(cfg, argparse.Namespace(days=15)); cmd_live_run(cfg, a)
    step("8. export snapshot"); cmd_export(cfg, a)
    step("9. status"); cmd_status(cfg, a)
    print("\nDemo finished. Look in", cfg["results_dir"], "and start the dashboard:  streamlit run app/dashboard.py -- --config config/demo.yaml")


def main(argv=None):
    p = argparse.ArgumentParser(prog="volab", description="Volatility Forecasting Lab")
    p.add_argument("--config", default=None, help="path to a YAML config (default config/config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    c = sub.add_parser("check-yfinance"); c.add_argument("--ticker"); c.set_defaults(fn=cmd_check_yfinance)
    c = sub.add_parser("ingest"); c.add_argument("--full", action="store_true", help="re-fetch the full history")
    c.add_argument("--faults", action="store_true", help="(synthetic only) inject known data faults"); c.set_defaults(fn=cmd_ingest)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    c = sub.add_parser("evaluate"); c.add_argument("--fast", action="store_true"); c.add_argument("--ablation", action="store_true"); c.set_defaults(fn=cmd_evaluate)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    c = sub.add_parser("freeze"); c.add_argument("--as-of", default=None, help="(demo) pretend today is this date"); c.set_defaults(fn=cmd_freeze)
    sub.add_parser("live-run").set_defaults(fn=cmd_live_run)
    c = sub.add_parser("live-backfill"); c.add_argument("--days", type=int, default=15); c.set_defaults(fn=cmd_live_backfill)
    c = sub.add_parser("benchmark"); c.add_argument("--sizes", default="100000,1000000,5000000"); c.set_defaults(fn=cmd_benchmark)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    a.fn(config.load(a.config), a)


if __name__ == "__main__":
    main()
