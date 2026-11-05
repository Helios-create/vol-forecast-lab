"""Streamlit dashboard.   Run:  streamlit run app/dashboard.py [-- --config config/demo.yaml]

Four pages: Data health | Model comparison | Live scoreboard | Stock explorer.

It reads from PostgreSQL when it can. If the database is not reachable (for example when
the dashboard is deployed to Streamlit Community Cloud) it falls back to the CSV files
written by `volab export` and `volab evaluate`. Nothing here modifies any data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volab import config, db  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ARGS, _ = ap.parse_known_args()
CFG = config.load(ARGS.config)
RESULTS = Path(CFG["results_dir"])
SNAP = RESULTS / "snapshot"

st.set_page_config(page_title="Volatility Forecasting Lab", layout="wide")


@st.cache_data(ttl=60)
def table(name: str, sql: str) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, where_it_came_from). DB first, CSV snapshot as fallback."""
    try:
        with db.connect(CFG) as conn:
            return db.query_df(conn, sql), "database"
    except Exception:  # noqa: BLE001
        f = SNAP / f"{name}.csv"
        return (pd.read_csv(f), "snapshot CSV") if f.exists() else (pd.DataFrame(), "missing")


@st.cache_data(ttl=60)
def result(fname: str) -> pd.DataFrame:
    f = RESULTS / fname
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


st.title("Volatility Forecasting Lab")
if CFG["data_source"] == "synthetic":
    st.error("SYNTHETIC DATA: these numbers are not real and must never be quoted.")
st.caption("Forecasting the next 5 trading days' volatility of Indian large caps. A forecasting study, not a trading system, and not financial advice.")

page = st.sidebar.radio("Page", ["Data health", "Model comparison", "Live scoreboard", "Stock explorer"])

# ------------------------------------------------------------------ Data health
if page == "Data health":
    runs, src = table("pipeline_runs", "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 50")
    rej, _ = table("rejects_by_rule", "SELECT rule_name, count(*) AS n FROM rejects GROUP BY 1 ORDER BY 2 DESC")
    flg, _ = table("flags_by_rule", "SELECT rule_name, count(*) AS n FROM data_flags GROUP BY 1 ORDER BY 2 DESC")
    st.caption(f"source: {src}")
    if runs.empty:
        st.info("No pipeline runs yet. Run `volab ingest`.")
    else:
        last = runs[runs["step"] == "ingest"].head(1)
        c1, c2, c3 = st.columns(3)
        if not last.empty:
            c1.metric("Last ingest status", str(last["status"].iloc[0]))
            c2.metric("Rows loaded (last ingest)", f"{int(last['rows_out'].iloc[0]):,}")
            c3.metric("Last run finished", str(last["finished_at"].iloc[0])[:19])
        left, right = st.columns(2)
        left.subheader("Rejected rows by rule")
        left.bar_chart(rej.set_index("rule_name")["n"]) if not rej.empty else left.write("none")
        right.subheader("Flagged (kept) rows by rule")
        right.bar_chart(flg.set_index("rule_name")["n"]) if not flg.empty else right.write("none")
        st.subheader("Recent pipeline runs")
        st.dataframe(runs.drop(columns=["id"], errors="ignore"), width="stretch")

# ------------------------------------------------------------------ Model comparison
elif page == "Model comparison":
    overall, regime = result("metrics_overall.csv"), result("metrics_by_regime.csv")
    dmsum, imp, pred = result("dm_summary.csv"), result("gbm_importance.csv"), result("predictions.csv")
    if overall.empty:
        st.info("No results yet. Run `volab evaluate`.")
    else:
        bench = CFG["benchmark_model"]
        st.subheader("Out-of-sample accuracy (lower is better)")
        st.dataframe(overall.round(4), width="stretch")
        st.bar_chart(overall.set_index("model")["rmse"])
        st.subheader(f"Is it real? Diebold-Mariano tests vs {bench}")
        st.caption("Counts of stocks where a model is significantly better/worse than the benchmark. Trust the *_holm columns (multiple-testing corrected).")
        st.dataframe(dmsum, width="stretch")
        st.subheader("By market regime")
        st.dataframe(regime.round(4), width="stretch")
        if not imp.empty:
            st.subheader("What the gradient boosting model relies on")
            st.bar_chart(imp.set_index("feature")["mse_increase"])
        if not pred.empty:
            st.subheader("Rolling 63-day RMSE over time")
            pred["trade_date"] = pd.to_datetime(pred["trade_date"])
            cols = [c for c in ["naive", "ewma", "har", "gbm", "garch"] if c in pred.columns]
            roll = pd.DataFrame({c: (pred.assign(e=(pred["y"] - pred[c]) ** 2).groupby("trade_date")["e"].mean()
                                     .rolling(63).mean() ** 0.5) for c in cols})
            st.line_chart(roll.dropna())

# ------------------------------------------------------------------ Live scoreboard
elif page == "Live scoreboard":
    sc, src = table("scores", "SELECT * FROM scores ORDER BY as_of_date")
    fc, _ = table("forecasts", "SELECT * FROM forecasts")
    st.caption(f"source: {src}")
    if sc.empty:
        st.info("No scored forecasts yet. They appear after the nightly job has run for more than 5 trading days.")
    else:
        sc["is_backfill"] = sc["is_backfill"].astype(bool)
        live_n = int((~sc["is_backfill"]).sum())
        st.warning(f"{live_n} genuinely live scored forecasts, {int(sc['is_backfill'].sum())} backfilled. "
                   "Backfilled rows were written after the fact, so they are NOT proof of live performance. "
                   "With a small sample no conclusion is possible: treat this as a log, not a result.")
        which = st.radio("Show", ["Live only", "Backfilled only", "All"], horizontal=True)
        d = sc if which == "All" else sc[sc["is_backfill"] == (which == "Backfilled only")]
        if d.empty:
            st.info("Nothing in this selection yet.")
        else:
            s = d.groupby("model_name")["error"].agg(n="count", rmse=lambda e: float(np.sqrt(np.mean(e ** 2))), bias="mean").round(4)
            st.dataframe(s, width="stretch")
            d = d.assign(trade_date=pd.to_datetime(d["as_of_date"]), sq=d["error"] ** 2)
            by_day = d.groupby(["trade_date", "model_name"])["sq"].mean().unstack() ** 0.5
            st.line_chart(by_day)
    st.caption(f"Forecasts written so far: {len(fc):,}")

# ------------------------------------------------------------------ Stock explorer
else:
    prices, src = table("clean_prices", "SELECT ticker, trade_date, open, high, low, close, volume FROM clean_prices ORDER BY ticker, trade_date")
    if prices.empty:
        st.info("No price data found.")
    else:
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        ticker = st.selectbox("Stock", [t for t in prices["ticker"].unique() if t != CFG["index_ticker"]])
        g = prices[prices["ticker"] == ticker].set_index("trade_date")
        g = g.assign(vol_annualised_pct=np.sqrt(np.log(g["high"] / g["low"]) ** 2 / (4 * np.log(2)) * 252) * 100)
        c1, c2 = st.columns(2)
        c1.subheader("Close price (split-adjusted)")
        c1.line_chart(g["close"])
        c2.subheader("Daily range-based volatility (annualised %, 22-day average)")
        c2.line_chart(g["vol_annualised_pct"].rolling(22).mean())
        pred = result("predictions.csv")
        if not pred.empty:
            p = pred[pred["ticker"] == ticker].copy()
            p["trade_date"] = pd.to_datetime(p["trade_date"])
            st.subheader("Out-of-sample: realised vs forecast (log variance of the next 5 days)")
            cols = [c for c in ["y", "har", "gbm", "naive"] if c in p.columns]
            st.line_chart(p.set_index("trade_date")[cols])
