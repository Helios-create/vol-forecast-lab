# Volatility Forecasting Lab

A Python-based volatility forecasting system
with a PostgreSQL data pipeline, statistical
benchmarking and machine learning models.

> **Research question:** Can machine learning beat simple statistical benchmarks at forecasting
> 5-day-ahead stock volatility of Indian large caps, under leakage-free evaluation?

| Data engineering | Data science |
|---|---|
| Python extraction with retries and per-ticker failure isolation | Volatility proxy, target and feature engineering in pandas |
| Validation rules with a **rejects** table (Rejected records are stored separately for inspection) | Naive, EWMA, HAR, GARCH and gradient-boosting models |
| Idempotent PostgreSQL loads (`COPY` + `ON CONFLICT` upsert) | Walk-forward evaluation with a purge/embargo |
| Linux cron scheduling, logging, exit codes | Diebold-Mariano tests with HAC errors and Holm correction |
| Measured scale benchmark (INSERT vs batched vs COPY, index vs none) | Regime analysis, ablation, permutation importance |
| Tests (pytest) including a leakage test and DB tests | A live forward test that cannot be leaked |

## Status: what has and has not been verified

* **Verified end to end** on synthetic data in a real PostgreSQL 16 (37 tests pass; the whole `volab demo` runs).
* The **real Yahoo Finance path was not run** where this was built (no internet access to Yahoo there). It is covered by
  tests against a fake yfinance (formats, retries, failure isolation) and by checking the library's signature, but the
  first real run is yours. `volab check-yfinance` tells you in 5 seconds whether it works on your machine.
* Results on synthetic data prove the code works. They say **nothing** about real markets and must never be quoted.

## Quick start (details in `docs/EXECUTION.md`)

```bash
bash scripts/setup_wsl.sh          # once: PostgreSQL + virtualenv + packages (WSL2, Ubuntu 24.04)
source .venv/bin/activate
make test                          # expect: all tests pass
make demo                          # whole system on SYNTHETIC data, no internet needed
make dashboard-demo                # look at it in your browser

volab check-yfinance               # does the real data source work?
volab init-db
volab ingest --full                # REAL data: fetch -> validate -> load
volab evaluate --fast              # quick sanity run on real data
# commit docs/DESIGN.md (pre-registration), THEN:
make evaluate                      # the real study (10-25 minutes)
volab report                       # writes results/MODEL_CARD.md
volab freeze && volab live-run     # start the live forward test
```

## How it works

```
Yahoo Finance (yfinance)                   synthetic generator (offline demo/tests)
        |  extract.py   retries, exponential backoff, one bad ticker never kills the run
        v
 data/raw/*.csv   replayable copy on disk
        |  load.store_raw
        v
 raw_prices  (append-only, exactly what the source said)
        |  validate.py  reject rules + flags
        v
 rejects (why a row was refused)   data_flags (suspicious but kept)
        |  load.adjust_ohlc  split/dividend adjustment
        v
 clean_prices  (UPSERT: re-running never duplicates)
        |  features.py  Parkinson variance, target, features (past-only)
        v
 evaluate.py   walk-forward + purge -> naive | EWMA | HAR | GBM | GARCH
        |        metrics, Diebold-Mariano (HAC, Holm), regimes, ablation
        v
 results/*.csv  ->  report.py -> results/MODEL_CARD.md   ->  app/dashboard.py

 live.py (nightly): freeze models -> forecast (immutable) -> score when the outcome arrives
```

### Methodology
1. Daily volatility proxy from the high-low range (Parkinson 1980): `pv = ln(H/L)^2 / (4 ln 2)`.
2. Target: `y = log(mean pv over the next 5 trading days)`.
3. Models see only the past; every test forecast comes from a model fitted on labels that were already fully known.
4. A model only counts as better if a Diebold-Mariano test says so, after correcting for testing 20 stocks.

## Commands

| Command | What it does |
|---|---|
| `volab init-db` | create the tables (safe to repeat) |
| `volab check-yfinance` | fetch 10 days of one ticker to prove the data source works |
| `volab ingest [--full]` | fetch, validate, adjust, load. Default is incremental (last stored date minus 7 days) |
| `volab status` | row counts, rejects by rule, last pipeline runs |
| `volab evaluate [--ablation] [--fast]` | the modelling study. `--fast` = refit every 63 days, no GARCH |
| `volab report` | generate `results/MODEL_CARD.md` from the saved results |
| `volab freeze` | fit HAR + GBM on all labelled data, save as a versioned frozen model |
| `volab live-run` | write today's forecasts, score the ones whose outcome is now known |
| `volab live-backfill --days N` | recovery/demo: forecast past dates (flagged `is_backfill`) |
| `volab benchmark` | scale benchmark on synthetic rows |
| `volab export` | CSV snapshots of key tables (for the deployed dashboard) |
| `volab demo` | everything above on synthetic data (needs `--config config/demo.yaml`) |

## Repository map

```
config/config.yaml    real-data settings (tickers, dates, protocol)        config/demo.yaml   synthetic demo
sql/schema.sql        tables + CHECK constraints                           sql/analysis_queries.sql  SQL practice
src/volab/            config, db, synth, extract, validate, load, pipeline, features, models,
                      evaluate, report, live, benchmark, cli
app/dashboard.py      Streamlit dashboard (4 pages)
scripts/              setup_wsl.sh, run_nightly.sh, init_databases.sql
tests/                37 tests: validation rules, leakage, models/statistics, walk-forward purge, DB, live test, extraction
docs/                 EXECUTION.md (do this) | STUDY_GUIDE.md (understand this) | DESIGN.md (pre-registration)
```

## Limitations
* **Survivorship bias:** the 20 stocks are large caps *today*.
* One market, one volatility proxy (range-based, daily data), one horizon.
* Stocks are correlated, so per-stock significance tests are not independent.
* yfinance is an unofficial wrapper around Yahoo data: use for personal/educational work, expect occasional failures, and adjusted prices can be revised.
* Special sessions (for example Muhurat trading) can create odd bars; the weekend and flat-bar rules catch some, not all.
* The live test is only as long as the nightly job has been running. A few days of scores prove the mechanism, not skill.
* This is a forecasting study, **not a trading system and not financial advice.**

## Tests
`make test` runs everything. DB tests need the `volab_test` database (created by `scripts/setup_wsl.sh`);
without it they are skipped. The most important test is `tests/test_features_leakage.py`: it rewrites every price after
a cutoff date and asserts no feature before the cutoff changes. It was verified to fail when a leak is deliberately introduced.
