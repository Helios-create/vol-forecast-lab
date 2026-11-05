-- Volatility Forecasting Lab: database schema.
-- Safe to run many times (IF NOT EXISTS everywhere). Run with: volab init-db
--
-- Layers (data only ever flows downward):
--   raw_prices   -> exactly what the source returned, append-only, never edited
--   clean_prices -> validated + split/dividend-adjusted, one row per (ticker, day)
--   rejects      -> rows that FAILED a rule, with the rule name and the original row
--   data_flags   -> rows kept but suspicious (big moves, missing days) for human review
--   forecasts / scores / model_runs / pipeline_runs -> modelling and operations logs

CREATE TABLE IF NOT EXISTS raw_prices (
    id          BIGSERIAL PRIMARY KEY,
    batch_id    TEXT        NOT NULL,          -- one id per ingest run
    ticker      TEXT        NOT NULL,
    trade_date  DATE        NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    adj_close   DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    source      TEXT        NOT NULL,          -- 'yfinance' or 'synthetic'
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS raw_prices_key_idx ON raw_prices (ticker, trade_date);

-- The CHECK constraints are the SECOND line of defence: even if validate.py had a
-- bug, the database itself refuses an impossible bar.
CREATE TABLE IF NOT EXISTS clean_prices (
    ticker      TEXT        NOT NULL,
    trade_date  DATE        NOT NULL,
    open        DOUBLE PRECISION NOT NULL CHECK (open  > 0),
    high        DOUBLE PRECISION NOT NULL CHECK (high  > 0),
    low         DOUBLE PRECISION NOT NULL CHECK (low   > 0),
    close       DOUBLE PRECISION NOT NULL CHECK (close > 0),
    volume      DOUBLE PRECISION NOT NULL CHECK (volume >= 0),
    source      TEXT        NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date),
    CHECK (high >= low),
    CHECK (high >= open AND high >= close),
    CHECK (low  <= open AND low  <= close)
);
CREATE INDEX IF NOT EXISTS clean_prices_date_idx ON clean_prices (trade_date);

CREATE TABLE IF NOT EXISTS rejects (
    id          BIGSERIAL PRIMARY KEY,
    batch_id    TEXT        NOT NULL,
    ticker      TEXT,
    trade_date  DATE,
    rule_name   TEXT        NOT NULL,
    reason      TEXT        NOT NULL,
    row_json    JSONB,                          -- the original row, so nothing is lost
    rejected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rejects_rule_idx ON rejects (rule_name);

CREATE TABLE IF NOT EXISTS data_flags (
    id          BIGSERIAL PRIMARY KEY,
    batch_id    TEXT        NOT NULL,
    ticker      TEXT,
    trade_date  DATE,
    rule_name   TEXT        NOT NULL,
    detail      TEXT,
    flagged_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          BIGSERIAL PRIMARY KEY,
    step        TEXT        NOT NULL,           -- ingest | forecast | score | freeze ...
    status      TEXT        NOT NULL,           -- ok | partial | failed
    rows_in     INTEGER,
    rows_out    INTEGER,
    message     TEXT,
    started_at  TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id         BIGSERIAL PRIMARY KEY,
    model_name     TEXT        NOT NULL,
    model_version  TEXT        NOT NULL,
    params_json    JSONB,
    train_end_date DATE,
    git_commit     TEXT,
    data_source    TEXT,
    metrics_json   JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A forecast is written ONCE per (model, version, ticker, as_of_date, horizon) and never
-- changed (ON CONFLICT DO NOTHING). That immutability is what makes the live test honest.
CREATE TABLE IF NOT EXISTS forecasts (
    model_name        TEXT        NOT NULL,
    model_version     TEXT        NOT NULL,
    ticker            TEXT        NOT NULL,
    as_of_date        DATE        NOT NULL,
    horizon_days      INTEGER     NOT NULL,
    predicted_log_var DOUBLE PRECISION NOT NULL,
    is_backfill       BOOLEAN     NOT NULL DEFAULT FALSE,  -- TRUE if made after the fact
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_name, model_version, ticker, as_of_date, horizon_days)
);

CREATE TABLE IF NOT EXISTS scores (
    model_name        TEXT        NOT NULL,
    model_version     TEXT        NOT NULL,
    ticker            TEXT        NOT NULL,
    as_of_date        DATE        NOT NULL,
    horizon_days      INTEGER     NOT NULL,
    predicted_log_var DOUBLE PRECISION NOT NULL,
    actual_log_var    DOUBLE PRECISION NOT NULL,
    error             DOUBLE PRECISION NOT NULL,   -- actual - predicted
    is_backfill       BOOLEAN     NOT NULL DEFAULT FALSE,
    scored_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_name, model_version, ticker, as_of_date, horizon_days)
);
