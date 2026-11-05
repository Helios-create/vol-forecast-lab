-- SQL practice and data-health queries. Run in psql:   psql -h localhost -U volab -d volab -f sql/analysis_queries.sql
-- (password: volab).  Read each query, predict the output, THEN run it. These cover the SQL that
-- data-engineering and data-science interviews ask: GROUP BY, window functions, CTEs, joins, indexes.

-- 1. Rows per ticker and the date range we hold (GROUP BY, aggregate functions)
SELECT ticker, count(*) AS days, min(trade_date) AS first_day, max(trade_date) AS last_day
FROM clean_prices
GROUP BY ticker
ORDER BY ticker;

-- 2. Rejected rows by rule, per ticker (what is wrong with the data, and where?)
SELECT rule_name, ticker, count(*) AS n
FROM rejects
GROUP BY rule_name, ticker
ORDER BY n DESC;

-- 3. 7-day rolling average volume per ticker (WINDOW FUNCTION with a frame)
SELECT ticker, trade_date, volume,
       avg(volume) OVER (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS vol_7d
FROM clean_prices
WHERE ticker = (SELECT min(ticker) FROM clean_prices WHERE ticker NOT LIKE '^%')
ORDER BY trade_date DESC
LIMIT 20;

-- 4. The 10 widest-range days in the whole dataset (the most turbulent days)
SELECT ticker, trade_date, round((100 * (high - low) / close)::numeric, 2) AS range_pct
FROM clean_prices
WHERE ticker NOT LIKE '^%'
ORDER BY (high - low) / close DESC
LIMIT 10;

-- 5. Daily cross-sectional volatility rank (RANK window function): which stock was most volatile each day?
WITH pv AS (
    SELECT ticker, trade_date, power(ln(high / low), 2) / (4 * ln(2)) AS park_var
    FROM clean_prices
    WHERE ticker NOT LIKE '^%' AND high > low
)
SELECT trade_date, ticker, park_var,
       rank() OVER (PARTITION BY trade_date ORDER BY park_var DESC) AS vol_rank
FROM pv
ORDER BY trade_date DESC, vol_rank
LIMIT 20;

-- 6. Daily return and 22-day realised volatility with LAG (needs the previous row)
WITH r AS (
    SELECT ticker, trade_date, close,
           ln(close / lag(close) OVER (PARTITION BY ticker ORDER BY trade_date)) AS ret
    FROM clean_prices
)
SELECT ticker, trade_date, round((ret * 100)::numeric, 3) AS ret_pct,
       round((stddev_samp(ret) OVER (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 21 PRECEDING AND CURRENT ROW) * sqrt(252) * 100)::numeric, 2) AS vol22_annual_pct
FROM r
WHERE ticker NOT LIKE '^%'
ORDER BY trade_date DESC
LIMIT 20;

-- 7. Live scoreboard: RMSE per model, separating genuine live forecasts from backfilled ones
SELECT model_name, is_backfill, count(*) AS n,
       round(sqrt(avg(power(error, 2)))::numeric, 4) AS rmse, round(avg(error)::numeric, 4) AS bias
FROM scores
GROUP BY model_name, is_backfill
ORDER BY is_backfill, rmse;

-- 8. Data-quality dashboard in one query (UNION ALL of the three quality tables)
SELECT 'rejected' AS kind, rule_name, count(*) AS n FROM rejects   GROUP BY rule_name
UNION ALL
SELECT 'flagged'  AS kind, rule_name, count(*) AS n FROM data_flags GROUP BY rule_name
ORDER BY kind, n DESC;

-- 9. Why indexes matter: ask PostgreSQL how it executes a query (look for Seq Scan vs Index Scan)
EXPLAIN (ANALYZE)
SELECT avg(close) FROM clean_prices
WHERE ticker = (SELECT min(ticker) FROM clean_prices WHERE ticker NOT LIKE '^%')
  AND trade_date BETWEEN '2020-01-01' AND '2020-12-31';
