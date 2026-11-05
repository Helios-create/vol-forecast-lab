# STUDY GUIDE: understand every part

How to use this: read a part, open the file it names, read the code, then do the "Check yourself" box.
Parts 1-2 take about 2 hours, 3-4 about 3 hours, 5 is interview practice.

---
## Part 0. Vocabulary you must own
* **Volatility / variance:** how much a price moves. Variance = volatility squared. We forecast variance on a log scale.
* **OHLCV:** a day's Open, High, Low, Close and Volume.
* **Adjusted price:** history rescaled for splits and dividends so that a 1-for-2 split does not look like a -50% crash.
* **ETL:** Extract (get data), Transform (clean/validate), Load (store). Ours: `extract.py`, `validate.py`/`load.adjust_ohlc`, `load.py`.
* **Idempotent:** running it twice gives the same state as running it once. Essential for jobs that retry after failures.
* **Upsert:** INSERT, but if the key already exists UPDATE instead. In SQL: `INSERT ... ON CONFLICT (...) DO UPDATE`.
* **Data leakage:** letting a model, feature or tuning decision use information that would not have been available at forecast time. The classic way student projects look better than they are.
* **Walk-forward validation:** train on the past, test on the next block, move forward, repeat. The honest way to test time-series models.
* **Purge / embargo:** dropping training rows whose labels overlap the period you are about to forecast.
* **Benchmark:** a simple model you must beat to justify a complex one.

---
## Part 1. The data layer

### `sql/schema.sql`: how the database is organised
Open it. Notice the **layers**: `raw_prices` (exactly what the source said, append-only) -> `clean_prices` (validated and adjusted)
with `rejects` and `data_flags` beside it. Why keep raw? Because if you later find a bug in validation, you can re-derive
clean data from raw instead of re-downloading. Why `rejects` with a copy of the original row (`row_json`)? So nothing is ever
silently deleted: every refusal is explainable.
* `PRIMARY KEY (ticker, trade_date)` on `clean_prices` creates a B-tree index on that pair: it is what makes "one stock, a date range" queries fast, and it is what `ON CONFLICT` uses to detect duplicates.
* The `CHECK` constraints are a **second line of defence**: even if `validate.py` had a bug, PostgreSQL itself refuses a bar with high < low (`tests/test_db_pipeline.py` proves it).

### `extract.py`
* `fetch_yfinance`: one ticker at a time, `retries` attempts with exponential backoff (`sleep(2 ** attempt)` = 1s, 2s, 4s). A ticker that still fails goes into `failed`; the others continue. That isolation is a core production habit.
* `_normalize_yf`: converts Yahoo's timezone-aware index into a plain local date. A subtle bug source: a UTC conversion could shift Indian dates by a day.
* Every download is written to `data/raw/*.csv` **before** anything touches the database, so a failed load can be replayed.
* `fetch_synthetic` reuses the same output format, which is why the whole system can be tested offline.

### `validate.py`
One rule = one boolean mask over all rows (True = bad). Read `reject_masks`. Then ask yourself, for each rule, "could a real, correct bar ever break this?":
| Rule | Reject or flag? | Why |
|---|---|---|
| high < max(open, close, low), low > min(open, close, high), prices <= 0, nulls | reject | physically impossible |
| duplicate (ticker, date) | reject the earlier row | keep the latest revision |
| weekend date | reject | NSE normally trades Mon-Fri (special sessions are an edge case) |
| 3+ identical bars in a row | reject days 2..n | filler data, not trading |
| high == low and volume == 0 | reject | filler row for a day with no trading |
| daily move > 20% | **flag only** | real crashes and results days exist; deleting them would make volatility look calmer than it was |
| index traded but the stock has no row | flag | you want to know, but you cannot invent data |
`validate()` returns three things: clean rows, rejects (one row per bad row **per broken rule**) and flags.
**Check yourself:** why does a negative price also trigger `low_above_others`? (Because `min(open, close, high)` is the negative number and low is above it. One broken row can break several rules.)

### `load.py`
* `adjust_ohlc`: factor = `adj_close / close`; multiply open, high, low, close by it. All four use the same factor, so each bar stays internally consistent. Volume is not adjusted here.
* `upsert_clean`: (1) `COPY` the batch into a TEMP staging table, (2) one `INSERT ... SELECT ... ON CONFLICT DO UPDATE`. COPY is PostgreSQL's bulk path, far faster than one INSERT per row (your `volab benchmark` measures by how much). `RETURNING (xmax = 0)` is a PostgreSQL trick: `xmax = 0` means the row was freshly inserted rather than updated, which lets us report `inserted` vs `updated` counts.

### `pipeline.py`
Everything that touches the database runs in **one transaction** (`with db.connect(cfg) as conn:`): either the whole batch lands or none of it. `_starts` decides the incremental window: for each ticker, last stored date minus 7 days (vendors sometimes revise recent bars; re-fetching a week and upserting corrects them without duplicating).

**Check yourself (data layer)**
1. Run `volab --config config/demo.yaml ingest --full` twice. What changes in `clean_prices`? In `raw_prices`? Why?
2. Add a deliberately bad row to a copy of a raw CSV and predict which rule fires.
3. Explain "idempotent" using `ON CONFLICT`.

---
## Part 2. Features and the target (`features.py`)

**Parkinson variance.** For a price following Brownian motion with daily variance sigma^2, `E[ ln(High/Low)^2 ] = 4 ln2 * sigma^2`. Dividing by `4 ln 2` therefore gives an (approximately) unbiased estimate of that day's variance from just the high and low. It is more efficient than a squared return, which uses a single number (close vs previous close).
Caveats you should state: it assumes no overnight gaps and continuous trading, so it understates variance on gap days.

**The target and the shift trick.** `pv.rolling(5).mean()` at row `i` averages rows `i-4 .. i`. We want, at row `t`, the average of `t+1 .. t+5`. That is the rolling value at row `t+5`, so `.shift(-5)` moves it back to row `t`. The last 5 rows have no target (the future has not happened), which is correct: they are exactly the rows we forecast live. `tests/test_features_leakage.py::test_target_is_log_of_mean_variance_over_next_h_days` pins this down by hand.

**Why log?** Variance is positive and right-skewed; the log makes errors roughly symmetric, turns multiplicative errors into additive ones, and makes the HAR regression behave.

**Features** (all computed from data up to and including day t; read the docstring at the top of the file). `lpv5` and `lpv22` are the HAR "weekly" and "monthly" terms. `lewma` is an exponentially weighted average. The index features give the market backdrop.

**The leakage test.** `test_features_at_t_do_not_change_when_the_future_changes` rebuilds the panel after rewriting every bar after a cutoff date (new levels, new ranges, new volumes) and asserts all features at or before the cutoff are identical. Note the comment in `_mutate_future`: simply multiplying a bar by a constant would not change `ln(High/Low)`, so the first version of the test would have passed even with a leak. We also confirmed the test fails when a deliberate leak (`rolling(..., center=True)`) is introduced. **A test you have never seen fail is not yet a test.**

**Check yourself**
1. What would go wrong if `y` used `shift(-4)`? (An off-by-one: the target would include day t itself, which the features already contain.)
2. Why is `FLOOR = 1e-8` needed, and what happens to a day with `High == Low`?

---
## Part 3. The models (`models.py`)

All models predict the same quantity: the log of next-5-day mean variance.

| Model | In one line | Idea |
|---|---|---|
| naive | `lpv5` | tomorrow-ish looks like the recent past (volatility is persistent) |
| ewma | `lewma` | the same, with exponentially fading memory (RiskMetrics uses 0.94) |
| har | OLS on `lpv, lpv5, lpv22` per stock | volatility has memory at several horizons (day, week, month); the standard benchmark |
| gbm | HistGradientBoosting, one pooled model | can capture non-linearities and interactions; stock id as categorical |
| garch | GARCH(1,1) on returns | a classic parametric model; uses returns only |

**HAR.** `y = b0 + b1*lpv + b2*lpv5 + b3*lpv22`. `fit_har` solves it with `np.linalg.lstsq` (ordinary least squares). `tests/test_models.py::test_har_recovers_known_coefficients` simulates data with known coefficients and checks they are recovered.

**GBM.** Boosting builds many shallow trees, each correcting the previous ones' errors. Risks: overfitting (hence a validation slice, shallow trees, `min_samples_leaf`, L2 regularisation) and extrapolation (trees predict within the range seen in training: a crisis bigger than anything in training will be under-forecast). `select_gbm_params` picks hyperparameters using only the last 20% of the *training* dates, with the same purge as the walk-forward.
`permutation_importance` shuffles one feature at a time on the validation slice and measures how much the error grows: a model-agnostic measure of reliance.

**GARCH(1,1).** Conditional variance recursion (code: `garch_avg_forecast`):
`s[t] = omega + alpha * r[t-1]^2 + beta * s[t-1]`. k days ahead the expected variance decays toward the long-run level `V = omega / (1 - alpha - beta)` as `(alpha+beta)^(k-1)`; we average k = 1..5. `test_garch_forecast_matches_the_arch_library` checks our hand-written recursion equals the `arch` package's forecast for the same parameters. Because returns-based variance and range-based variance differ by a scale, we multiply by a constant estimated on the **training window only**.

**Check yourself**
1. Why one pooled GBM but a separate HAR per stock? (Pooling gives the GBM more rows to learn non-linear structure; per-stock HAR has only 4 parameters, so little data is needed.)
2. Which model do you expect to win on a calm year, and which in March 2020? Write the guess in DESIGN.md before looking.

---
## Part 4. Evaluation (`evaluate.py`): the heart of the project

**Why not a random train/test split?** Volatility is persistent and targets overlap, so a random split puts almost the same information in train and test; the model looks great and is useless. Time order must be respected.

**Walk-forward with a purge** (read the docstring at the top):
```
... training rows ... | gap of h days | test block (21 days) | next block ...
                      ^ cutoff = t0 - h trading days
```
Why the gap? The label at date `t` averages variance over `t+1..t+5`. If `t` is within 5 days of the first test day `t0`, its label contains information from the test block. **Purging** removes those rows. `walk_forward` computes `cutoff = calendar[pos[t0] - h]`, and `tests/test_evaluate.py` asserts `pos[cutoff] + h <= pos[first_test_date]` for every fold.

**Metrics.** With `e = y - yhat` (log scale): RMSE and MAE are the usual; bias is the mean of e (systematic over/under-forecasting).
**QLIKE** is the standard loss for variance forecasts: `L = actual_var/forecast_var - ln(actual_var/forecast_var) - 1`. Since `actual_var/forecast_var = exp(e)`, the code is `exp(e) - e - 1`. It is zero only when the forecast is exact and, unlike squared error, it penalises under-forecasting variance more than over-forecasting, which is what risk managers care about.

**Diebold-Mariano (DM) test.** Question: is model A's lower loss real or luck? Let `d_t = loss_bench(t) - loss_model(t)`. Test whether `mean(d) = 0`. Because a 5-day target overlaps with the next day's, the errors are autocorrelated (a moving average of order h-1 = 4), so ordinary standard errors are far too small and produce false "significant" results. **HAC (Newey-West) standard errors** with `h-1 = 4` lags correct that. In code: regress `d` on a constant with `cov_type="HAC"`; the t-statistic of the constant is the DM statistic. `test_dm_has_roughly_correct_size_with_overlapping_errors` checks that, with no true difference, a 5% test rejects about 5% of the time.

**Holm correction.** You run one test per stock per model (20+ tests). At the 5% level you would expect one false positive by chance. Holm sorts p-values and compares the k-th smallest to `alpha / (m - k + 1)`, controlling the chance of *any* false positive. Trust the `*_holm` columns. (Stocks are correlated, so even Holm is only approximate: say so.)

**Regimes.** `add_regime` labels each test day `high_vol` if the index's 22-day log variance exceeds the 75th percentile **of the training period**. (Computing the threshold on the test period would leak.) Typical story: ML helps when volatility is unusual or non-linear; simple models are hard to beat in calm times.

**Ablation.** Re-run the GBM with feature groups removed (`gbm_no_market`, `gbm_no_volume`, `gbm_har_only`). If removing market features does not hurt, the market features do not help.

### How to read your result files
| File | Look for |
|---|---|
| `metrics_overall.csv` | RMSE/MAE/QLIKE per model; `rmse_vs_har` is the % change relative to HAR (negative = better) |
| `dm_summary.csv` | per model: in how many stocks significantly better/worse than HAR, raw and Holm-corrected |
| `dm_tests.csv` | the per-stock detail behind it |
| `metrics_by_regime.csv` | does the ranking change in `high_vol`? |
| `gbm_importance.csv` | which features the GBM relies on |
| `folds.csv` | proof of the purge (cutoff vs first test date) |
| `predictions.csv` | every out-of-sample forecast: plot `y` vs models yourself |

### What to write in your Interpretation (three honest outcomes)
1. **GBM clearly better (Holm-significant in most stocks):** say by how much, in which regime, and that its features (ablation) explain why; note extrapolation risk.
2. **Mixed (better in high-vol, not elsewhere):** this is the most interesting finding; explain a mechanism (non-linear response to market stress).
3. **HAR not beaten:** report it plainly: "a 4-parameter regression was not beaten by gradient boosting." That is a legitimate, respected result and shows you did not p-hack.
Never write "the model predicts the market"; it forecasts *volatility*, not direction.

---
## Part 5. Live test, scale, tests, and interview practice

### Live forward test (`live.py`)
`freeze` saves a versioned model (`<git>-<data_end>`) and records `data_end`, the last date it saw. `forecast` writes one row per (model, version, stock, date, horizon) with `ON CONFLICT DO NOTHING`, so a forecast can never be edited after the fact. **Frozen models are only allowed to forecast dates >= data_end** (otherwise they would be forecasting days they already saw; `test_live_forecasts_are_immutable_gated_and_scored` checks this). `score` joins forecasts to the realised target once 5 days have passed. `is_backfill` marks forecasts written after the fact; they are shown separately and never presented as live evidence.

### Scale benchmark (`benchmark.py`)
Measures rows/second for one-INSERT-per-row vs `executemany` vs `COPY` on the same sample, COPY at larger sizes, and a range query with and without an index via `EXPLAIN (ANALYZE)`. Use *your measured numbers*. For 230 million rows reason from them: bulk load via COPY in chunks, index after loading, partition by date, keep the working set in memory, analyse after big loads.

### The test suite
Run `python -m pytest -v` and read the names: they are the project's specification. Exercise: break something on purpose (change `shift(-h)` to `shift(-h+1)`; remove the purge; use `center=True`) and watch which test fails.

### Interview questions with answer outlines
1. **Why forecast volatility, not price?** Prices are close to unpredictable; volatility is persistent (clusters), so it is forecastable, and it drives risk, position sizing and option pricing.
2. **What is data leakage and where could it creep in here?** Using future information. Candidates: centered rolling windows, scaling on the full sample, tuning on the test period, training labels overlapping the forecast period, regime thresholds from the test period. Each is prevented (and tested or documented).
3. **Why walk-forward and what is the purge?** Time order matters; targets overlap h days, so rows within h days of the test block carry its information; remove them.
4. **What is HAR and why a good benchmark?** Linear regression on daily/weekly/monthly average variance; few parameters, hard to beat, standard in the literature.
5. **How do you know a model really beats the benchmark?** Diebold-Mariano test with HAC errors (overlapping forecast errors are autocorrelated), Holm correction across stocks.
6. **What does a p-value mean?** Probability of seeing a difference at least this large if the models were truly equally accurate. Not the probability that the model is better.
7. **What is the multiple-testing problem?** Many tests => some will look significant by chance; Holm controls it.
8. **How is the pipeline idempotent, and what if a run fails halfway?** Upsert with a primary key; the database work is in one transaction, so a failed run leaves nothing half-loaded, and rerunning is safe.
9. **Why COPY?** One bulk operation instead of a round trip per row; measured X times faster on my machine.
10. **How do you treat outliers and bad data?** Reject only the impossible (audit trail in `rejects`), flag the suspicious (extreme moves can be real), never silently drop.
11. **How would this handle 230 million rows?** COPY in chunks, composite index on (ticker, date) built after loading, partition by date range, `ANALYZE`, and measure; my benchmark shows the index effect.
12. **What would you monitor in production?** Data freshness and row counts, reject rates by rule, feature drift, rolling forecast error versus benchmark, job failures.
13. **What are the limits of this study?** Survivorship bias, one market, a range-based proxy, correlated stocks, a short live sample, unofficial data source.
14. **Did you build this yourself?** Answer truthfully and specifically: it was developed with AI assistance; you ran it on real data, tested it (including breaking it on purpose), studied each module, and extended it with [your extension]. Then prove it by explaining any line they point at.

---
## Study schedule (about 8-10 hours total)
Day 1: Parts 0-1 + demo. Day 2: Part 2 + break the leakage test. Day 3: Part 3 + run HAR by hand on a small sample in a notebook. Day 4: Part 4 with your real results open. Day 5: Part 5, practise all 14 answers aloud.
