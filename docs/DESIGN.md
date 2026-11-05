# DESIGN: Pre-registered evaluation plan

> **Rule:** fill in and `git commit` this file BEFORE you run `volab evaluate` on real data.
> The git timestamp is your proof that you fixed the plan before seeing any result. Do not edit
> the protocol sections afterwards. If you change your mind, add a dated "Amendment" at the bottom
> and say why. Delete these grey instructions (the lines starting with `>`) when you commit.

Author: Aditya Shreedhara  |  Date committed: ____-__-__  |  Commit hash: ________

## 1. Research question
Can machine learning (gradient boosting) beat simple statistical benchmarks (naive persistence, EWMA, HAR)
at forecasting the 5-day-ahead volatility of Indian large-cap stocks, under leakage-free walk-forward evaluation,
and in which market regimes does each model succeed or fail?

## 2. Hypotheses (write your honest prior BEFORE you look at results)
- H1: HAR is hard to beat. I expect: ______________________________________________
- H2: Gradient boosting helps most in: [ ] high-volatility periods  [ ] calm periods  [ ] nowhere  because ______________
- H3: Market-wide features (index volatility) add value for individual stocks: [ ] yes [ ] no [ ] unsure

## 3. Data
- Source: Yahoo Finance via yfinance, daily bars, split/dividend adjusted using adj_close / close.
- Universe: the 20 tickers in `config/config.yaml` plus the Nifty 50 index (`^NSEI`) for market features.
- Period: 2015-01-01 to the data available when I run the study.
- Known bias: survivorship (companies that are large today).
- Tickers removed after checking with `volab check-yfinance` (if any): ______________________

## 4. Target and features
- Daily variance proxy (Parkinson 1980): `pv = ln(High/Low)^2 / (4 ln 2)`, floored at 1e-8.
- Target: `y_t = log( mean(pv_{t+1..t+5}) )`, horizon h = 5 trading days. Forecast made at the close of day t.
- Features (past-only): log pv today, log 5-day mean, log 22-day mean, log EWMA(0.94), |return|, 5-day return,
  volume / 22-day average volume, index 5- and 22-day log variance, stock id (categorical, GBM only).

## 5. Models
1. naive: the trailing 5-day mean variance
2. ewma: exponentially weighted variance, decay 0.94
3. har: per-stock linear regression of y on log pv, log 5-day, log 22-day (Corsi 2009)
4. gbm: pooled HistGradientBoostingRegressor, hyperparameters chosen by the grid in `config.yaml`
   on a validation slice inside the training window only
5. garch: GARCH(1,1) on returns, scaled to the Parkinson scale with a training-window constant
- Ablations of the GBM: without market features, without the volume feature, HAR features only.

## 6. Validation protocol
- Test period starts: **2021-01-01**. Everything earlier is training/validation.
- Walk-forward: refit every **21** trading days; inside a block the models are frozen.
- Purge: training rows must have date <= (first test day of the block) minus 5 trading days, so that no training
  label overlaps the period being forecast.
- Scalers/encoders/hyperparameters are fitted on training data only. The test period influences no choice.
- A forecast is evaluated only if every model produced one for that stock-day (fair comparison).

## 7. Metrics and what counts as "better"
- Primary: RMSE of log variance, pooled and per stock. Secondary: MAE, QLIKE (variance-level loss), bias.
- Significance: Diebold-Mariano test per stock against HAR, squared-error loss, Newey-West (HAC) errors with 4 lags,
  Holm correction across stocks. A model is "significantly better" only if its Holm-adjusted p-value < 0.05.
- Regimes: high-vol = index 22-day log variance above the **75th percentile of the training period**.
- I will report all results including negative ones.

## 8. Live forward test
- Models frozen with `volab freeze`; forecasts written nightly with `volab live-run`; immutable once written.
- Backfilled forecasts are flagged and never presented as live evidence.

## 9. Analysis I will NOT do
No tuning on the test period. No dropping stocks or dates after seeing results. No claim of trading profit.

## Amendments
(date, what changed, why)
