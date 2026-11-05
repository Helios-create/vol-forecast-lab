# # Research Methodology

This document describes the research question, dataset,
forecasting models, validation strategy and statistical
evaluation methodology used in Vol Forecast Lab.

## 1. Research question
Can machine learning (gradient boosting) beat simple statistical benchmarks (naive persistence, EWMA, HAR)
at forecasting the 5-day-ahead volatility of Indian large-cap stocks, under leakage-free walk-forward evaluation,
and in which market regimes does each model succeed or fail?

## 2. Research Hypotheses
- H1: HAR provides a strong benchmark for 5-day-ahead volatility forecasting.
- H2: Gradient boosting may perform differently across high-volatility and normal market regimes.
- H3: Market-wide volatility features may improve forecasting performance for individual stocks.

## 3. Data
- Source: Yahoo Finance via yfinance, daily bars, split/dividend adjusted using adj_close / close.
- Universe: the 20 tickers in `config/config.yaml` plus the Nifty 50 index (`^NSEI`) for market features.
- Period: 2015-01-01 to the data available when I run the study.
- Known bias: survivorship (companies that are large today).
- Ticker availability: Pending verification.

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

## 6. Validation Protocol

- **Test period:** Begins on 2021-01-01. All earlier observations are used for training and validation.
- **Walk-forward evaluation:** Models are refitted every 21 trading days and remain fixed within each test block.
- **Purging:** Training observations are restricted to dates at least 5 trading days before the first test date of each block. This prevents overlap between training labels and the forecast period.
- **Data preprocessing:** Scalers, encoders and hyperparameters are fitted using training data only. Test-period observations are excluded from model selection.
- **Fair comparison:** Evaluation is restricted to stock-day observations for which every model produces a forecast.

## 7. Evaluation Metrics and Statistical Testing
- **Primary metric:** RMSE of log variance, calculated across all observations and separately for each stock.
- **Secondary metrics:** MAE, QLIKE (variance-level loss) and forecast bias.
- **Statistical significance:** Diebold-Mariano tests are conducted per stock against HAR using squared-error loss and Newey-West HAC standard errors with 4 lags. Holm correction is applied across stocks. A model is considered significantly better when its Holm-adjusted p-value is below 0.05.
- **Market regimes:** High-volatility periods are identified when the index 22-day log variance exceeds the 75th percentile calculated from the training period.
- All results, including negative results, are reported.

## 8. Live Forward Testing

- Models are frozen using `volab freeze`, and forecasts are generated nightly using `volab live-run`.
- Generated forecasts remain immutable after being recorded.
- Backfilled forecasts are explicitly flagged and excluded from live performance evidence.

## 9. Evaluation Constraints
The evaluation follows these constraints:

- No hyperparameter tuning using the test period.
- No removal of stocks or dates based on observed test results.
- No claims of trading profitability based solely on forecasting performance.


