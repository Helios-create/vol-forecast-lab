<!-- EXAMPLE OUTPUT from `make demo` on SYNTHETIC data. Shows what the generated model card looks like. Not real results. -->
# Model Card: 5-day-ahead volatility forecasting

> **SYNTHETIC DATA. These numbers are NOT real and must never appear on a resume.**

_Generated 2026-10-29T00:14:58+00:00 from commit `nogit`. Facts below are produced by the code; the
**Interpretation** section is yours to write._

## Setup
- Data source: **synthetic**, 6 stocks, test period 2021-01-01 to 2025-12-24 (7794 out-of-sample stock-days)
- Target: log of the mean Parkinson variance over the next 5 trading days
- Walk-forward: refit every 63 trading days; training labels purged by 5 days; test starts 2021-01-01
- Benchmark for significance tests: **har**
- GBM hyperparameters (chosen inside the training window): `{'learning_rate': 0.05, 'max_depth': 3, 'max_iter': 150, 'min_samples_leaf': 50, 'l2_regularization': 1.0}`

## Accuracy (lower is better; `rmse_vs_har` is the % change in RMSE relative to har)
| model | rmse | mae | qlike | bias | n | rmse_vs_har |
|---|---|---|---|---|---|---|
| naive | 0.5012 | 0.3964 | 0.1349 | -0.0050 | 7794 | 7.1063 |
| ewma | 0.5300 | 0.4183 | 0.1411 | -0.1065 | 7794 | 13.2632 |
| har | 0.4679 | 0.3674 | 0.1237 | 0.0204 | 7794 | 0.0000 |
| gbm | 0.4909 | 0.3816 | 0.1445 | 0.0453 | 7794 | 4.9144 |
| gbm_no_market | 0.4928 | 0.3840 | 0.1456 | 0.0434 | 7794 | 5.3176 |
| gbm_no_volume | 0.4973 | 0.3864 | 0.1480 | 0.0426 | 7794 | 6.2729 |
| gbm_har_only | 0.4950 | 0.3856 | 0.1472 | 0.0487 | 7794 | 5.7933 |

Lowest RMSE overall: **har** (0.4679).

## Is the difference real? Diebold-Mariano tests vs har (per stock, squared-error loss, HAC errors)
`better_raw` = number of stocks where the model is significantly better at 5% without correction; `better_holm` = after Holm
multiple-testing correction (the number to trust).
| model | tickers | better_raw | worse_raw | better_holm | worse_holm |
|---|---|---|---|---|---|
| ewma | 6 | 0 | 6 | 0 | 6 |
| gbm | 6 | 0 | 2 | 0 | 1 |
| gbm_har_only | 6 | 0 | 6 | 0 | 6 |
| gbm_no_market | 6 | 0 | 6 | 0 | 1 |
| gbm_no_volume | 6 | 0 | 4 | 0 | 1 |
| naive | 6 | 0 | 4 | 0 | 4 |

## Where does each model do well or badly? (market regimes)
High-volatility regime threshold (index 22-day log variance, training 75th percentile): -10.141
| regime | model | rmse | mae | qlike | bias | n | rmse_vs_har |
|---|---|---|---|---|---|---|---|
| high_vol | naive | 0.5154 | 0.4061 | 0.1368 | -0.0521 | 2772 | 3.6525 |
| high_vol | ewma | 0.5872 | 0.4629 | 0.1661 | -0.1492 | 2772 | 18.0896 |
| high_vol | har | 0.4972 | 0.3883 | 0.1427 | 0.0389 | 2772 | 0.0000 |
| high_vol | gbm | 0.5660 | 0.4392 | 0.2027 | 0.0525 | 2772 | 13.8420 |
| high_vol | gbm_no_market | 0.5639 | 0.4381 | 0.2064 | 0.1078 | 2772 | 13.4194 |
| high_vol | gbm_no_volume | 0.5778 | 0.4489 | 0.2097 | 0.0453 | 2772 | 16.2020 |
| high_vol | gbm_har_only | 0.5667 | 0.4400 | 0.2084 | 0.1183 | 2772 | 13.9833 |
| normal | naive | 0.4931 | 0.3911 | 0.1339 | 0.0210 | 5022 | 9.3631 |
| normal | ewma | 0.4956 | 0.3936 | 0.1273 | -0.0829 | 5022 | 9.9051 |
| normal | har | 0.4509 | 0.3559 | 0.1133 | 0.0102 | 5022 | 0.0000 |
| normal | gbm | 0.4440 | 0.3498 | 0.1124 | 0.0414 | 5022 | -1.5300 |
| normal | gbm_no_market | 0.4487 | 0.3542 | 0.1120 | 0.0078 | 5022 | -0.4891 |
| normal | gbm_no_volume | 0.4467 | 0.3519 | 0.1140 | 0.0411 | 5022 | -0.9475 |
| normal | gbm_har_only | 0.4506 | 0.3555 | 0.1134 | 0.0102 | 5022 | -0.0789 |

## What does the gradient boosting model rely on? (permutation importance, validation slice)
| feature | mse_increase | std |
|---|---|---|
| lpv5 | 0.0619 | 0.0040 |
| idx_lpv5 | 0.0128 | 0.0009 |
| lpv22 | 0.0120 | 0.0014 |
| lewma | 0.0116 | 0.0009 |
| ticker_code | 0.0047 | 0.0021 |
| idx_lpv22 | 0.0047 | 0.0010 |
| vol_ratio | 0.0032 | 0.0012 |
| lpv | 0.0016 | 0.0012 |
| ret5 | 0.0000 | 0.0002 |
| abs_ret | -0.0003 | 0.0005 |

## Interpretation  (WRITE THIS YOURSELF)
- Which model wins overall, and by how much? Is it statistically significant (Holm column)?
- In which regime does the ranking change? Why might that be?
- If the benchmark was not beaten, say so plainly. That is a valid finding.

## Limitations (keep these, they make the project credible)
- Survivorship bias: only stocks that are large today were studied.
- One market (NSE large caps) and one volatility proxy (range-based, daily data).
- Stocks are correlated; per-stock tests are not independent. Holm correction does not fix that fully.
- yfinance is an unofficial data source; adjusted prices can be revised.
- Live forward-test sample is small unless the nightly job has run for a long time.
- This is a forecasting study, not a trading system, and not financial advice.
