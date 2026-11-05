"""The forecasting models. All predict y = log of next-h-day mean Parkinson variance.

    naive   persistence: "next week will look like the last 5 days"      (parameter-free)
    ewma    RiskMetrics exponentially weighted average of past variance  (parameter-free)
    har     Heterogeneous AutoRegressive model (Corsi 2009): a plain linear regression
            of y on [today, last-5-day, last-22-day] log variance, fitted per stock.
            It is the standard benchmark in the volatility literature: simple but hard to beat.
    gbm     Gradient boosting (scikit-learn HistGradientBoostingRegressor), ONE pooled
            model across all stocks with the stock id as a categorical feature.
    garch   GARCH(1,1) fitted on returns; a classic statistical model with a different
            information set (returns only), included for comparison. Optional.

Models never see the test period when they are fitted: evaluate.py enforces that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .features import FEATURES_ALL, FEATURES_HAR

# ---------------------------------------------------------------- parameter-free models

def predict_naive(df: pd.DataFrame) -> np.ndarray:
    return df["lpv5"].to_numpy()


def predict_ewma(df: pd.DataFrame) -> np.ndarray:
    return df["lewma"].to_numpy()


# ---------------------------------------------------------------- HAR (linear regression)

def fit_har(train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Ordinary least squares per ticker: y = b0 + b1*lpv + b2*lpv5 + b3*lpv22."""
    coefs = {}
    for ticker, g in train.groupby("ticker"):
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in FEATURES_HAR])
        beta, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        coefs[ticker] = beta
    return coefs


def predict_har(coefs: dict[str, np.ndarray], df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan)
    tick = df["ticker"].to_numpy()
    for ticker, beta in coefs.items():
        m = tick == ticker
        if m.any():
            X = np.column_stack([np.ones(m.sum())] + [df.loc[m, c].to_numpy() for c in FEATURES_HAR])
            out[m] = X @ beta
    return out


# ---------------------------------------------------------------- gradient boosting

DEFAULT_GBM = {"learning_rate": 0.05, "max_depth": 4, "max_iter": 300, "min_samples_leaf": 50, "l2_regularization": 1.0}


def make_gbm(params: dict, features: list[str]) -> HistGradientBoostingRegressor:
    cat = [f == "ticker_code" for f in features]       # the stock id is categorical, not a number
    return HistGradientBoostingRegressor(
        categorical_features=cat if any(cat) else None, random_state=0, early_stopping=False, **params
    )


def fit_gbm(train: pd.DataFrame, params: dict, features: list[str] = FEATURES_ALL):
    model = make_gbm(params, features)
    model.fit(train[features], train["y"])
    return model


def select_gbm_params(train: pd.DataFrame, grid: list[dict], horizon: int, features: list[str] = FEATURES_ALL, val_frac: float = 0.2):
    """Choose hyperparameters using ONLY the training window.

    The last `val_frac` of the training DATES is held out as validation. Rows whose
    5-day target would reach into the validation period are purged from the fitting
    set (the same embargo used in the walk-forward), so there is no leakage even here.
    Returns (best_params, scores_table, importance_table).
    """
    dates = np.sort(train["trade_date"].unique())
    val_start = dates[int(len(dates) * (1 - val_frac))]
    purge_cut = dates[max(int(np.searchsorted(dates, val_start)) - horizon, 0)]
    fit_rows = train[train["trade_date"] <= purge_cut]
    val_rows = train[train["trade_date"] >= val_start]

    scores, best, best_rmse, best_model = [], None, np.inf, None
    for params in grid:
        m = fit_gbm(fit_rows, params, features)
        rmse = float(np.sqrt(np.mean((val_rows["y"] - m.predict(val_rows[features])) ** 2)))
        scores.append({**params, "val_rmse": rmse})
        if rmse < best_rmse:
            best, best_rmse, best_model = params, rmse, m

    imp = permutation_importance(best_model, val_rows[features], val_rows["y"], scoring="neg_mean_squared_error",
                                 n_repeats=5, random_state=0)
    importance = pd.DataFrame({"feature": features, "mse_increase": imp.importances_mean, "std": imp.importances_std})
    return best, pd.DataFrame(scores), importance.sort_values("mse_increase", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- GARCH(1,1)

def fit_garch_params(returns: np.ndarray):
    """Fit a zero-mean GARCH(1,1) with `arch`. Returns (omega, alpha, beta) in %^2 units."""
    from arch import arch_model

    res = arch_model(returns * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
    p = res.params
    return float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])


def garch_avg_forecast(returns: np.ndarray, omega: float, alpha: float, beta: float, horizon: int, s0: float) -> np.ndarray:
    """Average conditional variance over the next `horizon` days, for EVERY day t (using returns up to t only).

    Recursion (all in %^2):   s[t] = omega + alpha*r[t-1]^2 + beta*s[t-1]       (variance of day t)
    One step ahead from t:    s_next = omega + alpha*r[t]^2 + beta*s[t]
    k steps ahead:            E[s_{t+k}] = V + (alpha+beta)^(k-1) * (s_next - V),   V = omega / (1 - alpha - beta)
    The forecast is the mean of these over k = 1..horizon.
    """
    r = np.nan_to_num(returns * 100.0)
    n = len(r)
    s = np.empty(n)
    s[0] = s0
    for t in range(1, n):
        s[t] = omega + alpha * r[t - 1] ** 2 + beta * s[t - 1]
    s_next = omega + alpha * r ** 2 + beta * s
    persist = min(alpha + beta, 0.9999)
    long_run = omega / (1.0 - persist)
    decay = np.mean([persist ** k for k in range(horizon)])
    return long_run + (s_next - long_run) * decay


def garch_predict(g: pd.DataFrame, cutoff, horizon: int) -> pd.Series:
    """Log-variance forecast for every row of one ticker's frame `g` (sorted by date).

    Parameters are fitted on rows with trade_date <= cutoff ONLY. The conditional variance is
    then filtered forward over all rows (causal: each day uses returns up to that day).
    A scale constant (training mean of Parkinson variance / mean of squared returns) maps
    close-to-close variance onto the Parkinson scale so the log-errors are comparable.
    """
    train = g[(g["trade_date"] <= cutoff) & g["ret"].notna()]
    if len(train) < 250:
        return pd.Series(np.nan, index=g.index)
    r_train = train["ret"].to_numpy()
    omega, alpha, beta = fit_garch_params(r_train)
    avg = garch_avg_forecast(g["ret"].to_numpy(), omega, alpha, beta, horizon, s0=float(np.var(r_train * 100.0)))
    scale = train["pv"].mean() / np.mean(r_train ** 2)
    return pd.Series(np.log(np.maximum(avg / 1e4 * scale, 1e-10)), index=g.index)
