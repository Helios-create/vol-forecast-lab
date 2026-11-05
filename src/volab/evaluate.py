"""Leakage-free evaluation: walk-forward forecasts, metrics, significance tests, regimes.

THE PROTOCOL (this is what docs/DESIGN.md pre-registers)

  Calendar:  ... train ... | purge (h days) | test block 1 | test block 2 | ...

  * The test period starts at `test_start`. It is cut into blocks of `refit_every`
    trading days (21 = about one month).
  * Before each block, every model is REFITTED using only rows whose label was already
    fully known on the first day of the block:  train rows have  date <= t0 - h  trading days.
    Why h? The label of a row at date t averages variance over t+1..t+h. If t > t0 - h,
    that label already contains information from the block we are about to forecast.
    That would be leakage. Removing those rows is called PURGING (or an embargo).
  * Inside a block the model is frozen, just like a deployed model between retrainings.
  * Hyperparameters were chosen earlier on a validation slice INSIDE the training window.
    The test period never influences any choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from . import models
from .features import FEATURES_ALL, usable

GBM_VARIANTS = {   # feature sets for the ablation study: "what does each group of features add?"
    "gbm": FEATURES_ALL,
    "gbm_no_market": [f for f in FEATURES_ALL if not f.startswith("idx_")],
    "gbm_no_volume": [f for f in FEATURES_ALL if f != "vol_ratio"],
    "gbm_har_only": ["lpv", "lpv5", "lpv22", "ticker_code"],
}


# ---------------------------------------------------------------- walk-forward

def walk_forward(panel: pd.DataFrame, cfg: dict, gbm_params: dict, ablation: bool = False,
                 use_garch: bool = True, log=print):
    """Produce out-of-sample forecasts for every model. Returns (predictions, folds).

    predictions: one row per (ticker, date) with the realised y and one column per model.
    folds: one row per refit block, with the dates that prove the purge was respected.
    """
    h, step = cfg["horizon"], cfg["refit_every"]
    test_start = pd.Timestamp(cfg["test_start"])

    calendar = np.sort(panel["trade_date"].unique())                 # every trading date in the data
    pos = {d: i for i, d in enumerate(calendar)}
    rows = usable(panel)                                             # rows with all features and a target
    test_dates = [d for d in np.sort(rows["trade_date"].unique()) if d >= np.datetime64(test_start)]
    blocks = [test_dates[i:i + step] for i in range(0, len(test_dates), step)]
    variants = {k: v for k, v in GBM_VARIANTS.items() if ablation or k == "gbm"}

    by_ticker = {t: g.sort_values("trade_date").reset_index(drop=True) for t, g in panel.groupby("ticker")}
    preds, folds = [], []
    for bi, block in enumerate(blocks):
        t0 = block[0]
        cutoff = calendar[pos[t0] - h]                               # last date whose label is fully known at t0
        train = rows[rows["trade_date"] <= cutoff]
        test = rows[rows["trade_date"].isin(block)]
        if train.empty or test.empty:
            continue
        out = test[["ticker", "trade_date", "y", "idx_lpv22"]].copy()
        out["naive"] = models.predict_naive(test)
        out["ewma"] = models.predict_ewma(test)
        out["har"] = models.predict_har(models.fit_har(train), test)
        for name, feats in variants.items():
            out[name] = models.fit_gbm(train, gbm_params, feats).predict(test[feats])
        if use_garch:
            col = pd.Series(np.nan, index=out.index)
            for ticker, g in by_ticker.items():
                fc = models.garch_predict(g, cutoff, h)
                fc.index = g["trade_date"]
                m = out["ticker"] == ticker
                col[m] = out.loc[m, "trade_date"].map(fc).to_numpy()
            out["garch"] = col
        preds.append(out)
        folds.append({"block": bi, "first_test_date": t0, "train_cutoff": cutoff,
                      "last_train_label_date": cutoff, "n_train": len(train), "n_test": len(test)})
        if bi % 5 == 0:
            log(f"  walk-forward block {bi + 1}/{len(blocks)}  (test from {pd.Timestamp(t0).date()}, {len(train)} training rows)")

    predictions = pd.concat(preds, ignore_index=True)
    return predictions, pd.DataFrame(folds)


# ---------------------------------------------------------------- metrics

def model_columns(pred: pd.DataFrame) -> list[str]:
    return [c for c in pred.columns if c not in ("ticker", "trade_date", "y", "idx_lpv22", "regime")]


def _metrics(e: np.ndarray) -> dict:
    # e = actual - predicted (log variance). exp(e) = actual_var / predicted_var.
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "qlike": float(np.mean(np.exp(e) - e - 1.0)),    # standard loss for variance forecasts
            "bias": float(np.mean(e)), "n": int(len(e))}


def metrics_table(pred: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """RMSE / MAE / QLIKE / bias per model (optionally per ticker, per regime...)."""
    rows = []
    groups = pred.groupby(by) if by else [((), pred)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in model_columns(pred):
            ok = g[m].notna()
            e = (g.loc[ok, "y"] - g.loc[ok, m]).to_numpy()
            if len(e):
                rows.append({**dict(zip(by or [], key)), "model": m, **_metrics(e)})
    return pd.DataFrame(rows)


def add_skill(table: pd.DataFrame, bench: str, by: list[str] | None = None) -> pd.DataFrame:
    """Add rmse_vs_<bench> = % change in RMSE relative to the benchmark (negative = better)."""
    t = table.copy()
    keys = by or []
    ref = t[t.model == bench].set_index(keys)["rmse"] if keys else pd.Series({0: t.loc[t.model == bench, "rmse"].iloc[0]})
    if keys:
        t["rmse_vs_" + bench] = [100 * (r.rmse / ref.loc[tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]] - 1) for _, r in t.iterrows()]
    else:
        t["rmse_vs_" + bench] = 100 * (t["rmse"] / ref.iloc[0] - 1)
    return t


# ---------------------------------------------------------------- Diebold-Mariano test

def dm_test(loss_bench: np.ndarray, loss_model: np.ndarray, horizon: int):
    """Diebold-Mariano test: is `model` really more accurate than `bench`, or is it luck?

    d_t = loss_bench(t) - loss_model(t)      (positive means the model was better on day t)
    H0: mean(d) = 0 (equal accuracy).  We regress d on a constant and use HAC
    (Newey-West) standard errors with `horizon - 1` lags, because forecasts for
    overlapping h-day targets make the errors autocorrelated; ordinary standard errors
    would be far too small and produce false 'significant' results.
    Returns (t_statistic, two-sided p-value). NaN if there is nothing to test.
    """
    d = np.asarray(loss_bench, dtype=float) - np.asarray(loss_model, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 30 or np.allclose(d, 0.0):
        return float("nan"), float("nan")
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 0)})
    return float(res.tvalues[0]), float(res.pvalues[0])


def dm_table(pred: pd.DataFrame, bench: str, horizon: int) -> pd.DataFrame:
    """DM test of every model against `bench`, per ticker, with Holm multiple-testing correction.

    Why Holm? With 20 stocks you run 20 tests per model; at a 5% threshold about one will look
    'significant' by pure chance. Holm adjusts the p-values to control that.
    """
    rows = []
    for model in [m for m in model_columns(pred) if m != bench]:
        part = []
        for ticker, g in pred.groupby("ticker"):
            ok = g[model].notna() & g[bench].notna()
            lb, lm = (g.loc[ok, "y"] - g.loc[ok, bench]) ** 2, (g.loc[ok, "y"] - g.loc[ok, model]) ** 2
            stat, p = dm_test(lb.to_numpy(), lm.to_numpy(), horizon)
            part.append({"ticker": ticker, "model": model, "bench": bench,
                         "mse_bench": float(lb.mean()), "mse_model": float(lm.mean()),
                         "pct_mse_change": 100 * float(lm.mean() / lb.mean() - 1), "dm_stat": stat, "p_value": p})
        part = pd.DataFrame(part)
        valid = part["p_value"].notna()
        part["p_holm"] = np.nan
        if valid.any():
            part.loc[valid, "p_holm"] = multipletests(part.loc[valid, "p_value"], method="holm")[1]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def dm_summary(dm: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One line per model: how many stocks is it significantly better / worse than the benchmark?"""
    out = []
    for model, g in dm.groupby("model"):
        better = (g["dm_stat"] > 0)
        out.append({
            "model": model, "tickers": len(g),
            "better_raw": int((better & (g.p_value < alpha)).sum()), "worse_raw": int((~better & (g.p_value < alpha)).sum()),
            "better_holm": int((better & (g.p_holm < alpha)).sum()), "worse_holm": int((~better & (g.p_holm < alpha)).sum()),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- regimes

def add_regime(pred: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, float]:
    """Label each test day 'high_vol' or 'normal' by the market index's 22-day variance.

    The threshold (a quantile) is computed on TRAINING dates only: a regime definition must
    not peek at the test period either.
    """
    train_idx = panel.loc[panel["trade_date"] < pd.Timestamp(cfg["test_start"])].drop_duplicates("trade_date")["idx_lpv22"].dropna()
    thr = float(train_idx.quantile(cfg["regime_quantile"]))
    out = pred.copy()
    out["regime"] = np.where(out["idx_lpv22"] > thr, "high_vol", "normal")
    return out, thr


# ---------------------------------------------------------------- orchestration

def evaluate_panel(panel: pd.DataFrame, cfg: dict, ablation: bool = False, log=print) -> dict:
    """Run the whole study on a prepared panel. Returns a dict of result tables."""
    h = cfg["horizon"]
    test_start = pd.Timestamp(cfg["test_start"])
    rows = usable(panel)
    cutoff_dates = np.sort(rows["trade_date"].unique())
    # Training window for hyperparameter selection: everything whose label was known before test_start.
    train0 = rows[rows["trade_date"] < test_start]
    if len(train0) < 500:
        raise ValueError(f"only {len(train0)} usable training rows before test_start={cfg['test_start']}; need more history")
    log("Selecting GBM hyperparameters on a validation slice inside the training window...")
    best, grid_scores, importance = models.select_gbm_params(train0, cfg["gbm_grid"], h)
    log(f"  chosen: {best}")

    log("Walk-forward evaluation (refit every %d trading days)..." % cfg["refit_every"])
    pred, folds = walk_forward(panel, cfg, best, ablation=ablation, use_garch=cfg["models"]["garch"], log=log)

    # Compare models only on rows where ALL of them produced a forecast (fairness).
    pred = pred.dropna(subset=model_columns(pred)).reset_index(drop=True)
    bench = cfg["benchmark_model"]
    pred, thr = add_regime(pred, panel, cfg)

    overall = add_skill(metrics_table(pred), bench)
    by_ticker = metrics_table(pred, by=["ticker"])
    by_regime = add_skill(metrics_table(pred, by=["regime"]), bench, by=["regime"])
    dm = dm_table(pred, bench, h)
    return {"predictions": pred, "folds": folds, "overall": overall, "by_ticker": by_ticker,
            "by_regime": by_regime, "dm": dm, "dm_summary": dm_summary(dm),
            "grid_scores": grid_scores, "importance": importance, "gbm_params": best,
            "regime_threshold": thr, "n_test_rows": len(pred),
            "test_range": (str(pred["trade_date"].min().date()), str(pred["trade_date"].max().date()))}
