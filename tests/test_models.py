"""Model and statistics tests: do the maths pieces behave as theory says they should?"""
import numpy as np
import pandas as pd
import pytest

from volab import evaluate, models


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"ticker": "X", "lpv": rng.normal(size=n), "lpv5": rng.normal(size=n), "lpv22": rng.normal(size=n)})
    df["y"] = 0.1 + 0.3 * df.lpv + 0.2 * df.lpv5 + 0.4 * df.lpv22 + rng.normal(0, 0.1, n)
    beta = models.fit_har(df)["X"]
    assert np.allclose(beta, [0.1, 0.3, 0.2, 0.4], atol=0.02)
    pred = models.predict_har({"X": beta}, df)
    assert np.sqrt(np.mean((df.y - pred) ** 2)) < 0.11


def test_garch_forecast_matches_the_arch_library():
    """Our hand-written recursion must equal the `arch` package's forecast for the same parameters."""
    from arch import arch_model
    omega, alpha, beta = 0.05, 0.08, 0.90
    rng = np.random.default_rng(1)
    am = arch_model(rng.standard_normal(2500), mean="Zero", vol="GARCH", p=1, q=1)
    sim = am.simulate([omega, alpha, beta], 2500)["data"].to_numpy() / 100.0      # returns in decimals
    res = arch_model(sim * 100, mean="Zero", vol="GARCH", p=1, q=1).fix([omega, alpha, beta])
    theirs = res.forecast(horizon=5, reindex=False).variance.to_numpy()[-1].mean()
    ours = models.garch_avg_forecast(sim, omega, alpha, beta, 5, s0=float(np.var(sim * 100)))[-1]
    assert np.isclose(ours, theirs, rtol=1e-4)


def test_dm_detects_a_truly_better_model():
    rng = np.random.default_rng(2)
    bench = rng.chisquare(1, 600) + 0.3
    model = bench - 0.15 + rng.normal(0, 0.1, 600)        # consistently lower loss
    stat, p = evaluate.dm_test(bench, model, horizon=5)
    assert stat > 0 and p < 0.01


def test_dm_returns_nan_for_identical_forecasts():
    x = np.random.default_rng(3).random(100)
    stat, p = evaluate.dm_test(x, x, horizon=5)
    assert np.isnan(stat) and np.isnan(p)


def test_dm_has_roughly_correct_size_with_overlapping_errors():
    """Under H0 (equal accuracy) a 5% test should reject about 5% of the time, even when errors overlap."""
    rng = np.random.default_rng(4)
    rejects = 0
    sims = 300
    for _ in range(sims):
        e = rng.normal(size=504)
        d = np.convolve(e, np.ones(5), mode="valid")       # MA(4): the autocorrelation that h=5 overlap creates
        _, p = evaluate.dm_test(d + 1.0, np.ones_like(d) + 0.0 * d + 1.0, horizon=5) if False else evaluate.dm_test(d, np.zeros_like(d), horizon=5)
        rejects += p < 0.05
    assert 0.02 <= rejects / sims <= 0.12


def test_metrics_and_qlike_are_zero_for_perfect_forecasts():
    pred = pd.DataFrame({"ticker": "A", "trade_date": pd.date_range("2020-01-01", periods=50), "y": np.linspace(-9, -7, 50)})
    pred["perfect"] = pred["y"]
    t = evaluate.metrics_table(pred).iloc[0]
    assert t.rmse == 0 and t.qlike == 0 and t.bias == 0


def test_gbm_selection_uses_only_training_rows(panel, base_cfg):
    from volab.features import usable
    train = usable(panel)
    train = train[train.trade_date < pd.Timestamp(base_cfg["test_start"])]
    best, scores, imp = models.select_gbm_params(train, base_cfg["gbm_grid"], base_cfg["horizon"])
    assert best in base_cfg["gbm_grid"] and len(scores) == len(base_cfg["gbm_grid"])
    assert set(imp["feature"]) >= {"lpv5", "lpv22"}
