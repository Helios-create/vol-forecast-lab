"""The most important tests in the project: no feature may use information from the future."""
import numpy as np
import pandas as pd

from volab import features
from volab.features import FEATURES_ALL


def _mutate_future(prices, cutoff, seed=3):
    """Scramble EVERY bar after `cutoff`: new levels, new ranges, new volume (bars stay valid).

    Note: scaling a bar by a constant would NOT change its range ratio ln(High/Low), so the
    variance proxy would not move and the test would prove nothing. We redraw the ranges.
    """
    rng = np.random.default_rng(seed)
    p = prices.copy()
    m = (p["trade_date"] > cutoff).to_numpy()
    k = int(m.sum())
    level = rng.uniform(0.5, 2.0, k)
    o = p.loc[m, "open"].to_numpy() * level
    c = p.loc[m, "close"].to_numpy() * level
    p.loc[m, "open"], p.loc[m, "close"], p.loc[m, "adj_close"] = o, c, c
    p.loc[m, "high"] = np.maximum(o, c) * (1 + rng.uniform(0.0, 0.15, k))
    p.loc[m, "low"] = np.minimum(o, c) * (1 - rng.uniform(0.0, 0.15, k))
    p.loc[m, "volume"] = rng.uniform(1, 1e6, k)
    return p


def test_features_at_t_do_not_change_when_the_future_changes(prices, base_cfg):
    cutoff = pd.Timestamp("2019-03-15")
    def build(p):
        return features.build_panel(p, base_cfg["index_ticker"], base_cfg["tickers"], base_cfg["horizon"], base_cfg["ewma_lambda"])
    a, b = build(prices), build(_mutate_future(prices, cutoff))
    a, b = a[a.trade_date <= cutoff].reset_index(drop=True), b[b.trade_date <= cutoff].reset_index(drop=True)
    for col in FEATURES_ALL:
        assert np.allclose(a[col], b[col], equal_nan=True), f"feature {col} leaks future information"


def test_target_does_use_the_future_as_designed(prices, base_cfg):
    """Sanity check of the test itself: the TARGET must change when the future changes."""
    cutoff, h = pd.Timestamp("2019-03-15"), base_cfg["horizon"]
    build = lambda p: features.build_panel(p, base_cfg["index_ticker"], base_cfg["tickers"], h, base_cfg["ewma_lambda"])  # noqa: E731
    a, b = build(prices), build(_mutate_future(prices, cutoff))
    near = (a.trade_date <= cutoff) & (a.trade_date > cutoff - pd.Timedelta(days=5))
    assert not np.allclose(a.loc[near, "y"], b.loc[near, "y"], equal_nan=True)


def test_target_is_log_of_mean_variance_over_next_h_days():
    h = 5
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    spread = rng.uniform(0.01, 0.03, 60)
    rows = pd.DataFrame({"ticker": "A", "trade_date": dates, "open": close, "close": close,
                         "high": close * (1 + spread), "low": close * (1 - spread), "volume": 1000.0})
    idx = rows.assign(ticker="^I")
    panel = features.build_panel(pd.concat([rows, idx]), "^I", ["A"], h, 0.94)
    pv = features.parkinson_var(rows["high"], rows["low"]).to_numpy()
    t = 30
    expected = np.log(pv[t + 1:t + 1 + h].mean())
    assert np.isclose(panel.loc[t, "y"], expected)
    assert panel["y"].tail(h).isna().all()             # the last h days have no target yet
