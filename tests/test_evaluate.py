"""Walk-forward protocol tests: the purge must hold, and forecasts must be sane."""
import numpy as np
import pandas as pd

from volab import evaluate


def test_walk_forward_purges_training_labels(panel, base_cfg):
    pred, folds = evaluate.walk_forward(panel, base_cfg, {"learning_rate": 0.1, "max_depth": 3, "max_iter": 30, "min_samples_leaf": 50, "l2_regularization": 1.0}, log=lambda *_: None)
    calendar = np.sort(panel["trade_date"].unique())
    pos = {d: i for i, d in enumerate(calendar)}
    h = base_cfg["horizon"]
    assert len(folds) >= 3
    for _, f in folds.iterrows():
        # the last training label's window (cutoff+1 .. cutoff+h) must end on or before the first test day
        assert pos[np.datetime64(f.train_cutoff)] + h <= pos[np.datetime64(f.first_test_date)]
    assert pred["trade_date"].min() >= pd.Timestamp(base_cfg["test_start"])
    for m in ["naive", "ewma", "har", "gbm"]:
        assert np.isfinite(pred[m]).all()


def test_blocks_do_not_overlap_and_cover_the_test_period(panel, base_cfg):
    pred, folds = evaluate.walk_forward(panel, base_cfg, {"learning_rate": 0.1, "max_depth": 3, "max_iter": 20, "min_samples_leaf": 50, "l2_regularization": 1.0}, log=lambda *_: None)
    assert not pred.duplicated(["ticker", "trade_date"]).any()
    assert (folds["first_test_date"].diff().dropna() > pd.Timedelta(0)).all()


def test_har_beats_naive_on_persistent_synthetic_volatility(panel, base_cfg):
    """On data built to be persistent, a fitted regression should beat a raw 5-day average (sanity check of the pipeline)."""
    res = evaluate.evaluate_panel(panel, base_cfg, ablation=False, log=lambda *_: None)
    o = res["overall"].set_index("model")["rmse"]
    assert o["har"] < o["naive"]
    assert {"overall", "by_ticker", "by_regime", "dm", "dm_summary", "importance"} <= set(res)
    assert set(res["by_regime"]["regime"]) <= {"normal", "high_vol"}
