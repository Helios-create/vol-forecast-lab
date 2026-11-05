"""One test per validation rule, plus a test that the validator finds EXACTLY the injected faults."""
import numpy as np
import pandas as pd

from volab import synth, validate
from volab.load import adjust_ohlc


def bars(rows):
    df = pd.DataFrame(rows, columns=["ticker", "trade_date", "open", "high", "low", "close", "volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adj_close"] = df["close"]
    return df


def rule_counts(df, **kw):
    r = validate.validate(df, **kw)
    return r.rejects["rule_name"].value_counts().to_dict(), r


GOOD = ("A", "2020-01-06", 100, 105, 98, 102, 1000)


def test_clean_row_passes():
    counts, r = rule_counts(bars([GOOD]))
    assert counts == {} and len(r.clean) == 1


def test_null_value():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 100, 105, 98, None, 1000)]))
    assert counts.get("null_values") == 1


def test_non_positive_price():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 0, 105, 98, 102, 1000)]))
    assert counts.get("non_positive_price") == 1


def test_negative_volume():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 100, 105, 98, 102, -1)]))
    assert counts.get("negative_volume") == 1


def test_high_below_others():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 100, 99, 98, 102, 1000)]))
    assert counts.get("high_below_others") == 1


def test_low_above_others():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 100, 105, 101, 102, 1000)]))
    assert counts.get("low_above_others") == 1


def test_duplicate_key_keeps_last():
    df = bars([("A", "2020-01-06", 100, 105, 98, 102, 1000), ("A", "2020-01-06", 100, 105, 98, 103, 1000)])
    counts, r = rule_counts(df)
    assert counts == {"duplicate_key": 1}
    assert r.clean["close"].iloc[0] == 103          # the later row survives


def test_weekend_rejected():
    counts, _ = rule_counts(bars([("A", "2020-01-04", 100, 105, 98, 102, 1000)]))  # a Saturday
    assert counts.get("weekend_date") == 1


def test_stale_run_rejects_all_but_first():
    flat = [("A", d, 100, 100, 100, 100, 5) for d in ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]]
    counts, r = rule_counts(bars(flat), stale_run=3)
    assert counts == {"stale_ohlc": 3} and len(r.clean) == 1


def test_two_identical_days_are_not_stale():
    flat = [("A", d, 100, 100, 100, 100, 5) for d in ["2020-01-06", "2020-01-07"]]
    counts, _ = rule_counts(bars(flat), stale_run=3)
    assert counts == {}


def test_flat_zero_volume_filler_row_rejected_but_flat_with_volume_only_flagged():
    counts, _ = rule_counts(bars([("A", "2020-01-06", 100, 100, 100, 100, 0)]))
    assert counts == {"flat_zero_volume": 1}
    counts, r = rule_counts(bars([("A", "2020-01-06", 100, 100, 100, 100, 500)]))
    assert counts == {} and (r.flags["rule_name"] == "zero_range").sum() == 1


def test_extreme_move_is_flagged_not_rejected():
    df = bars([("A", "2020-01-06", 100, 101, 99, 100, 1), ("A", "2020-01-07", 130, 135, 128, 130, 1)])
    counts, r = rule_counts(df, extreme_move=0.20)
    assert counts == {} and len(r.clean) == 2
    assert (r.flags["rule_name"] == "extreme_move").sum() == 1


def test_missing_trading_day_flagged():
    idx = [("^I", d, 100, 101, 99, 100, 0) for d in ["2020-01-06", "2020-01-07", "2020-01-08"]]
    stock = [("A", d, 10, 11, 9, 10, 1) for d in ["2020-01-06", "2020-01-08"]]
    _, r = rule_counts(bars(idx + stock), index_ticker="^I")
    miss = r.flags[r.flags["rule_name"] == "missing_trading_day"]
    assert len(miss) == 1 and miss["trade_date"].iloc[0] == pd.Timestamp("2020-01-07")


def test_validator_finds_exactly_the_injected_faults(prices):
    bad, expected, _ = synth.inject_faults(prices)
    r = validate.validate(bad, index_ticker="^NSEI")
    got = r.rejects["rule_name"].value_counts().to_dict()
    got.update(r.flags["rule_name"].value_counts().to_dict())
    for rule, n in expected.items():
        assert got.get(rule, 0) == n, f"{rule}: expected {n}, got {got.get(rule, 0)}"
    assert set(got) <= set(expected)            # and nothing unexpected was reported


def test_adjust_ohlc_makes_prices_continuous_across_a_split(prices):
    split_ticker = prices.ticker.unique()[3]     # synth splits the 3rd stock (index is element 0)
    adj = adjust_ohlc(prices[prices.ticker == split_ticker]).sort_values("trade_date")
    jump = np.log(adj["close"]).diff().abs().max()
    raw_jump = np.log(prices[prices.ticker == split_ticker].sort_values("trade_date")["close"]).diff().abs().max()
    assert raw_jump > 0.5 and jump < 0.3         # raw shows the -50% split jump, adjusted does not
    assert (adj["high"] >= adj["low"]).all()
