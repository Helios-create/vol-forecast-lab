"""Data validation: decide, for every row, keep / reject / keep-but-flag.

DESIGN PRINCIPLES (good talking points)
  * REJECT only what is provably impossible (high below low, negative price...).
    A rejected row is not deleted: it goes to the `rejects` table WITH the rule name
    and the original row, so you can always audit what was refused and why.
  * FLAG what is merely suspicious (a 25% daily move can be real: results day, a crash).
    Deleting real extreme days would bias volatility models toward calm, which is the
    exact opposite of what a volatility study needs.
  * One rule = one small function returning a boolean mask (True = bad row). Easy to
    read, easy to unit-test (tests/test_validate.py has one test per rule).
  * Validation happens on the RAW prices, before split/dividend adjustment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]

REASONS = {
    "null_values": "A required value (open/high/low/close/volume) is missing.",
    "non_positive_price": "A price is zero or negative.",
    "negative_volume": "Volume is negative.",
    "high_below_others": "High is lower than open, close or low.",
    "low_above_others": "Low is higher than open, close or high.",
    "duplicate_key": "Another row exists for the same ticker and date; the later one is kept.",
    "weekend_date": "Date is a Saturday or Sunday (NSE normally trades Mon-Fri).",
    "stale_ohlc": "Open/high/low/close identical to the previous day for 3+ days in a row.",
    "flat_zero_volume": "High equals low and volume is zero: a filler row for a day with no real trading.",
}


@dataclass
class ValidationResult:
    clean: pd.DataFrame      # rows that passed every REJECT rule
    rejects: pd.DataFrame    # one row per (bad row, broken rule)
    flags: pd.DataFrame      # suspicious-but-kept observations
    summary: dict            # counts per rule, for logging


def _stale_mask(df: pd.DataFrame, run_len: int) -> pd.Series:
    """True for days 2..n of a run of >= run_len identical OHLC bars (per ticker)."""
    s = df.sort_values(["ticker", "trade_date"])
    same_as_prev = s[OHLC].eq(s[OHLC].shift()).all(axis=1) & s["ticker"].eq(s["ticker"].shift())
    run_id = (~same_as_prev).cumsum()                       # a new id each time the bar changes
    run_size = s.groupby(run_id)["ticker"].transform("size")
    return (same_as_prev & (run_size >= run_len)).reindex(df.index, fill_value=False)


def reject_masks(df: pd.DataFrame, stale_run: int = 3) -> dict[str, pd.Series]:
    """All REJECT rules as {rule_name: boolean mask (True = row is bad)}."""
    o, h, l, c = (df[x] for x in OHLC)
    return {
        "null_values": df[OHLC + ["volume"]].isna().any(axis=1),
        "non_positive_price": (df[OHLC] <= 0).any(axis=1),
        "negative_volume": df["volume"] < 0,
        "high_below_others": h < pd.concat([o, c, l], axis=1).max(axis=1),
        "low_above_others": l > pd.concat([o, c, h], axis=1).min(axis=1),
        "duplicate_key": df.duplicated(["ticker", "trade_date"], keep="last"),
        "weekend_date": df["trade_date"].dt.day_of_week >= 5,
        "stale_ohlc": _stale_mask(df, stale_run),
        "flat_zero_volume": (h == l) & (df["volume"] == 0),
    }


def _flags(df: pd.DataFrame, survivors: pd.DataFrame, index_ticker: str | None, extreme_move: float) -> pd.DataFrame:
    """Suspicious-but-kept observations. Returns columns: ticker, trade_date, rule_name, detail."""
    out = []

    # 1. Extreme daily moves (computed on the split-adjusted close so splits do not trigger it).
    s = survivors.sort_values(["ticker", "trade_date"])
    ref = s["adj_close"].where(s["adj_close"] > 0, s["close"])
    move = ref.groupby(s["ticker"]).pct_change()
    big = s[move.abs() > extreme_move]
    for idx, row in big.iterrows():
        out.append((row.ticker, row.trade_date, "extreme_move", f"daily move {move[idx]:+.1%}"))

    # 2. Missing trading days: dates the index traded that a stock has no row for.
    #    "Present" means ANY row in the input (even a rejected one), so a bad row is
    #    reported once (in rejects) and not a second time here.
    if index_ticker is not None:
        calendar = pd.DatetimeIndex(df.loc[df.ticker == index_ticker, "trade_date"].unique()).sort_values()
        for ticker, g in df[df.ticker != index_ticker].groupby("ticker"):
            have = pd.DatetimeIndex(g["trade_date"].unique())
            expected = calendar[(calendar >= have.min()) & (calendar <= have.max())]
            for d in expected.difference(have):
                out.append((ticker, d, "missing_trading_day", "index traded, this ticker has no row"))

    # 3. Zero-range days with real volume: legitimate but they make the variance proxy zero, so the
    #    log is floored (see features.FLOOR). Flagged so you know how many there are.
    flat = survivors[survivors["high"] == survivors["low"]]
    for _, row in flat.iterrows():
        out.append((row.ticker, row.trade_date, "zero_range", "high == low with volume > 0"))

    # 4. Suspicious adjustment factor (adjusted close should not exceed the raw close).
    ratio = survivors["adj_close"] / survivors["close"]
    odd = survivors[survivors["adj_close"].notna() & ((ratio <= 0) | (ratio > 1.05))]
    for _, row in odd.iterrows():
        out.append((row.ticker, row.trade_date, "suspicious_adjustment", f"adj/close = {row.adj_close / row.close:.3f}"))

    return pd.DataFrame(out, columns=["ticker", "trade_date", "rule_name", "detail"])


def validate(df: pd.DataFrame, index_ticker: str | None = None, extreme_move: float = 0.20, stale_run: int = 3) -> ValidationResult:
    """Split a raw batch into clean rows, rejects and flags."""
    df = df.copy().reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    masks = reject_masks(df, stale_run)

    any_bad = pd.Series(False, index=df.index)
    reject_parts = []
    for name, mask in masks.items():
        mask = mask.fillna(False)
        any_bad |= mask
        if mask.any():
            part = df[mask]
            records = part.astype(object).where(part.notna(), None).to_dict("records")
            reject_parts.append(pd.DataFrame({
                "ticker": part["ticker"].values,
                "trade_date": part["trade_date"].values,
                "rule_name": name,
                "reason": REASONS[name],
                "row_json": [json.dumps(r, default=str) for r in records],
            }))
    rejects = pd.concat(reject_parts, ignore_index=True) if reject_parts else pd.DataFrame(
        columns=["ticker", "trade_date", "rule_name", "reason", "row_json"])

    survivors = df[~any_bad].reset_index(drop=True)
    flags = _flags(df, survivors, index_ticker, extreme_move)

    summary = {name: int(mask.fillna(False).sum()) for name, mask in masks.items()}
    summary.update({"rows_in": len(df), "rows_clean": len(survivors),
                    "flags": flags["rule_name"].value_counts().to_dict() if len(flags) else {}})
    return ValidationResult(survivors, rejects, flags, summary)
