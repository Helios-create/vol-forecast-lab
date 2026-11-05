"""Saving results to disk and writing the MODEL_CARD.md from them.

The model card is generated FROM the saved result files, so the numbers in it can
never drift from the numbers the code actually produced. You still have to read it and
add your own interpretation: the generator provides the facts, you provide the insight.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def git_commit(root: str) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def save_results(res: dict, cfg: dict, extra: dict | None = None) -> Path:
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res["predictions"].to_csv(out / "predictions.csv", index=False)
    for key, fname in [("folds", "folds.csv"), ("overall", "metrics_overall.csv"), ("by_ticker", "metrics_by_ticker.csv"),
                       ("by_regime", "metrics_by_regime.csv"), ("dm", "dm_tests.csv"), ("dm_summary", "dm_summary.csv"),
                       ("grid_scores", "gbm_grid_scores.csv"), ("importance", "gbm_importance.csv")]:
        res[key].to_csv(out / fname, index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(cfg["_root"]),
        "data_source": cfg["data_source"],
        "tickers": cfg["tickers"], "horizon": cfg["horizon"], "test_start": cfg["test_start"],
        "refit_every": cfg["refit_every"], "benchmark_model": cfg["benchmark_model"],
        "gbm_params": res["gbm_params"], "regime_threshold_log_var": res["regime_threshold"],
        "n_test_rows": res["n_test_rows"], "test_range": res["test_range"], **(extra or {}),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return out


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal DataFrame -> Markdown table (avoids an extra dependency)."""
    if df is None or df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_model_card(cfg: dict) -> Path:
    d = Path(cfg["results_dir"])
    if not (d / "run_summary.json").exists():
        raise FileNotFoundError("No results yet. Run `volab evaluate` first.")
    s = json.loads((d / "run_summary.json").read_text())
    overall = pd.read_csv(d / "metrics_overall.csv")
    regime = pd.read_csv(d / "metrics_by_regime.csv")
    dmsum = pd.read_csv(d / "dm_summary.csv")
    imp = pd.read_csv(d / "gbm_importance.csv")
    bench = s["benchmark_model"]

    synthetic = s["data_source"] == "synthetic"
    banner = ("> **SYNTHETIC DATA. These numbers are NOT real and must never appear on a resume.**\n\n" if synthetic else "")
    best = overall.sort_values("rmse").iloc[0]

    txt = f"""# Model Card: 5-day-ahead volatility forecasting

{banner}_Generated {s['created_at']} from commit `{s['git_commit']}`. Facts below are produced by the code; the
**Interpretation** section is yours to write._

## Setup
- Data source: **{s['data_source']}**, {len(s['tickers'])} stocks, test period {s['test_range'][0]} to {s['test_range'][1]} ({s['n_test_rows']} out-of-sample stock-days)
- Target: log of the mean Parkinson variance over the next {s['horizon']} trading days
- Walk-forward: refit every {s['refit_every']} trading days; training labels purged by {s['horizon']} days; test starts {s['test_start']}
- Benchmark for significance tests: **{bench}**
- GBM hyperparameters (chosen inside the training window): `{s['gbm_params']}`

## Accuracy (lower is better; `rmse_vs_{bench}` is the % change in RMSE relative to {bench})
{_md(overall)}
Lowest RMSE overall: **{best['model']}** ({best['rmse']:.4f}).

## Is the difference real? Diebold-Mariano tests vs {bench} (per stock, squared-error loss, HAC errors)
`better_raw` = number of stocks where the model is significantly better at 5% without correction; `better_holm` = after Holm
multiple-testing correction (the number to trust).
{_md(dmsum)}
## Where does each model do well or badly? (market regimes)
High-volatility regime threshold (index 22-day log variance, training 75th percentile): {s['regime_threshold_log_var']:.3f}
{_md(regime)}
## What does the gradient boosting model rely on? (permutation importance, validation slice)
{_md(imp)}
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
"""
    out = d / "MODEL_CARD.md"
    out.write_text(txt)
    return out
