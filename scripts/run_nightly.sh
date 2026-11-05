#!/usr/bin/env bash
# Nightly job: ingest new prices, write today's forecasts, score matured ones.
# Exits non-zero if ANY step fails (set -e), so cron/logs show a clear failure.
#
# crontab example (6:30 PM IST on weekdays; NSE closes 3:30 PM):
#   30 18 * * 1-5 /full/path/to/vol-forecast-lab/scripts/run_nightly.sh >> /full/path/to/vol-forecast-lab/logs/cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs
echo "[$(date -Is)] nightly start"
volab ingest
volab live-run
echo "[$(date -Is)] nightly done"
