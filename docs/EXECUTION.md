# EXECUTION GUIDE: how to run it, make it good, and use it

The code is finished and tested. Your job is to **run it on real data, understand it, extend it a little, write it up honestly,
and apply**. This guide is written for the window 29 Oct to 12 Nov 2026 (Primenumbers closes 12 Nov, QuickHyre 15 Nov).

## 0. The plan in one table

| When | Do this | Time | Checkpoint |
|---|---|---|---|
| **Thu 29 - Fri 30 Oct** | Section 1-3: install, tests, demo, read STUDY_GUIDE parts 1-2 | 3-4 h | `make test` passes, you ran the demo |
| **Sat 31 Oct** | Section 4: real data (`check-yfinance`, `ingest --full`), inspect rejects, `volab benchmark`, git + GitHub | 3 h | real data in PostgreSQL, benchmark table saved |
| **Sun 1 Nov** | **Apply to Primenumbers** (data-engineering story is complete). Section 5: edit + commit DESIGN.md, `evaluate --fast` | 3 h | application sent, DESIGN.md committed |
| **Mon 2 - Wed 4 Nov** | Full `evaluate --ablation`, `report`, read every result file with STUDY_GUIDE part 4 | 4-5 h | you can explain every number |
| **Thu 5 - Sat 7 Nov** | Section 6: one extension of your own, `freeze`, start nightly live run | 5 h | extension amended in DESIGN.md and run |
| **Sun 8 - Tue 10 Nov** | Interpretation paragraph, README with real numbers, demo video, interview practice | 5 h | MODEL_CARD interpretation written |
| **Wed 11 - Thu 12 Nov** | Resume, application, **apply to QuickHyre** | 2 h | applications sent with 3 days of buffer |

Your subject blocks stay untouched; use the project slots from your timetable. If you fall behind, jump to Section 9 (minimum path).

## 1. Install (WSL2 + Ubuntu 24.04), once
1. Windows PowerShell **as Administrator**: `wsl --install -d Ubuntu-24.04`, reboot, create a Linux username/password.
2. Put the zip in your Linux home. In Windows Explorer open `\\wsl$\Ubuntu-24.04\home\<your-linux-user>` and paste `vol-forecast-lab.zip`.
3. In the Ubuntu terminal:
   ```bash
   cd ~ && sudo apt-get install -y unzip && unzip vol-forecast-lab.zip && cd vol-forecast-lab
   bash scripts/setup_wsl.sh        # asks for your sudo password; takes a few minutes
   source .venv/bin/activate
   ```
   It must be **Ubuntu 24.04** (Python 3.12): the pinned package versions need Python 3.11+.
4. After every Windows restart, PostgreSQL must be started again: `sudo service postgresql start`.

## 2. Prove it works
```bash
make test            # expect: all tests pass (DB tests need PostgreSQL running)
make demo            # about 2-3 minutes; SYNTHETIC data; prints every stage
make dashboard-demo  # open http://localhost:8501 in your Windows browser
```
Read the demo output slowly. Every stage is something you will explain in an interview: ingest with injected faults (the
validator finds exactly what was broken), a second ingest (idempotent), evaluation, model card, frozen models, simulated live days, scores.

## 3. Understand before you proceed
Read `docs/STUDY_GUIDE.md` parts 1 and 2 (data layer, validation, loading) now, and keep the SQL file open:
```bash
PGPASSWORD=volab psql -h localhost -U volab -d volab_demo -f sql/analysis_queries.sql
```
Predict each query's output before running it.

## 4. Real data
```bash
git init && git add -A && git commit -m "Initial project"     # first commit
volab check-yfinance                  # must print OK. If not, see Troubleshooting
volab init-db
volab ingest --full                   # downloads ~11 years x 21 symbols; a few minutes
volab status
```
Then **look at what the validator found**:
```bash
PGPASSWORD=volab psql -h localhost -U volab -d volab -c "select rule_name, ticker, count(*) from rejects group by 1,2 order by 3 desc limit 20"
PGPASSWORD=volab psql -h localhost -U volab -d volab -c "select rule_name, ticker, trade_date, detail from data_flags order by trade_date desc limit 30"
```
Questions to answer in your notes: Which tickers have the most rejects/flags and why? Do the extreme-move flags match real
events (March 2020 crash, results days)? Is any ticker missing history (a symbol that changed)? Remove broken symbols from
`config/config.yaml`, and record the removal in DESIGN.md section 3.

Then the scale benchmark (about 5-10 minutes; use `--sizes 100000,1000000` to be quick):
```bash
volab benchmark
```
Keep `results/benchmark.csv`: it is your measured answer to "how would this scale to 230 million rows?".

Publish the code to GitHub (create an empty repo on github.com first):
```bash
git remote add origin https://github.com/<you>/vol-forecast-lab.git && git branch -M main && git push -u origin main
```
`.gitignore` already excludes data, logs, models and results. **Do not publish downloaded Yahoo price data**; publish code and result summaries.

## 5. Pre-register, then evaluate
1. Open `docs/DESIGN.md`. Fill sections 2 and 3 honestly (your hypotheses BEFORE seeing results). Leave the protocol unless you have a reason.
2. `git add docs/DESIGN.md config/config.yaml && git commit -m "Pre-register evaluation plan"`. This commit must come BEFORE the next commands.
3. Sanity run: `volab evaluate --fast` (a few minutes). Check nothing is absurd (an RMSE on log variance somewhere around 0.3-0.8 is typical, though that is a rough guide; NaNs or absurd numbers mean a data problem).
4. The real run: `volab evaluate --ablation` (10-25 minutes depending on your laptop). Then `volab report`.
5. Open `results/MODEL_CARD.md` and the CSVs. Use STUDY_GUIDE part 4 to read them. Write the **Interpretation** section yourself.

Never re-run until the result "looks better". If you find a real bug, fix it, record it in DESIGN.md Amendments, and re-run once.

## 6. Make it yours: one extension (this is what separates you from a copy)
Pick ONE, write it as an Amendment in DESIGN.md **before** running, then run and report it, including if it fails.

| Extension | Effort | How |
|---|---|---|
| **Forecast combination** (average HAR and GBM): often beats both | 1-2 h | in `evaluate.walk_forward`, after the models, add `out["har_gbm_avg"] = (out["har"] + out["gbm"]) / 2`. Metrics, tests and the dashboard pick up the new column automatically. |
| **Horizon sweep** h = 1, 5, 22 | 2 h | change `horizon` in a copy of the config (`config/h22.yaml`, with its own `results_dir`), run `volab --config config/h22.yaml evaluate`. Compare tables. |
| **QLIKE-based DM test** | 2-3 h | add a `loss` argument to `dm_table` using the QLIKE formula from `_metrics`. |
| **Semivariance/leverage feature** (down-day volatility) | 3-4 h | new column in `features.build_panel`, add to `FEATURES_ALL`, extend the leakage test, re-run the ablation. |
| **Second universe** (mid caps) | 2 h | new config with different tickers and database; does the ranking of models hold? |

Whichever you pick: add or extend a test for it, and be able to say what you expected and what happened.

## 7. Go live (the forward test)
```bash
volab freeze                          # fits and saves the models; prints the version
volab live-run                        # writes today's forecasts, scores any that matured
```
Automating it (the nightly job runs `ingest` then `live-run`):
* **Simplest and reliable:** run `bash scripts/run_nightly.sh` yourself each evening after 6 PM IST (NSE closes 3:30 PM). Dated lines appear in the output.
* **Automatic while WSL is running:** `sudo service cron start`, then `crontab -e` and add
  `30 18 * * 1-5 /home/<you>/vol-forecast-lab/scripts/run_nightly.sh >> /home/<you>/vol-forecast-lab/logs/cron.log 2>&1`.
  cron inside WSL only runs while WSL is running (your laptop on and WSL started).
* On the dashboard's Live scoreboard, **live** forecasts and **backfilled** ones are shown separately. By 12 Nov you will have at most
  a few scored live forecasts: say so plainly ("live log started on X, N scored forecasts, too few to conclude"). That honesty is the point.

## 8. Packaging for your application
1. **README.md:** keep it, add your real results table, your Interpretation paragraph, and the benchmark table.
2. **Demo video (2 minutes, OBS):** run `volab status`, show the rejects, show `make test`, show the dashboard model comparison and the live scoreboard, state the main finding and one limitation.
3. **Resume bullets** (fill brackets with your own measured numbers only):
   - Built and tested a Python/PostgreSQL pipeline ingesting daily data for [N] NSE stocks with [N] validation rules, a rejects audit table, idempotent COPY/upsert loads and cron scheduling on Linux.
   - Compared naive, EWMA, HAR, GARCH and gradient-boosting volatility forecasts using leakage-free walk-forward validation (purged labels); [model] [did / did not] beat HAR (Diebold-Mariano, Holm-corrected, [N] of [N] stocks).
   - Benchmarked bulk loading on [N] synthetic rows: COPY [X]x faster than row-by-row inserts; an index cut a range query from [A] ms to [B] ms.
   - Added [your extension]; found [what happened].
4. **Application answer (4-5 lines):** what the project asks, the single most interesting result (negative results count), what the live log shows so far, what you want to learn.

**Honesty note (important):** this scaffold was built with AI assistance and you then ran, tested, studied and extended it. Describe it accurately
("developed with AI assistance, then validated and extended by me") if asked, and only claim what you can explain line by line. Interviewers probe;
being able to explain `purge`, `HAC` and `ON CONFLICT` in your own words is what protects you. Prepare with STUDY_GUIDE part 5.

## 9. Minimum path if time runs out
Install -> `make test` -> `check-yfinance` -> `ingest --full` -> benchmark -> commit DESIGN.md -> `evaluate --fast` -> `report` -> README -> apply.
That already gives you a real data pipeline, a real result, and an honest story. Drop the extension and the live run before anything else.

## 10. Troubleshooting
| Symptom | Fix |
|---|---|
| `connection refused` / DB tests skipped | `sudo service postgresql start`; check `psql -h localhost -U volab -d volab` (password `volab`) |
| `database "volab_demo" does not exist` | re-run `bash scripts/setup_wsl.sh` (it creates volab, volab_test, volab_demo) |
| `check-yfinance` prints FAILED | wait a minute and retry (rate limit); `pip install -U yfinance`; check your internet inside WSL (`ping 8.8.8.8`); confirm the symbol on finance.yahoo.com (NSE symbols end in `.NS`) |
| a ticker has far fewer rows than the others | its history is shorter or the symbol changed after a merger/demerger; drop it from `config.yaml` and note it in DESIGN.md |
| `evaluate` says too few training rows | `start_date` is too late or too many tickers failed; check `volab status` |
| pip install fails on a package | you are not on Ubuntu 24.04 / Python 3.12; check `python --version` |
| evaluate is slow | `--fast`, or lower `max_iter` in `gbm_grid`, or set `models.garch: false` |
| dashboard shows "No results yet" | run `volab evaluate` first (the dashboard reads `results/`) |
| Windows laptop slept and the nightly job missed days | run `volab ingest` (it catches up on all missing days), then `volab live-run`; missed forecast days cannot be recreated as live forecasts |
