# Shortcuts. Run `make help` to list them. Activate the virtualenv first: source .venv/bin/activate
.PHONY: help setup test demo dashboard-demo init ingest evaluate evaluate-fast report freeze live benchmark dashboard status

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ -/'

setup:            ## one-time WSL setup (PostgreSQL, venv, packages)
	bash scripts/setup_wsl.sh

test:             ## run all tests (DB tests need the volab_test database)
	python -m pytest -q

demo:             ## whole system on SYNTHETIC data (no internet needed)
	volab --config config/demo.yaml demo

dashboard-demo:   ## dashboard on the demo data
	streamlit run app/dashboard.py -- --config config/demo.yaml

init:             ## create tables in the real database
	volab init-db

ingest:           ## fetch + validate + load (REAL data)
	volab ingest

evaluate:         ## the full modelling study (slow: 10-20 min)
	volab evaluate --ablation

evaluate-fast:    ## quick version for checking things work
	volab evaluate --fast

report:           ## write results/MODEL_CARD.md
	volab report

freeze:           ## fit + save the models for live forecasting
	volab freeze

live:             ## today's forecasts + scoring (what the nightly job runs)
	volab live-run

benchmark:        ## scale benchmark on synthetic rows
	volab benchmark

dashboard:        ## dashboard on real data
	streamlit run app/dashboard.py

status:           ## what is in the database
	volab status
