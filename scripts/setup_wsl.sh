#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu 24.04 (Python 3.12). Run from the project root:  bash scripts/setup_wsl.sh
# It installs PostgreSQL, creates the database users, creates a virtualenv and installs the project.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing system packages (asks for your sudo password)"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip postgresql postgresql-client git

echo "==> Starting PostgreSQL"
sudo service postgresql start

echo "==> Creating database user and databases (skips ones that already exist)"
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='volab'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE USER volab WITH PASSWORD 'volab' CREATEDB;"
for db in volab volab_test volab_demo; do
  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE $db OWNER volab;"
done

echo "==> Creating virtualenv and installing the project"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo
echo "Setup finished. Next:"
echo "  source .venv/bin/activate"
echo "  make test        # all tests should pass"
echo "  make demo        # full system on synthetic data"
