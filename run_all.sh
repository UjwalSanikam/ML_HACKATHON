#!/usr/bin/env bash
# Reproduce every result in results/ and every artifact in artifacts/ from the raw data.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY -m pytest -q tests
$PY scripts/01_eda.py
$PY scripts/02_build_dataset.py
$PY scripts/03_train_compare.py
$PY scripts/04_bayes_opt.py --model lightgbm --trials 50
$PY scripts/05_hmm_eval.py --trials 40
$PY scripts/06_figures.py
$PY scripts/07_report_assets.py
