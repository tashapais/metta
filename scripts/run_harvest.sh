#!/bin/bash
# Run diagnostics harvesting from metta repo
# Usage: cd /Users/tashapais/Documents/Github/metta && bash scripts/run_harvest.sh

set -e

echo "Activating metta virtual environment..."
source .venv/bin/activate

echo "Installing wandb if needed..."
python -m ensurepip --upgrade 2>/dev/null || true
pip install wandb -q

echo ""
echo "Running diagnostics harvesting..."
python scripts/harvest_diagnostics.py

echo ""
echo "✓ Data saved to diagnostics_data/"
echo "Check diagnostics_data/ for CSV files with metrics from each run"
