#!/usr/bin/env bash
# wait_and_update.sh
# Polls for the 4 inter-agent InfoNCE result JSONs and runs the update script
# when all 4 are present.
#
# Expected files (written by paper_exp_contrastive.py seed 0, then seeds 1-4
# appended by repeated runs):
#   results_contrastive_baseline_18agents.json
#   results_contrastive_contrastive_18agents.json
#   results_contrastive_baseline_24agents.json
#   results_contrastive_contrastive_24agents.json
#
# Each JSON is a list of per-seed dicts. We wait until all 4 files exist AND
# each contains exactly 5 seeds before running the update.

set -euo pipefail

RESULTS_DIR="/home/devuser/metta/v3_experiments"
REQUIRED_SEEDS=5
CHECK_INTERVAL=120   # seconds between polls
LOGFILE="/tmp/wait_and_update_interagent.log"

FILES=(
    "results_contrastive_baseline_18agents.json"
    "results_contrastive_contrastive_18agents.json"
    "results_contrastive_baseline_24agents.json"
    "results_contrastive_contrastive_24agents.json"
)

echo "[$(date -u +%FT%TZ)] wait_and_update.sh started (inter-agent InfoNCE)" | tee -a "$LOGFILE"

while true; do
    all_ready=true
    for f in "${FILES[@]}"; do
        fpath="$RESULTS_DIR/$f"
        if [[ ! -f "$fpath" ]]; then
            echo "[$(date -u +%FT%TZ)] Missing: $f" | tee -a "$LOGFILE"
            all_ready=false
            continue
        fi
        # Count seeds: each seed is one object in the JSON array
        n_seeds=$(python3 -c "import json,sys; data=json.load(open('$fpath')); print(len(data))" 2>/dev/null || echo 0)
        if [[ "$n_seeds" -lt "$REQUIRED_SEEDS" ]]; then
            echo "[$(date -u +%FT%TZ)] $f has $n_seeds/$REQUIRED_SEEDS seeds" | tee -a "$LOGFILE"
            all_ready=false
        else
            echo "[$(date -u +%FT%TZ)] $f OK ($n_seeds seeds)" | tee -a "$LOGFILE"
        fi
    done

    if $all_ready; then
        echo "[$(date -u +%FT%TZ)] All 4 result files ready with $REQUIRED_SEEDS seeds each. Running update..." | tee -a "$LOGFILE"
        cd "$RESULTS_DIR"
        python3 update_paper_table3.py 2>&1 | tee -a "$LOGFILE"
        echo "[$(date -u +%FT%TZ)] Done. Exiting." | tee -a "$LOGFILE"
        exit 0
    fi

    echo "[$(date -u +%FT%TZ)] Not ready yet. Sleeping ${CHECK_INTERVAL}s..." | tee -a "$LOGFILE"
    sleep $CHECK_INTERVAL
done
