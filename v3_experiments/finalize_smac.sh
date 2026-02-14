#!/bin/bash
# Wait for SMAC results to be updated, then finalize paper

RESULTS_FILE="/home/ubuntu/metta/v3_experiments/results_smac.json"
INITIAL_MTIME=$(stat -c %Y "$RESULTS_FILE" 2>/dev/null || echo "0")

echo "Monitoring $RESULTS_FILE (initial mtime: $INITIAL_MTIME)"
echo "Checking every 5 minutes..."

while true; do
    CURRENT_MTIME=$(stat -c %Y "$RESULTS_FILE" 2>/dev/null || echo "0")
    if [ "$CURRENT_MTIME" != "$INITIAL_MTIME" ]; then
        echo "SMAC results updated! New mtime: $CURRENT_MTIME"
        echo "Results file contents:"
        cat "$RESULTS_FILE"
        echo ""
        echo "SMAC EXPERIMENTS COMPLETE - ready for final paper update"
        exit 0
    fi

    # Show progress from SMAC output
    LAST_STEP=$(grep "\[" /tmp/claude-1001/-home-ubuntu-metta/tasks/b257c99.output 2>/dev/null | grep "Step" | tail -1)
    echo "$(date '+%H:%M:%S') Still waiting... Last logged: $LAST_STEP"

    sleep 300
done
