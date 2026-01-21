#!/usr/bin/env bash
#
# Queue Tribal Village experiments to run after metta experiments complete.
# Monitors GPU usage and launches tribal village when GPUs are free.
#

set -euo pipefail

METTA_DIR="/home/ubuntu/metta"
TRIBAL_DIR="/home/ubuntu/tribal-village"
LOG_FILE="/home/ubuntu/metta/train_dir/tribal_queue.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_metta_running() {
    # Check if any v2_gc_crl experiments are still running
    pgrep -f "v2_gc_crl_experiments" > /dev/null 2>&1
}

get_free_gpus() {
    # Get GPUs with less than 1GB memory used (essentially free)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F', ' '$2 < 1000 {print $1}'
}

launch_tribal_experiment() {
    local gpu=$1
    local mode=$2
    local seed=$3

    log "Launching tribal village $mode seed $seed on GPU $gpu"

    cd "$METTA_DIR"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash "$TRIBAL_DIR/scripts/train_contrastive.sh" \
        --"$mode" \
        --seed "$seed" \
        --gpu "$gpu" \
        --steps 100000000 \
        > "$TRIBAL_DIR/train_dir/tribal_${mode}_s${seed}_gpu${gpu}.log" 2>&1 &

    echo $!
}

log "=============================================="
log "Tribal Village Queue Monitor Started"
log "=============================================="
log "Waiting for metta experiments to complete..."

# Wait for metta experiments to finish
while check_metta_running; do
    # Show progress every 5 minutes
    sleep 300
    if check_metta_running; then
        log "Metta experiments still running..."
        # Show current progress
        for f in "$METTA_DIR"/train_dir/v2_arena_*/logs/script.log "$METTA_DIR"/train_dir/v2_nav_*/logs/script.log; do
            if [[ -f "$f" ]]; then
                tail -1 "$f" 2>/dev/null | grep -oE "epoch [0-9]+ .* \([0-9.]+%\)" || true
            fi
        done
    fi
done

log "Metta experiments completed!"
log "Waiting 30 seconds for cleanup..."
sleep 30

# Get list of free GPUs
FREE_GPUS=$(get_free_gpus)
log "Free GPUs: $FREE_GPUS"

# Define experiments to run
declare -a EXPERIMENTS=(
    "baseline:1"
    "baseline:2"
    "baseline:3"
    "gc-crl:1"
    "gc-crl:2"
    "gc-crl:3"
)

# Track PIDs
declare -A GPU_PIDS

job_idx=0
total_jobs=${#EXPERIMENTS[@]}

log "Launching $total_jobs tribal village experiments..."

# Initial launch on all free GPUs
for gpu in $FREE_GPUS; do
    if [[ $job_idx -lt $total_jobs ]]; then
        exp="${EXPERIMENTS[$job_idx]}"
        mode="${exp%:*}"
        seed="${exp#*:}"

        pid=$(launch_tribal_experiment "$gpu" "$mode" "$seed")
        GPU_PIDS[$gpu]=$pid
        ((job_idx++))

        sleep 10  # Small delay between launches
    fi
done

# Monitor and launch remaining experiments as GPUs free up
while [[ $job_idx -lt $total_jobs ]]; do
    sleep 60

    for gpu in ${!GPU_PIDS[@]}; do
        pid=${GPU_PIDS[$gpu]}
        if ! kill -0 "$pid" 2>/dev/null; then
            # GPU is free, launch next experiment
            if [[ $job_idx -lt $total_jobs ]]; then
                exp="${EXPERIMENTS[$job_idx]}"
                mode="${exp%:*}"
                seed="${exp#*:}"

                pid=$(launch_tribal_experiment "$gpu" "$mode" "$seed")
                GPU_PIDS[$gpu]=$pid
                ((job_idx++))
            fi
        fi
    done
done

log "All tribal village experiments launched!"
log "Monitor with: tail -f $TRIBAL_DIR/train_dir/*.log"

# Wait for all to complete
log "Waiting for all experiments to complete..."
wait

log "=============================================="
log "All tribal village experiments completed!"
log "=============================================="
