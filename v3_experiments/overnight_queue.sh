#!/bin/bash
# Overnight experiment queue manager
# This script monitors running experiments and starts new ones when resources are available

set -e
cd /home/ubuntu/metta
source .venv/bin/activate

LOG_FILE="/tmp/overnight_queue.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

count_running() {
    local pattern=$1
    ps aux | grep "$pattern" | grep -v grep | wc -l
}

wait_for_gpu_memory() {
    local gpu_id=$1
    local min_free_mb=$2
    while true; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        local total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id)
        local free=$((total - used))
        if [ "$free" -gt "$min_free_mb" ]; then
            return 0
        fi
        sleep 60
    done
}

log "Starting overnight queue manager"
log "Current directory: $(pwd)"

# Phase 1: Wait for current experiments to finish
log "Phase 1: Monitoring current experiments..."

while true; do
    mujoco_count=$(count_running "run_mujoco")
    nav_count=$(count_running "run_navigation")
    craftax_count=$(count_running "run_craftax")
    sweep_count=$(count_running "wandb agent")

    log "Status: MuJoCo=$mujoco_count, Navigation=$nav_count, Craftax=$craftax_count, Sweep=$sweep_count"

    # Check if Craftax baseline finished, start contrastive
    if [ "$craftax_count" -eq 0 ]; then
        if [ ! -f "/tmp/craftax_contrastive_started" ]; then
            log "Starting Craftax contrastive experiments..."
            CUDA_VISIBLE_DEVICES=0 nohup python v3_experiments/run_craftax.py --contrastive --total_timesteps 1000000 > /tmp/craftax_contrastive.log 2>&1 &
            touch /tmp/craftax_contrastive_started
            log "Craftax contrastive started (PID: $!)"
        fi
    fi

    # If MuJoCo finished and there's GPU space, start more Navigation sweep agents
    if [ "$mujoco_count" -eq 0 ] && [ "$sweep_count" -lt 6 ]; then
        log "MuJoCo finished, starting more sweep agents..."
        for i in $(seq $((sweep_count + 1)) 6); do
            nohup wandb agent tashapais/metta/6bqtxo02 > /tmp/sweep_agent_$i.log 2>&1 &
            log "Started sweep agent $i"
            sleep 2
        done
    fi

    # Check if everything is done
    if [ "$mujoco_count" -eq 0 ] && [ "$nav_count" -eq 0 ] && [ "$craftax_count" -eq 0 ] && [ "$sweep_count" -eq 0 ]; then
        log "All experiments completed!"
        break
    fi

    sleep 300  # Check every 5 minutes
done

log "Overnight queue manager finished"
