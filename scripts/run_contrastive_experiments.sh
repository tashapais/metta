#!/bin/bash
# Script to run all contrastive learning paper experiments
# Usage: ./scripts/run_contrastive_experiments.sh [--seed N] [--dry-run] [experiment_name]

set -e

# Ensure uv is in PATH and CUDA is configured
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# GPU configuration
NUM_GPUS=4

# Wandb configuration
WANDB_PROJECT="metta"
WANDB_ENTITY="tashapais"

# All experiments from the paper (8 total)
EXPERIMENTS=(
    "baseline_ppo"
    "ppo_plus_contrastive"
    "ablation_no_projection"
    "ablation_temp_0.05"
    "ablation_temp_0.5"
    "ablation_coef_0.01"
    "ablation_embed_64"
    "ablation_fixed_offset"
)

# Parse command line arguments
EXPERIMENT=""
DRY_RUN=false
SEED=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# If specific experiment provided, run only that one
if [ -n "$EXPERIMENT" ]; then
    EXPERIMENTS=("$EXPERIMENT")
fi

echo "=========================================="
echo "Contrastive Learning Paper Experiments"
echo "=========================================="
echo "Experiments to run: ${EXPERIMENTS[*]}"
echo "Number of GPUs: $NUM_GPUS"
echo "Seed: $SEED"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Create log directory
LOG_DIR="logs/contrastive_experiments_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Array to track background PIDs
declare -a PIDS=()
declare -a PID_EXPERIMENTS=()

# Function to run an experiment on a specific GPU
run_experiment() {
    local exp=$1
    local gpu_id=$2
    local log_file="$LOG_DIR/${exp}_seed${SEED}.log"

    RUN_ID="${exp}.seed${SEED}.$(date +%m_%d_%y)"
    CMD="uv run ./tools/run.py train contrastive_paper_experiments experiment_name=$exp training_env.num_workers=30 training_env.auto_workers=false evaluator.epoch_interval=0 system.seed=$SEED wandb.enabled=True wandb.project=$WANDB_PROJECT wandb.entity=$WANDB_ENTITY wandb.run_id=$RUN_ID"

    echo "  [GPU $gpu_id] Starting: $exp seed=$SEED (log: $log_file)"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] CUDA_VISIBLE_DEVICES=$gpu_id $CMD"
    else
        CUDA_VISIBLE_DEVICES=$gpu_id $CMD > "$log_file" 2>&1 &
        PIDS+=($!)
        PID_EXPERIMENTS+=("$exp")
    fi
}

# Function to wait for a GPU slot to become available
wait_for_gpu_slot() {
    while [ ${#PIDS[@]} -ge $NUM_GPUS ]; do
        # Check each PID to see if it has completed
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" && echo "  Completed: ${PID_EXPERIMENTS[$i]}" || echo "  FAILED: ${PID_EXPERIMENTS[$i]} (check log)"
                unset 'PIDS[i]'
                unset 'PID_EXPERIMENTS[i]'
                # Re-index arrays
                PIDS=("${PIDS[@]}")
                PID_EXPERIMENTS=("${PID_EXPERIMENTS[@]}")
                return
            fi
        done
        sleep 5
    done
}

echo "----------------------------------------"
echo "Launching experiments in parallel..."
echo "----------------------------------------"

# Launch experiments across GPUs
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    gpu_id=$((i % NUM_GPUS))

    # Wait for a GPU slot if all are busy
    wait_for_gpu_slot

    run_experiment "$exp" "$gpu_id"
done

# Wait for all remaining experiments to complete
echo ""
echo "----------------------------------------"
echo "Waiting for remaining experiments to complete..."
echo "----------------------------------------"

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && echo "  Completed: ${PID_EXPERIMENTS[$i]}" || echo "  FAILED: ${PID_EXPERIMENTS[$i]} (check log)"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "1. Check wandb for logged metrics"
echo "2. Compare learning curves between baseline and ablations"
echo "3. Review individual logs: ls $LOG_DIR/"
echo ""
