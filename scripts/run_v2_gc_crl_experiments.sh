#!/usr/bin/env bash
#
# V2 Goal-Conditioned CRL Experiments
#
# Runs the new GC-CRL experiments (fundamentally different from auxiliary contrastive).
# Uses 3 seeds across 4 A100 GPUs.
#
# Key differences from v1 (auxiliary contrastive):
# - GC-CRL uses contrastive learning as PRIMARY objective (not auxiliary)
# - Much higher coefficient (0.1 vs 0.00068)
# - Dedicated dual encoders (SA + Goal)
# - Logsumexp regularization
#

set -euo pipefail

# Ensure CUDA is configured
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

NUM_SEEDS=3
TOTAL_TIMESTEPS=100000000
NUM_GPUS=4
PROJECT_NAME="metta"
WANDB_ENTITY="${WANDB_ENTITY:-tashapais}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METTA_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${METTA_DIR}/train_dir/v2_gc_crl_experiments_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "V2 Goal-Conditioned CRL Experiments"
echo "=============================================="
echo "Seeds: $NUM_SEEDS"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "GPUs: $NUM_GPUS"
echo "Log dir: $LOG_DIR"
echo ""

# V2 experiments - GC-CRL focused
declare -a EXPERIMENTS=(
    "arena_baseline"
    "arena_gc_crl"
    "navigation_baseline"
    "navigation_gc_crl"
)

# Build job queue
declare -a JOB_QUEUE=()
for experiment in "${EXPERIMENTS[@]}"; do
    for seed in $(seq 1 "$NUM_SEEDS"); do
        JOB_QUEUE+=("${experiment}:${seed}")
    done
done

total_jobs=${#JOB_QUEUE[@]}
echo "Total jobs: $total_jobs"
echo "Experiments: ${EXPERIMENTS[*]}"
echo ""

# Track GPU PIDs
declare -a GPU_PIDS=("" "" "" "")

job_idx=0

cd "$METTA_DIR"

while [[ $job_idx -lt $total_jobs ]]; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [[ -z "${GPU_PIDS[$gpu]}" ]] || ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
            if [[ $job_idx -lt $total_jobs ]]; then
                job="${JOB_QUEUE[$job_idx]}"
                experiment="${job%:*}"
                seed="${job#*:}"
                log_file="${LOG_DIR}/${experiment}_seed${seed}.log"
                run_name="v2_${experiment}_seed${seed}"
                run_id="${experiment}_seed${seed}_$(date +%s)"

                echo "[$((job_idx + 1))/$total_jobs] GPU $gpu: $experiment (seed $seed)"

                CUDA_VISIBLE_DEVICES=$gpu nohup uv run ./tools/run.py train v2_gc_crl_experiments \
                    experiment="$experiment" \
                    run="$run_name" \
                    trainer.total_timesteps="$TOTAL_TIMESTEPS" \
                    training_env.num_workers=30 \
                    training_env.auto_workers=false \
                    system.seed="$seed" \
                    wandb.enabled=True \
                    wandb.project="$PROJECT_NAME" \
                    wandb.entity="$WANDB_ENTITY" \
                    wandb.run_id="$run_id" \
                    > "$log_file" 2>&1 &

                GPU_PIDS[$gpu]=$!
                ((job_idx++))
                sleep 5  # Small delay between launches
            fi
        fi
    done
    sleep 30
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo ""

# Wait for all jobs
wait

echo "=============================================="
echo "All V2 GC-CRL experiments completed!"
echo "Logs: $LOG_DIR"
echo "=============================================="
