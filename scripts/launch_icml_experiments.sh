#!/bin/bash
# Launch ICML experiments across 2x A100 GPUs
# Each experiment runs 100M timesteps
# Two experiments run in parallel (one per GPU), sequentially within each GPU
set -euo pipefail

METTA_DIR=/home/ubuntu/metta
PYTHON="${METTA_DIR}/.venv/bin/python"
RUN_CMD="${METTA_DIR}/tools/run.py"
RECIPE="v3_icml_experiments"
LOG_DIR="${METTA_DIR}/logs/icml_experiments"

# CUDA 12.2 nvcc + g++-12 needed for cortex C++ extension compilation
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:/usr/local/bin:/usr/bin:$PATH
export CXX=g++-12
export CC=gcc-12

mkdir -p "$LOG_DIR"

# Run two experiments in parallel (one per GPU), wait for both to finish
run_pair() {
    local exp0=$1 seed0=$2 exp1=$3 seed1=$4
    local log0="${LOG_DIR}/${exp0}_seed${seed0}_gpu0.log"
    local log1="${LOG_DIR}/${exp1}_seed${seed1}_gpu1.log"

    echo "[$(date)] GPU0: ${exp0} seed=${seed0} | GPU1: ${exp1} seed=${seed1}"

    CUDA_VISIBLE_DEVICES=0 $PYTHON $RUN_CMD ${RECIPE}.${exp0} \
        run="${exp0}_s${seed0}" system.seed=${seed0} \
        > "$log0" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 $PYTHON $RUN_CMD ${RECIPE}.${exp1} \
        run="${exp1}_s${seed1}" system.seed=${seed1} \
        > "$log1" 2>&1 &
    local pid1=$!

    wait $pid0 || echo "WARNING: ${exp0} seed=${seed0} exited non-zero"
    wait $pid1 || echo "WARNING: ${exp1} seed=${seed1} exited non-zero"
    echo "[$(date)] Pair complete"
}

# ============================================================================
# Phase 1: 10-seed baseline + PPO+C (highest priority)
# ============================================================================
echo "=== Phase 1: 10-seed key conditions ==="
for seed in 1 2 3 4 5 6 7 8 9 10; do
    run_pair "train_baseline_10seed" $seed "train_ppoc_10seed" $seed
done

# ============================================================================
# Phase 2: Regularizer baselines + therapeutic window
# ============================================================================
echo "=== Phase 2: Regularizer baselines + therapeutic window ==="
for seed in 1 2 3; do
    run_pair "train_l2_regularizer" $seed "train_ppoc_coef_003" $seed
    run_pair "train_l2_regularizer_high" $seed "train_ppoc_coef_005" $seed
    run_pair "train_matched_capacity" $seed "train_ppoc_coef_05" $seed
done

# ============================================================================
# Phase 3: Complexity scaling (Arena 6 and 12 agents)
# ============================================================================
echo "=== Phase 3: Complexity scaling ==="
for seed in 1 2 3; do
    run_pair "train_baseline_6agent" $seed "train_ppoc_6agent" $seed
    run_pair "train_baseline_12agent" $seed "train_ppoc_12agent" $seed
done

echo "=== All experiments complete at $(date) ==="
