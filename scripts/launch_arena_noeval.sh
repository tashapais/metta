#!/bin/bash
# Launch Arena experiments WITHOUT evaluation (eval causes deadlocks)
# We track hearts.created from the training progress logger instead
set -euo pipefail

METTA_DIR=/home/ubuntu/metta
PYTHON="${METTA_DIR}/.venv/bin/python"
RUN_CMD="${METTA_DIR}/tools/run.py"
RECIPE="v3_icml_experiments"
LOG_DIR="${METTA_DIR}/logs/icml_experiments"

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:/usr/local/bin:/usr/bin:$PATH
export CXX=g++-12
export CC=gcc-12

mkdir -p "$LOG_DIR"

run_pair() {
    local exp0=$1 seed0=$2 exp1=$3 seed1=$4
    echo "[$(date)] GPU0: ${exp0} seed=${seed0} | GPU1: ${exp1} seed=${seed1}"

    CUDA_VISIBLE_DEVICES=0 $PYTHON $RUN_CMD ${RECIPE}.${exp0} \
        run="${exp0}_s${seed0}" system.seed=${seed0} \
        > "${LOG_DIR}/${exp0}_seed${seed0}_gpu0.log" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 $PYTHON $RUN_CMD ${RECIPE}.${exp1} \
        run="${exp1}_s${seed1}" system.seed=${seed1} \
        > "${LOG_DIR}/${exp1}_seed${seed1}_gpu1.log" 2>&1 &
    local pid1=$!

    wait $pid0 || echo "WARNING: ${exp0} seed=${seed0} exited non-zero"
    wait $pid1 || echo "WARNING: ${exp1} seed=${seed1} exited non-zero"
    echo "[$(date)] Pair complete"
}

# Seeds 2-10 for baseline+PPO+C (seed 1 done, seed 2 baseline incomplete)
echo "=== Phase 1: 10-seed baseline + PPO+C (seeds 2-10) ==="
for seed in 2 3 4 5 6 7 8 9 10; do
    run_pair "train_baseline_10seed" $seed "train_ppoc_10seed" $seed
done

# Phase 2: Regularizer baselines
echo "=== Phase 2: Regularizer baselines ==="
for seed in 1 2 3; do
    run_pair "train_l2_regularizer" $seed "train_matched_capacity" $seed
done

# Phase 3: Therapeutic window
echo "=== Phase 3: Therapeutic window ==="
for seed in 1 2 3; do
    run_pair "train_ppoc_coef_003" $seed "train_ppoc_coef_005" $seed
done
for seed in 1 2 3; do
    run_pair "train_ppoc_coef_05" $seed "train_l2_regularizer_high" $seed
done

# Phase 4: Complexity scaling
echo "=== Phase 4: Complexity scaling ==="
for seed in 1 2 3; do
    run_pair "train_baseline_6agent" $seed "train_ppoc_6agent" $seed
    run_pair "train_baseline_12agent" $seed "train_ppoc_12agent" $seed
done

echo "=== All experiments complete at $(date) ==="
