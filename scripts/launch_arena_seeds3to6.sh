#!/bin/bash
# Launch Arena seeds 3-6 only (baseline + PPO+C) = 8 runs, ~4 hrs
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

echo "=== Arena seeds 3-6: baseline + PPO+C ==="
for seed in 3 4 5 6; do
    run_pair "train_baseline_10seed" $seed "train_ppoc_10seed" $seed
done

echo "=== All experiments complete at $(date) ==="
