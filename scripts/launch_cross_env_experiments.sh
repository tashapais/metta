#!/bin/bash
# Launch ONLY the cross-environment experiments we're MISSING
#
# Already have (don't repeat):
#   - MuJoCo Ant baseline + contrastive: 3 seeds (0-2)
#   - MuJoCo Swimmer baseline + contrastive: 3 seeds each
#   - MuJoCo GC-CRL buggy: Ant 2 seeds, Swimmer 3 seeds
#   - Craftax baseline + contrastive + GC-CRL: 3 seeds each
#
# Need:
#   - MuJoCo Ant baseline: 7 MORE seeds (3-9) to get 10 total
#   - MuJoCo Ant contrastive: 7 MORE seeds (3-9) to get 10 total
#   - MuJoCo Ant L2: 10 seeds (new method)
#   - MuJoCo Ant matched-capacity: 3 seeds (new method)
#
# Total: 27 new MuJoCo runs, ~5 min each = ~2.5 hrs serial
set -euo pipefail

METTA_DIR=/home/ubuntu/metta
PYTHON="${METTA_DIR}/.venv/bin/python"
LOG_DIR="${METTA_DIR}/logs/icml_cross_env"

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:/usr/local/bin:/usr/bin:$PATH
export CXX=g++-12
export CC=gcc-12

export WANDB_MODE=disabled
mkdir -p "$LOG_DIR"

echo "=== MuJoCo Ant: Additional baseline seeds (3-9) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -u ${METTA_DIR}/v3_experiments/run_mujoco.py \
    --task ant --method baseline --num_seeds 7 --seed_offset 3 \
    --total_timesteps 10000000 --num_envs 32 --video_interval 9999 \
    > "${LOG_DIR}/mujoco_ant_baseline_extra.log" 2>&1
echo "[$(date)] Done"

echo "=== MuJoCo Ant: Additional contrastive seeds (3-9) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -u ${METTA_DIR}/v3_experiments/run_mujoco.py \
    --task ant --method contrastive --num_seeds 7 --seed_offset 3 \
    --total_timesteps 10000000 --num_envs 32 --video_interval 9999 \
    > "${LOG_DIR}/mujoco_ant_contrastive_extra.log" 2>&1
echo "[$(date)] Done"

echo "=== MuJoCo Ant: L2 regularizer (10 seeds) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -u ${METTA_DIR}/v3_experiments/run_mujoco.py \
    --task ant --method l2 --num_seeds 10 \
    --total_timesteps 10000000 --num_envs 32 --video_interval 9999 \
    > "${LOG_DIR}/mujoco_ant_l2.log" 2>&1
echo "[$(date)] Done"

echo "=== MuJoCo Ant: Matched-capacity (3 seeds) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -u ${METTA_DIR}/v3_experiments/run_mujoco.py \
    --task ant --method matched_capacity --num_seeds 3 \
    --total_timesteps 10000000 --num_envs 32 --video_interval 9999 \
    > "${LOG_DIR}/mujoco_ant_matched_capacity.log" 2>&1
echo "[$(date)] Done"

echo "=== All cross-environment experiments complete at $(date) ==="
