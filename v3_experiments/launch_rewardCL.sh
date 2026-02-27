#!/bin/bash
# =============================================================================
# Launch: Reward-Conditioned SupCon (Experiment 4b)
# =============================================================================
# Two conditions × 5 seeds across 2 GPUs.
#
# individual_rewardCL : per-agent rewards + SupCon over contribution rank
#                       → shows CL helps when reward signal is already diverse
# shared_rewardCL     : shared team rewards + SupCon over contribution rank
#                       → KEY TEST: rank labels from env (not shared reward),
#                         so SupCon has a differentiated signal even when PPO
#                         gradient is homogenised. Can it prevent collapse?
#
# By default uses GPUs 2 and 3. Override with GPU_A / GPU_B env vars:
#   GPU_A=4 GPU_B=5 bash launch_rewardCL.sh
#
# Usage:
#   cd /home/ubuntu/metta
#   bash v3_experiments/launch_rewardCL.sh
# =============================================================================

set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG_DIR="v3_experiments/logs_reward_ablation"
mkdir -p "$LOG_DIR"

GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}

echo "============================================================"
echo "  Reward-Conditioned SupCon Launch — $(date)"
echo "  GPU_A=$GPU_A (individual_rewardCL)  GPU_B=$GPU_B (shared_rewardCL)"
echo "============================================================"

# GPU_A: individual rewards + reward-conditioned SupCon
CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON $SCRIPT \
    --reward_type individual --reward_cl \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/individual_rewardCL_18ag_seed0-4.log" 2>&1 &
PID_A=$!
echo "[GPU $GPU_A] individual_rewardCL  18-agent  seeds 0-4  (PID $PID_A)"

# GPU_B: shared rewards + reward-conditioned SupCon  (THE KEY TEST)
CUDA_VISIBLE_DEVICES=$GPU_B $PYTHON $SCRIPT \
    --reward_type shared --reward_cl \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/shared_rewardCL_18ag_seed0-4.log" 2>&1 &
PID_B=$!
echo "[GPU $GPU_B] shared_rewardCL      18-agent  seeds 0-4  (PID $PID_B)"

echo ""
echo "Waiting for completion... (logs in $LOG_DIR/)"
wait $PID_A $PID_B
echo ""
echo "=== Reward-Conditioned SupCon Complete at $(date) ==="
