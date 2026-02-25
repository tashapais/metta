#!/bin/bash
# Launch contrastive vs baseline paper experiments
# GPU 0: MAPPO baseline (5 seeds)
# GPU 1: MAPPO + InfoNCE contrastive (5 seeds)
# GPU 2: available for additional runs

set -e
cd /home/devuser/metta

PYTHON="/home/devuser/metta/.venv/bin/python"
LOG_DIR="/tmp/experiment_logs"
mkdir -p "$LOG_DIR"

TOTAL_TIMESTEPS=5000000  # 5M steps per seed (~2h total with 5 seeds on 1 GPU)

echo "$(date): Starting contrastive paper experiments" | tee "$LOG_DIR/contrastive_master.log"
echo "  GPU 0: MAPPO baseline (5 seeds)" | tee -a "$LOG_DIR/contrastive_master.log"
echo "  GPU 1: MAPPO+Contrastive (5 seeds)" | tee -a "$LOG_DIR/contrastive_master.log"

# GPU 0: Baseline
(
    echo "$(date): [GPU0] Starting baseline runs" | tee -a "$LOG_DIR/contrastive_master.log"
    CUDA_VISIBLE_DEVICES=0 $PYTHON v3_experiments/paper_exp_contrastive.py \
        --method baseline \
        --gpu 0 \
        --num_seeds 5 \
        --num_agents 18 \
        --total_timesteps $TOTAL_TIMESTEPS \
        >> "$LOG_DIR/contrastive_baseline.log" 2>&1
    echo "$(date): [GPU0] Baseline DONE" | tee -a "$LOG_DIR/contrastive_master.log"
) &
GPU0_PID=$!

# GPU 1: Contrastive
(
    echo "$(date): [GPU1] Starting contrastive runs" | tee -a "$LOG_DIR/contrastive_master.log"
    CUDA_VISIBLE_DEVICES=1 $PYTHON v3_experiments/paper_exp_contrastive.py \
        --method contrastive \
        --gpu 1 \
        --num_seeds 5 \
        --num_agents 18 \
        --total_timesteps $TOTAL_TIMESTEPS \
        >> "$LOG_DIR/contrastive_contrastive.log" 2>&1
    echo "$(date): [GPU1] Contrastive DONE" | tee -a "$LOG_DIR/contrastive_master.log"
) &
GPU1_PID=$!

echo "$(date): GPU0 PID=$GPU0_PID, GPU1 PID=$GPU1_PID" | tee -a "$LOG_DIR/contrastive_master.log"
wait $GPU0_PID $GPU1_PID
echo "$(date): ALL CONTRASTIVE EXPERIMENTS COMPLETE" | tee -a "$LOG_DIR/contrastive_master.log"
