#!/bin/bash
# Run all experiments for the contrastive learning paper
# This script queues experiments to run overnight

set -e
cd /home/ubuntu/metta
source .venv/bin/activate

LOG_DIR="/tmp/experiment_logs"
mkdir -p $LOG_DIR

echo "$(date): Starting experiment queue" | tee $LOG_DIR/master.log

# ============================================================================
# MUJOCO EXPERIMENTS (GPU 0)
# ============================================================================
echo "$(date): === MUJOCO EXPERIMENTS ===" | tee -a $LOG_DIR/master.log

# Ant - 3 seeds each
for seed in 0 1 2; do
    echo "$(date): Starting MuJoCo Ant baseline seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=0 python v3_experiments/run_mujoco.py --task ant --method baseline --num_envs 32 --total_timesteps 10000000 > $LOG_DIR/mujoco_ant_baseline_s${seed}.log 2>&1 &
    PID_ANT_B=$!

    echo "$(date): Starting MuJoCo Ant contrastive seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=0 python v3_experiments/run_mujoco.py --task ant --method contrastive --num_envs 32 --total_timesteps 10000000 > $LOG_DIR/mujoco_ant_contrastive_s${seed}.log 2>&1 &
    PID_ANT_C=$!

    # Wait for this seed to complete before next
    wait $PID_ANT_B $PID_ANT_C
    echo "$(date): Completed MuJoCo Ant seed $seed" | tee -a $LOG_DIR/master.log
done

# Swimmer - 3 seeds each
for seed in 0 1 2; do
    echo "$(date): Starting MuJoCo Swimmer baseline seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=0 python v3_experiments/run_mujoco.py --task swimmer --method baseline --num_envs 32 --total_timesteps 10000000 > $LOG_DIR/mujoco_swimmer_baseline_s${seed}.log 2>&1 &
    PID_SW_B=$!

    echo "$(date): Starting MuJoCo Swimmer contrastive seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=0 python v3_experiments/run_mujoco.py --task swimmer --method contrastive --num_envs 32 --total_timesteps 10000000 > $LOG_DIR/mujoco_swimmer_contrastive_s${seed}.log 2>&1 &
    PID_SW_C=$!

    wait $PID_SW_B $PID_SW_C
    echo "$(date): Completed MuJoCo Swimmer seed $seed" | tee -a $LOG_DIR/master.log
done

# ============================================================================
# NAVIGATION EXPERIMENTS (GPU 1)
# ============================================================================
echo "$(date): === NAVIGATION EXPERIMENTS ===" | tee -a $LOG_DIR/master.log

# Baseline - 3 seeds
for seed in 0 1 2; do
    echo "$(date): Starting Navigation baseline seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_navigation.py --method baseline --num_envs 16 --num_agents 12 --total_timesteps 10000000 --seed $seed > $LOG_DIR/nav_baseline_s${seed}.log 2>&1 &
done

# Contrastive coefficient sweep
for coef in 0.0001 0.0005 0.001 0.005 0.01; do
    echo "$(date): Starting Navigation contrastive coef=$coef" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_navigation.py --method contrastive --num_envs 16 --num_agents 12 --total_timesteps 10000000 --contrastive_coef $coef --seed 100 > $LOG_DIR/nav_coef_${coef}.log 2>&1 &
done

# Temperature sweep
for temp in 0.05 0.1 0.2 0.5; do
    echo "$(date): Starting Navigation contrastive temp=$temp" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_navigation.py --method contrastive --num_envs 16 --num_agents 12 --total_timesteps 10000000 --temperature $temp --seed 200 > $LOG_DIR/nav_temp_${temp}.log 2>&1 &
done

# Embedding dimension sweep
for dim in 32 64 128 256; do
    echo "$(date): Starting Navigation contrastive embed_dim=$dim" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_navigation.py --method contrastive --num_envs 16 --num_agents 12 --total_timesteps 10000000 --embedding_dim $dim --seed 300 > $LOG_DIR/nav_embed_${dim}.log 2>&1 &
done

# Wait for all navigation experiments
wait
echo "$(date): Completed all Navigation experiments" | tee -a $LOG_DIR/master.log

# ============================================================================
# CRAFTAX EXPERIMENTS (GPU 1) - May fail without CUDA toolkit
# ============================================================================
echo "$(date): === CRAFTAX EXPERIMENTS (may fail) ===" | tee -a $LOG_DIR/master.log

export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"

for seed in 0 1 2; do
    echo "$(date): Starting Craftax baseline seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_craftax.py --total_timesteps 1000000 --num_seeds 1 > $LOG_DIR/craftax_baseline_s${seed}.log 2>&1 || echo "Craftax baseline seed $seed failed" | tee -a $LOG_DIR/master.log &

    echo "$(date): Starting Craftax contrastive seed $seed" | tee -a $LOG_DIR/master.log
    CUDA_VISIBLE_DEVICES=1 python v3_experiments/run_craftax.py --contrastive --total_timesteps 1000000 --num_seeds 1 > $LOG_DIR/craftax_contrastive_s${seed}.log 2>&1 || echo "Craftax contrastive seed $seed failed" | tee -a $LOG_DIR/master.log &
done

wait
echo "$(date): === ALL EXPERIMENTS COMPLETE ===" | tee -a $LOG_DIR/master.log

# Summary
echo ""
echo "$(date): === SUMMARY ===" | tee -a $LOG_DIR/master.log
echo "Check wandb for results: https://wandb.ai/tashapais/metta" | tee -a $LOG_DIR/master.log
echo "Logs saved in: $LOG_DIR" | tee -a $LOG_DIR/master.log
