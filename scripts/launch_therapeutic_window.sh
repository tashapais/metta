#!/bin/bash
# Therapeutic window experiments: alpha = 0.003, 0.005, 0.05
# 5 seeds each, on MuJoCo Ant
# Run on GPU 0 (has room alongside Arena)

cd /home/ubuntu/metta

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:/usr/local/bin:/usr/bin:$PATH
export CXX=g++-12
export CC=gcc-12
export WANDB_MODE=disabled

for alpha in 0.003 0.005 0.05; do
    alpha_label=$(echo $alpha | tr '.' 'p')
    logfile="logs/icml_cross_env/mujoco_ant_alpha_${alpha_label}.log"
    echo "[$(date)] Starting alpha=${alpha} (5 seeds) -> ${logfile}"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u v3_experiments/run_mujoco.py \
        --task ant --method contrastive --contrastive_coef ${alpha} \
        --num_seeds 5 --total_timesteps 10000000 --num_envs 32 --video_interval 9999 \
        > "${logfile}" 2>&1
    echo "[$(date)] alpha=${alpha} done."
done

echo "[$(date)] All therapeutic window experiments done."
