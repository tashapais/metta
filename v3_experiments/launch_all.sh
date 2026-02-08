#!/bin/bash
# Negative sampling ablation: same-agent vs other-agent negatives
# GPU 0: Same-agent negatives (intra_trajectory), 3 seeds
# GPU 1: Other-agent negatives (all/cross-agent), 3 seeds

set -e
cd /home/ubuntu/metta

PYTHON="/home/ubuntu/metta/.venv/bin/python"
export PATH="/usr/local/cuda-12.2/bin:/home/ubuntu/metta/.venv/bin:/usr/bin:/usr/local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.2
LOG_DIR="/tmp/experiment_logs"
mkdir -p $LOG_DIR

echo "$(date): Starting negative sampling ablation (6 runs)" | tee $LOG_DIR/master.log

# GPU 0: Same-agent negatives (3 seeds)
(
    for seed in 0 1 2; do
        echo "$(date): [GPU0] same-agent seed $seed" | tee -a $LOG_DIR/master.log
        CUDA_VISIBLE_DEVICES=0 $PYTHON tools/run.py \
            recipes.experiment.v2_gc_crl_experiments.train_arena_contrastive_same_agent \
            run=neg_ablation_same_agent_s${seed} \
            system.seed=${seed} \
            wandb.enabled=true wandb.project=metta wandb.entity=tashapais \
            training_env.auto_workers=false training_env.num_workers=14 \
            >> $LOG_DIR/neg_same_agent_s${seed}.log 2>&1
        echo "$(date): [GPU0] same-agent seed $seed DONE" | tee -a $LOG_DIR/master.log
    done
    echo "$(date): [GPU0] ALL DONE" | tee -a $LOG_DIR/master.log
) &
GPU0_PID=$!

# GPU 1: Other-agent negatives (3 seeds)
(
    for seed in 0 1 2; do
        echo "$(date): [GPU1] other-agent seed $seed" | tee -a $LOG_DIR/master.log
        CUDA_VISIBLE_DEVICES=1 $PYTHON tools/run.py \
            recipes.experiment.v2_gc_crl_experiments.train_arena_contrastive_other_agent \
            run=neg_ablation_other_agent_s${seed} \
            system.seed=${seed} \
            wandb.enabled=true wandb.project=metta wandb.entity=tashapais \
            training_env.auto_workers=false training_env.num_workers=14 \
            >> $LOG_DIR/neg_other_agent_s${seed}.log 2>&1
        echo "$(date): [GPU1] other-agent seed $seed DONE" | tee -a $LOG_DIR/master.log
    done
    echo "$(date): [GPU1] ALL DONE" | tee -a $LOG_DIR/master.log
) &
GPU1_PID=$!

echo "$(date): GPU0 PID=$GPU0_PID, GPU1 PID=$GPU1_PID" | tee -a $LOG_DIR/master.log
wait $GPU0_PID $GPU1_PID
echo "$(date): ALL 6 ABLATION RUNS COMPLETE" | tee -a $LOG_DIR/master.log
