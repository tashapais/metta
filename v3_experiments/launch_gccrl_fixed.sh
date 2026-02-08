#!/bin/bash
# GC-CRL Arena rerun with optimizer fix (encoders now actually train)
# GPU 0: Seeds 0-1, GPU 1: Seed 2

set -e
cd /home/ubuntu/metta

PYTHON="/home/ubuntu/metta/.venv/bin/python"
export PATH="/usr/local/cuda-12.2/bin:/home/ubuntu/metta/.venv/bin:/usr/bin:/usr/local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.2
LOG_DIR="/tmp/experiment_logs"
mkdir -p $LOG_DIR

echo "$(date): Starting GC-CRL fixed rerun (3 seeds)" | tee $LOG_DIR/gccrl_fixed.log

# GPU 0: Seeds 0 and 1
(
    for seed in 0 1; do
        echo "$(date): [GPU0] GC-CRL fixed seed $seed" | tee -a $LOG_DIR/gccrl_fixed.log
        CUDA_VISIBLE_DEVICES=0 $PYTHON tools/run.py \
            recipes.experiment.v2_gc_crl_experiments.train_arena_gc_crl \
            run=gccrl_fixed_s${seed} \
            system.seed=${seed} \
            wandb.enabled=true wandb.project=metta wandb.entity=tashapais \
            training_env.auto_workers=false training_env.num_workers=14 \
            >> $LOG_DIR/gccrl_fixed_s${seed}.log 2>&1
        echo "$(date): [GPU0] GC-CRL fixed seed $seed DONE" | tee -a $LOG_DIR/gccrl_fixed.log
    done
    echo "$(date): [GPU0] ALL DONE" | tee -a $LOG_DIR/gccrl_fixed.log
) &
GPU0_PID=$!

# GPU 1: Seed 2
(
    echo "$(date): [GPU1] GC-CRL fixed seed 2" | tee -a $LOG_DIR/gccrl_fixed.log
    CUDA_VISIBLE_DEVICES=1 $PYTHON tools/run.py \
        recipes.experiment.v2_gc_crl_experiments.train_arena_gc_crl \
        run=gccrl_fixed_s2 \
        system.seed=2 \
        wandb.enabled=true wandb.project=metta wandb.entity=tashapais \
        training_env.auto_workers=false training_env.num_workers=14 \
        >> $LOG_DIR/gccrl_fixed_s2.log 2>&1
    echo "$(date): [GPU1] GC-CRL fixed seed 2 DONE" | tee -a $LOG_DIR/gccrl_fixed.log
    echo "$(date): [GPU1] ALL DONE" | tee -a $LOG_DIR/gccrl_fixed.log
) &
GPU1_PID=$!

echo "$(date): GPU0 PID=$GPU0_PID, GPU1 PID=$GPU1_PID" | tee -a $LOG_DIR/gccrl_fixed.log
wait $GPU0_PID $GPU1_PID
echo "$(date): ALL 3 GC-CRL FIXED RUNS COMPLETE" | tee -a $LOG_DIR/gccrl_fixed.log
