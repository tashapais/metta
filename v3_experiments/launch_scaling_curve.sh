#!/bin/bash
# =============================================================================
# Batch 4: Full scaling curve — individual vs shared at 6 and 18 agents
# Completes the dataset: 6, 12, 18, 24 agents × individual/shared rewards
# All 8 GPUs, 20 seeds total across 3 rounds
# =============================================================================
set -euo pipefail
cd /home/ubuntu/metta
PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG="v3_experiments/logs_new"
export WANDB_API_KEY="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"

echo "=== Scaling Curve Batch — $(date) ==="

# --- Round 1: 6ag individual s0-4, 6ag shared s0-2 (8 GPUs, ~15 min) ---
echo "Round 1: 6ag seeds"
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 0 > "$LOG/ind_6ag_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 1 > "$LOG/ind_6ag_s1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 2 > "$LOG/ind_6ag_s2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 3 > "$LOG/ind_6ag_s3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 4 > "$LOG/ind_6ag_s4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 0 > "$LOG/shr_6ag_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 1 > "$LOG/shr_6ag_s1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 2 > "$LOG/shr_6ag_s2.log" 2>&1 &
wait; echo "=== Round 1 done $(date) ==="

# --- Round 2: 6ag shared s3-4, 18ag individual s0-4, 18ag shared s0 (7 GPUs, ~30 min) ---
echo "Round 2: 6ag shared s3-4 + 18ag seeds"
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 3 > "$LOG/shr_6ag_s3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 4 > "$LOG/shr_6ag_s4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 0 > "$LOG/ind_18ag_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 1 > "$LOG/ind_18ag_s1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 2 > "$LOG/ind_18ag_s2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 3 > "$LOG/ind_18ag_s3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type individual --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 4 > "$LOG/ind_18ag_s4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 0 > "$LOG/shr_18ag_s0.log" 2>&1 &
wait; echo "=== Round 2 done $(date) ==="

# --- Round 3: 18ag shared s1-4 (4 GPUs, ~30 min) ---
echo "Round 3: 18ag shared s1-4"
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 1 > "$LOG/shr_18ag_s1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 2 > "$LOG/shr_18ag_s2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 3 > "$LOG/shr_18ag_s3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT --reward_type shared --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 4 > "$LOG/shr_18ag_s4.log" 2>&1 &
wait; echo "=== Round 3 done $(date) ==="

# --- Merge all seed files ---
echo "=== Merging ==="
.venv/bin/python3 - <<'PYEOF'
import json, glob, numpy as np, os

result_dir = "v3_experiments"
conditions = [
    ("individual", 6), ("shared", 6),
    ("individual", 18), ("shared", 18),
]
for cond, n_ag in conditions:
    files = sorted(glob.glob(f"{result_dir}/results_reward_{cond}_{n_ag}agents_s*.json"))
    if not files: print(f"WARNING: no files for {cond} {n_ag}ag"); continue
    merged = []
    for f in files:
        merged.extend(json.load(open(f)))
    out = f"{result_dir}/results_reward_{cond}_{n_ag}agents.json"
    json.dump(merged, open(out,"w"), indent=2)
    pa = [r["probe_accuracy"] for r in merged]
    er = [r["effrank_per_agent"] for r in merged]
    wr = [r["win_rate"] for r in merged]
    print(f"{cond}_{n_ag}ag (n={len(merged)}): probe={np.mean(pa):.3f}±{np.std(pa):.3f}  effrank/n={np.mean(er):.3f}  winrate={np.mean(wr):.3f}")
PYEOF
echo "=== All done $(date) ==="
