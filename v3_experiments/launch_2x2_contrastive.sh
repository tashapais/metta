#!/bin/bash
# =============================================================================
# Launch: 2x2 contrastive ablation (individual/shared × no-CL/CL)
# 3 new conditions × 5 seeds = 15 total seeds across 8 A100 GPUs
#
# Round 1 (8 GPUs simultaneously):
#   GPU 0: individual+CL, 12ag, seed 0
#   GPU 1: individual+CL, 12ag, seed 1
#   GPU 2: individual+CL, 12ag, seed 2
#   GPU 3: individual+CL, 12ag, seed 3
#   GPU 4: individual+CL, 12ag, seed 4
#   GPU 5: individual+CL, 24ag, seed 0
#   GPU 6: shared+CL,     12ag, seed 0
#   GPU 7: individual+CL, 24ag, seed 1
#
# Round 2 (after round 1, 7 seeds remaining):
#   GPU 0: individual+CL, 24ag, seed 2
#   GPU 1: individual+CL, 24ag, seed 3
#   GPU 2: individual+CL, 24ag, seed 4
#   GPU 3: shared+CL,     12ag, seed 1
#   GPU 4: shared+CL,     12ag, seed 2
#   GPU 5: shared+CL,     12ag, seed 3
#   GPU 6: shared+CL,     12ag, seed 4
# =============================================================================

set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG="v3_experiments/logs_new"
mkdir -p "$LOG"

export WANDB_API_KEY="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"

echo "============================================================"
echo "  2x2 Contrastive Ablation — Round 1 — $(date)"
echo "  8 GPUs × 8 seeds in parallel"
echo "============================================================"

# --- Round 1 ---
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/ind_cl_12ag_s0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/ind_cl_12ag_s1.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/ind_cl_12ag_s2.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/ind_cl_12ag_s3.log" 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/ind_cl_12ag_s4.log" 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/ind_cl_24ag_s0.log" 2>&1 &
PID5=$!

CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/shr_cl_12ag_s0.log" 2>&1 &
PID6=$!

CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/ind_cl_24ag_s1.log" 2>&1 &
PID7=$!

echo "[GPU 0] individual+CL 12ag seed 0  (PID $PID0)"
echo "[GPU 1] individual+CL 12ag seed 1  (PID $PID1)"
echo "[GPU 2] individual+CL 12ag seed 2  (PID $PID2)"
echo "[GPU 3] individual+CL 12ag seed 3  (PID $PID3)"
echo "[GPU 4] individual+CL 12ag seed 4  (PID $PID4)"
echo "[GPU 5] individual+CL 24ag seed 0  (PID $PID5)"
echo "[GPU 6] shared+CL    12ag seed 0  (PID $PID6)"
echo "[GPU 7] individual+CL 24ag seed 1  (PID $PID7)"
echo ""

wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
echo "=== Round 1 complete at $(date) ==="

echo ""
echo "============================================================"
echo "  2x2 Contrastive Ablation — Round 2 — $(date)"
echo "  7 seeds remaining"
echo "============================================================"

# --- Round 2 ---
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/ind_cl_24ag_s2.log" 2>&1 &
R2_0=$!

CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/ind_cl_24ag_s3.log" 2>&1 &
R2_1=$!

CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --contrastive --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/ind_cl_24ag_s4.log" 2>&1 &
R2_2=$!

CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/shr_cl_12ag_s1.log" 2>&1 &
R2_3=$!

CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/shr_cl_12ag_s2.log" 2>&1 &
R2_4=$!

CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/shr_cl_12ag_s3.log" 2>&1 &
R2_5=$!

CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --contrastive --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/shr_cl_12ag_s4.log" 2>&1 &
R2_6=$!

echo "[GPU 0] individual+CL 24ag seed 2  (PID $R2_0)"
echo "[GPU 1] individual+CL 24ag seed 3  (PID $R2_1)"
echo "[GPU 2] individual+CL 24ag seed 4  (PID $R2_2)"
echo "[GPU 3] shared+CL    12ag seed 1  (PID $R2_3)"
echo "[GPU 4] shared+CL    12ag seed 2  (PID $R2_4)"
echo "[GPU 5] shared+CL    12ag seed 3  (PID $R2_5)"
echo "[GPU 6] shared+CL    12ag seed 4  (PID $R2_6)"
echo ""

wait $R2_0 $R2_1 $R2_2 $R2_3 $R2_4 $R2_5 $R2_6
echo "=== Round 2 complete at $(date) ==="

echo ""
echo "=== Merging per-seed result files ==="
.venv/bin/python - <<'PYEOF'
import json, glob, numpy as np

result_dir = "/home/ubuntu/metta/v3_experiments"
conditions = [
    ("individual_contrastive", 12),
    ("individual_contrastive", 24),
    ("shared_contrastive",     12),
]

for cond, n_ag in conditions:
    pattern = f"{result_dir}/results_reward_{cond}_{n_ag}agents_s*.json"
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: no files for {cond} {n_ag}ag — {pattern}")
        continue
    merged = []
    for f in files:
        with open(f) as fh:
            merged.extend(json.load(fh))
    out = f"{result_dir}/results_reward_{cond}_{n_ag}agents.json"
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)

    er  = [r["effrank_per_agent"] for r in merged]
    ad  = [r["final_act_div"]     for r in merged]
    wr  = [r["win_rate"]          for r in merged]
    pa  = [r["probe_accuracy"]    for r in merged]
    pc  = [r["probe_chance"]      for r in merged]
    print(f"\n{cond} ({n_ag}ag, n={len(merged)}):")
    print(f"  EffRank/n : {np.mean(er):.3f} ± {np.std(er):.3f}")
    print(f"  Act. Div  : {np.mean(ad):.4f} ± {np.std(ad):.4f}")
    print(f"  Win Rate  : {np.mean(wr):.3f} ± {np.std(wr):.3f}")
    print(f"  Probe Acc : {np.mean(pa):.3f} ± {np.std(pa):.3f}  (chance={np.mean(pc):.3f})")
    print(f"  → saved to {out}")
PYEOF

echo ""
echo "=== All done at $(date) ==="
