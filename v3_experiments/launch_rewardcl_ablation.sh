#!/bin/bash
# =============================================================================
# Launch: Reward-conditioned SupCon ablation (Batch 3)
# Tests whether SupCon with per-agent rank labels can rescue representations
# under shared rewards — the contrastive objective that directly optimizes
# for what the probe measures.
#
# Round 1 (8 GPUs):
#   GPU 0-4: shared+rewardCL, 12ag, seeds 0-4
#   GPU 5-7: individual+rewardCL, 12ag, seeds 0-2
# Round 2 (7 seeds):
#   GPU 0-1: individual+rewardCL, 12ag, seeds 3-4
#   GPU 2-4: shared+rewardCL, 24ag, seeds 0-2
#   GPU 5-6: individual+rewardCL, 24ag, seeds 0-1
# Round 3 (5 seeds):
#   GPU 0-2: shared+rewardCL, 24ag, seeds 3-4 + individual+rewardCL 24ag seeds 2-4
# =============================================================================

set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG="v3_experiments/logs_new"
mkdir -p "$LOG"

export WANDB_API_KEY="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"

echo "============================================================"
echo "  Reward-Conditioned SupCon Ablation — Round 1 — $(date)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/shr_rcl_12ag_s0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/shr_rcl_12ag_s1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/shr_rcl_12ag_s2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/shr_rcl_12ag_s3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/shr_rcl_12ag_s4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/ind_rcl_12ag_s0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/ind_rcl_12ag_s1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/ind_rcl_12ag_s2.log" 2>&1 &

echo "[GPU 0] shared+rewardCL    12ag seed 0"
echo "[GPU 1] shared+rewardCL    12ag seed 1"
echo "[GPU 2] shared+rewardCL    12ag seed 2"
echo "[GPU 3] shared+rewardCL    12ag seed 3"
echo "[GPU 4] shared+rewardCL    12ag seed 4"
echo "[GPU 5] individual+rewardCL 12ag seed 0"
echo "[GPU 6] individual+rewardCL 12ag seed 1"
echo "[GPU 7] individual+rewardCL 12ag seed 2"

wait
echo "=== Round 1 complete at $(date) ==="

echo "============================================================"
echo "  Reward-Conditioned SupCon — Round 2 — $(date)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/ind_rcl_12ag_s3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 12 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/ind_rcl_12ag_s4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/shr_rcl_24ag_s0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/shr_rcl_24ag_s1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/shr_rcl_24ag_s2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/ind_rcl_24ag_s0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 1 \
    > "$LOG/ind_rcl_24ag_s1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 2 \
    > "$LOG/ind_rcl_24ag_s2.log" 2>&1 &

echo "[GPU 0] individual+rewardCL 12ag seed 3"
echo "[GPU 1] individual+rewardCL 12ag seed 4"
echo "[GPU 2] shared+rewardCL    24ag seed 0"
echo "[GPU 3] shared+rewardCL    24ag seed 1"
echo "[GPU 4] shared+rewardCL    24ag seed 2"
echo "[GPU 5] individual+rewardCL 24ag seed 0"
echo "[GPU 6] individual+rewardCL 24ag seed 1"
echo "[GPU 7] individual+rewardCL 24ag seed 2"

wait
echo "=== Round 2 complete at $(date) ==="

echo "============================================================"
echo "  Reward-Conditioned SupCon — Round 3 — $(date)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/shr_rcl_24ag_s3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/shr_rcl_24ag_s4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/ind_rcl_24ag_s3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type individual --reward_cl --num_agents 24 \
    --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/ind_rcl_24ag_s4.log" 2>&1 &

echo "[GPU 0] shared+rewardCL    24ag seed 3"
echo "[GPU 1] shared+rewardCL    24ag seed 4"
echo "[GPU 2] individual+rewardCL 24ag seed 3"
echo "[GPU 3] individual+rewardCL 24ag seed 4"

wait
echo "=== Round 3 complete at $(date) ==="

echo ""
echo "=== Merging rewardCL result files ==="
.venv/bin/python - <<'PYEOF'
import json, glob, numpy as np

result_dir = "/home/ubuntu/metta/v3_experiments"
conditions = [
    ("individual_rewardCL", 12),
    ("individual_rewardCL", 24),
    ("shared_rewardCL",     12),
    ("shared_rewardCL",     24),
]

for cond, n_ag in conditions:
    pattern = f"{result_dir}/results_reward_{cond}_{n_ag}agents_s*.json"
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: no files for {cond} {n_ag}ag")
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
    print(f"  → {out}")
PYEOF

echo "=== All done at $(date) ==="
