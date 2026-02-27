#!/bin/bash
# =============================================================================
# Launch: Individual vs Shared Rewards Ablation (Experiment 4)
# =============================================================================
# Runs 4 conditions × 5 seeds across 8 A100 GPUs.
# Each GPU runs one condition sequentially (5 seeds).
# CUDA_VISIBLE_DEVICES handles GPU routing; --gpu 0 always within each process.
#
# All runs logged to wandb project: representation-collapse
# Results: metta/v3_experiments/results_reward_<cond>_<N>agents.json
#
# Usage:
#   cd /home/ubuntu/metta
#   bash v3_experiments/launch_reward_ablation.sh
# =============================================================================

set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG_DIR="v3_experiments/logs_reward_ablation"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Reward Ablation Launch — $(date)"
echo "  8 GPUs × parallel conditions"
echo "============================================================"
echo ""
echo "--- Launching 8 runs in parallel ---"

# Note: CUDA_VISIBLE_DEVICES remaps the selected GPU to index 0 in the process.
# So always pass --gpu 0; CUDA_VISIBLE_DEVICES controls which physical GPU.

# GPU 0: individual rewards, 18 agents (seeds 0–4)
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    --reward_type individual \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/individual_18ag_seed0-4.log" 2>&1 &
PID0=$!
echo "[GPU 0] individual 18-agent      seeds 0-4  (PID $PID0)"

# GPU 1: shared rewards, 18 agents (seeds 0–4)
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
    --reward_type shared \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/shared_18ag_seed0-4.log" 2>&1 &
PID1=$!
echo "[GPU 1] shared 18-agent          seeds 0-4  (PID $PID1)"

# GPU 2: individual + contrastive, 18 agents (seeds 0–4)
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
    --reward_type individual --contrastive \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/individual_contrastive_18ag_seed0-4.log" 2>&1 &
PID2=$!
echo "[GPU 2] individual+contrastive   seeds 0-4  (PID $PID2)"

# GPU 3: shared + contrastive, 18 agents (seeds 0–4)
CUDA_VISIBLE_DEVICES=3 $PYTHON $SCRIPT \
    --reward_type shared --contrastive \
    --gpu 0 --num_agents 18 --num_seeds 5 \
    > "$LOG_DIR/shared_contrastive_18ag_seed0-4.log" 2>&1 &
PID3=$!
echo "[GPU 3] shared+contrastive       seeds 0-4  (PID $PID3)"

# GPU 4: individual rewards, 12 agents — extends scaling story
CUDA_VISIBLE_DEVICES=4 $PYTHON $SCRIPT \
    --reward_type individual \
    --gpu 0 --num_agents 12 --num_seeds 5 \
    > "$LOG_DIR/individual_12ag_seed0-4.log" 2>&1 &
PID4=$!
echo "[GPU 4] individual 12-agent      seeds 0-4  (PID $PID4)"

# GPU 5: shared rewards, 12 agents
CUDA_VISIBLE_DEVICES=5 $PYTHON $SCRIPT \
    --reward_type shared \
    --gpu 0 --num_agents 12 --num_seeds 5 \
    > "$LOG_DIR/shared_12ag_seed0-4.log" 2>&1 &
PID5=$!
echo "[GPU 5] shared 12-agent          seeds 0-4  (PID $PID5)"

# GPU 6: individual rewards, 24 agents — extends scaling story
CUDA_VISIBLE_DEVICES=6 $PYTHON $SCRIPT \
    --reward_type individual \
    --gpu 0 --num_agents 24 --num_seeds 5 \
    > "$LOG_DIR/individual_24ag_seed0-4.log" 2>&1 &
PID6=$!
echo "[GPU 6] individual 24-agent      seeds 0-4  (PID $PID6)"

# GPU 7: shared rewards, 24 agents
CUDA_VISIBLE_DEVICES=7 $PYTHON $SCRIPT \
    --reward_type shared \
    --gpu 0 --num_agents 24 --num_seeds 5 \
    > "$LOG_DIR/shared_24ag_seed0-4.log" 2>&1 &
PID7=$!
echo "[GPU 7] shared 24-agent          seeds 0-4  (PID $PID7)"

echo ""
echo "All 8 runs launched. Waiting for completion..."
echo "(Logs: $LOG_DIR/)"
echo ""

# Wait for all jobs
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
echo ""
echo "=== All complete at $(date) ==="

# -------------------------------------------------------------------
# Aggregate and print summary
# -------------------------------------------------------------------
echo ""
echo "=== Results Summary ==="
$PYTHON - <<'PYEOF'
import json, glob, numpy as np, os

result_dir = "/home/ubuntu/metta/v3_experiments"
files = sorted(glob.glob(f"{result_dir}/results_reward_*.json"))

for fpath in files:
    with open(fpath) as f:
        data = json.load(f)
    if not data:
        continue
    cond = data[0]["condition"]
    n_ag = data[0]["num_agents"]

    er   = [d["effrank_per_agent"] for d in data]
    ad   = [d["final_act_div"]     for d in data]
    wr   = [d["win_rate"]          for d in data]
    pa   = [d["probe_accuracy"]    for d in data]
    pc   = [d["probe_chance"]      for d in data]
    lift = [d["probe_lift"]        for d in data]

    print(f"\n{cond} ({n_ag} agents, n={len(data)}):")
    print(f"  EffRank/n  : {np.mean(er):.3f} +/- {np.std(er):.3f}")
    print(f"  Act.Div    : {np.mean(ad):.4f} +/- {np.std(ad):.4f}")
    print(f"  Win Rate   : {np.mean(wr):.3f} +/- {np.std(wr):.3f}")
    print(f"  Probe Acc  : {np.mean(pa):.3f} +/- {np.std(pa):.3f}  "
          f"(chance={np.mean(pc):.3f}, lift={np.mean(lift):+.3f})")
PYEOF

echo ""
echo "=== Reward Ablation Complete ==="
