#!/bin/bash
# =============================================================================
# Separate Encoder Ablation (Exp 7)
# Tests whether giving each agent its own encoder (no shared gradients)
# fixes semantic collapse under shared team rewards.
#
# Conditions (12 agents, 5 seeds, 5M timesteps):
#   GPU 0-4: shared rewards + separate encoders (seeds 0-4)
#   GPU 5-7: individual rewards + separate encoders (seeds 0-2)
#   → Round 2: individual rewards + separate encoders (seeds 3-4)
#
# Prediction: If reward structure is the cause (not gradient sharing),
#   shared+sepenc → probe at chance (same as shared+shared_encoder)
# Counterevidence: If gradient sharing is the cause,
#   shared+sepenc → probe above chance
# =============================================================================

set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG="v3_experiments/logs_new"
mkdir -p "$LOG"

export WANDB_API_KEY="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"

echo "=== Separate Encoder Ablation — Round 1 — $(date) ==="

# GPU 0-4: shared + separate encoders, seeds 0-4
for i in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$i WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type shared --separate_encoders \
        --num_agents 12 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/shr_sepenc_12ag_s${i}.log" 2>&1 &
    echo "[GPU $i] shared+sepenc 12ag seed $i"
done

# GPU 5-7: individual + separate encoders, seeds 0-2
for i in 0 1 2; do
    gpu=$((i+5))
    CUDA_VISIBLE_DEVICES=$gpu WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type individual --separate_encoders \
        --num_agents 12 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/ind_sepenc_12ag_s${i}.log" 2>&1 &
    echo "[GPU $gpu] individual+sepenc 12ag seed $i"
done

wait
echo "=== Round 1 done $(date) ==="

# Round 2: individual + separate encoders, seeds 3-4
echo "=== Separate Encoder Ablation — Round 2 — $(date) ==="

for i in 3 4; do
    gpu=$((i-3))
    CUDA_VISIBLE_DEVICES=$gpu WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type individual --separate_encoders \
        --num_agents 12 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/ind_sepenc_12ag_s${i}.log" 2>&1 &
    echo "[GPU $gpu] individual+sepenc 12ag seed $i"
done

wait
echo "=== Round 2 done $(date) ==="

# Merge
echo "=== Merging sepenc results ==="
.venv/bin/python3 - <<'PYEOF'
import json, glob, numpy as np

result_dir = "v3_experiments"
for cond, label in [("shared_sepenc", "shared+sepenc"), ("individual_sepenc", "ind+sepenc")]:
    files = sorted(glob.glob(f"{result_dir}/results_reward_{cond}_12agents_s*.json"))
    if not files:
        print(f"WARNING: no files for {label}")
        continue
    merged = []
    for f in files:
        merged.extend(json.load(open(f)))
    out = f"{result_dir}/results_reward_{cond}_12agents.json"
    json.dump(merged, open(out, "w"), indent=2)
    pa = [r['probe_accuracy'] for r in merged]
    er = [r['effrank_per_agent'] for r in merged]
    wr = [r['win_rate'] for r in merged]
    print(f"{label} (n={len(merged)}): probe={np.mean(pa):.3f}±{np.std(pa):.3f}  eff/n={np.mean(er):.3f}±{np.std(er):.3f}  wr={np.mean(wr):.3f}")
PYEOF

echo "=== All done $(date) ==="
