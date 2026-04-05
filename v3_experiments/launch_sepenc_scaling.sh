#!/bin/bash
# =============================================================================
# Exp 7b: Separate Encoder at 6ag and 18ag
# Tests whether the non-monotonic probe failure (Exp 6: ind+shared_enc at 6ag
# and 18ag also probes at chance) is caused by gradient sharing or game dynamics.
#
# If sep_enc rescues at 6ag/18ag: gradient sharing is causal at all team sizes.
# If sep_enc fails: game dynamics (no role demand) explains non-monotonic failures.
# =============================================================================
set -euo pipefail
cd /home/ubuntu/metta

PYTHON=".venv/bin/python -u"
SCRIPT="v3_experiments/paper_exp_reward_type.py"
LOG="v3_experiments/logs_new"
mkdir -p "$LOG"
export WANDB_API_KEY="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"

echo "=== Exp 7b: SepEnc Scaling — Round 1 (6ag, 8 GPUs) — $(date) ==="
# GPUs 0-4: ind+sep_enc 6ag seeds 0-4
for i in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$i WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type individual --separate_encoders \
        --num_agents 6 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/ind_sepenc_6ag_s${i}.log" 2>&1 &
    echo "[GPU $i] ind+sepenc 6ag seed $i"
done
# GPUs 5-7: shr+sep_enc 6ag seeds 0-2
for i in 0 1 2; do
    gpu=$((i+5))
    CUDA_VISIBLE_DEVICES=$gpu WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type shared --separate_encoders \
        --num_agents 6 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/shr_sepenc_6ag_s${i}.log" 2>&1 &
    echo "[GPU $gpu] shr+sepenc 6ag seed $i"
done
wait; echo "=== Round 1 done $(date) ==="

echo "=== Round 2 (shr+sep 6ag s3-4 + ind+sep 18ag s0-4 + shr+sep 18ag s0) — $(date) ==="
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --separate_encoders --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 3 \
    > "$LOG/shr_sepenc_6ag_s3.log" 2>&1 &
echo "[GPU 0] shr+sepenc 6ag seed 3"
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --separate_encoders --num_agents 6 --gpu 0 --num_seeds 1 --seed_start 4 \
    > "$LOG/shr_sepenc_6ag_s4.log" 2>&1 &
echo "[GPU 1] shr+sepenc 6ag seed 4"
for i in 0 1 2 3 4; do
    gpu=$((i+2))
    CUDA_VISIBLE_DEVICES=$gpu WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type individual --separate_encoders \
        --num_agents 18 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/ind_sepenc_18ag_s${i}.log" 2>&1 &
    echo "[GPU $gpu] ind+sepenc 18ag seed $i"
done
CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
    --reward_type shared --separate_encoders --num_agents 18 --gpu 0 --num_seeds 1 --seed_start 0 \
    > "$LOG/shr_sepenc_18ag_s0.log" 2>&1 &
echo "[GPU 7] shr+sepenc 18ag seed 0"
wait; echo "=== Round 2 done $(date) ==="

echo "=== Round 3 (shr+sep 18ag s1-4) — $(date) ==="
for i in 1 2 3 4; do
    gpu=$((i-1))
    CUDA_VISIBLE_DEVICES=$gpu WANDB_API_KEY=$WANDB_API_KEY $PYTHON $SCRIPT \
        --reward_type shared --separate_encoders --num_agents 18 --gpu 0 --num_seeds 1 --seed_start $i \
        > "$LOG/shr_sepenc_18ag_s${i}.log" 2>&1 &
    echo "[GPU $gpu] shr+sepenc 18ag seed $i"
done
wait; echo "=== Round 3 done $(date) ==="

echo "=== Merging Exp 7b results ==="
.venv/bin/python3 - <<'PYEOF'
import json, glob, numpy as np

result_dir = "v3_experiments"
for cond, n_ag in [("individual_sepenc",6),("shared_sepenc",6),
                   ("individual_sepenc",18),("shared_sepenc",18)]:
    files = sorted(glob.glob(f"{result_dir}/results_reward_{cond}_{n_ag}agents_s*.json"))
    if not files: print(f"WARNING: no files for {cond} {n_ag}ag"); continue
    d = []
    for f in files: d.extend(json.load(open(f)))
    json.dump(d, open(f"{result_dir}/results_reward_{cond}_{n_ag}agents.json","w"), indent=2)
    pa=[r['probe_accuracy'] for r in d]; er=[r['effrank_per_agent'] for r in d]
    print(f"{cond} {n_ag}ag (n={len(d)}): probe={np.mean(pa):.3f}±{np.std(pa):.3f}  eff/n={np.mean(er):.3f}±{np.std(er):.3f}")
PYEOF
echo "=== All done $(date) ==="
