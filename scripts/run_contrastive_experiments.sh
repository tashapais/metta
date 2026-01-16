#!/bin/bash
# Script to run all contrastive learning paper experiments
# Usage: ./scripts/run_contrastive_experiments.sh [experiment_name] [--dry-run]

set -e

# Wandb configuration
WANDB_PROJECT="metta"
WANDB_ENTITY="tashapais"

# All experiments from the paper
EXPERIMENTS=(
    "baseline_ppo"
    "ppo_plus_contrastive"
    "ablation_no_projection"
    "ablation_temp_0.05"
    "ablation_temp_0.5"
    "ablation_coef_0.01"
    "ablation_fixed_offset"
)

# Parse command line arguments
EXPERIMENT=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# If specific experiment provided, run only that one
if [ -n "$EXPERIMENT" ]; then
    EXPERIMENTS=("$EXPERIMENT")
fi

echo "=========================================="
echo "Contrastive Learning Paper Experiments"
echo "=========================================="
echo "Experiments to run: ${EXPERIMENTS[*]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Run each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------"
    echo "Running experiment: $exp"
    echo "----------------------------------------"

    # Construct the command with unique run_id for each experiment
    RUN_ID="${exp}.$(date +%m_%d_%y)"
    CMD="uv run ./tools/run.py train contrastive_paper_experiments experiment_name=$exp wandb.enabled=True wandb.project=$WANDB_PROJECT wandb.entity=$WANDB_ENTITY wandb.run_id=$RUN_ID"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
    else
        echo "  Running: $CMD"
        $CMD
    fi

    echo ""
    echo "Completed experiment: $exp"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check wandb for logged metrics"
echo "2. Compare learning curves between baseline and ablations"
echo ""
