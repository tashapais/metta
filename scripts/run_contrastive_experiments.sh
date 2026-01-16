#!/bin/bash
# Script to run all contrastive learning paper experiments
# Usage: ./scripts/run_contrastive_experiments.sh [experiment_name] [--seeds N] [--dry-run]

set -e

RECIPE="recipes.experiment.contrastive_paper_experiments"
NUM_SEEDS=5  # Run 5 seeds per condition as per paper statistical requirements

# Parse command line arguments
EXPERIMENT=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
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

# If specific experiment provided, run only that one
if [ -n "$EXPERIMENT" ]; then
    EXPERIMENTS=("$EXPERIMENT")
fi

echo "=========================================="
echo "Contrastive Learning Paper Experiments"
echo "=========================================="
echo "Recipe: $RECIPE"
echo "Seeds per experiment: $NUM_SEEDS"
echo "Experiments to run: ${EXPERIMENTS[*]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Run each experiment with multiple seeds
for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------"
    echo "Running experiment: $exp"
    echo "----------------------------------------"

    for seed in $(seq 1 $NUM_SEEDS); do
        echo ""
        echo "  Seed $seed/$NUM_SEEDS"

        # Construct wandb run name
        RUN_NAME="${exp}_seed${seed}"

        # Construct the command
        CMD="python -m metta.tools.train \
            --recipe $RECIPE \
            --train-fn train \
            --experiment-name $exp \
            --seed $seed \
            --wandb-run-name $RUN_NAME \
            --wandb-tags contrastive_paper,${exp},seed_${seed}"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] $CMD"
        else
            echo "  Running: $CMD"
            eval $CMD
        fi
    done

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
echo "2. Analyze results using notebooks/contrastive_paper_analysis.ipynb"
echo "3. Generate figures for the paper"
echo ""
