"""
Harvest diagnostic metrics from completed WandB runs.

This script fetches contrastive learning diagnostic metrics from WandB
to analyze representation quality, collapse patterns, and failure modes.
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Run names from completed experiments
RUNS = {
    # Arena Baseline & Main (3 seeds)
    "baseline_ppo_seed1": "baseline_ppo.01_21_26",
    "baseline_ppo_seed2": "baseline_ppo.seed2.01_21_26",
    "baseline_ppo_seed3": "baseline_ppo.seed3.01_21_26",
    "ppo_contrastive_seed1": "ppo_plus_contrastive.01_21_26",
    "ppo_contrastive_seed2": "ppo_plus_contrastive.seed2.01_21_26",
    "ppo_contrastive_seed3": "ppo_plus_contrastive.seed3.01_21_26",

    # Ablations (3 seeds each)
    "no_projection_seed1": "ablation_no_projection.01_21_26",
    "no_projection_seed2": "ablation_no_projection.seed2.01_21_26",
    "no_projection_seed3": "ablation_no_projection.seed3.01_21_26",
    "temp_0.05_seed1": "ablation_temp_0.05.01_21_26",
    "temp_0.05_seed2": "ablation_temp_0.05.seed2.01_21_26",
    "temp_0.05_seed3": "ablation_temp_0.05.seed3.01_21_26",
    "temp_0.5_seed1": "ablation_temp_0.5.01_21_26",
    "temp_0.5_seed2": "ablation_temp_0.5.seed2.01_21_26",
    "temp_0.5_seed3": "ablation_temp_0.5.seed3.01_21_26",
    "coef_0.01_seed1": "ablation_coef_0.01.01_21_26",
    "coef_0.01_seed2": "ablation_coef_0.01.seed2.01_21_26",
    "coef_0.01_seed3": "ablation_coef_0.01.seed3.01_21_26",
    "embed_64_seed1": "ablation_embed_64.01_21_26",
    "embed_64_seed2": "ablation_embed_64.seed2.01_21_26",
    "embed_64_seed3": "ablation_embed_64.seed3.01_21_26",
    "fixed_offset_seed1": "ablation_fixed_offset.01_21_26",
    "fixed_offset_seed2": "ablation_fixed_offset.seed2.01_21_26",
    "fixed_offset_seed3": "ablation_fixed_offset.seed3.01_21_26",

    # GC-CRL experiments
    "arena_baseline_gccrl": "genial-water-44",
    "arena_gccrl": "desert-monkey-45",
    "nav_baseline_gccrl": "deft-plant-46",
    "nav_gccrl": "comfy-moon-47",
}

PROJECT = "tashapais/metta"

def fetch_run_metrics(run_name: str, metrics: list[str]) -> pd.DataFrame:
    """Fetch specific metrics from a WandB run."""
    api = wandb.Api()
    try:
        run = api.run(f"{PROJECT}/{run_name}")
        history = run.history(keys=metrics)
        return history
    except Exception as e:
        print(f"Error fetching {run_name}: {e}")
        return pd.DataFrame()

def compute_similarity_gap(df: pd.DataFrame) -> pd.Series:
    """Compute similarity gap: positive_sim - negative_sim."""
    if "losses/positive_sim_mean" in df.columns and "losses/negative_sim_mean" in df.columns:
        return df["losses/positive_sim_mean"] - df["losses/negative_sim_mean"]
    return pd.Series([np.nan] * len(df))

def detect_collapse(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    """Detect representation collapse (similarity gap near 0)."""
    gap = compute_similarity_gap(df)
    if gap.empty:
        return False
    # Check if gap stays below threshold for majority of training
    return (gap.abs() < threshold).sum() > len(gap) * 0.5

def main():
    """Harvest diagnostic metrics from all completed runs."""

    output_dir = Path("diagnostics_data")
    output_dir.mkdir(exist_ok=True)

    # Metrics to fetch
    contrastive_metrics = [
        "losses/positive_sim_mean",
        "losses/negative_sim_mean",
        "losses/positive_sim_std",
        "losses/negative_sim_std",
        "losses/num_pairs",
        "losses/delta_mean",
        "losses/contrastive_loss",
        "overview/reward",
        "_step",
    ]

    results = {}

    print("Fetching diagnostic metrics from WandB...")
    print("=" * 60)

    for run_id, run_name in RUNS.items():
        print(f"\nProcessing: {run_id} ({run_name})")

        df = fetch_run_metrics(run_name, contrastive_metrics)

        if df.empty:
            print(f"  ⚠️  No data found")
            continue

        # Compute derived metrics
        df["similarity_gap"] = compute_similarity_gap(df)

        # Save to CSV
        output_file = output_dir / f"{run_id}.csv"
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to {output_file}")

        # Detect patterns
        collapsed = detect_collapse(df)
        if collapsed:
            print(f"  ⚠️  COLLAPSE DETECTED in {run_id}")

        # Summary statistics
        if "similarity_gap" in df.columns and not df["similarity_gap"].isna().all():
            final_gap = df["similarity_gap"].iloc[-100:].mean()  # Last 100 steps
            print(f"  Final similarity gap: {final_gap:.2f}")

        results[run_id] = {
            "run_name": run_name,
            "collapsed": collapsed,
            "data": df,
        }

    # Generate summary report
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # Group by experiment type
    print("\n📊 Representation Collapse Analysis:")
    for run_id, data in results.items():
        if data["collapsed"]:
            print(f"  ❌ {run_id}: COLLAPSED")
        else:
            print(f"  ✓ {run_id}: Healthy")

    print("\n💾 All data saved to:", output_dir)
    print("\nYou can now analyze these metrics for the paper!")

    return results

if __name__ == "__main__":
    results = main()
