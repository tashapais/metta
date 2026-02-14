"""
Run remaining SMAC maps (corridor, 3s_vs_5z) and merge with partial results.
"""
import argparse
import json
import sys
import os

# Re-use everything from the main script
sys.path.insert(0, os.path.dirname(__file__))
from paper_exp_smac import train_and_measure, aggregate_seed_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    # Load partial results
    partial_path = "/home/ubuntu/metta/v3_experiments/results_smac_partial.json"
    with open(partial_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} partial results: {[r['map_name'] for r in all_results]}")

    # Run remaining maps
    remaining_maps = ["corridor", "3s_vs_5z"]

    for map_name in remaining_maps:
        config = {
            "map_name": map_name,
            "num_envs": 1,
            "num_steps": 128,
            "total_timesteps": args.total_timesteps,
            "minibatch_size": 512,
            "update_epochs": 10,
            "lr": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "hidden_dim": 128,
            "embedding_dim": 64,
            "gpu": args.gpu,
        }

        print(f"\n{'='*60}")
        print(f"Running SMAC map: {map_name} ({args.num_seeds} seeds)")
        print(f"{'='*60}")
        sys.stdout.flush()

        seed_results = []
        for seed in range(args.num_seeds):
            print(f"\n--- Seed {seed}/{args.num_seeds} ---")
            sys.stdout.flush()
            result = train_and_measure(config, seed=seed)
            seed_results.append(result)
            print(f"Seed {seed} result: {json.dumps(result, indent=2)}")
            sys.stdout.flush()

        aggregated = aggregate_seed_results(seed_results)
        all_results.append(aggregated)
        print(f"\nAggregated ({map_name}): {json.dumps(aggregated, indent=2)}")
        sys.stdout.flush()

    # Save complete results
    output_path = "/home/ubuntu/metta/v3_experiments/results_smac.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
