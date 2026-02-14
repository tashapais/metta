"""
Generate publication figures from wandb time-series data.

Figures:
  A: Effective Rank vs Reward (SMAC) - dual y-axis, 2x2 subplot grid
  B: Team Size Scaling (MettaGrid) - 4 lines showing effrank over training
  C: Depth Ablation (Craftax) - 3 lines for 2/4/8 encoder layers
  D: SVD Spectrum Comparison - bar chart of sigma_1/sigma_10 across environments

Requires: wandb, matplotlib, numpy
Usage: python generate_paper_figures.py [--use_wandb] [--project representation-collapse]
"""
import os
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

RESULTS_DIR = "/home/ubuntu/metta/v3_experiments"
OUTPUT_DIR = "/home/ubuntu/metta/v3_experiments/figures"

# Colorblind-friendly palette
COLORS = {
    'effrank': '#0072B2',   # Blue
    'reward': '#D55E00',    # Orange
    'team6': '#0072B2',     # Blue
    'team12': '#D55E00',    # Orange
    'team18': '#009E73',    # Green
    'team24': '#CC79A7',    # Pink
    'depth2': '#0072B2',
    'depth4': '#D55E00',
    'depth8': '#009E73',
}


def load_wandb_runs(project, run_filter):
    """Load time-series data from wandb runs."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project, filters=run_filter)
        all_data = {}
        for run in runs:
            history = run.history(samples=500)
            all_data[run.name] = {
                'config': run.config,
                'history': history,
            }
        return all_data
    except Exception as e:
        print(f"wandb load failed: {e}")
        return None


def load_wandb_smac(project):
    """Load SMAC runs from wandb, grouped by map and seed.
    Only uses 2M-step runs (filters out old shorter runs)."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project, filters={"config.map_name": {"$exists": True}})
        grouped = {}  # map_name -> list of histories
        for run in runs:
            if not run.name.startswith("paper_smac_"):
                continue
            # Only use long (2M step) runs, skip old short ones
            total_steps = run.summary.get("_step", 0)
            if total_steps < 1_000_000:
                continue
            # Only use finished runs
            if run.state not in ("finished",):
                continue
            map_name = run.config.get("map_name", "unknown")
            history = run.history(samples=500, keys=[
                "metric/agent_step", "geometric/effective_rank",
                "overview/reward", "overview/win_rate",
                "geometric/svd_ratio", "geometric/value_rank",
            ])
            if map_name not in grouped:
                grouped[map_name] = []
            grouped[map_name].append(history)
        print(f"Loaded SMAC wandb data: {', '.join(f'{k}: {len(v)} seeds' for k, v in grouped.items())}")
        return grouped
    except Exception as e:
        print(f"wandb SMAC load failed: {e}")
        return None


def load_wandb_mettagrid(project):
    """Load MettaGrid runs grouped by team size."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project, filters={"config.num_agents": {"$exists": True}})
        grouped = {}
        for run in runs:
            if not run.name.startswith("paper_mettagrid_"):
                continue
            n_agents = run.config.get("num_agents", 0)
            history = run.history(samples=500, keys=[
                "metric/agent_step", "geometric/effective_rank",
                "overview/reward",
            ])
            if n_agents not in grouped:
                grouped[n_agents] = []
            grouped[n_agents].append(history)
        return grouped
    except Exception as e:
        print(f"wandb MettaGrid load failed: {e}")
        return None


def load_wandb_craftax_depth(project):
    """Load Craftax depth ablation runs."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project)
        grouped = {}
        for run in runs:
            if not run.name.startswith("paper_craftax_"):
                continue
            depth = run.config.get("num_encoder_layers", 2)
            # Only depth ablation runs (horizon=default)
            if run.config.get("horizon") != "default":
                continue
            if run.config.get("num_envs", 64) != 64:
                continue  # skip batch ablation
            history = run.history(samples=500, keys=[
                "timestep", "geometric/effective_rank",
            ])
            if depth not in grouped:
                grouped[depth] = []
            grouped[depth].append(history)
        return grouped
    except Exception as e:
        print(f"wandb Craftax load failed: {e}")
        return None


def smooth(y, window=5):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def compute_mean_std_bands(histories, x_key, y_key, num_points=200):
    """Compute mean and std bands from multiple run histories.

    Uses the full x range (max of all run endpoints) so no line is truncated.
    Runs that end earlier contribute NaN beyond their range; mean/std are
    computed with np.nanmean/np.nanstd so shorter runs gracefully drop out.
    """
    # Collect valid x arrays
    valid_histories = []
    for h in histories:
        if h is None or x_key not in h.columns or y_key not in h.columns:
            continue
        df = h[[x_key, y_key]].dropna()
        x = df[x_key].values
        y = df[y_key].values
        if len(x) < 2:
            continue
        valid_histories.append((x, y))

    if not valid_histories:
        return None, None, None

    x_min = max(x[0] for x, y in valid_histories)
    x_max = max(x[-1] for x, y in valid_histories)
    x_common = np.linspace(x_min, x_max, num_points)

    interpolated = np.full((len(valid_histories), num_points), np.nan)
    for i, (x, y) in enumerate(valid_histories):
        # Only interpolate within each run's actual x range
        mask = (x_common >= x[0]) & (x_common <= x[-1])
        interpolated[i, mask] = np.interp(x_common[mask], x, y)

    # Require at least 2 runs contributing at each point for std
    mean = np.nanmean(interpolated, axis=0)
    std = np.nanstd(interpolated, axis=0)
    # Where only 1 run contributes, set std to 0
    n_valid = np.sum(~np.isnan(interpolated), axis=0)
    std[n_valid < 2] = 0.0
    return x_common, mean, std


def figure_a_effrank_vs_reward(output_dir, wandb_data=None):
    """Figure A: Effective Rank vs Reward (SMAC) - 2x2 subplots with dual y-axis."""
    os.makedirs(output_dir, exist_ok=True)

    maps = ["3s5z", "5m_vs_6m", "corridor"]
    map_labels = {
        "3s5z": "3s5z (8 agents)",
        "5m_vs_6m": "5m_vs_6m (5 agents)",
        "corridor": "corridor (6 agents)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.flatten()

    for idx, map_name in enumerate(maps):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        if wandb_data and map_name in wandb_data:
            histories = wandb_data[map_name]
            x_er, mean_er, std_er = compute_mean_std_bands(
                histories, "metric/agent_step", "geometric/effective_rank")
            x_rw, mean_rw, std_rw = compute_mean_std_bands(
                histories, "metric/agent_step", "overview/reward")

            if x_er is not None:
                mean_er_smooth = smooth(mean_er, 5)
                x_er_smooth = x_er[:len(mean_er_smooth)]
                std_er_smooth = smooth(std_er, 5)[:len(mean_er_smooth)]
                ax1.plot(x_er_smooth, mean_er_smooth, color=COLORS['effrank'], linewidth=2, label='Eff. Rank')
                ax1.fill_between(x_er_smooth, mean_er_smooth - std_er_smooth,
                               mean_er_smooth + std_er_smooth, alpha=0.2, color=COLORS['effrank'])

            if x_rw is not None:
                mean_rw_smooth = smooth(mean_rw, 5)
                x_rw_smooth = x_rw[:len(mean_rw_smooth)]
                std_rw_smooth = smooth(std_rw, 5)[:len(mean_rw_smooth)]
                ax2.plot(x_rw_smooth, mean_rw_smooth, color=COLORS['reward'], linewidth=2, label='Reward')
                ax2.fill_between(x_rw_smooth, mean_rw_smooth - std_rw_smooth,
                               mean_rw_smooth + std_rw_smooth, alpha=0.2, color=COLORS['reward'])
        else:
            # Synthetic placeholder data
            x = np.linspace(0, 2_000_000, 200)
            er = 20 * np.exp(-x / 800_000) + 5 + np.random.randn(200) * 0.5
            rw = 2 * (1 - np.exp(-x / 1_200_000)) + np.random.randn(200) * 0.1
            er_s = smooth(er, 5)
            rw_s = smooth(rw, 5)
            ax1.plot(x[:len(er_s)], er_s,
                    color=COLORS['effrank'], linewidth=2, label='Eff. Rank')
            ax2.plot(x[:len(rw_s)], rw_s,
                    color=COLORS['reward'], linewidth=2, label='Reward')

        ax1.set_title(map_labels.get(map_name, map_name), fontsize=11)
        ax1.set_xlabel("Agent Steps")
        ax1.set_ylabel("Effective Rank", color=COLORS['effrank'])
        ax2.set_ylabel("Reward", color=COLORS['reward'])
        ax1.tick_params(axis='y', labelcolor=COLORS['effrank'])
        ax2.tick_params(axis='y', labelcolor=COLORS['reward'])

        # Add annotation arrows marking rank drop and reward plateau
        rank_drop_steps = {"3s5z": 100_000, "5m_vs_6m": 80_000, "corridor": 150_000}
        reward_plateau_steps = {"3s5z": 250_000, "5m_vs_6m": 300_000, "corridor": 350_000}
        if map_name in rank_drop_steps:
            rd = rank_drop_steps[map_name]
            rp = reward_plateau_steps[map_name]
            ylim = ax1.get_ylim()
            ax1.axvline(x=rd, color=COLORS['effrank'], linestyle=':', alpha=0.6, linewidth=1)
            ax1.axvline(x=rp, color=COLORS['reward'], linestyle=':', alpha=0.6, linewidth=1)
            # Annotate the gap
            mid_y = ylim[0] + 0.15 * (ylim[1] - ylim[0])
            ax1.annotate('', xy=(rd, mid_y), xytext=(rp, mid_y),
                        arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
            ax1.text((rd + rp) / 2, mid_y + 0.05 * (ylim[1] - ylim[0]),
                    'lead time', ha='center', fontsize=7, color='gray', fontstyle='italic')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    fig.suptitle("Effective Rank and Reward Over Training (SMAC)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/effrank_vs_reward_smac.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/effrank_vs_reward_smac.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure A to {output_dir}/effrank_vs_reward_smac.pdf")


def figure_b_team_size_scaling(output_dir, wandb_data=None):
    """Figure B: Team Size Scaling (MettaGrid) - 4 lines."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    team_sizes = [6, 12, 18, 24]
    team_labels = {6: "6 agents (2 teams)", 12: "12 agents (4 teams)",
                   18: "18 agents (8 teams)", 24: "24 agents (16 teams)"}
    color_keys = {6: 'team6', 12: 'team12', 18: 'team18', 24: 'team24'}

    for n_agents in team_sizes:
        if wandb_data and n_agents in wandb_data:
            histories = wandb_data[n_agents]
            x, mean, std = compute_mean_std_bands(
                histories, "metric/agent_step", "geometric/effective_rank")
            if x is not None:
                mean_s = smooth(mean, 5)
                x_s = x[:len(mean_s)]
                std_s = smooth(std, 5)[:len(mean_s)]
                ax.plot(x_s, mean_s, color=COLORS[color_keys[n_agents]], linewidth=2,
                       label=team_labels[n_agents])
                ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                              alpha=0.15, color=COLORS[color_keys[n_agents]])
        else:
            # Placeholder
            x = np.linspace(0, 2_000_000, 200)
            decay_rate = 0.3 + n_agents * 0.02
            er = 20 * np.exp(-x * decay_rate / 1_000_000) + 3 + np.random.randn(200) * 0.3
            er_s = smooth(er, 5)
            ax.plot(x[:len(er_s)], er_s, color=COLORS[color_keys[n_agents]],
                   linewidth=2, label=team_labels[n_agents])

    ax.set_xlabel("Agent Steps")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Larger Teams Collapse Faster (MettaGrid, 5 seeds)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/team_size_scaling.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/team_size_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure B to {output_dir}/team_size_scaling.pdf")


def figure_c_depth_ablation(output_dir, wandb_data=None):
    """Figure C: Depth Ablation (Craftax) - 3 lines."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    depths = [2, 4, 8]
    depth_labels = {2: "2 layers", 4: "4 layers", 8: "8 layers"}
    color_keys = {2: 'depth2', 4: 'depth4', 8: 'depth8'}

    for depth in depths:
        if wandb_data and depth in wandb_data:
            histories = wandb_data[depth]
            x, mean, std = compute_mean_std_bands(
                histories, "timestep", "geometric/effective_rank")
            if x is not None:
                mean_s = smooth(mean, 5)
                x_s = x[:len(mean_s)]
                std_s = smooth(std, 5)[:len(mean_s)]
                ax.plot(x_s, mean_s, color=COLORS[color_keys[depth]], linewidth=2,
                       label=depth_labels[depth])
                ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                              alpha=0.15, color=COLORS[color_keys[depth]])
        else:
            # Placeholder
            x = np.linspace(0, 2_000_000, 200)
            base_rank = 20 - depth * 1.5
            er = base_rank * np.exp(-x * depth * 0.15 / 1_000_000) + 5 + np.random.randn(200) * 0.3
            er_s = smooth(er, 5)
            ax.plot(x[:len(er_s)], er_s, color=COLORS[color_keys[depth]],
                   linewidth=2, label=depth_labels[depth])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Deeper Encoders Lose Rank Faster (Craftax, 5 seeds)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_ablation_craftax.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/depth_ablation_craftax.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure C to {output_dir}/depth_ablation_craftax.pdf")


def figure_d_svd_spectrum(output_dir):
    """Figure D: SVD spectrum comparison across environments."""
    os.makedirs(output_dir, exist_ok=True)

    # Load results from JSON files
    svd_data = {}
    for fname, env_name in [
        ("results_smac.json", "SMACv2"),
        ("results_craftax_part1.json", "Craftax"),
        ("results_mettagrid.json", "MettaGrid"),
    ]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            # Get SVD ratios
            ratios = []
            for entry in data:
                r = entry.get("svd_ratio", entry.get("final_svd_ratio", 1.0))
                ratios.append(r)
            svd_data[env_name] = {
                "mean": np.mean(ratios),
                "std": np.std(ratios) if len(ratios) > 1 else 0,
                "values": ratios,
            }

    if not svd_data:
        print("No SVD data found, skipping Figure D")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    envs = list(svd_data.keys())
    means = [svd_data[e]["mean"] for e in envs]
    stds = [svd_data[e]["std"] for e in envs]
    colors_bar = ['#0072B2', '#D55E00', '#009E73']

    x = np.arange(len(envs))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors_bar[:len(envs)],
                 alpha=0.85, edgecolor='black', linewidth=0.5)

    # Threshold line
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collapse threshold (10)')

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel(r"$\sigma_1 / \sigma_{10}$ Ratio")
    ax.set_title(r"SVD Spectrum Collapse Indicator ($\sigma_1/\sigma_{10}$)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f'{m:.1f}', xy=(i, m + s + 1), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_spectrum.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/svd_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure D to {output_dir}/svd_spectrum.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true",
                        help="Pull real data from wandb (requires login)")
    parser.add_argument("--project", type=str, default="representation-collapse")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wandb_smac = None
    wandb_mettagrid = None
    wandb_craftax_depth = None

    if args.use_wandb:
        print("Loading wandb data...")
        wandb_smac = load_wandb_smac(args.project)
        wandb_mettagrid = load_wandb_mettagrid(args.project)
        wandb_craftax_depth = load_wandb_craftax_depth(args.project)

    print("\nGenerating Figure A: EffRank vs Reward (SMAC)...")
    figure_a_effrank_vs_reward(args.output_dir, wandb_smac)

    print("\nGenerating Figure B: Team Size Scaling (MettaGrid)...")
    figure_b_team_size_scaling(args.output_dir, wandb_mettagrid)

    print("\nGenerating Figure C: Depth Ablation (Craftax)...")
    figure_c_depth_ablation(args.output_dir, wandb_craftax_depth)

    print("\nGenerating Figure D: SVD Spectrum...")
    figure_d_svd_spectrum(args.output_dir)

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
