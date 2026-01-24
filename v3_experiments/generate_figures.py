"""
Generate figures for the paper from experiment metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import os

# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'ppo': '#1f77b4',  # Blue
    'contrastive': '#ff7f0e',  # Orange
    'atc': '#2ca02c',  # Green
    'icm': '#d62728',  # Red
    'rnd': '#9467bd',  # Purple
}


def load_metrics(metrics_dir, method):
    """Load stacked metrics for a method."""
    filepath = f"{metrics_dir}/{method}_all_seeds.npz"
    if os.path.exists(filepath):
        return dict(np.load(filepath))
    return None


def smooth(data, window=5):
    """Apply moving average smoothing."""
    kernel = np.ones(window) / window
    if len(data.shape) == 1:
        return np.convolve(data, kernel, mode='valid')
    else:
        return np.array([np.convolve(d, kernel, mode='valid') for d in data])


def plot_learning_curves(metrics_dir, output_dir, smoothing=5):
    """Generate learning curves comparison plot."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = ['ppo', 'contrastive']
    labels = ['PPO Baseline', 'PPO + Contrastive (Ours)']

    for method, label in zip(methods, labels):
        data = load_metrics(metrics_dir, method)
        if data is None:
            print(f"Warning: No data for {method}")
            continue

        rewards = data['mean_reward']  # Shape: (num_seeds, num_updates)

        # Smooth the data
        rewards_smooth = smooth(rewards, window=smoothing)

        # Compute mean and std across seeds
        mean = rewards_smooth.mean(axis=0)
        std = rewards_smooth.std(axis=0)

        # X-axis: timesteps
        num_updates = len(mean)
        timesteps = np.linspace(0, 1_000_000, num_updates)

        # Plot mean with confidence band
        ax.plot(timesteps, mean, label=label, color=COLORS[method], linewidth=2)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=COLORS[method])

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Learning Curves on Craftax (5 seeds, ±1σ)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1_000_000)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves_craftax.pdf")
    plt.savefig(f"{output_dir}/learning_curves_craftax.png")
    plt.close()
    print(f"Saved learning curves to {output_dir}/learning_curves_craftax.pdf")


def plot_variance_comparison(metrics_dir, output_dir):
    """Generate variance comparison plot showing individual seed trajectories."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    methods = ['ppo', 'contrastive']
    titles = ['PPO Baseline', 'PPO + Contrastive (Ours)']

    for ax, method, title in zip(axes, methods, titles):
        data = load_metrics(metrics_dir, method)
        if data is None:
            continue

        rewards = data['mean_reward']
        rewards_smooth = smooth(rewards, window=10)

        num_updates = rewards_smooth.shape[1]
        timesteps = np.linspace(0, 1_000_000, num_updates)

        # Plot individual seeds
        for i, seed_rewards in enumerate(rewards_smooth):
            ax.plot(timesteps, seed_rewards, alpha=0.7, linewidth=1.5,
                   label=f'Seed {i}' if method == 'ppo' else None)

        # Compute final reward stats
        final_rewards = rewards[:, -1]
        mean_final = final_rewards.mean()
        std_final = final_rewards.std()

        ax.set_xlabel('Timesteps')
        ax.set_title(f'{title}\nFinal: {mean_final:.4f} ± {std_final:.4f}')
        ax.set_xlim(0, 1_000_000)

    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_comparison_craftax.pdf")
    plt.savefig(f"{output_dir}/variance_comparison_craftax.png")
    plt.close()
    print(f"Saved variance comparison to {output_dir}/variance_comparison_craftax.pdf")


def plot_teaser_figure(metrics_dir, output_dir):
    """Generate teaser figure for first page of paper."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 3))

    # Left panel: Learning curves
    ax1 = fig.add_subplot(121)

    methods = ['ppo', 'contrastive']
    labels = ['PPO', 'PPO + Contrastive']

    final_rewards = {}
    for method, label in zip(methods, labels):
        data = load_metrics(metrics_dir, method)
        if data is None:
            continue

        rewards = smooth(data['mean_reward'], window=5)
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)

        num_updates = len(mean)
        timesteps = np.linspace(0, 1, num_updates)  # Normalized

        ax1.plot(timesteps, mean, label=label, color=COLORS[method], linewidth=2)
        ax1.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=COLORS[method])

        final_rewards[method] = data['mean_reward'][:, -1]

    ax1.set_xlabel('Training Progress')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curves')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)

    # Right panel: Final reward distribution (bar chart with error bars)
    ax2 = fig.add_subplot(122)

    if final_rewards:
        methods_plot = list(final_rewards.keys())
        means = [final_rewards[m].mean() for m in methods_plot]
        stds = [final_rewards[m].std() for m in methods_plot]
        colors = [COLORS[m] for m in methods_plot]

        x = np.arange(len(methods_plot))
        bars = ax2.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['PPO', 'Contrastive'])
        ax2.set_ylabel('Final Reward')
        ax2.set_title('Final Performance')

        # Add std annotation
        for i, (m, s) in enumerate(zip(means, stds)):
            ax2.annotate(f'σ={s:.4f}', xy=(i, m + s + 0.001), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/teaser_figure.pdf")
    plt.savefig(f"{output_dir}/teaser_figure.png")
    plt.close()
    print(f"Saved teaser figure to {output_dir}/teaser_figure.pdf")


def plot_baseline_comparison(output_dir):
    """Generate bar chart comparing all baselines."""
    os.makedirs(output_dir, exist_ok=True)

    # Data from our experiments
    methods = ['PPO', 'Contrastive\n(Ours)', 'ATC', 'ICM', 'RND']
    final_rewards = [0.0146, 0.0149, 0.0148, 0.0134, 0.0131]
    stds = [0.0019, 0.0023, 0.0027, 0.0009, 0.0012]
    colors = [COLORS['ppo'], COLORS['contrastive'], COLORS['atc'], COLORS['icm'], COLORS['rnd']]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(methods))
    bars = ax.bar(x, final_rewards, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Final Reward')
    ax.set_title('Craftax Baseline Comparison (1M timesteps, 5 seeds)')

    # Highlight best performer
    best_idx = np.argmax(final_rewards)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # Add value labels
    for i, (v, s) in enumerate(zip(final_rewards, stds)):
        ax.annotate(f'{v:.4f}', xy=(i, v + s + 0.0005), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/baseline_comparison.pdf")
    plt.savefig(f"{output_dir}/baseline_comparison.png")
    plt.close()
    print(f"Saved baseline comparison to {output_dir}/baseline_comparison.pdf")


def main():
    metrics_dir = "v3_experiments/metrics"
    output_dir = "v3_experiments/figures"

    print("Generating figures...")

    # Check if metrics exist
    if os.path.exists(f"{metrics_dir}/ppo_all_seeds.npz"):
        plot_learning_curves(metrics_dir, output_dir)
        plot_variance_comparison(metrics_dir, output_dir)
        plot_teaser_figure(metrics_dir, output_dir)
    else:
        print(f"Metrics not found in {metrics_dir}. Run experiments first.")

    # Always generate baseline comparison (uses hardcoded data from experiments)
    plot_baseline_comparison(output_dir)

    print("\nDone! Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
