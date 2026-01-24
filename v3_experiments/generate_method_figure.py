"""
Generate method diagram for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import matplotlib
matplotlib.use('Agg')

def draw_method_figure():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Colors
    agent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Title
    ax.text(5, 3.8, 'Geometric Temporal Contrastive Learning in Multi-Agent RL',
            ha='center', va='top', fontsize=12, fontweight='bold')

    # Left panel: Multi-agent trajectories
    ax.text(1.5, 3.3, 'Multi-Agent\nTrajectories', ha='center', va='top', fontsize=10, fontweight='bold')

    # Draw agent trajectories
    for i, color in enumerate(agent_colors):
        y_base = 2.5 - i * 0.6
        # Draw trajectory line
        x_points = np.linspace(0.3, 2.7, 6)
        y_points = y_base + 0.15 * np.sin(np.linspace(0, 2*np.pi, 6) + i)
        ax.plot(x_points, y_points, color=color, linewidth=2, alpha=0.8)
        # Draw state circles
        for j, (x, y) in enumerate(zip(x_points, y_points)):
            circle = plt.Circle((x, y), 0.08, color=color, alpha=0.8)
            ax.add_patch(circle)
        ax.text(0.1, y_base, f'Agent {i+1}', fontsize=8, va='center', color=color)

    # Middle panel: Geometric sampling
    ax.text(5, 3.3, 'Geometric Temporal\nPositive Sampling', ha='center', va='top', fontsize=10, fontweight='bold')

    # Draw anchor and positive
    ax.add_patch(plt.Circle((4, 2), 0.15, color='#1f77b4', zorder=5))
    ax.text(4, 2, 't', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.text(4, 1.7, 'Anchor', ha='center', fontsize=8)

    ax.add_patch(plt.Circle((5.5, 2), 0.15, color='#2ca02c', zorder=5))
    ax.text(5.5, 2, 't+Δt', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    ax.text(5.5, 1.7, 'Positive', ha='center', fontsize=8)

    # Draw arrow between them
    ax.annotate('', xy=(5.3, 2), xytext=(4.2, 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Geometric distribution visualization
    x_geo = np.arange(1, 15)
    p = 0.023  # 1 - gamma
    y_geo = (1-p)**(x_geo-1) * p
    y_geo = y_geo / max(y_geo) * 0.5  # Normalize for display

    for i, (x, y) in enumerate(zip(x_geo[:8], y_geo[:8])):
        bar_x = 4.2 + i * 0.2
        ax.bar(bar_x, y, width=0.15, bottom=2.5, color='#9467bd', alpha=0.7)
    ax.text(5, 3.1, 'P(Δt) ~ Geo(1-γ)', ha='center', fontsize=8)

    # Right panel: Loss computation
    ax.text(8.5, 3.3, 'InfoNCE Loss', ha='center', va='top', fontsize=10, fontweight='bold')

    # Draw similarity computation
    ax.add_patch(FancyBboxPatch((7.2, 1.8), 2.6, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#f0f0f0', edgecolor='black', linewidth=1))

    ax.text(8.5, 2.7, 'sim(anchor, pos) ↑', ha='center', fontsize=8, color='green')
    ax.text(8.5, 2.3, 'sim(anchor, neg) ↓', ha='center', fontsize=8, color='red')
    ax.text(8.5, 1.95, 'negatives from other agents', ha='center', fontsize=7, style='italic')

    # Add arrows connecting panels
    ax.annotate('', xy=(3.5, 2), xytext=(3, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7, 2), xytext=(6, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Bottom text
    ax.text(5, 0.3,
            'Key insight: Multi-agent trajectories provide natural negative samples, enabling contrastive learning without augmentation',
            ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig('v3_experiments/figures/method_figure.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('v3_experiments/figures/method_figure.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved method figure to v3_experiments/figures/method_figure.pdf")


if __name__ == "__main__":
    draw_method_figure()
