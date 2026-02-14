"""
Generate a MettaGrid screenshot showing 12 agents mid-game.

Uses Simulation class + grid_objects() API to extract grid state,
then renders to matplotlib with color-coded tiles.
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import to_rgba

from mettagrid.builder.envs import make_arena
from mettagrid.simulator.simulator import Simulation


# Color palette for grid objects
TILE_COLORS = {
    'wall': '#4a4a4a',
    'empty': '#f0f0f0',
    'agent.team0': '#e41a1c',
    'agent.team1': '#377eb8',
    'chest': '#8B4513',
    'heart': '#ff4444',
    'charger': '#FFD700',
    'junction': '#4169E1',
    'mine': '#808080',
    'generator': '#9ACD32',
    'altar': '#DDA0DD',
    'converter': '#FF8C00',
}


def get_team_color(team_id):
    """Get color for a team."""
    team_colors = ['#e41a1c', '#377eb8', '#2ca02c', '#984ea3']
    return team_colors[team_id % len(team_colors)]


def render_grid(sim, ax, step_num):
    """Render the grid state to a matplotlib axes."""
    try:
        grid_objects_dict = sim.grid_objects()
    except AttributeError:
        # Fallback: render from observations
        obs = sim._c_sim.observations()
        ax.text(0.5, 0.5, f"Step {step_num}\n{obs.shape[0]} agents",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        return

    # grid_objects() returns a dict {id: obj_dict}
    if not grid_objects_dict:
        ax.text(0.5, 0.5, "No grid objects", ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        return

    objects = list(grid_objects_dict.values())

    max_r = max(obj.get('r', 0) for obj in objects) + 1
    max_c = max(obj.get('c', 0) for obj in objects) + 1

    # Draw background
    ax.set_xlim(-0.5, max_c - 0.5)
    ax.set_ylim(-0.5, max_r - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Fill with empty tiles
    for r in range(max_r):
        for c in range(max_c):
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0.5,
                            edgecolor='#cccccc', facecolor=TILE_COLORS['empty'])
            ax.add_patch(rect)

    # Draw objects
    for obj in objects:
        r = obj.get('r', 0)
        c = obj.get('c', 0)
        obj_type = obj.get('type_name', obj.get('type', 'unknown')).lower()

        if 'wall' in obj_type:
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0,
                            facecolor=TILE_COLORS['wall'])
            ax.add_patch(rect)
        elif 'agent' in obj_type:
            team_id = obj.get('team', obj.get('tag', obj.get('group', 0)))
            # tag is typically 0-5 for team 0, 6+ for team 1
            if isinstance(team_id, int) and team_id >= 6:
                team_id = 1
            elif isinstance(team_id, int):
                team_id = 0
            color = get_team_color(team_id)
            circle = Circle((c, r), 0.35, linewidth=1.5,
                           edgecolor='black', facecolor=color)
            ax.add_patch(circle)
            # Agent direction indicator
            direction = obj.get('direction', 0)
            dx, dy = [(0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0)][direction % 4]
            ax.arrow(c, r, dx, dy, head_width=0.12, head_length=0.08,
                    fc='white', ec='white', linewidth=1)
        elif 'chest' in obj_type:
            rect = Rectangle((c - 0.35, r - 0.35), 0.7, 0.7, linewidth=1,
                            edgecolor='#654321', facecolor=TILE_COLORS['chest'])
            ax.add_patch(rect)
        elif 'heart' in obj_type:
            circle = Circle((c, r), 0.25, linewidth=1,
                           edgecolor='#cc0000', facecolor=TILE_COLORS['heart'])
            ax.add_patch(circle)
        elif 'charger' in obj_type or 'generator' in obj_type:
            rect = Rectangle((c - 0.3, r - 0.3), 0.6, 0.6, linewidth=1,
                            edgecolor='#b8860b', facecolor=TILE_COLORS['charger'])
            ax.add_patch(rect)
            ax.text(c, r, '+', ha='center', va='center', fontsize=8,
                   fontweight='bold', color='black')
        elif 'junction' in obj_type:
            rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, linewidth=1.5,
                            edgecolor='#1a3c6e', facecolor=TILE_COLORS['junction'])
            ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])


def generate_screenshot(output_dir, num_agents=12, seed=42, run_steps=200):
    """Generate and save MettaGrid screenshot."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_agents}-agent arena...")
    cfg = make_arena(num_agents=num_agents)
    sim = Simulation(cfg, seed=seed)

    # Run for some steps with random actions
    n_actions = len(sim.action_ids)
    for step in range(run_steps):
        actions = np.random.randint(0, n_actions, size=num_agents)
        sim._c_sim.actions()[:] = actions
        sim.step()

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        render_grid(sim, ax, run_steps)
        ax.set_title(f"MettaGrid Arena ({num_agents} agents, step {run_steps})",
                    fontsize=13, fontweight='bold', pad=10)
    except Exception as e:
        print(f"Grid rendering failed ({e}), using observation-based rendering...")
        # Fallback: visualize observation matrix
        obs = sim._c_sim.observations()
        ax.imshow(obs.mean(axis=0) if obs.ndim > 2 else obs, cmap='viridis', aspect='auto')
        ax.set_title(f"MettaGrid Observations ({num_agents} agents)", fontsize=13, fontweight='bold')

    # Add legend
    legend_items = [
        ('Team 0', '#e41a1c'), ('Team 1', '#377eb8'),
        ('Wall', '#4a4a4a'), ('Chest', '#8B4513'),
        ('Heart', '#ff4444'), ('Charger', '#FFD700'),
        ('Junction', '#4169E1'),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.plot([], [], 's', color=color, markersize=8, label=label)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mettagrid_screenshot.png", dpi=300, bbox_inches='tight')
    plt.close()
    sim.close()
    print(f"Saved MettaGrid screenshot to {output_dir}/mettagrid_screenshot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=12)
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/metta/v3_experiments/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_screenshot(args.output_dir, num_agents=args.num_agents, seed=args.seed)
