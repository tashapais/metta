"""
Generate teaser figure: PCA embedding trajectory visualization.

Three panels:
  (a) Healthy representations - embeddings spread across space, agents distinguishable
  (b) Collapsed representations - embeddings clustered in low-dim subspace
  (c) Expansion near goals - color by timestep, show expansion ratio difference

Uses SMAC 3s_vs_5z (the map with clearest 5.7x lead time).
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

if "SC2PATH" not in os.environ:
    default_sc2 = os.path.expanduser("~/StarCraftII")
    if os.path.isdir(default_sc2):
        os.environ["SC2PATH"] = default_sc2

from smac.env import StarCraft2Env


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.encoder(obs)
        embedding = self.embedding_head(features)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value, embedding

    def get_action_and_value(self, obs, avail_actions):
        logits, value, embedding = self.forward(obs)
        logits = logits.masked_fill(avail_actions == 0, -1e10)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, embedding


def collect_embeddings(policy, env, n_agents, obs_dim, device, num_episodes=50):
    """Collect per-agent embeddings over episodes."""
    all_episode_embs = []
    all_episode_won = []

    for ep in range(num_episodes):
        env.reset()
        obs = np.array([env.get_obs_agent(a) for a in range(n_agents)])
        avail = np.array([env.get_avail_agent_actions(a) for a in range(n_agents)])

        episode_embs = []
        done = False
        step = 0
        info = {}

        while not done and step < env.episode_limit:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            avail_t = torch.tensor(avail, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, embeddings = policy.get_action_and_value(obs_t, avail_t)

            episode_embs.append(embeddings.cpu().numpy())  # (n_agents, embed_dim)
            reward, done, info = env.step(actions.cpu().numpy().tolist())
            obs = np.array([env.get_obs_agent(a) for a in range(n_agents)])
            avail = np.array([env.get_avail_agent_actions(a) for a in range(n_agents)])
            step += 1

        if len(episode_embs) >= 5:
            all_episode_embs.append(np.stack(episode_embs))  # (T, n_agents, embed_dim)
            all_episode_won.append(info.get("battle_won", False))

    return all_episode_embs, all_episode_won


def train_short(map_name, total_steps, device, seed=0):
    """Train a policy for a given number of steps and return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = StarCraft2Env(map_name=map_name, seed=seed)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    policy = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

    steps = 0
    while steps < total_steps:
        env.reset()
        obs = np.array([env.get_obs_agent(a) for a in range(n_agents)])
        avail = np.array([env.get_avail_agent_actions(a) for a in range(n_agents)])
        done = False
        ep_obs, ep_avail, ep_actions, ep_rewards, ep_values = [], [], [], [], []

        while not done and steps < total_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            avail_t = torch.tensor(avail, dtype=torch.float32, device=device)
            logits, value, _ = policy(obs_t)
            logits = logits.masked_fill(avail_t == 0, -1e10)
            dist = Categorical(logits=logits)
            action = dist.sample()

            ep_obs.append(obs_t)
            ep_avail.append(avail_t)
            ep_actions.append(action)
            ep_values.append(value)

            reward, done, info = env.step(action.cpu().numpy().tolist())
            ep_rewards.append(reward)
            obs = np.array([env.get_obs_agent(a) for a in range(n_agents)])
            avail = np.array([env.get_avail_agent_actions(a) for a in range(n_agents)])
            steps += n_agents

        # Simple REINFORCE update
        if ep_rewards:
            returns = []
            G = 0
            for r in reversed(ep_rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = 0
            for t, (obs_t, avail_t, act, val) in enumerate(zip(ep_obs, ep_avail, ep_actions, ep_values)):
                logits, _, _ = policy(obs_t)
                logits = logits.masked_fill(avail_t == 0, -1e10)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(act)
                loss += -(log_probs * returns[t]).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()

    return policy, env, n_agents, obs_dim


def generate_teaser(output_dir, map_name="3s_vs_5z", device_id=0):
    """Generate the 3-panel teaser figure."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print("Training early-stage policy (healthy representations)...")
    policy_early, env_early, n_agents, obs_dim = train_short(map_name, 50_000, device, seed=0)
    embs_early, won_early = collect_embeddings(policy_early, env_early, n_agents, obs_dim, device, num_episodes=30)
    env_early.close()

    print("Training late-stage policy (collapsed representations)...")
    policy_late, env_late, _, _ = train_short(map_name, 500_000, device, seed=42)
    embs_late, won_late = collect_embeddings(policy_late, env_late, n_agents, obs_dim, device, num_episodes=30)
    env_late.close()

    # Fit PCA on combined embeddings — 3 components for panel (c)
    all_flat = []
    for ep_embs in embs_early + embs_late:
        all_flat.append(ep_embs.reshape(-1, ep_embs.shape[-1]))
    all_flat = np.concatenate(all_flat, axis=0)
    pca3 = PCA(n_components=3)
    pca3.fit(all_flat)
    pca2 = PCA(n_components=2)
    pca2.fit(all_flat)

    # --- Create figure: all 3 panels in a single row, panel (c) gets extra width ---
    fig = plt.figure(figsize=(18, 5))
    cmap = cm.viridis
    agent_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']

    # Use gridspec: panels (a) and (b) get equal width, panel (c) gets 1.4x width
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.3)

    # Panel (a): Healthy representations (early training) — 2D
    ax = fig.add_subplot(gs[0])
    ax.set_title("(a) Healthy Representations\n(Early Training)", fontsize=11, fontweight='bold')
    for ep_idx, ep_embs in enumerate(embs_early[:8]):
        projected = pca2.transform(ep_embs.reshape(-1, ep_embs.shape[-1]))
        projected = projected.reshape(ep_embs.shape[0], ep_embs.shape[1], 2)
        for agent_id in range(min(n_agents, len(agent_colors))):
            color = agent_colors[agent_id]
            ax.scatter(projected[:, agent_id, 0], projected[:, agent_id, 1],
                      c=color, s=10, alpha=0.4,
                      label=f'Agent {agent_id}' if ep_idx == 0 else None)
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)
    ax.legend(fontsize=6, loc='upper right', ncol=1, markerscale=2)

    # Panel (b): Collapsed representations (late training) — 2D
    ax = fig.add_subplot(gs[1])
    ax.set_title("(b) Collapsed Representations\n(Late Training)", fontsize=11, fontweight='bold')
    for ep_idx, ep_embs in enumerate(embs_late[:8]):
        projected = pca2.transform(ep_embs.reshape(-1, ep_embs.shape[-1]))
        projected = projected.reshape(ep_embs.shape[0], ep_embs.shape[1], 2)
        for agent_id in range(min(n_agents, len(agent_colors))):
            color = agent_colors[agent_id]
            ax.scatter(projected[:, agent_id, 0], projected[:, agent_id, 1],
                      c=color, s=10, alpha=0.4)
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)

    # Panel (c): 3D embedding trajectory — wider panel
    ax = fig.add_subplot(gs[2], projection='3d')
    ax.set_title("(c) Embedding Trajectories\nNear Goal (3D PCA)", fontsize=11, fontweight='bold', pad=10)

    # Find the longest successful episode from early training (most movement near goal)
    best_ep = None
    best_len = 0
    for i, (ep_embs, won) in enumerate(zip(embs_early, won_early)):
        if len(ep_embs) > best_len:
            best_ep = ep_embs
            best_len = len(ep_embs)

    if best_ep is None and embs_early:
        best_ep = embs_early[0]

    if best_ep is not None:
        # Project all agent embeddings into 3D for this episode
        proj_3d = pca3.transform(best_ep.reshape(-1, best_ep.shape[-1]))
        proj_3d = proj_3d.reshape(best_ep.shape[0], best_ep.shape[1], 3)  # (T, n_agents, 3)
        T = proj_3d.shape[0]
        timesteps_norm = np.linspace(0, 1, T)

        for agent_id in range(min(n_agents, len(agent_colors))):
            traj = proj_3d[:, agent_id, :]  # (T, 3)
            colors = cmap(timesteps_norm)
            # Draw trajectory line with increasing width near goal
            for t in range(T - 1):
                alpha = 0.3 + 0.7 * timesteps_norm[t]
                lw = 1.0 + 3.0 * timesteps_norm[t]
                ax.plot3D(traj[t:t+2, 0], traj[t:t+2, 1], traj[t:t+2, 2],
                         color=colors[t], linewidth=lw, alpha=alpha)
            # Scatter points with increasing size near goal
            sizes = 10 + 80 * timesteps_norm ** 2
            ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2],
                        c=timesteps_norm, cmap='viridis', s=sizes,
                        alpha=0.7, edgecolors='none')
            # Mark start and end
            ax.scatter3D([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                        c='blue', s=120, marker='o', edgecolors='black', linewidths=1.5, zorder=10,
                        label='Start' if agent_id == 0 else None)
            ax.scatter3D([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                        c='yellow', s=120, marker='*', edgecolors='black', linewidths=1.0, zorder=10,
                        label='Goal' if agent_id == 0 else None)

    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.08, shrink=0.8)
    cbar.set_label("Timestep", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Start", "Mid", "Goal"])
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)
    ax.set_zlabel("PC 3", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=25, azim=135)
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/teaser_embedding_geometry.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/teaser_embedding_geometry.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved teaser figure to {output_dir}/teaser_embedding_geometry.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--map", type=str, default="3s_vs_5z")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/metta/v3_experiments/figures")
    args = parser.parse_args()
    generate_teaser(args.output_dir, map_name=args.map, device_id=args.gpu)
