"""
SMAC (StarCraft Multi-Agent Challenge) experiments for contrastive learning paper.
Runs PPO baseline and PPO+Contrastive on SMAC micromanagement scenarios.

Requires StarCraft II headless binary and SMAC maps.
Set SC2PATH=/path/to/StarCraftII before running.

Usage:
    python run_smac.py --map 3m --method baseline --num_seeds 1
    python run_smac.py --map 8m_vs_9m --method contrastive --num_seeds 1
    python run_smac.py --map 3s5z --method contrastive --total_timesteps 5000000
"""

import argparse
import time
from typing import Dict, Tuple, Optional
from collections import defaultdict
import os

# Auto-detect SC2PATH if not set
if "SC2PATH" not in os.environ:
    default_sc2 = os.path.expanduser("~/StarCraftII")
    if os.path.isdir(default_sc2):
        os.environ["SC2PATH"] = default_sc2
    else:
        raise RuntimeError(
            "SC2PATH not set and ~/StarCraftII not found. "
            "Download StarCraft II headless from: "
            "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
        )

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from smac.env import StarCraft2Env


class VectorizedSMAC:
    """Wrapper to vectorize multiple SMAC environments.

    Handles episode boundaries, available action masking, and auto-reset.

    SMAC interface:
    - obs: each agent gets a local observation vector
    - state: global state (not used for decentralized exec)
    - actions: discrete, per agent
    - available_actions: binary mask per agent per step
    """

    def __init__(self, map_name: str, num_envs: int, seed: int = 0):
        self.map_name = map_name
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            env = StarCraft2Env(map_name=map_name, seed=seed + i)
            self.envs.append(env)

        # Get env info from first env
        env_info = self.envs[0].get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_dim = env_info["obs_shape"]
        self.state_dim = env_info["state_shape"]
        self.episode_limit = env_info["episode_limit"]

        self._needs_reset = np.ones(num_envs, dtype=bool)
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros(num_envs)

    def reset(self):
        """Reset all environments. Returns obs and available_actions."""
        all_obs = []
        all_avail = []
        for i, env in enumerate(self.envs):
            env.reset()
            obs = np.array([env.get_obs_agent(a) for a in range(self.n_agents)])
            avail = np.array([env.get_avail_agent_actions(a) for a in range(self.n_agents)])
            all_obs.append(obs)
            all_avail.append(avail)
            self._needs_reset[i] = False
            self.episode_steps[i] = 0
            self.episode_returns[i] = 0.0

        # (num_envs, n_agents, obs_dim), (num_envs, n_agents, n_actions)
        return np.stack(all_obs), np.stack(all_avail)

    def step(self, actions: np.ndarray):
        """
        Step all environments.

        Args:
            actions: (num_envs, n_agents) integer actions

        Returns:
            obs: (num_envs, n_agents, obs_dim)
            avail_actions: (num_envs, n_agents, n_actions)
            rewards: (num_envs,) shared team reward
            dones: (num_envs,) episode terminated
            infos: list of dicts with battle stats
        """
        all_obs = []
        all_avail = []
        all_rewards = np.zeros(self.num_envs)
        all_dones = np.zeros(self.num_envs, dtype=bool)
        all_infos = []

        for i, env in enumerate(self.envs):
            if self._needs_reset[i]:
                env.reset()
                obs = np.array([env.get_obs_agent(a) for a in range(self.n_agents)])
                avail = np.array([env.get_avail_agent_actions(a) for a in range(self.n_agents)])
                all_obs.append(obs)
                all_avail.append(avail)
                all_infos.append({})
                self._needs_reset[i] = False
                self.episode_steps[i] = 0
                self.episode_returns[i] = 0.0
                continue

            # Step with list of actions
            reward, terminated, info = env.step(actions[i].tolist())
            self.episode_steps[i] += 1
            self.episode_returns[i] += reward

            obs = np.array([env.get_obs_agent(a) for a in range(self.n_agents)])
            avail = np.array([env.get_avail_agent_actions(a) for a in range(self.n_agents)])

            all_obs.append(obs)
            all_avail.append(avail)
            all_rewards[i] = reward
            all_dones[i] = terminated

            ep_info = {
                "episode_length": self.episode_steps[i],
                "battle_won": info.get("battle_won", False),
                "episode_return": self.episode_returns[i],
            }
            all_infos.append(ep_info)

            if terminated or self.episode_steps[i] >= self.episode_limit:
                all_dones[i] = True
                self._needs_reset[i] = True

        return (
            np.stack(all_obs),
            np.stack(all_avail),
            all_rewards,
            all_dones,
            all_infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


class ActorCritic(nn.Module):
    """Actor-Critic for discrete action SMAC with action masking and contrastive embedding."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Embedding head for contrastive learning
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

        # Actor head (logits over discrete actions)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        embedding = self.embedding_head(features)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value, embedding

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            obs: (batch, obs_dim)
            avail_actions: (batch, n_actions) binary mask
            action: optional (batch,) for log_prob computation
        """
        logits, value, embedding = self.forward(obs)

        # Mask unavailable actions with large negative logits
        logits = logits.masked_fill(avail_actions == 0, -1e10)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, embedding


class ContrastiveLoss:
    """InfoNCE contrastive loss with geometric temporal sampling."""

    def __init__(
        self,
        temperature: float = 0.19,
        discount: float = 0.977,
        contrastive_coef: float = 0.00068,
        embedding_dim: int = 64,
        use_projection_head: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.temperature = temperature
        self.discount = discount
        self.contrastive_coef = contrastive_coef
        self.embedding_dim = embedding_dim
        self.use_projection_head = use_projection_head
        self.device = device
        self.projection_head = None
        self._projection_input_dim = None

    def _init_projection_head(self, input_dim: int):
        if self.projection_head is None or self._projection_input_dim != input_dim:
            self._projection_input_dim = input_dim
            self.projection_head = nn.Linear(input_dim, self.embedding_dim).to(self.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (num_steps, batch_size, embedding_dim)
            dones: (num_steps, batch_size)
        """
        num_steps, batch_size, embed_dim = embeddings.shape

        if self.use_projection_head:
            self._init_projection_head(embed_dim)
            embeddings = self.projection_head(embeddings)
            embed_dim = self.embedding_dim

        embeddings = embeddings.permute(1, 0, 2)  # (batch, steps, dim)

        prob = max(1.0 - self.discount, 1e-8)

        batch_indices = []
        anchor_steps = []
        positive_steps = []
        sampled_deltas = []
        num_samples_per_traj = min(8, num_steps // 4)

        for traj_idx in range(batch_size):
            for _ in range(num_samples_per_traj):
                max_anchor = int(num_steps * 0.75)
                anchor_step = int(torch.randint(0, max(1, max_anchor), (1,)).item())
                max_future = num_steps - anchor_step - 1
                if max_future < 1:
                    continue
                delta = int(np.random.geometric(prob))
                delta = min(delta, max_future)
                delta = max(delta, 1)
                positive_step = anchor_step + delta

                batch_indices.append(traj_idx)
                anchor_steps.append(anchor_step)
                positive_steps.append(positive_step)
                sampled_deltas.append(float(delta))

        num_pairs = len(batch_indices)
        if num_pairs < 2:
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0, "negative_sim_mean": 0.0,
                "positive_sim_std": 0.0, "negative_sim_std": 0.0,
                "num_pairs": 0, "delta_mean": 0.0,
            }

        batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        anchor_idx_tensor = torch.tensor(anchor_steps, device=self.device, dtype=torch.long)
        positive_idx_tensor = torch.tensor(positive_steps, device=self.device, dtype=torch.long)

        anchor_embeddings = F.normalize(embeddings[batch_idx_tensor, anchor_idx_tensor], dim=-1)
        positive_embeddings = F.normalize(embeddings[batch_idx_tensor, positive_idx_tensor], dim=-1)

        similarities = anchor_embeddings @ positive_embeddings.T
        positive_logits = similarities.diagonal().unsqueeze(1)
        mask = torch.eye(num_pairs, device=self.device, dtype=torch.bool)
        negative_logits = similarities[~mask].view(num_pairs, num_pairs - 1)

        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.temperature
        labels = torch.zeros(num_pairs, dtype=torch.long, device=self.device)
        infonce_loss = F.cross_entropy(logits, labels, reduction="mean")

        metrics = {
            "positive_sim_mean": positive_logits.mean().item(),
            "negative_sim_mean": negative_logits.mean().item(),
            "positive_sim_std": positive_logits.std().item() if num_pairs > 1 else 0.0,
            "negative_sim_std": negative_logits.std().item() if num_pairs > 1 else 0.0,
            "num_pairs": num_pairs,
            "delta_mean": sum(sampled_deltas) / len(sampled_deltas),
        }
        return infonce_loss * self.contrastive_coef, metrics


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def train(config: dict, seed: int = 0):
    """Run a single training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = f"smac_{config['map_name']}_{config['method']}.seed{seed}"
    wandb.init(
        project="metta",
        entity="tashapais",
        name=run_name,
        config={**config, "seed": seed},
        reinit=True,
    )

    # Create vectorized SMAC environments
    print(f"Creating {config['num_envs']} parallel SMAC environments ({config['map_name']})...")
    env = VectorizedSMAC(config["map_name"], config["num_envs"], seed=seed)

    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    agents_per_step = config["num_envs"] * n_agents

    print(f"Map: {config['map_name']}")
    print(f"Num envs: {config['num_envs']}, Agents per env: {n_agents}")
    print(f"Obs dim: {obs_dim}, Actions: {n_actions}")
    print(f"Effective batch: {agents_per_step} agents/step, {agents_per_step * config['num_steps']} samples/update")

    # Create policy (shared across agents, parameter sharing)
    policy = ActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    # Contrastive loss
    contrastive_loss_fn = None
    if config["contrastive_coef"] > 0:
        contrastive_loss_fn = ContrastiveLoss(
            temperature=config["temperature"],
            discount=config["discount"],
            contrastive_coef=config["contrastive_coef"],
            embedding_dim=config["embedding_dim"],
            use_projection_head=config["use_projection_head"],
            device=device,
        )

    # Rollout storage
    obs_buf = torch.zeros((config["num_steps"], agents_per_step, obs_dim), device=device)
    avail_buf = torch.zeros((config["num_steps"], agents_per_step, n_actions), device=device)
    actions_buf = torch.zeros((config["num_steps"], agents_per_step), dtype=torch.long, device=device)
    log_probs_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    rewards_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    dones_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    values_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    embeddings_buf = torch.zeros((config["num_steps"], agents_per_step, config["embedding_dim"]), device=device)

    # Initialize
    obs, avail_actions = env.reset()  # (num_envs, n_agents, obs_dim), (num_envs, n_agents, n_actions)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32, device=device)

    total_timesteps = 0
    num_updates = 0
    recent_returns = []
    recent_ep_lengths = []
    recent_win_rates = []
    episode_returns = defaultdict(float)

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        # Collect rollout
        for step in range(config["num_steps"]):
            obs_flat = obs.reshape(agents_per_step, obs_dim)
            avail_flat = avail_actions.reshape(agents_per_step, n_actions)

            with torch.no_grad():
                actions, log_probs, _, values, embeddings = policy.get_action_and_value(
                    obs_flat, avail_flat
                )

            obs_buf[step] = obs_flat
            avail_buf[step] = avail_flat
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs
            values_buf[step] = values
            embeddings_buf[step] = embeddings

            # Reshape actions for env: (num_envs*n_agents,) -> (num_envs, n_agents)
            actions_env = actions.reshape(config["num_envs"], n_agents).cpu().numpy()

            next_obs, next_avail, rewards, dones, infos = env.step(actions_env)

            # SMAC gives shared team reward; broadcast to all agents
            rewards_expanded = np.repeat(rewards[:, np.newaxis], n_agents, axis=1)
            dones_expanded = np.repeat(dones[:, np.newaxis], n_agents, axis=1)

            rewards_flat = torch.tensor(rewards_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)
            dones_flat = torch.tensor(dones_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)

            rewards_buf[step] = rewards_flat
            dones_buf[step] = dones_flat

            # Track episode stats
            for i in range(config["num_envs"]):
                episode_returns[i] += rewards[i]
                if dones[i]:
                    ep_return = episode_returns[i]
                    ep_length = infos[i].get("episode_length", 0)
                    won = infos[i].get("battle_won", False)
                    recent_returns.append(ep_return)
                    recent_ep_lengths.append(ep_length)
                    recent_win_rates.append(1.0 if won else 0.0)
                    episode_returns[i] = 0

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            avail_actions = torch.tensor(next_avail, dtype=torch.float32, device=device)
            total_timesteps += agents_per_step

        # Compute advantages
        obs_flat = obs.reshape(agents_per_step, obs_dim)
        avail_flat = avail_actions.reshape(agents_per_step, n_actions)
        with torch.no_grad():
            _, _, _, next_value, _ = policy.get_action_and_value(obs_flat, avail_flat)

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, next_value,
            gamma=config["gamma"], gae_lambda=config["gae_lambda"],
        )

        # Flatten for training
        batch_size = config["num_steps"] * agents_per_step
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_avail = avail_buf.reshape(batch_size, n_actions)
        b_actions = actions_buf.reshape(batch_size)
        b_log_probs = log_probs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        for epoch in range(config["update_epochs"]):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_indices = indices[start:end]

                mb_obs = b_obs[mb_indices]
                mb_avail = b_avail[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]

                _, new_log_probs, entropy, new_values, _ = policy.get_action_and_value(
                    mb_obs, mb_avail, mb_actions
                )

                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()

        # Contrastive loss (separate backward pass on embeddings)
        contrastive_metrics = {}
        if contrastive_loss_fn is not None:
            c_loss, contrastive_metrics = contrastive_loss_fn.compute_loss(
                embeddings_buf, dones_buf
            )
            if c_loss.item() > 0:
                optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()

        num_updates += 1

        # Logging
        if num_updates % 10 == 0:
            elapsed = time.time() - start_time
            sps = total_timesteps / elapsed

            mean_return = np.mean(recent_returns[-100:]) if recent_returns else 0.0
            mean_ep_length = np.mean(recent_ep_lengths[-100:]) if recent_ep_lengths else 0.0
            win_rate = np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0

            log_dict = {
                "metric/agent_step": total_timesteps,
                "metric/epoch": num_updates,
                "overview/reward": mean_return,
                "overview/episode_length": mean_ep_length,
                "overview/win_rate": win_rate,
                "overview/sps": sps,
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy_loss.item(),
            }

            if contrastive_metrics:
                log_dict.update({
                    "losses/contrastive_loss": contrastive_metrics.get("positive_sim_mean", 0) * config["contrastive_coef"],
                    "losses/positive_sim_mean": contrastive_metrics.get("positive_sim_mean", 0),
                    "losses/negative_sim_mean": contrastive_metrics.get("negative_sim_mean", 0),
                    "losses/positive_sim_std": contrastive_metrics.get("positive_sim_std", 0),
                    "losses/negative_sim_std": contrastive_metrics.get("negative_sim_std", 0),
                    "losses/num_pairs": contrastive_metrics.get("num_pairs", 0),
                    "losses/delta_mean": contrastive_metrics.get("delta_mean", 0),
                })

            wandb.log(log_dict, step=total_timesteps)

            print(
                f"Step {total_timesteps:,} | "
                f"Return: {mean_return:.2f} | "
                f"WinRate: {win_rate:.1%} | "
                f"EpLen: {mean_ep_length:.0f} | "
                f"SPS: {sps:,.0f}"
            )

    env.close()
    wandb.finish()

    return {
        "final_return": np.mean(recent_returns[-100:]) if recent_returns else 0.0,
        "final_win_rate": np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0,
        "total_timesteps": total_timesteps,
    }


# Maps grouped by difficulty (from SMAC paper)
EASY_MAPS = ["2m_vs_1z", "2s_vs_1sc", "3m", "8m"]
HARD_MAPS = ["3s5z", "5m_vs_6m", "8m_vs_9m", "MMM", "1c3s5z", "3s_vs_3z", "2s3z"]
SUPER_HARD_MAPS = ["3s5z_vs_3s6z", "27m_vs_30m", "MMM2", "corridor", "6h_vs_8z", "bane_vs_bane"]

ALL_MAPS = EASY_MAPS + HARD_MAPS + SUPER_HARD_MAPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="3m", choices=ALL_MAPS)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "contrastive"])
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per rollout per env")
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    args = parser.parse_args()

    config = {
        "map_name": args.map,
        "method": args.method,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "num_steps": args.num_steps,
        "minibatch_size": args.minibatch_size,
        "update_epochs": 10,
        "lr": args.lr,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 10.0,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "contrastive_coef": 0.00068 if args.method == "contrastive" else 0.0,
        "temperature": 0.19,
        "discount": 0.977,
        "use_projection_head": True,
    }

    print("=" * 60)
    print(f"SMAC: {args.map} - {args.method.upper()}")
    print("=" * 60)
    print(f"Num envs: {args.num_envs}")
    print(f"Steps/rollout: {args.num_steps}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Contrastive: {config['contrastive_coef'] > 0}")
    print("=" * 60)

    results = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---")
        result = train(config, seed=seed)
        results.append(result)

    final_returns = [r["final_return"] for r in results]
    final_win_rates = [r["final_win_rate"] for r in results]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final return: {np.mean(final_returns):.2f} +/- {np.std(final_returns):.2f}")
    print(f"Final win rate: {np.mean(final_win_rates):.1%} +/- {np.std(final_win_rates):.1%}")


if __name__ == "__main__":
    main()
