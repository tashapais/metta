"""
Multi-Agent MuJoCo experiments for contrastive learning paper.
Runs PPO baseline and PPO+Contrastive on Multi-Agent Ant and Swimmer.

Uses vectorized environments for high throughput on GPU.

Usage:
    python run_mujoco.py --task ant --method baseline --num_seeds 1
    python run_mujoco.py --task ant --method contrastive --num_seeds 1
"""

import argparse
import time
from typing import Dict, Tuple, Optional
from collections import defaultdict
import os

# Use EGL for headless rendering (must be set before importing mujoco)
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import wandb

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium_robotics import mamujoco_v1


def make_single_env(task: str, seed: int, render_mode: Optional[str] = None):
    """Factory function for creating a single MA-MuJoCo environment."""
    def _init():
        if task == "ant":
            env = mamujoco_v1.parallel_env("Ant", agent_conf="2x4", render_mode=render_mode)
        elif task == "swimmer":
            env = mamujoco_v1.parallel_env("Swimmer", agent_conf="2x1", render_mode=render_mode)
        else:
            raise ValueError(f"Unknown task: {task}")
        return env
    return _init


class VectorizedMAMuJoCo:
    """Wrapper to vectorize multi-agent MuJoCo environments."""

    def __init__(self, task: str, num_envs: int, seed: int = 0):
        self.task = task
        self.num_envs = num_envs
        self.envs = [make_single_env(task, seed + i)() for i in range(num_envs)]

        # Get env info from first env
        self.agents = self.envs[0].agents
        self.num_agents = len(self.agents)
        self.obs_space = self.envs[0].observation_space(self.agents[0])
        self.action_space = self.envs[0].action_space(self.agents[0])
        self.obs_dim = self.obs_space.shape[0]
        self.action_dim = self.action_space.shape[0]

    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        all_obs = []
        for i, env in enumerate(self.envs):
            obs_dict, _ = env.reset(seed=seed + i if seed else None)
            # Stack observations from all agents: (num_agents, obs_dim)
            obs = np.stack([obs_dict[agent] for agent in self.agents])
            all_obs.append(obs)
        # Shape: (num_envs, num_agents, obs_dim)
        return np.stack(all_obs)

    def step(self, actions: np.ndarray):
        """
        Step all environments.

        Args:
            actions: (num_envs, num_agents, action_dim)

        Returns:
            obs: (num_envs, num_agents, obs_dim)
            rewards: (num_envs, num_agents)
            dones: (num_envs, num_agents)
            infos: list of dicts
        """
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []

        for i, env in enumerate(self.envs):
            # Convert actions to dict
            action_dict = {agent: actions[i, j] for j, agent in enumerate(self.agents)}

            obs_dict, reward_dict, term_dict, trunc_dict, info = env.step(action_dict)

            obs = np.stack([obs_dict[agent] for agent in self.agents])
            rewards = np.array([reward_dict[agent] for agent in self.agents])
            dones = np.array([term_dict[agent] or trunc_dict[agent] for agent in self.agents])

            all_obs.append(obs)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_infos.append(info)

            # Auto-reset if any agent is done (PettingZoo doesn't do this)
            if dones.any():
                reset_obs, _ = env.reset()
                # Replace obs with reset obs so next step starts fresh
                all_obs[-1] = np.stack([reset_obs[agent] for agent in self.agents])

        return (
            np.stack(all_obs),
            np.stack(all_rewards),
            np.stack(all_dones),
            all_infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def make_eval_env(task: str):
    """Create evaluation environment with rendering enabled."""
    if task == "ant":
        env = mamujoco_v1.parallel_env("Ant", agent_conf="2x4", render_mode="rgb_array")
    elif task == "swimmer":
        env = mamujoco_v1.parallel_env("Swimmer", agent_conf="2x1", render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown task: {task}")
    return env


def record_video(policy, task: str, device: torch.device, max_steps: int = 500) -> np.ndarray:
    """Record a video of the policy acting in the environment."""
    env = make_eval_env(task)
    obs_dict, _ = env.reset()
    agents = env.agents

    frames = []

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_list = [obs_dict[agent] for agent in agents]
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)

        with torch.no_grad():
            actions, _, _, _, _ = policy.get_action_and_value(obs_tensor)

        actions_np = actions.cpu().numpy()
        actions_dict = {agent: actions_np[i] for i, agent in enumerate(agents)}

        obs_dict, rewards_dict, terminations, truncations, _ = env.step(actions_dict)

        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()

    if len(frames) == 0:
        return None

    video = np.stack(frames)
    video = video.transpose(0, 3, 1, 2)  # THWC -> TCHW
    return video


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous control with contrastive embedding."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Embedding head for contrastive learning
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        embedding = self.embedding_head(features)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
        value = self.critic(features).squeeze(-1)
        return action_mean, action_std, value, embedding

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value, embedding = self.forward(obs)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
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
        Compute InfoNCE contrastive loss.

        Args:
            embeddings: (num_steps, batch_size, embedding_dim) where batch_size = num_envs * num_agents
            dones: (num_steps, batch_size)
        """
        num_steps, batch_size, embed_dim = embeddings.shape

        if self.use_projection_head:
            self._init_projection_head(embed_dim)
            embeddings = self.projection_head(embeddings)
            embed_dim = self.embedding_dim

        # Transpose to (batch_size, num_steps, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)

        # Sample positive pairs using geometric distribution
        prob = max(1.0 - self.discount, 1e-8)

        batch_indices = []
        anchor_steps = []
        positive_steps = []
        sampled_deltas = []

        # Sample multiple pairs per trajectory
        num_samples_per_traj = min(8, num_steps // 4)

        for traj_idx in range(batch_size):
            for _ in range(num_samples_per_traj):
                max_anchor = int(num_steps * 0.75)
                anchor_step = int(torch.randint(0, max(1, max_anchor), (1,)).item())
                max_future = num_steps - anchor_step - 1

                if max_future < 1:
                    continue

                # Sample geometric offset
                delta = int(np.random.geometric(prob))
                if delta > max_future:
                    delta = max_future
                if delta < 1:
                    delta = 1

                positive_step = anchor_step + delta

                batch_indices.append(traj_idx)
                anchor_steps.append(anchor_step)
                positive_steps.append(positive_step)
                sampled_deltas.append(float(delta))

        num_pairs = len(batch_indices)
        if num_pairs < 2:
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "positive_sim_std": 0.0,
                "negative_sim_std": 0.0,
                "num_pairs": 0,
                "delta_mean": 0.0,
            }

        batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        anchor_idx_tensor = torch.tensor(anchor_steps, device=self.device, dtype=torch.long)
        positive_idx_tensor = torch.tensor(positive_steps, device=self.device, dtype=torch.long)

        anchor_embeddings = embeddings[batch_idx_tensor, anchor_idx_tensor]
        positive_embeddings = embeddings[batch_idx_tensor, positive_idx_tensor]

        # Normalize
        anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)

        # Compute similarities
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


class MujocoSAEncoder(nn.Module):
    """State-Action encoder for GC-CRL. 4-layer MLP with LayerNorm + SiLU."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024, embed_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out]:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.ln1(self.fc1(x)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        x = F.silu(self.ln4(self.fc4(x)))
        return self.fc_out(x)


class MujocoGEncoder(nn.Module):
    """Goal encoder for GC-CRL. 4-layer MLP with LayerNorm + SiLU."""

    def __init__(self, goal_dim: int, hidden_dim: int = 1024, embed_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(goal_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out]:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(layer.bias)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.ln1(self.fc1(goal)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        x = F.silu(self.ln4(self.fc4(x)))
        return self.fc_out(x)


class GCCRLLoss:
    """Goal-Conditioned Contrastive RL loss for MuJoCo.

    Uses dual encoders (SA + G) with Euclidean distance Q-values,
    InfoNCE loss, and logsumexp regularization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        embed_dim: int = 64,
        contrastive_coef: float = 0.1,
        logsumexp_coef: float = 0.1,
        discount: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.contrastive_coef = contrastive_coef
        self.logsumexp_coef = logsumexp_coef
        self.discount = discount
        self.device = device

        self.sa_encoder = MujocoSAEncoder(state_dim, action_dim, hidden_dim, embed_dim).to(device)
        self.g_encoder = MujocoGEncoder(state_dim, hidden_dim, embed_dim).to(device)

    def parameters(self):
        return list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())

    def compute_loss(
        self,
        obs_buf: torch.Tensor,
        actions_buf: torch.Tensor,
        dones_buf: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GC-CRL loss over rollout buffer.

        Args:
            obs_buf: (num_steps, batch_size, obs_dim)
            actions_buf: (num_steps, batch_size, action_dim)
            dones_buf: (num_steps, batch_size)
        """
        num_steps, batch_size, obs_dim = obs_buf.shape

        # Sample (state, action, goal) triples
        prob = max(1.0 - self.discount, 1e-8)
        num_samples_per_traj = min(8, num_steps // 4)

        all_states = []
        all_actions = []
        all_goals = []

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

                goal_step = anchor_step + delta

                # Check episode boundary
                episode_boundary = False
                for t in range(anchor_step, goal_step):
                    if dones_buf[t, traj_idx] > 0.5:
                        episode_boundary = True
                        break
                if episode_boundary:
                    continue

                all_states.append(obs_buf[anchor_step, traj_idx])
                all_actions.append(actions_buf[anchor_step, traj_idx])
                all_goals.append(obs_buf[goal_step, traj_idx])

        if len(all_states) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "gc_categorical_accuracy": 0.0,
                "gc_logits_pos": 0.0,
                "gc_logits_neg": 0.0,
                "gc_infonce_loss": 0.0,
                "gc_logsumexp_reg": 0.0,
            }

        states = torch.stack(all_states)
        actions = torch.stack(all_actions)
        goals = torch.stack(all_goals)
        n = states.shape[0]

        # Encode
        sa_repr = self.sa_encoder(states, actions)
        g_repr = self.g_encoder(goals)

        # Pairwise negative Euclidean distance logits
        diff = sa_repr[:, None, :] - g_repr[None, :, :]  # [N, N, embed_dim]
        logits = -torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # [N, N]

        # InfoNCE
        diag_logits = torch.diag(logits)
        logsumexp = torch.logsumexp(logits, dim=1)
        infonce_loss = -torch.mean(diag_logits - logsumexp)

        # Logsumexp regularization
        logsumexp_reg = self.logsumexp_coef * torch.mean(logsumexp ** 2)

        total_loss = (infonce_loss + logsumexp_reg) * self.contrastive_coef

        # Metrics
        with torch.no_grad():
            I = torch.eye(n, device=self.device)
            correct = (torch.argmax(logits, dim=1) == torch.arange(n, device=self.device)).float()
            categorical_accuracy = correct.mean().item()
            logits_pos = torch.sum(logits * I) / torch.sum(I)
            logits_neg = torch.sum(logits * (1 - I)) / torch.sum(1 - I)

        metrics = {
            "gc_categorical_accuracy": categorical_accuracy,
            "gc_logits_pos": logits_pos.item(),
            "gc_logits_neg": logits_neg.item(),
            "gc_infonce_loss": infonce_loss.item(),
            "gc_logsumexp_reg": logsumexp_reg.item(),
        }

        return total_loss, metrics


class MujocoMatchedCapacityLoss:
    """Matched-capacity control: same dual-encoder architecture as GC-CRL, random-target MSE."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        embed_dim: int = 64,
        capacity_coef: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity_coef = capacity_coef
        self.device = device

        self.sa_encoder = MujocoSAEncoder(state_dim, action_dim, hidden_dim, embed_dim).to(device)
        self.g_encoder = MujocoGEncoder(state_dim, hidden_dim, embed_dim).to(device)
        # Fixed random projection (not trained)
        self.random_projection = nn.Linear(state_dim, embed_dim, bias=False).to(device)
        self.random_projection.requires_grad_(False)

    def parameters(self):
        return list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())

    def compute_loss(
        self,
        obs_buf: torch.Tensor,
        actions_buf: torch.Tensor,
        dones_buf: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_steps, batch_size, obs_dim = obs_buf.shape
        obs_flat = obs_buf.reshape(-1, obs_dim)
        actions_flat = actions_buf.reshape(-1, actions_buf.shape[-1])

        sa_repr = self.sa_encoder(obs_flat, actions_flat)
        g_repr = self.g_encoder(obs_flat)

        with torch.no_grad():
            target = self.random_projection(obs_flat)

        sa_loss = F.mse_loss(sa_repr, target)
        g_loss = F.mse_loss(g_repr, target)
        total_loss = self.capacity_coef * (sa_loss + g_loss)

        return total_loss, {
            "mc_sa_mse": sa_loss.item(),
            "mc_g_mse": g_loss.item(),
            "mc_total_loss": total_loss.item(),
        }


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

    # Initialize wandb
    run_name = f"mujoco_{config['task']}_{config['method']}.seed{seed}"
    wandb.init(
        project="metta",
        entity="tashapais",
        name=run_name,
        config={**config, "seed": seed},
        reinit=True,
    )

    # Create vectorized environment
    print(f"Creating {config['num_envs']} parallel environments...")
    env = VectorizedMAMuJoCo(config["task"], config["num_envs"], seed=seed)

    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    # Effective batch size per step = num_envs * num_agents
    agents_per_step = config["num_envs"] * num_agents

    print(f"Task: {config['task']}")
    print(f"Num envs: {config['num_envs']}, Agents per env: {num_agents}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Effective batch: {agents_per_step} agents/step, {agents_per_step * config['num_steps']} samples/update")

    # Create policy
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    # GC-CRL loss
    gc_crl_loss_fn = None
    gc_crl_optimizer = None
    if config["method"] == "gc_crl":
        gc_crl_loss_fn = GCCRLLoss(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config["gc_hidden_dim"],
            embed_dim=config["gc_embed_dim"],
            contrastive_coef=config["gc_contrastive_coef"],
            logsumexp_coef=config["gc_logsumexp_coef"],
            discount=config["gc_discount"],
            device=device,
        )
        gc_crl_optimizer = torch.optim.Adam(gc_crl_loss_fn.parameters(), lr=config["gc_lr"], eps=1e-5)

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

    # Matched-capacity loss
    mc_loss_fn = None
    mc_optimizer = None
    if config["method"] == "matched_capacity":
        mc_loss_fn = MujocoMatchedCapacityLoss(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config["mc_hidden_dim"],
            embed_dim=config["mc_embed_dim"],
            capacity_coef=config["mc_coef"],
            device=device,
        )
        mc_optimizer = torch.optim.Adam(mc_loss_fn.parameters(), lr=config["lr"], eps=1e-5)

    # Storage
    obs_buf = torch.zeros((config["num_steps"], agents_per_step, obs_dim), device=device)
    actions_buf = torch.zeros((config["num_steps"], agents_per_step, action_dim), device=device)
    log_probs_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    rewards_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    dones_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    values_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    embeddings_buf = torch.zeros((config["num_steps"], agents_per_step, config["embedding_dim"]), device=device)

    # Initialize
    obs = env.reset(seed=seed)  # (num_envs, num_agents, obs_dim)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    total_timesteps = 0
    num_updates = 0
    recent_returns = []
    episode_returns = defaultdict(float)

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        # Collect rollout
        for step in range(config["num_steps"]):
            # Flatten obs: (num_envs, num_agents, obs_dim) -> (num_envs * num_agents, obs_dim)
            obs_flat = obs.reshape(agents_per_step, obs_dim)

            with torch.no_grad():
                actions, log_probs, _, values, embeddings = policy.get_action_and_value(obs_flat)

            # Store
            obs_buf[step] = obs_flat
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs
            values_buf[step] = values
            embeddings_buf[step] = embeddings

            # Reshape actions for env: (num_envs * num_agents, action_dim) -> (num_envs, num_agents, action_dim)
            actions_env = actions.reshape(config["num_envs"], num_agents, action_dim).cpu().numpy()

            # Step environment
            next_obs, rewards, dones, infos = env.step(actions_env)

            # Flatten rewards and dones
            rewards_flat = torch.tensor(rewards.reshape(agents_per_step), dtype=torch.float32, device=device)
            dones_flat = torch.tensor(dones.reshape(agents_per_step), dtype=torch.float32, device=device)

            rewards_buf[step] = rewards_flat
            dones_buf[step] = dones_flat

            # Track episode returns (sum across agents per env)
            env_rewards = rewards.sum(axis=1)  # (num_envs,)
            env_dones = dones.any(axis=1)  # (num_envs,)

            for i in range(config["num_envs"]):
                episode_returns[i] += env_rewards[i]
                if env_dones[i]:
                    recent_returns.append(episode_returns[i])
                    episode_returns[i] = 0

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            total_timesteps += agents_per_step

        # Compute advantages
        obs_flat = obs.reshape(agents_per_step, obs_dim)
        with torch.no_grad():
            _, _, _, next_value, _ = policy.get_action_and_value(obs_flat)

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, next_value,
            gamma=config["gamma"], gae_lambda=config["gae_lambda"]
        )

        # Flatten for training
        batch_size = config["num_steps"] * agents_per_step
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_actions = actions_buf.reshape(batch_size, action_dim)
        b_log_probs = log_probs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        for epoch in range(config["update_epochs"]):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_indices = indices[start:end]

                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]

                _, new_log_probs, entropy, new_values, _ = policy.get_action_and_value(mb_obs, mb_actions)

                # Policy loss
                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()

        # Contrastive loss (recompute embeddings with grad so gradients flow to encoder)
        contrastive_metrics = {}
        if contrastive_loss_fn is not None:
            # Recompute embeddings for full rollout buffer with grad
            # obs_buf: (num_steps, agents_per_step, obs_dim)
            c_obs_flat = obs_buf.reshape(-1, obs_dim)  # (num_steps * agents_per_step, obs_dim)
            _, _, _, _, c_embeddings_flat = policy.get_action_and_value(c_obs_flat)
            c_embeddings = c_embeddings_flat.reshape(config["num_steps"], agents_per_step, -1)
            c_loss, contrastive_metrics = contrastive_loss_fn.compute_loss(
                c_embeddings, dones_buf
            )
            if c_loss.item() > 0:
                optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()

        # L2 activation regularization (recompute embeddings with grad)
        l2_metrics = {}
        if config["l2_coef"] > 0:
            # Sample a minibatch of obs and recompute embeddings with grad
            l2_indices = torch.randperm(batch_size, device=device)[:config["minibatch_size"]]
            l2_obs = b_obs[l2_indices]
            _, _, _, _, l2_embeddings = policy.get_action_and_value(l2_obs)
            l2_loss = config["l2_coef"] * torch.mean(l2_embeddings ** 2)
            optimizer.zero_grad()
            l2_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            optimizer.step()
            l2_metrics = {"l2_loss": l2_loss.item()}

        # Matched-capacity loss (separate optimizer)
        mc_metrics = {}
        if mc_loss_fn is not None:
            mc_loss, mc_metrics = mc_loss_fn.compute_loss(obs_buf, actions_buf, dones_buf)
            if mc_loss.requires_grad:
                mc_optimizer.zero_grad()
                mc_loss.backward()
                nn.utils.clip_grad_norm_(mc_loss_fn.parameters(), config["max_grad_norm"])
                mc_optimizer.step()

        # GC-CRL loss (separate optimizer)
        gc_crl_metrics = {}
        if gc_crl_loss_fn is not None:
            gc_loss, gc_crl_metrics = gc_crl_loss_fn.compute_loss(
                obs_buf, actions_buf, dones_buf
            )
            if gc_loss.requires_grad:
                gc_crl_optimizer.zero_grad()
                gc_loss.backward()
                nn.utils.clip_grad_norm_(gc_crl_loss_fn.parameters(), config["max_grad_norm"])
                gc_crl_optimizer.step()

        num_updates += 1

        # Logging
        if num_updates % 10 == 0:
            elapsed = time.time() - start_time
            sps = total_timesteps / elapsed

            mean_return = np.mean(recent_returns[-100:]) if recent_returns else 0.0

            log_dict = {
                "metric/agent_step": total_timesteps,
                "metric/epoch": num_updates,
                "overview/reward": mean_return,
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

            if l2_metrics:
                log_dict.update({"losses/" + k: v for k, v in l2_metrics.items()})

            if mc_metrics:
                log_dict.update({"losses/" + k: v for k, v in mc_metrics.items()})

            if gc_crl_metrics:
                log_dict.update({
                    "losses/gc_categorical_accuracy": gc_crl_metrics.get("gc_categorical_accuracy", 0),
                    "losses/gc_logits_pos": gc_crl_metrics.get("gc_logits_pos", 0),
                    "losses/gc_logits_neg": gc_crl_metrics.get("gc_logits_neg", 0),
                    "losses/gc_infonce_loss": gc_crl_metrics.get("gc_infonce_loss", 0),
                    "losses/gc_logsumexp_reg": gc_crl_metrics.get("gc_logsumexp_reg", 0),
                })

            wandb.log(log_dict, step=total_timesteps)

            print(
                f"Step {total_timesteps:,} | "
                f"Return: {mean_return:.2f} | "
                f"SPS: {sps:,.0f} | "
                f"PG Loss: {pg_loss.item():.4f}"
            )

        # Record video periodically
        if num_updates % config.get("video_interval", 100) == 0:
            print(f"Recording video at step {total_timesteps:,}...")
            try:
                video = record_video(policy, config["task"], device, max_steps=500)
                if video is not None:
                    wandb.log(
                        {"video/episode": wandb.Video(video, fps=30, format="mp4")},
                        step=total_timesteps,
                    )
                    print("Video logged to wandb")
            except Exception as e:
                print(f"Failed to record video: {e}")

    env.close()
    wandb.finish()

    return {
        "final_return": np.mean(recent_returns[-100:]) if recent_returns else 0.0,
        "total_timesteps": total_timesteps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ant", choices=["ant", "swimmer"])
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "contrastive", "gc_crl", "l2", "matched_capacity"])
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--seed_offset", type=int, default=0, help="Starting seed (seeds run from offset to offset+num_seeds-1)")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_steps", type=int, default=256, help="Steps per rollout per env")
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--video_interval", type=int, default=50, help="Record video every N updates")
    # GC-CRL hyperparameters
    parser.add_argument("--gc_hidden_dim", type=int, default=1024, help="GC-CRL encoder hidden dim")
    parser.add_argument("--gc_embed_dim", type=int, default=64, help="GC-CRL embedding dim")
    parser.add_argument("--gc_contrastive_coef", type=float, default=0.1, help="GC-CRL loss coefficient")
    parser.add_argument("--gc_logsumexp_coef", type=float, default=0.1, help="GC-CRL logsumexp regularization")
    parser.add_argument("--gc_discount", type=float, default=0.99, help="GC-CRL geometric goal sampling discount")
    parser.add_argument("--gc_lr", type=float, default=3e-4, help="GC-CRL encoder learning rate")
    # Contrastive coefficient override (for therapeutic window experiments)
    parser.add_argument("--contrastive_coef", type=float, default=None, help="Override contrastive coefficient (default: 0.00068 for contrastive method)")
    # L2 regularizer
    parser.add_argument("--l2_coef", type=float, default=0.00068, help="L2 activation regularization coefficient")
    # Matched capacity
    parser.add_argument("--mc_hidden_dim", type=int, default=1024, help="Matched-capacity encoder hidden dim")
    parser.add_argument("--mc_embed_dim", type=int, default=64, help="Matched-capacity embedding dim")
    parser.add_argument("--mc_coef", type=float, default=0.1, help="Matched-capacity loss coefficient")
    args = parser.parse_args()

    config = {
        "task": args.task,
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
        "max_grad_norm": 0.5,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "video_interval": args.video_interval,
        "contrastive_coef": (args.contrastive_coef if args.contrastive_coef is not None else 0.00068) if args.method == "contrastive" else 0.0,
        "temperature": 0.19,
        "discount": 0.977,
        "use_projection_head": True,
        # GC-CRL config
        "gc_hidden_dim": args.gc_hidden_dim,
        "gc_embed_dim": args.gc_embed_dim,
        "gc_contrastive_coef": args.gc_contrastive_coef,
        "gc_logsumexp_coef": args.gc_logsumexp_coef,
        "gc_discount": args.gc_discount,
        "gc_lr": args.gc_lr,
        # L2 regularizer
        "l2_coef": args.l2_coef if args.method == "l2" else 0.0,
        # Matched capacity
        "mc_hidden_dim": args.mc_hidden_dim,
        "mc_embed_dim": args.mc_embed_dim,
        "mc_coef": args.mc_coef,
    }

    print("=" * 60)
    print(f"Multi-Agent MuJoCo: {args.task.upper()} - {args.method.upper()}")
    print("=" * 60)
    print(f"Num envs: {args.num_envs}")
    print(f"Steps/rollout: {args.num_steps}")
    print(f"Samples/update: {args.num_envs * 2 * args.num_steps:,}")  # 2 agents per env
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Contrastive: {config['contrastive_coef'] > 0}")
    print(f"L2 Regularizer: {config['l2_coef'] > 0} (coef={config['l2_coef']})")
    print(f"Matched Capacity: {args.method == 'matched_capacity'}")
    print(f"GC-CRL: {args.method == 'gc_crl'}")
    print("=" * 60)

    results = []
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        print(f"\n--- Seed {seed} ---")
        result = train(config, seed=seed)
        results.append(result)

    final_returns = [r["final_return"] for r in results]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final return: {np.mean(final_returns):.2f} +/- {np.std(final_returns):.2f}")


if __name__ == "__main__":
    main()
