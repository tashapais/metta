"""
Navigation environment experiments for contrastive learning paper.
Runs PPO baseline and PPO+Contrastive on Navigation task.

Uses vectorized environments for high throughput.

Usage:
    python run_navigation.py --method baseline --num_envs 16 --total_timesteps 10000000
    python run_navigation.py --method contrastive --num_envs 16 --total_timesteps 10000000
    python run_navigation.py --method gccrl --num_envs 16 --total_timesteps 10000000
"""

import argparse
import time
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from mettagrid import Simulator
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
import mettagrid.builder.envs as eb


class VectorizedNavigation:
    """Vectorized wrapper for multiple Navigation environments."""

    def __init__(self, num_envs: int, num_agents: int = 12, seed: int = 0, max_episode_steps: int = 500):
        self.num_envs = num_envs
        self.num_agents_per_env = num_agents
        self.max_episode_steps = max_episode_steps

        # Create environments
        self.envs: List[MettaGridPufferEnv] = []
        for i in range(num_envs):
            config = eb.make_navigation(num_agents=num_agents)
            config.game.max_steps = max_episode_steps
            # Add rewards for collecting hearts
            config.game.agent.rewards.inventory["heart"] = 1
            config.game.agent.rewards.inventory_max["heart"] = 100
            # IMPORTANT: Place assemblers in the map (they generate hearts)
            # Scale map size and assemblers with agent count
            map_size = max(20, int(num_agents * 2))
            config.game.map_builder.width = map_size
            config.game.map_builder.height = map_size
            config.game.map_builder.objects = {"assembler": num_agents * 2}  # 2 assemblers per agent

            sim = Simulator()
            env = MettaGridPufferEnv(sim, config, seed=seed + i)
            self.envs.append(env)

        # Get env info from first env
        self.obs_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.obs_shape = self.obs_space.shape[1:]  # Per-agent obs shape
        self.num_actions = self.action_space.nvec[0]  # Actions per agent

        # Track episode state
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((num_envs, num_agents), dtype=np.float32)
        self._needs_reset = np.ones(num_envs, dtype=bool)

    @property
    def total_agents(self) -> int:
        return self.num_envs * self.num_agents_per_env

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset all environments.

        Returns:
            obs: (num_envs, num_agents, *obs_shape)
        """
        all_obs = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed + i if seed else None)
            all_obs.append(obs)
            self.episode_steps[i] = 0
            self.episode_returns[i] = 0.0
            self._needs_reset[i] = False

        return np.stack(all_obs)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments.

        Args:
            actions: (num_envs, num_agents)

        Returns:
            obs: (num_envs, num_agents, *obs_shape)
            rewards: (num_envs, num_agents)
            dones: (num_envs,) - True if any agent done
            truncs: (num_envs,) - True if episode truncated
            infos: dict with episode info
        """
        all_obs = []
        all_rewards = []
        all_dones = []
        all_truncs = []

        episode_infos = []

        for i, env in enumerate(self.envs):
            if self._needs_reset[i]:
                obs, info = env.reset()
                rewards = np.zeros(self.num_agents_per_env, dtype=np.float32)
                dones = np.zeros(self.num_agents_per_env, dtype=bool)
                truncs = np.zeros(self.num_agents_per_env, dtype=bool)
                self.episode_steps[i] = 0
                self.episode_returns[i] = 0.0
                self._needs_reset[i] = False
            else:
                obs, rewards, dones, truncs, info = env.step(actions[i])

            self.episode_steps[i] += 1
            self.episode_returns[i] += rewards

            # Check for episode end (any agent done or max steps)
            env_done = dones.any()
            env_trunc = truncs.any() or (self.episode_steps[i] >= self.max_episode_steps)

            if env_done or env_trunc:
                episode_infos.append({
                    'episode_return': self.episode_returns[i].sum(),
                    'episode_length': self.episode_steps[i],
                    'per_agent_returns': self.episode_returns[i].copy(),
                })
                self._needs_reset[i] = True

            all_obs.append(obs)
            all_rewards.append(rewards)
            all_dones.append(env_done)
            all_truncs.append(env_trunc)

        obs = np.stack(all_obs)
        rewards = np.stack(all_rewards)
        dones = np.array(all_dones)
        truncs = np.array(all_truncs)

        infos = {'episodes': episode_infos}
        return obs, rewards, dones, truncs, infos

    def close(self):
        for env in self.envs:
            env.close()


class NavigationPolicy(nn.Module):
    """Policy network for Navigation environment with tokenized observations."""

    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape  # (seq_len, token_dim) e.g., (200, 3)
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Embedding for tokenized observations
        # obs_shape is (seq_len, token_dim) where token_dim typically has (type, x, y) info
        self.seq_len, self.token_dim = obs_shape

        # Simple MLP to process flattened tokens
        obs_flat_dim = self.seq_len * self.token_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch_size, seq_len, token_dim)

        Returns:
            logits: (batch_size, num_actions)
            embeddings: (batch_size, hidden_dim)
        """
        obs = obs.float() / 255.0  # Normalize uint8 observations
        embeddings = self.encoder(obs)
        logits = self.policy_head(embeddings)
        return logits, embeddings

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.0
        embeddings = self.encoder(obs)
        return self.value_head(embeddings)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        logits, embeddings = self.forward(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.value_head(embeddings), embeddings


class ContrastiveLoss:
    """InfoNCE contrastive loss with geometric temporal sampling."""

    def __init__(
        self,
        temperature: float = 0.19,
        discount: float = 0.977,
        contrastive_coef: float = 0.001,
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
            embeddings: (num_steps, batch_size, embedding_dim)
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

        num_samples_per_traj = min(8, num_steps // 4)

        for traj_idx in range(batch_size):
            for _ in range(num_samples_per_traj):
                max_anchor = int(num_steps * 0.75)
                anchor_step = int(torch.randint(0, max(1, max_anchor), (1,)).item())
                max_future = num_steps - anchor_step - 1

                if max_future < 1:
                    continue

                delta = int(np.random.geometric(prob))
                if delta > max_future:
                    delta = max_future
                if delta < 1:
                    delta = 1

                positive_step = anchor_step + delta

                batch_indices.append(traj_idx)
                anchor_steps.append(anchor_step)
                positive_steps.append(positive_step)
                sampled_deltas.append(delta)

        if len(batch_indices) == 0:
            return torch.tensor(0.0, device=self.device), {
                "contrastive_loss": 0.0,
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "num_pairs": 0,
            }

        batch_indices = torch.tensor(batch_indices, device=self.device)
        anchor_steps = torch.tensor(anchor_steps, device=self.device)
        positive_steps = torch.tensor(positive_steps, device=self.device)

        anchors = embeddings[batch_indices, anchor_steps]
        positives = embeddings[batch_indices, positive_steps]

        # Normalize
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)

        # Compute similarities
        positive_sim = (anchors * positives).sum(dim=-1) / self.temperature
        all_sims = torch.mm(anchors, positives.t()) / self.temperature

        # InfoNCE loss
        labels = torch.arange(len(anchors), device=self.device)
        loss = F.cross_entropy(all_sims, labels)

        metrics = {
            "contrastive_loss": loss.item(),
            "positive_sim_mean": positive_sim.mean().item() * self.temperature,
            "negative_sim_mean": (all_sims.sum() - positive_sim.sum()).item() / max(1, all_sims.numel() - len(anchors)) * self.temperature,
            "num_pairs": len(anchors),
            "delta_mean": np.mean(sampled_deltas) if sampled_deltas else 0,
        }

        return loss * self.contrastive_coef, metrics


def train(
    method: str = "baseline",
    num_envs: int = 16,
    num_agents: int = 12,
    total_timesteps: int = 10_000_000,
    num_steps: int = 128,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    hidden_dim: int = 256,
    # Contrastive hyperparameters
    temperature: float = 0.19,
    contrastive_coef: float = 0.001,
    embedding_dim: int = 64,
    discount: float = 0.977,
    seed: int = 0,
    video_interval: int = 50,
    wandb_project: str = "metta",
):
    """Train Navigation with PPO and optional contrastive learning."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = VectorizedNavigation(
        num_envs=num_envs,
        num_agents=num_agents,
        seed=seed,
        max_episode_steps=500,
    )

    obs_shape = env.obs_shape
    num_actions = env.num_actions
    batch_size = env.total_agents

    print(f"Obs shape: {obs_shape}, Num actions: {num_actions}")
    print(f"Batch size: {batch_size} ({num_envs} envs x {num_agents} agents)")

    # Create policy
    policy = NavigationPolicy(obs_shape, num_actions, hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Create contrastive loss if needed
    contrastive_loss_fn = None
    if method == "contrastive":
        contrastive_loss_fn = ContrastiveLoss(
            temperature=temperature,
            discount=discount,
            contrastive_coef=contrastive_coef,
            embedding_dim=embedding_dim,
            device=device,
        )

    # Initialize wandb
    run_name = f"navigation_{method}.seed{seed}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "method": method,
            "num_envs": num_envs,
            "num_agents": num_agents,
            "total_timesteps": total_timesteps,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "temperature": temperature,
            "contrastive_coef": contrastive_coef,
            "embedding_dim": embedding_dim,
            "discount": discount,
            "seed": seed,
        },
        reinit="finish_previous",
    )

    # Storage
    obs_storage = torch.zeros((num_steps, batch_size) + obs_shape, dtype=torch.uint8, device=device)
    actions_storage = torch.zeros((num_steps, batch_size), dtype=torch.long, device=device)
    logprobs_storage = torch.zeros((num_steps, batch_size), device=device)
    rewards_storage = torch.zeros((num_steps, batch_size), device=device)
    dones_storage = torch.zeros((num_steps, batch_size), device=device)
    values_storage = torch.zeros((num_steps, batch_size), device=device)
    embeddings_storage = torch.zeros((num_steps, batch_size, hidden_dim), device=device)

    # Training loop
    global_step = 0
    start_time = time.time()
    num_updates = total_timesteps // (num_steps * batch_size)

    obs = env.reset(seed=seed)
    obs = torch.tensor(obs, device=device).reshape(batch_size, *obs_shape)
    done = torch.zeros(batch_size, device=device)

    episode_returns = []
    episode_lengths = []

    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(num_steps):
            global_step += batch_size

            with torch.no_grad():
                action, logprob, _, value, embeddings = policy.get_action_and_value(obs)

            obs_storage[step] = obs
            actions_storage[step] = action
            logprobs_storage[step] = logprob
            values_storage[step] = value.squeeze(-1)
            embeddings_storage[step] = embeddings
            dones_storage[step] = done

            # Step environment
            obs_np, rewards_np, dones_np, truncs_np, infos = env.step(
                action.cpu().numpy().reshape(num_envs, num_agents)
            )

            obs = torch.tensor(obs_np, device=device).reshape(batch_size, *obs_shape)
            rewards = torch.tensor(rewards_np, device=device).reshape(batch_size)
            done = torch.tensor(dones_np | truncs_np, device=device).repeat_interleave(num_agents).float()

            rewards_storage[step] = rewards

            # Log episode info
            for ep_info in infos.get('episodes', []):
                episode_returns.append(ep_info['episode_return'])
                episode_lengths.append(ep_info['episode_length'])

        # Compute advantages
        with torch.no_grad():
            next_value = policy.get_value(obs).squeeze(-1)
            advantages = torch.zeros_like(rewards_storage)
            lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]

                delta = rewards_storage[t] + gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values_storage

        # Flatten batch
        b_obs = obs_storage.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)

        # Optimize
        batch_inds = np.arange(num_steps * batch_size)
        minibatch_size = (num_steps * batch_size) // num_minibatches

        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)

            for start in range(0, num_steps * batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.squeeze(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Contrastive loss (computed on full rollout)
        contrastive_metrics = {}
        if contrastive_loss_fn is not None:
            c_loss, contrastive_metrics = contrastive_loss_fn.compute_loss(
                embeddings_storage, dones_storage
            )
            if c_loss.item() > 0:
                optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        sps = int(global_step / (time.time() - start_time))

        if len(episode_returns) > 0:
            mean_return = np.mean(episode_returns[-100:])
            mean_length = np.mean(episode_lengths[-100:])
        else:
            mean_return = 0
            mean_length = 0

        log_dict = {
            "metric/agent_step": global_step,
            "metric/epoch": update,
            "overview/episode_return": mean_return,
            "overview/episode_length": mean_length,
            "overview/sps": sps,
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
        }

        if contrastive_metrics:
            log_dict.update({
                "losses/contrastive_loss": contrastive_metrics.get("contrastive_loss", 0),
                "losses/positive_sim_mean": contrastive_metrics.get("positive_sim_mean", 0),
                "losses/negative_sim_mean": contrastive_metrics.get("negative_sim_mean", 0),
                "losses/num_pairs": contrastive_metrics.get("num_pairs", 0),
            })

        wandb.log(log_dict)

        if update % 10 == 0:
            print(f"Update {update}/{num_updates} | Step {global_step:,} | "
                  f"Return: {mean_return:.2f} | Length: {mean_length:.1f} | SPS: {sps}")

    env.close()
    wandb.finish()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "contrastive", "gccrl"])
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_agents", type=int, default=12)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.19)
    parser.add_argument("--contrastive_coef", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--discount", type=float, default=0.977)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video_interval", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default="metta")

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
