"""
SMACv2 experiments with geometric diagnostic metrics.

Paper experiments addressed:
  Exp 1: Representation collapse precedes failure (4 maps × 5 seeds)
  Exp 4: Embedding trajectory visualization (SMACv2 part)
  Exp 5: SVD spectrum analysis (SMACv2 part)
  Exp 6: Cross-environment consistency (SMACv2 part)

Usage:
    # Single run
    CUDA_VISIBLE_DEVICES=0 python run_smac_geom.py --map 3s5z --seed 0

    # All Exp 1 runs (sequential on one GPU)
    CUDA_VISIBLE_DEVICES=0 python run_smac_geom.py --run_all
"""

import argparse
import time
from typing import Dict, Tuple, Optional
from collections import defaultdict
import os
import json
import sys

# Auto-detect SC2PATH
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
from geometric_metrics import compute_geometric_metrics_torch, effective_rank


class VectorizedSMAC:
    """Wrapper to vectorize multiple SMAC environments."""

    def __init__(self, map_name: str, num_envs: int, seed: int = 0):
        self.map_name = map_name
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            env = StarCraft2Env(map_name=map_name, seed=seed + i)
            self.envs.append(env)

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
        all_obs, all_avail = [], []
        for i, env in enumerate(self.envs):
            env.reset()
            obs = np.array([env.get_obs_agent(a) for a in range(self.n_agents)])
            avail = np.array([env.get_avail_agent_actions(a) for a in range(self.n_agents)])
            all_obs.append(obs)
            all_avail.append(avail)
            self._needs_reset[i] = False
            self.episode_steps[i] = 0
            self.episode_returns[i] = 0.0
        return np.stack(all_obs), np.stack(all_avail)

    def step(self, actions: np.ndarray):
        all_obs, all_avail = [], []
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

        return np.stack(all_obs), np.stack(all_avail), all_rewards, all_dones, all_infos

    def close(self):
        for env in self.envs:
            env.close()


class ActorCritic(nn.Module):
    """Actor-Critic with exposed critic hidden layer for value rank computation."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        # Critic with exposed hidden layer
        self.critic_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.encoder(obs)
        embedding = self.embedding_head(features)
        logits = self.actor(features)
        critic_h = self.critic_hidden(features)
        value = self.critic_out(critic_h).squeeze(-1)
        return logits, value, embedding, critic_h

    def get_action_and_value(self, obs, avail_actions, action=None):
        logits, value, embedding, critic_h = self.forward(obs)
        logits = logits.masked_fill(avail_actions == 0, -1e10)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, embedding, critic_h


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
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
    """Run a single training run with geometric metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    run_name = f"smac_geom_{config['map_name']}_seed{seed}"
    wandb.init(
        project="repr-collapse-marl",
        entity=config.get("wandb_entity", "tashapais"),
        name=run_name,
        config={**config, "seed": seed},
        tags=["smac", config["map_name"], f"seed{seed}", "geometric-metrics"],
        reinit=True,
    )

    env = VectorizedSMAC(config["map_name"], config["num_envs"], seed=seed)
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    agents_per_step = config["num_envs"] * n_agents

    print(f"Map: {config['map_name']}, Agents: {n_agents}, Obs: {obs_dim}, Actions: {n_actions}")

    policy = ActorCritic(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_dim=config["hidden_dim"], embedding_dim=config["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    # Rollout storage
    obs_buf = torch.zeros((config["num_steps"], agents_per_step, obs_dim), device=device)
    avail_buf = torch.zeros((config["num_steps"], agents_per_step, n_actions), device=device)
    actions_buf = torch.zeros((config["num_steps"], agents_per_step), dtype=torch.long, device=device)
    log_probs_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    rewards_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    dones_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    values_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    embeddings_buf = torch.zeros((config["num_steps"], agents_per_step, config["embedding_dim"]), device=device)
    critic_hidden_buf = torch.zeros((config["num_steps"], agents_per_step, config["hidden_dim"]), device=device)

    # Track per-episode embeddings for expansion ratio (Exp 4)
    episode_embeddings = defaultdict(list)  # env_idx -> list of embeddings

    obs, avail_actions = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32, device=device)

    total_timesteps = 0
    num_updates = 0
    recent_returns, recent_ep_lengths, recent_win_rates = [], [], []
    episode_returns = defaultdict(float)

    # Track when metrics first cross thresholds (Exp 1)
    rank_threshold = 0.3 * config["embedding_dim"]
    rank_drop_timestep = None
    reward_stall_timestep = None
    best_reward = -float("inf")
    reward_window = []

    # Results storage
    results_log = []

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        # Collect rollout
        for step in range(config["num_steps"]):
            obs_flat = obs.reshape(agents_per_step, obs_dim)
            avail_flat = avail_actions.reshape(agents_per_step, n_actions)

            with torch.no_grad():
                actions, log_probs, _, values, embeddings, critic_h = policy.get_action_and_value(obs_flat, avail_flat)

            obs_buf[step] = obs_flat
            avail_buf[step] = avail_flat
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs
            values_buf[step] = values
            embeddings_buf[step] = embeddings
            critic_hidden_buf[step] = critic_h

            actions_env = actions.reshape(config["num_envs"], n_agents).cpu().numpy()
            next_obs, next_avail, rewards, dones, infos = env.step(actions_env)

            rewards_expanded = np.repeat(rewards[:, np.newaxis], n_agents, axis=1)
            dones_expanded = np.repeat(dones[:, np.newaxis], n_agents, axis=1)

            rewards_buf[step] = torch.tensor(rewards_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)
            dones_buf[step] = torch.tensor(dones_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)

            # Track episode stats
            for i in range(config["num_envs"]):
                # Store embeddings for expansion ratio
                agent0_emb = embeddings[i * n_agents].detach().cpu().numpy()
                episode_embeddings[i].append(agent0_emb)

                episode_returns[i] += rewards[i]
                if dones[i]:
                    ep_return = episode_returns[i]
                    ep_length = infos[i].get("episode_length", 0)
                    won = infos[i].get("battle_won", False)
                    recent_returns.append(ep_return)
                    recent_ep_lengths.append(ep_length)
                    recent_win_rates.append(1.0 if won else 0.0)
                    episode_returns[i] = 0

                    # Compute per-episode expansion ratio
                    if len(episode_embeddings[i]) > 5:
                        ep_embs = np.array(episode_embeddings[i])
                        from geometric_metrics import expansion_ratio as er_fn
                        ep_er = er_fn(ep_embs)
                        wandb.log({
                            "episodes/expansion_ratio": ep_er,
                            "episodes/won": 1.0 if won else 0.0,
                            "episodes/return": ep_return,
                        }, step=total_timesteps)
                    episode_embeddings[i] = []

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            avail_actions = torch.tensor(next_avail, dtype=torch.float32, device=device)
            total_timesteps += agents_per_step

        # Compute advantages
        obs_flat = obs.reshape(agents_per_step, obs_dim)
        avail_flat = avail_actions.reshape(agents_per_step, n_actions)
        with torch.no_grad():
            _, _, _, next_value, _, _ = policy.get_action_and_value(obs_flat, avail_flat)

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

                _, new_log_probs, entropy, new_values, _, _ = policy.get_action_and_value(
                    b_obs[mb_indices], b_avail[mb_indices], b_actions[mb_indices]
                )

                log_ratio = new_log_probs - b_log_probs[mb_indices]
                ratio = torch.exp(log_ratio)
                pg_loss1 = -b_advantages[mb_indices] * ratio
                pg_loss2 = -b_advantages[mb_indices] * torch.clamp(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - b_returns[mb_indices]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()

        num_updates += 1

        # Compute geometric metrics every 10 updates
        if num_updates % 10 == 0:
            # Get critic hidden for a batch
            with torch.no_grad():
                sample_obs = obs_buf[:, :min(512, agents_per_step), :].reshape(-1, obs_dim)
                sample_avail = avail_buf[:, :min(512, agents_per_step), :].reshape(-1, n_actions)
                _, _, _, _, _, sample_critic_h = policy.get_action_and_value(sample_obs, sample_avail)

            geom_metrics = compute_geometric_metrics_torch(
                embeddings_buf, values_buf, dones_buf,
                critic_hidden=sample_critic_h,
                num_envs=config["num_envs"], n_agents=n_agents,
            )

            # Track threshold crossings (Exp 1)
            eff_rank = geom_metrics["geom/effective_rank"]
            if rank_drop_timestep is None and eff_rank < rank_threshold:
                rank_drop_timestep = total_timesteps

            mean_return = np.mean(recent_returns[-100:]) if recent_returns else 0.0
            reward_window.append(mean_return)
            if len(reward_window) > 20:
                reward_window.pop(0)
                recent_improvement = abs(reward_window[-1] - reward_window[0])
                if reward_stall_timestep is None and recent_improvement < 0.01 and total_timesteps > config["total_timesteps"] * 0.2:
                    reward_stall_timestep = total_timesteps

            # Standard metrics
            elapsed = time.time() - start_time
            sps = total_timesteps / elapsed
            win_rate = np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0
            mean_ep_length = np.mean(recent_ep_lengths[-100:]) if recent_ep_lengths else 0.0

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
                # Exp 1 threshold tracking
                "exp1/rank_drop_timestep": rank_drop_timestep or 0,
                "exp1/reward_stall_timestep": reward_stall_timestep or 0,
            }
            log_dict.update(geom_metrics)

            wandb.log(log_dict, step=total_timesteps)

            print(
                f"Step {total_timesteps:,} | "
                f"Return: {mean_return:.2f} | "
                f"WinRate: {win_rate:.1%} | "
                f"EffRank: {eff_rank:.1f}/{config['embedding_dim']} | "
                f"ExpRatio: {geom_metrics['geom/expansion_ratio_mean']:.2f} | "
                f"SPS: {sps:,.0f}"
            )

            results_log.append({
                "timestep": total_timesteps,
                "reward": mean_return,
                "win_rate": win_rate,
                "effective_rank": eff_rank,
                "expansion_ratio": geom_metrics["geom/expansion_ratio_mean"],
                "sigma_ratio": geom_metrics["geom/sigma1_sigma10_ratio"],
            })

    env.close()

    # Compute lead time (Exp 1)
    lead_time_steps = 0
    lead_time_factor = 0.0
    if rank_drop_timestep and reward_stall_timestep:
        lead_time_steps = reward_stall_timestep - rank_drop_timestep
        if rank_drop_timestep > 0:
            lead_time_factor = reward_stall_timestep / rank_drop_timestep

    final_results = {
        "map": config["map_name"],
        "seed": seed,
        "n_agents": n_agents,
        "final_return": np.mean(recent_returns[-100:]) if recent_returns else 0.0,
        "final_win_rate": np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0,
        "rank_drop_timestep": rank_drop_timestep,
        "reward_stall_timestep": reward_stall_timestep,
        "lead_time_steps": lead_time_steps,
        "lead_time_factor": lead_time_factor,
        "results_log": results_log,
    }

    # Log summary
    wandb.summary.update({
        "final_return": final_results["final_return"],
        "final_win_rate": final_results["final_win_rate"],
        "rank_drop_timestep": rank_drop_timestep,
        "reward_stall_timestep": reward_stall_timestep,
        "lead_time_steps": lead_time_steps,
        "lead_time_factor": lead_time_factor,
    })

    wandb.finish()
    return final_results


# Experiment 1 maps (from the paper Table 1)
EXP1_MAPS = ["3s5z", "5m_vs_6m", "corridor", "3s_vs_5z"]
NUM_SEEDS = 5


def run_all_exp1(device="cuda:0"):
    """Run all Experiment 1 configurations sequentially."""
    all_results = []

    for map_name in EXP1_MAPS:
        for seed in range(NUM_SEEDS):
            config = {
                "map_name": map_name,
                "device": device,
                "num_envs": 8,
                "total_timesteps": 2_000_000,
                "num_steps": 128,
                "minibatch_size": 512,
                "update_epochs": 10,
                "lr": 5e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 10.0,
                "hidden_dim": 128,
                "embedding_dim": 64,
            }

            print(f"\n{'='*60}")
            print(f"Exp 1: {map_name} seed={seed}")
            print(f"{'='*60}")

            result = train(config, seed=seed)
            all_results.append(result)

            # Save intermediate results
            with open(f"results_smac_exp1_{map_name}.json", "w") as f:
                json.dump([r for r in all_results if r["map"] == map_name], f, indent=2, default=str)

    # Save all results
    with open("results_smac_exp1_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table (Exp 1 Table 1)
    print(f"\n{'='*60}")
    print("EXPERIMENT 1 RESULTS: Lead Time of EffRank vs Reward")
    print(f"{'='*60}")
    print(f"{'Map':<15} {'Agents':<8} {'Lead (steps)':<15} {'Lead (x faster)':<15}")
    print("-" * 55)

    for map_name in EXP1_MAPS:
        map_results = [r for r in all_results if r["map"] == map_name]
        n_agents = map_results[0]["n_agents"]
        lead_steps = np.mean([r["lead_time_steps"] for r in map_results])
        lead_factor = np.mean([r["lead_time_factor"] for r in map_results if r["lead_time_factor"] > 0])
        print(f"{map_name:<15} {n_agents:<8} {lead_steps:<15.0f} {lead_factor:<15.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="3s5z")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_all", action="store_true", help="Run all Exp 1 configurations")
    parser.add_argument("--wandb_entity", type=str, default="tashapais")
    args = parser.parse_args()

    if args.run_all:
        run_all_exp1(device=args.device)
        return

    config = {
        "map_name": args.map,
        "device": args.device,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "num_steps": 128,
        "minibatch_size": 512,
        "update_epochs": 10,
        "lr": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 10.0,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "wandb_entity": args.wandb_entity,
    }

    result = train(config, seed=args.seed)
    print(f"\nFinal return: {result['final_return']:.2f}")
    print(f"Final win rate: {result['final_win_rate']:.1%}")
    print(f"Lead time: {result['lead_time_steps']} steps ({result['lead_time_factor']:.1f}x)")


if __name__ == "__main__":
    main()
