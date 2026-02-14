"""
Paper experiments: SMAC (GPU 0)
Computes geometric metrics (effective rank, expansion ratio, value rank, SVD spectrum)
for Tables 1, 4, 6 of the paper.
"""
import argparse
import json
import time
import os
from collections import defaultdict
from typing import Dict, Tuple, Optional

if "SC2PATH" not in os.environ:
    default_sc2 = os.path.expanduser("~/StarCraftII")
    if os.path.isdir(default_sc2):
        os.environ["SC2PATH"] = default_sc2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from smac.env import StarCraft2Env


def compute_effective_rank(Z: torch.Tensor) -> float:
    """Compute effective rank via SVD entropy."""
    if Z.shape[0] < 2:
        return 1.0
    try:
        S = torch.linalg.svdvals(Z.float())
        S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-12)).sum()
        return float(torch.exp(entropy))
    except Exception:
        return 1.0


def compute_svd_ratio(Z: torch.Tensor) -> float:
    """Compute sigma_1 / sigma_10 ratio."""
    try:
        S = torch.linalg.svdvals(Z.float())
        if len(S) >= 10:
            return float(S[0] / (S[9] + 1e-10))
        return float(S[0] / (S[-1] + 1e-10))
    except Exception:
        return 1.0


def compute_expansion_ratio(embeddings: torch.Tensor) -> float:
    """Compute expansion ratio: var(late embeddings) / var(early embeddings)."""
    T = embeddings.shape[0]
    if T < 5:
        return 1.0
    early_cutoff = max(1, int(T * 0.2))
    late_cutoff = max(early_cutoff + 1, int(T * 0.8))
    early = embeddings[:early_cutoff].reshape(-1, embeddings.shape[-1])
    late = embeddings[late_cutoff:].reshape(-1, embeddings.shape[-1])
    if early.shape[0] < 2 or late.shape[0] < 2:
        return 1.0
    early_var = early.var(dim=0).sum().item()
    late_var = late.var(dim=0).sum().item()
    return late_var / (early_var + 1e-10)


class VectorizedSMAC:
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
        return logits, value, embedding, features

    def get_action_and_value(self, obs, avail_actions, action=None):
        logits, value, embedding, critic_hidden = self.forward(obs)
        logits = logits.masked_fill(avail_actions == 0, -1e10)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, embedding, critic_hidden


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


def train_and_measure(config, seed=0):
    """Train and collect all geometric metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"[{config['map_name']}] Using device: {device}")

    run_name = f"paper_smac_{config['map_name']}_seed{seed}"
    wandb.init(project="representation-collapse", name=run_name, config={**config, "seed": seed}, reinit=True)

    env = VectorizedSMAC(config["map_name"], config["num_envs"], seed=seed)
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    agents_per_step = config["num_envs"] * n_agents

    policy = ActorCritic(obs_dim, n_actions, config["hidden_dim"], config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    obs_buf = torch.zeros((config["num_steps"], agents_per_step, obs_dim), device=device)
    avail_buf = torch.zeros((config["num_steps"], agents_per_step, n_actions), device=device)
    actions_buf = torch.zeros((config["num_steps"], agents_per_step), dtype=torch.long, device=device)
    log_probs_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    rewards_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    dones_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    values_buf = torch.zeros((config["num_steps"], agents_per_step), device=device)
    embeddings_buf = torch.zeros((config["num_steps"], agents_per_step, config["embedding_dim"]), device=device)
    critic_hidden_buf = torch.zeros((config["num_steps"], agents_per_step, config["hidden_dim"]), device=device)

    obs, avail_actions = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32, device=device)

    total_timesteps = 0
    num_updates = 0
    recent_returns = []
    recent_win_rates = []
    episode_returns_accum = defaultdict(float)

    # Metric tracking
    eff_rank_history = []
    reward_history = []
    expansion_ratios_success = []
    expansion_ratios_failure = []
    svd_ratio_history = []
    value_rank_history = []

    # Episode embedding tracking
    episode_embeddings = defaultdict(list)  # env_idx -> list of embeddings per step

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        for step in range(config["num_steps"]):
            obs_flat = obs.reshape(agents_per_step, obs_dim)
            avail_flat = avail_actions.reshape(agents_per_step, n_actions)
            with torch.no_grad():
                actions, log_probs, _, values, embeddings, critic_hidden = policy.get_action_and_value(obs_flat, avail_flat)

            obs_buf[step] = obs_flat
            avail_buf[step] = avail_flat
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs
            values_buf[step] = values
            embeddings_buf[step] = embeddings
            critic_hidden_buf[step] = critic_hidden

            actions_env = actions.reshape(config["num_envs"], n_agents).cpu().numpy()
            next_obs, next_avail, rewards, dones, infos = env.step(actions_env)

            rewards_expanded = np.repeat(rewards[:, np.newaxis], n_agents, axis=1)
            dones_expanded = np.repeat(dones[:, np.newaxis], n_agents, axis=1)
            rewards_flat = torch.tensor(rewards_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)
            dones_flat = torch.tensor(dones_expanded.reshape(agents_per_step), dtype=torch.float32, device=device)
            rewards_buf[step] = rewards_flat
            dones_buf[step] = dones_flat

            # Track per-episode embeddings for expansion ratio
            for i in range(config["num_envs"]):
                agent_embs = embeddings[i * n_agents:(i + 1) * n_agents].mean(dim=0)
                episode_embeddings[i].append(agent_embs.detach().cpu())

            for i in range(config["num_envs"]):
                episode_returns_accum[i] += rewards[i]
                if dones[i]:
                    ep_return = episode_returns_accum[i]
                    won = infos[i].get("battle_won", False)
                    recent_returns.append(ep_return)
                    recent_win_rates.append(1.0 if won else 0.0)

                    # Compute expansion ratio for this episode
                    if len(episode_embeddings[i]) >= 5:
                        ep_embs = torch.stack(episode_embeddings[i])
                        exp_ratio = compute_expansion_ratio(ep_embs)
                        if won:
                            expansion_ratios_success.append(exp_ratio)
                        else:
                            expansion_ratios_failure.append(exp_ratio)

                    episode_embeddings[i] = []
                    episode_returns_accum[i] = 0

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            avail_actions = torch.tensor(next_avail, dtype=torch.float32, device=device)
            total_timesteps += agents_per_step

        # Compute advantages and do PPO update
        obs_flat = obs.reshape(agents_per_step, obs_dim)
        avail_flat = avail_actions.reshape(agents_per_step, n_actions)
        with torch.no_grad():
            _, _, _, next_value, _, _ = policy.get_action_and_value(obs_flat, avail_flat)

        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value, config["gamma"], config["gae_lambda"])

        batch_size = config["num_steps"] * agents_per_step
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_avail = avail_buf.reshape(batch_size, n_actions)
        b_actions = actions_buf.reshape(batch_size)
        b_log_probs = log_probs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for epoch in range(config["update_epochs"]):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_indices = indices[start:end]
                _, new_log_probs, entropy, new_values, _, _ = policy.get_action_and_value(
                    b_obs[mb_indices], b_avail[mb_indices], b_actions[mb_indices])
                log_ratio = new_log_probs - b_log_probs[mb_indices]
                ratio = torch.exp(log_ratio)
                pg_loss1 = -b_advantages[mb_indices] * ratio
                pg_loss2 = -b_advantages[mb_indices] * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_values - b_returns[mb_indices]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()

        num_updates += 1

        # Compute geometric metrics every 10 updates
        if num_updates % 10 == 0:
            with torch.no_grad():
                all_embs = embeddings_buf.reshape(-1, config["embedding_dim"])
                sample_idx = torch.randint(0, all_embs.shape[0], (min(1024, all_embs.shape[0]),))
                sample_embs = all_embs[sample_idx]

                eff_rank = compute_effective_rank(sample_embs)
                svd_ratio = compute_svd_ratio(sample_embs)

                all_critic = critic_hidden_buf.reshape(-1, config["hidden_dim"])
                sample_critic = all_critic[sample_idx]
                val_rank = compute_effective_rank(sample_critic)

            mean_return = np.mean(recent_returns[-100:]) if recent_returns else 0.0
            win_rate = np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0

            eff_rank_history.append((total_timesteps, eff_rank))
            reward_history.append((total_timesteps, mean_return))
            svd_ratio_history.append((total_timesteps, svd_ratio))
            value_rank_history.append((total_timesteps, val_rank))

            wandb.log({
                "metric/agent_step": total_timesteps,
                "overview/reward": mean_return,
                "overview/win_rate": win_rate,
                "geometric/effective_rank": eff_rank,
                "geometric/svd_ratio": svd_ratio,
                "geometric/value_rank": val_rank,
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
            }, step=total_timesteps)

            if num_updates % 50 == 0:
                elapsed = time.time() - start_time
                sps = total_timesteps / elapsed
                print(f"[{config['map_name']}] Step {total_timesteps:,} | Ret: {mean_return:.2f} | "
                      f"WR: {win_rate:.1%} | EffRank: {eff_rank:.1f} | ValRank: {val_rank:.1f} | "
                      f"SVD: {svd_ratio:.1f} | SPS: {sps:,.0f}")

    env.close()

    # Compute lead time: when does eff_rank drop vs when does reward stall
    lead_time_steps = 0
    lead_time_multiplier = 1.0

    if len(eff_rank_history) > 10 and len(reward_history) > 10:
        # Find effective rank drop point (when it drops below 0.5 * initial)
        initial_rank = np.mean([r for _, r in eff_rank_history[:5]])
        rank_threshold = 0.5 * initial_rank
        rank_drop_step = None
        for ts, rank in eff_rank_history:
            if rank < rank_threshold:
                rank_drop_step = ts
                break

        # Find reward stall point (when moving avg improvement < epsilon)
        reward_vals = [r for _, r in reward_history]
        reward_stall_step = None
        window = 10
        for i in range(window, len(reward_vals)):
            recent_improvement = np.mean(reward_vals[max(0,i-window):i]) - np.mean(reward_vals[max(0,i-2*window):max(0,i-window)])
            if abs(recent_improvement) < 0.01 and i > len(reward_vals) * 0.3:
                reward_stall_step = reward_history[i][0]
                break

        if rank_drop_step and reward_stall_step and rank_drop_step < reward_stall_step:
            lead_time_steps = reward_stall_step - rank_drop_step
            lead_time_multiplier = reward_stall_step / max(rank_drop_step, 1)

    # Compute final metrics
    final_eff_rank = eff_rank_history[-1][1] if eff_rank_history else 0
    final_value_rank = value_rank_history[-1][1] if value_rank_history else 0

    mean_exp_ratio_success = np.mean(expansion_ratios_success) if expansion_ratios_success else 1.0
    mean_exp_ratio_failure = np.mean(expansion_ratios_failure) if expansion_ratios_failure else 1.0

    final_svd_ratio = svd_ratio_history[-1][1] if svd_ratio_history else 1.0

    results = {
        "map_name": config["map_name"],
        "n_agents": n_agents,
        "lead_time_steps": int(lead_time_steps),
        "lead_time_multiplier": round(lead_time_multiplier, 1),
        "final_eff_rank": round(final_eff_rank, 1),
        "final_value_rank": round(final_value_rank, 1),
        "expansion_ratio_success": round(mean_exp_ratio_success, 2),
        "expansion_ratio_failure": round(mean_exp_ratio_failure, 2),
        "expansion_ratio_diff": round(mean_exp_ratio_success - mean_exp_ratio_failure, 2),
        "svd_ratio": round(final_svd_ratio, 1),
        "final_win_rate": round(np.mean(recent_win_rates[-100:]) if recent_win_rates else 0.0, 3),
        "final_reward": round(np.mean(recent_returns[-100:]) if recent_returns else 0.0, 2),
        # Threshold checks for Table 6
        "effrank_below_0.3d": bool(final_eff_rank < 0.3 * config["embedding_dim"]),
        "exp_ratio_below_1": bool(mean_exp_ratio_failure < 1.0),
        "svd_ratio_above_10": bool(final_svd_ratio > 10.0),
    }

    wandb.finish()
    return results


def aggregate_seed_results(seed_results):
    """Aggregate results across seeds: compute mean ± std for numeric fields."""
    if not seed_results:
        return {}
    aggregated = {}
    keys = seed_results[0].keys()
    for key in keys:
        values = [r[key] for r in seed_results]
        if isinstance(values[0], (int, float)) and key not in ("n_agents",):
            arr = np.array(values, dtype=float)
            aggregated[key] = round(float(arr.mean()), 3)
            aggregated[f"{key}_std"] = round(float(arr.std()), 3)
            aggregated[f"{key}_seeds"] = [round(float(v), 3) for v in values]
        elif isinstance(values[0], bool):
            aggregated[key] = bool(np.mean(values) > 0.5)
        else:
            aggregated[key] = values[0]
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    maps = ["3s5z", "5m_vs_6m", "corridor", "3s_vs_5z"]
    all_results = []

    for map_name in maps:
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

        seed_results = []
        for seed in range(args.num_seeds):
            print(f"\n--- Seed {seed}/{args.num_seeds} ---")
            result = train_and_measure(config, seed=seed)
            seed_results.append(result)
            print(f"Seed {seed} result: {json.dumps(result, indent=2)}")

        aggregated = aggregate_seed_results(seed_results)
        all_results.append(aggregated)
        print(f"\nAggregated ({map_name}): {json.dumps(aggregated, indent=2)}")

    # Save results
    output_path = "/home/ubuntu/metta/v3_experiments/results_smac.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
