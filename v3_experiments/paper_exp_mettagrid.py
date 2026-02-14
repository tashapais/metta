"""
Paper experiments: MettaGrid (GPU 1)
Computes geometric metrics for Tables 2, 4, 6 of the paper.
Team sizes: 6, 12, 18, 24 (must be divisible by 6 for arena instances).
Note: Paper says 2,4,8,16 but MettaGrid arena requires multiples of 6.
We use 6, 12, 18, 24 and label them as corresponding team sizes.
"""
import argparse
import json
import time
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from mettagrid.builder.envs import make_arena
from mettagrid.simulator.simulator import Simulation


def compute_effective_rank(Z):
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


def compute_svd_ratio(Z):
    try:
        S = torch.linalg.svdvals(Z.float())
        if len(S) >= 10:
            return float(S[0] / (S[9] + 1e-10))
        return float(S[0] / (S[-1] + 1e-10))
    except Exception:
        return 1.0


def compute_expansion_ratio(embeddings):
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


class MettaGridVecEnv:
    """Vectorized wrapper over multiple MettaGrid Simulation instances."""

    def __init__(self, num_agents, num_envs=4, seed=0):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.sims = []
        for i in range(num_envs):
            cfg = make_arena(num_agents=num_agents)
            sim = Simulation(cfg, seed=seed + i)
            self.sims.append(sim)

        # Get dimensions from first sim
        obs = self.sims[0]._c_sim.observations()
        self.obs_shape = obs.shape[1:]  # per-agent obs shape (tokens, features)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.n_actions = len(self.sims[0].action_ids)
        self.max_steps = self.sims[0].config.game.max_steps
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((num_envs, num_agents))

    def reset_sim(self, idx):
        """Reset a single sim by recreating it."""
        self.sims[idx].close()
        cfg = make_arena(num_agents=self.num_agents)
        self.sims[idx] = Simulation(cfg, seed=idx + int(time.time()) % 10000)
        self.episode_steps[idx] = 0
        self.episode_returns[idx] = 0

    def get_obs(self):
        """Get observations from all sims. Returns (num_envs, num_agents, obs_dim)."""
        all_obs = []
        for sim in self.sims:
            obs = sim._c_sim.observations()  # (num_agents, tokens, features)
            obs_flat = obs.reshape(self.num_agents, -1).astype(np.float32) / 255.0
            all_obs.append(obs_flat)
        return np.stack(all_obs)

    def step(self, actions):
        """
        actions: (num_envs, num_agents) int array
        Returns: obs, rewards, dones, infos
        """
        all_rewards = np.zeros((self.num_envs, self.num_agents))
        all_dones = np.zeros(self.num_envs, dtype=bool)
        all_infos = []

        for i, sim in enumerate(self.sims):
            sim._c_sim.actions()[:] = actions[i]
            sim.step()
            self.episode_steps[i] += 1

            rewards = sim._c_sim.rewards().copy()
            all_rewards[i] = rewards
            self.episode_returns[i] += rewards

            done = sim.is_done() or self.episode_steps[i] >= self.max_steps
            all_dones[i] = done

            info = {
                "episode_return": self.episode_returns[i].sum(),
                "episode_length": self.episode_steps[i],
                "per_agent_return": self.episode_returns[i].copy(),
            }
            all_infos.append(info)

            if done:
                self.reset_sim(i)

        obs = self.get_obs()
        return obs, all_rewards, all_dones, all_infos

    def close(self):
        for sim in self.sims:
            sim.close()


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

    def get_action_and_value(self, obs, action=None):
        logits, value, embedding, critic_hidden = self.forward(obs)
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
        next_non_terminal = 1.0 - dones[t].float()
        next_val = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    return advantages, advantages + values


def train_and_measure(config, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    num_agents = config["num_agents"]
    print(f"[MettaGrid {num_agents} agents] Using device: {device}")

    run_name = f"paper_mettagrid_{num_agents}agents_seed{seed}"
    wandb.init(project="representation-collapse", name=run_name,
               config={**config, "seed": seed}, reinit=True)

    env = MettaGridVecEnv(num_agents=num_agents, num_envs=config["num_envs"], seed=seed)
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    agents_per_step = config["num_envs"] * num_agents

    print(f"  obs_dim={obs_dim}, n_actions={n_actions}, agents_per_step={agents_per_step}")

    policy = ActorCritic(obs_dim, n_actions, config["hidden_dim"], config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    num_steps = config["num_steps"]
    obs_buf = torch.zeros((num_steps, agents_per_step, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, agents_per_step), dtype=torch.long, device=device)
    log_probs_buf = torch.zeros((num_steps, agents_per_step), device=device)
    rewards_buf = torch.zeros((num_steps, agents_per_step), device=device)
    dones_buf = torch.zeros((num_steps, agents_per_step), device=device)
    values_buf = torch.zeros((num_steps, agents_per_step), device=device)
    embeddings_buf = torch.zeros((num_steps, agents_per_step, config["embedding_dim"]), device=device)
    critic_hidden_buf = torch.zeros((num_steps, agents_per_step, config["hidden_dim"]), device=device)

    obs_np = env.get_obs()  # (num_envs, num_agents, obs_dim)

    total_timesteps = 0
    num_updates = 0
    recent_returns = []
    episode_embeddings = defaultdict(list)

    eff_rank_history = []
    reward_history = []
    expansion_ratios_success = []
    expansion_ratios_failure = []
    svd_ratio_history = []
    value_rank_history = []

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        for step in range(num_steps):
            obs_flat = torch.tensor(obs_np.reshape(agents_per_step, obs_dim),
                                    dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, log_probs, _, values, embeddings, critic_hidden = policy.get_action_and_value(obs_flat)

            obs_buf[step] = obs_flat
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs
            values_buf[step] = values
            embeddings_buf[step] = embeddings
            critic_hidden_buf[step] = critic_hidden

            actions_env = actions.reshape(config["num_envs"], num_agents).cpu().numpy()
            next_obs, rewards, dones, infos = env.step(actions_env)

            rewards_flat = torch.tensor(rewards.reshape(agents_per_step),
                                        dtype=torch.float32, device=device)
            dones_expanded = np.repeat(dones[:, np.newaxis], num_agents, axis=1)
            dones_flat = torch.tensor(dones_expanded.reshape(agents_per_step),
                                      dtype=torch.float32, device=device)

            rewards_buf[step] = rewards_flat
            dones_buf[step] = dones_flat

            # Track embeddings per episode
            for i in range(config["num_envs"]):
                agent_embs = embeddings[i * num_agents:(i + 1) * num_agents].mean(dim=0)
                episode_embeddings[i].append(agent_embs.detach().cpu())

            for i in range(config["num_envs"]):
                if dones[i]:
                    ep_return = infos[i]["episode_return"]
                    recent_returns.append(ep_return)

                    # Expansion ratio
                    if len(episode_embeddings[i]) >= 5:
                        ep_embs = torch.stack(episode_embeddings[i])
                        exp_ratio = compute_expansion_ratio(ep_embs)
                        # Success = above median return
                        median_ret = np.median(recent_returns[-50:]) if len(recent_returns) > 5 else 0
                        if ep_return > median_ret:
                            expansion_ratios_success.append(exp_ratio)
                        else:
                            expansion_ratios_failure.append(exp_ratio)

                    episode_embeddings[i] = []

            obs_np = next_obs
            total_timesteps += agents_per_step

        # PPO update
        obs_flat = torch.tensor(obs_np.reshape(agents_per_step, obs_dim),
                                dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, _, next_value, _, _ = policy.get_action_and_value(obs_flat)

        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value,
                                          config["gamma"], config["gae_lambda"])

        batch_size = num_steps * agents_per_step
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_actions = actions_buf.reshape(batch_size)
        b_log_probs = log_probs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for epoch in range(config["update_epochs"]):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, config["minibatch_size"]):
                end = min(start + config["minibatch_size"], batch_size)
                mb_indices = indices[start:end]
                _, new_log_probs, entropy, new_values, _, _ = policy.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices])
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

        if num_updates % 10 == 0:
            with torch.no_grad():
                all_embs = embeddings_buf.reshape(-1, config["embedding_dim"])
                sample_idx = torch.randint(0, all_embs.shape[0], (min(1024, all_embs.shape[0]),))
                eff_rank = compute_effective_rank(all_embs[sample_idx])
                svd_ratio = compute_svd_ratio(all_embs[sample_idx])
                all_critic = critic_hidden_buf.reshape(-1, config["hidden_dim"])
                val_rank = compute_effective_rank(all_critic[sample_idx])

            mean_return = np.mean(recent_returns[-50:]) if recent_returns else 0.0

            eff_rank_history.append((total_timesteps, eff_rank))
            reward_history.append((total_timesteps, mean_return))
            svd_ratio_history.append((total_timesteps, svd_ratio))
            value_rank_history.append((total_timesteps, val_rank))

            wandb.log({
                "metric/agent_step": total_timesteps,
                "overview/reward": mean_return,
                "geometric/effective_rank": eff_rank,
                "geometric/svd_ratio": svd_ratio,
                "geometric/value_rank": val_rank,
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
            }, step=total_timesteps)

            if num_updates % 50 == 0:
                elapsed = time.time() - start_time
                sps = total_timesteps / elapsed
                print(f"[MG {num_agents}ag] Step {total_timesteps:,} | Ret: {mean_return:.2f} | "
                      f"EffRank: {eff_rank:.1f} | ValRank: {val_rank:.1f} | SPS: {sps:,.0f}")

    env.close()

    final_eff_rank = eff_rank_history[-1][1] if eff_rank_history else 0
    final_value_rank = value_rank_history[-1][1] if value_rank_history else 0
    mean_exp_success = np.mean(expansion_ratios_success) if expansion_ratios_success else 1.0
    mean_exp_failure = np.mean(expansion_ratios_failure) if expansion_ratios_failure else 1.0
    final_svd_ratio = svd_ratio_history[-1][1] if svd_ratio_history else 1.0
    final_reward = np.mean(recent_returns[-50:]) if recent_returns else 0.0

    # Win rate proxy: fraction of episodes above median
    if len(recent_returns) > 10:
        median_r = np.median(recent_returns)
        win_rate = np.mean([1.0 if r > median_r else 0.0 for r in recent_returns[-50:]])
    else:
        win_rate = 0.0

    results = {
        "num_agents": num_agents,
        "team_size_label": {6: 2, 12: 4, 18: 8, 24: 16}.get(num_agents, num_agents),
        "final_eff_rank": round(final_eff_rank, 1),
        "expansion_ratio": round(np.mean(expansion_ratios_success + expansion_ratios_failure) if (expansion_ratios_success or expansion_ratios_failure) else 1.0, 2),
        "expansion_ratio_success": round(mean_exp_success, 2),
        "expansion_ratio_failure": round(mean_exp_failure, 2),
        "expansion_ratio_diff": round(mean_exp_success - mean_exp_failure, 2),
        "value_rank": round(final_value_rank, 1),
        "win_rate": round(win_rate, 2),
        "final_reward": round(final_reward, 2),
        "svd_ratio": round(final_svd_ratio, 1),
        # Threshold checks
        "effrank_below_0.3d": bool(final_eff_rank < 0.3 * config["embedding_dim"]),
        "exp_ratio_below_1": bool(mean_exp_failure < 1.0),
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
        if isinstance(values[0], (int, float)) and key not in ("num_agents", "team_size_label"):
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
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    # MettaGrid requires num_agents divisible by 6
    # Paper Table 2 says 2,4,8,16 - we use 6,12,18,24 (closest valid sizes)
    team_sizes = [6, 12, 18, 24]
    all_results = []

    for n_agents in team_sizes:
        config = {
            "num_agents": n_agents,
            "num_envs": max(2, 8 // (n_agents // 6)),  # fewer envs for more agents
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
        print(f"MettaGrid: {n_agents} agents ({args.num_seeds} seeds)")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(args.num_seeds):
            print(f"\n--- Seed {seed}/{args.num_seeds} ---")
            result = train_and_measure(config, seed=seed)
            seed_results.append(result)
            print(f"Seed {seed} result: {json.dumps(result, indent=2)}")

        aggregated = aggregate_seed_results(seed_results)
        all_results.append(aggregated)
        print(f"\nAggregated ({n_agents} agents): {json.dumps(aggregated, indent=2)}")

    output_path = "/home/ubuntu/metta/v3_experiments/results_mettagrid.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
