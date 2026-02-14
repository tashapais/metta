"""
Paper experiments: Tribal Village
Computes geometric metrics for Table 2 of the paper.
Team counts: 2, 4, 8 teams (× 125 agents = 256, 506, 1006 total agents including goblins).
"""
import argparse
import ctypes
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb


# ─── Geometric metric functions (same as MettagGrid) ───────────────────────

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


# ─── Tribal Village Vectorized Environment ──────────────────────────────────

TV_DIR = "/home/ubuntu/tribal-village"
OBS_CHANNELS = 96
OBS_W = 11
OBS_H = 11
ACTION_SPACE_SIZE = 308  # 11 verbs × 28 arguments


class TribalVillageVecEnv:
    """Vectorized wrapper over multiple Tribal Village environments.

    Each env instance loads the appropriate team-count .so library.
    """

    def __init__(self, num_teams, num_envs=2, max_steps=1000, seed=0):
        self.num_envs = num_envs
        self.num_teams = num_teams
        self.max_steps = max_steps

        # Load the team-specific library
        lib_path = os.path.join(TV_DIR, f"libtribal_village_teams{num_teams}.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()

        self.num_agents = self.lib.tribal_village_get_num_agents()
        self.obs_layers = self.lib.tribal_village_get_obs_layers()
        self.obs_width = self.lib.tribal_village_get_obs_width()
        self.obs_height = self.lib.tribal_village_get_obs_height()
        self.obs_dim = self.obs_layers * self.obs_width * self.obs_height
        self.n_actions = ACTION_SPACE_SIZE
        self.agents_per_team = 125

        print(f"  TV env: {num_teams} teams, {self.num_agents} agents, "
              f"obs={self.obs_layers}x{self.obs_width}x{self.obs_height}")

        # Create environment instances
        self.envs = []
        self.obs_bufs = []
        self.reward_bufs = []
        self.terminal_bufs = []
        self.truncation_bufs = []
        self.action_bufs = []

        for i in range(num_envs):
            env_ptr = self.lib.tribal_village_create()
            if not env_ptr:
                raise RuntimeError(f"Failed to create env {i}")

            # Allocate buffers for this env
            obs = np.zeros((self.num_agents, self.obs_layers, self.obs_width, self.obs_height),
                           dtype=np.uint8)
            rewards = np.zeros(self.num_agents, dtype=np.float32)
            terminals = np.zeros(self.num_agents, dtype=np.float32)
            truncations = np.zeros(self.num_agents, dtype=np.float32)
            actions = np.zeros(self.num_agents, dtype=np.uint8)

            self.envs.append(env_ptr)
            self.obs_bufs.append(obs)
            self.reward_bufs.append(rewards)
            self.terminal_bufs.append(terminals)
            self.truncation_bufs.append(truncations)
            self.action_bufs.append(actions)

        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((num_envs, self.num_agents))

        # Reset all envs
        for i in range(num_envs):
            self._reset_env(i)

    def _setup_ctypes(self):
        """Setup ctypes signatures for the Nim library."""
        self.lib.tribal_village_create.restype = ctypes.c_void_p
        self.lib.tribal_village_get_num_agents.restype = ctypes.c_int32
        self.lib.tribal_village_get_obs_layers.restype = ctypes.c_int32
        self.lib.tribal_village_get_obs_width.restype = ctypes.c_int32
        self.lib.tribal_village_get_obs_height.restype = ctypes.c_int32

        self.lib.tribal_village_reset_and_get_obs.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.tribal_village_reset_and_get_obs.restype = ctypes.c_int32

        self.lib.tribal_village_step_with_pointers.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.tribal_village_step_with_pointers.restype = ctypes.c_int32

        self.lib.tribal_village_destroy.argtypes = [ctypes.c_void_p]

    def _reset_env(self, idx):
        """Reset a single env."""
        obs = self.obs_bufs[idx]
        rewards = self.reward_bufs[idx]
        terminals = self.terminal_bufs[idx]
        truncations = self.truncation_bufs[idx]

        self.lib.tribal_village_reset_and_get_obs(
            self.envs[idx],
            obs.ctypes.data_as(ctypes.c_void_p),
            rewards.ctypes.data_as(ctypes.c_void_p),
            terminals.ctypes.data_as(ctypes.c_void_p),
            truncations.ctypes.data_as(ctypes.c_void_p),
        )
        self.episode_steps[idx] = 0
        self.episode_returns[idx] = 0

    def get_obs(self):
        """Get observations from all envs. Returns (num_envs, num_agents, obs_dim)."""
        all_obs = []
        for i in range(self.num_envs):
            obs_flat = self.obs_bufs[i].reshape(self.num_agents, -1).astype(np.float32) / 255.0
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

        for i in range(self.num_envs):
            # Set actions
            self.action_bufs[i][:] = actions[i].astype(np.uint8)

            # Step
            self.lib.tribal_village_step_with_pointers(
                self.envs[i],
                self.action_bufs[i].ctypes.data_as(ctypes.c_void_p),
                self.obs_bufs[i].ctypes.data_as(ctypes.c_void_p),
                self.reward_bufs[i].ctypes.data_as(ctypes.c_void_p),
                self.terminal_bufs[i].ctypes.data_as(ctypes.c_void_p),
                self.truncation_bufs[i].ctypes.data_as(ctypes.c_void_p),
            )
            self.episode_steps[i] += 1

            rewards = self.reward_bufs[i].copy()
            all_rewards[i] = rewards
            self.episode_returns[i] += rewards

            # Check done: all terminated or max steps
            all_terminated = np.all(self.terminal_bufs[i] > 0.5)
            done = all_terminated or self.episode_steps[i] >= self.max_steps
            all_dones[i] = done

            # Compute team-level returns for win rate
            team_returns = []
            for t in range(self.num_teams):
                start = t * self.agents_per_team
                end = start + self.agents_per_team
                team_returns.append(float(self.episode_returns[i, start:end].sum()))

            info = {
                "episode_return": float(self.episode_returns[i].sum()),
                "episode_length": int(self.episode_steps[i]),
                "team_returns": team_returns,
            }
            all_infos.append(info)

            if done:
                self._reset_env(i)

        obs = self.get_obs()
        return obs, all_rewards, all_dones, all_infos

    def close(self):
        for env_ptr in self.envs:
            if env_ptr:
                self.lib.tribal_village_destroy(env_ptr)
        self.envs = []


# ─── Actor-Critic with CNN encoder ──────────────────────────────────────────

class ActorCritic(nn.Module):
    """CNN-based actor-critic for spatial observations.

    Input: (batch, obs_channels * obs_w * obs_h) flattened
    Architecture: 2 conv layers on reshaped spatial obs + embedding head
    """

    def __init__(self, obs_channels, obs_w, obs_h, n_actions,
                 hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.obs_channels = obs_channels
        self.obs_w = obs_w
        self.obs_h = obs_h

        # CNN encoder for spatial observations
        self.conv1 = nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        conv_out_dim = 64 * obs_w * obs_h

        self.fc_encoder = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
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

    def forward(self, obs_flat):
        # Reshape to spatial: (batch, channels, w, h)
        batch_size = obs_flat.shape[0]
        x = obs_flat.view(batch_size, self.obs_channels, self.obs_w, self.obs_h)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(batch_size, -1)
        features = self.fc_encoder(x)
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


# ─── GAE computation ────────────────────────────────────────────────────────

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


# ─── Training loop ──────────────────────────────────────────────────────────

def train_and_measure(config, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    num_teams = config["num_teams"]
    print(f"[Tribal Village {num_teams} teams] Using device: {device}")

    run_name = f"paper_tribal_{num_teams}teams_seed{seed}"
    wandb.init(project="representation-collapse", name=run_name,
               config={**config, "seed": seed}, reinit=True)

    env = TribalVillageVecEnv(
        num_teams=num_teams,
        num_envs=config["num_envs"],
        max_steps=config["max_steps"],
        seed=seed,
    )
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    agents_per_step = config["num_envs"] * num_agents

    print(f"  obs_dim={obs_dim}, n_actions={n_actions}, agents_per_step={agents_per_step}")

    policy = ActorCritic(
        OBS_CHANNELS, OBS_W, OBS_H, n_actions,
        config["hidden_dim"], config["embedding_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"], eps=1e-5)

    num_steps = config["num_steps"]
    obs_buf = torch.zeros((num_steps, agents_per_step, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, agents_per_step), dtype=torch.long, device=device)
    log_probs_buf = torch.zeros((num_steps, agents_per_step), device=device)
    rewards_buf = torch.zeros((num_steps, agents_per_step), device=device)
    dones_buf = torch.zeros((num_steps, agents_per_step), device=device)
    values_buf = torch.zeros((num_steps, agents_per_step), device=device)
    embeddings_buf = torch.zeros((num_steps, agents_per_step, config["embedding_dim"]),
                                 device=device)
    critic_hidden_buf = torch.zeros((num_steps, agents_per_step, config["hidden_dim"]),
                                    device=device)

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
    team_win_counts = defaultdict(int)
    total_episodes = 0

    start_time = time.time()

    while total_timesteps < config["total_timesteps"]:
        for step in range(num_steps):
            obs_flat = torch.tensor(obs_np.reshape(agents_per_step, obs_dim),
                                    dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, log_probs, _, values, embeddings, critic_hidden = \
                    policy.get_action_and_value(obs_flat)

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
                    total_episodes += 1

                    # Track team wins
                    team_returns = infos[i]["team_returns"]
                    best_team = int(np.argmax(team_returns))
                    team_win_counts[best_team] += 1

                    # Expansion ratio
                    if len(episode_embeddings[i]) >= 5:
                        ep_embs = torch.stack(episode_embeddings[i])
                        exp_ratio = compute_expansion_ratio(ep_embs)
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
                sample_idx = torch.randint(0, all_embs.shape[0],
                                           (min(1024, all_embs.shape[0]),))
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
                print(f"[TV {num_teams}t] Step {total_timesteps:,} | Ret: {mean_return:.2f} | "
                      f"EffRank: {eff_rank:.1f} | ValRank: {val_rank:.1f} | SPS: {sps:,.0f}")

    env.close()

    final_eff_rank = eff_rank_history[-1][1] if eff_rank_history else 0
    final_value_rank = value_rank_history[-1][1] if value_rank_history else 0
    mean_exp_success = np.mean(expansion_ratios_success) if expansion_ratios_success else 1.0
    mean_exp_failure = np.mean(expansion_ratios_failure) if expansion_ratios_failure else 1.0
    final_svd_ratio = svd_ratio_history[-1][1] if svd_ratio_history else 1.0
    final_reward = np.mean(recent_returns[-50:]) if recent_returns else 0.0

    # Win rate: fraction of episodes where the best-performing team wins
    if total_episodes > 10:
        # Balanced win rate: each team should win ~1/num_teams of the time
        # We measure how often any single team dominates
        max_wins = max(team_win_counts.values()) if team_win_counts else 0
        win_rate = max_wins / total_episodes
    else:
        win_rate = 0.0

    results = {
        "num_teams": num_teams,
        "total_agents": num_teams * 125 + 6,
        "final_eff_rank": round(final_eff_rank, 1),
        "expansion_ratio": round(
            np.mean(expansion_ratios_success + expansion_ratios_failure)
            if (expansion_ratios_success or expansion_ratios_failure) else 1.0, 2
        ),
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
    """Aggregate results across seeds: compute mean +/- std for numeric fields."""
    if not seed_results:
        return {}
    aggregated = {}
    keys = seed_results[0].keys()
    for key in keys:
        values = [r[key] for r in seed_results]
        if isinstance(values[0], (int, float)) and key not in ("num_teams", "total_agents"):
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

    team_counts = [2, 4, 8]
    all_results = []

    for num_teams in team_counts:
        total_agents = num_teams * 125 + 6
        config = {
            "num_teams": num_teams,
            "num_envs": max(1, 4 // num_teams),  # fewer envs for more agents
            "num_steps": 64,  # shorter rollouts due to large agent count
            "max_steps": 1000,  # env episode length
            "total_timesteps": args.total_timesteps,
            "minibatch_size": 512,
            "update_epochs": 4,
            "lr": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "hidden_dim": 256,
            "embedding_dim": 64,
            "gpu": args.gpu,
        }

        print(f"\n{'='*60}")
        print(f"Tribal Village: {num_teams} teams, {total_agents} agents ({args.num_seeds} seeds)")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(args.num_seeds):
            print(f"\n--- Seed {seed}/{args.num_seeds} ---")
            result = train_and_measure(config, seed=seed)
            seed_results.append(result)
            print(f"Seed {seed} result: {json.dumps(result, indent=2)}")

        aggregated = aggregate_seed_results(seed_results)
        all_results.append(aggregated)
        print(f"\nAggregated ({num_teams} teams): {json.dumps(aggregated, indent=2)}")

    output_path = "/home/ubuntu/metta/v3_experiments/results_tribal.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
