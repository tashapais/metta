"""
Paper Experiment 4: Individual Rewards vs Shared Team Rewards
=============================================================
The central causal claim of the paper is that shared rewards cause gradient
averaging which collapses agent representations. This experiment isolates that
causal arrow by holding everything else constant and varying ONLY whether
agents receive shared (averaged) or individual (per-agent) rewards.

Conditions:
  - individual        : each agent receives its own reward from the environment
  - shared            : all agents receive the mean team reward (gradient averaging)
  - individual+contrastive : individual rewards + inter-agent InfoNCE
  - shared+contrastive     : shared rewards + InfoNCE  (same as Exp 3 baseline/contrastive)

At end of training, runs a downstream representation probe:
  Labels each agent as "high_performer" (top half of team) or "low_performer"
  (bottom half) based on their cumulative episode return.
  Trains a logistic regression probe from frozen embeddings → label.
  High probe accuracy = representations encode who is contributing/specialising.
  Low probe accuracy  = representations have collapsed.

Usage (one GPU per call, 5 seeds each):
    cd /home/devuser/metta
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3_experiments/paper_exp_reward_type.py \
        --reward_type individual --gpu 0 --num_seeds 5
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python v3_experiments/paper_exp_reward_type.py \
        --reward_type shared --gpu 1 --num_seeds 5
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python v3_experiments/paper_exp_reward_type.py \
        --reward_type individual --contrastive --gpu 2 --num_seeds 5
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python v3_experiments/paper_exp_reward_type.py \
        --reward_type shared --contrastive --gpu 3 --num_seeds 5

wandb project: representation-collapse
Run names: paper_reward_{reward_type}[_contrastive]_18agents_seed{N}
"""
import argparse
import json
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical
import wandb

from mettagrid.builder.envs import make_arena
from mettagrid.simulator.simulator import Simulation

# ---------------------------------------------------------------------------
# Geometric diagnostics (identical to paper_exp_contrastive.py)
# ---------------------------------------------------------------------------

def compute_effective_rank(Z: torch.Tensor) -> float:
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
    try:
        S = torch.linalg.svdvals(Z.float())
        if len(S) >= 10:
            return float(S[0] / (S[9] + 1e-10))
        return float(S[0] / (S[-1] + 1e-10))
    except Exception:
        return 1.0


def compute_action_diversity(logits: torch.Tensor) -> float:
    """Mean pairwise KL divergence between agents' action distributions."""
    if logits.shape[1] < 2:
        return 0.0
    probs     = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    kl_matrix = (
        probs.unsqueeze(2) *
        (log_probs.unsqueeze(2) - log_probs.unsqueeze(1))
    ).sum(-1)
    n = kl_matrix.shape[1]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=logits.device)
    return float(kl_matrix[:, off_diag].mean())


def compute_expansion_ratio(embeddings: torch.Tensor) -> float:
    T = embeddings.shape[0]
    if T < 5:
        return 1.0
    early_cutoff = max(1, int(T * 0.2))
    late_cutoff  = max(early_cutoff + 1, int(T * 0.8))
    early = embeddings[:early_cutoff].reshape(-1, embeddings.shape[-1])
    late  = embeddings[late_cutoff:].reshape(-1, embeddings.shape[-1])
    if early.shape[0] < 2 or late.shape[0] < 2:
        return 1.0
    return float(late.var(dim=0).sum() / (early.var(dim=0).sum() + 1e-10))


# ---------------------------------------------------------------------------
# Inter-agent InfoNCE contrastive loss (identical to paper_exp_contrastive.py)
# ---------------------------------------------------------------------------

def inter_agent_infonce_loss(
    emb_buf: torch.Tensor,   # [T, E, n_agents, d]
    temperature: float = 0.19,
    gamma_c: float = 0.977,
    n_samples: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    T, E, n, d = emb_buf.shape
    if n < 2 or T < 2:
        return torch.tensor(0.0, device=device)

    t_a  = torch.randint(0, T - 1,  (n_samples,), device=device)
    e_i  = torch.randint(0, E,       (n_samples,), device=device)
    ag   = torch.randint(0, n,       (n_samples,), device=device)

    prob   = max(1.0 - gamma_c, 1e-8)
    deltas = torch.distributions.Geometric(torch.tensor(prob)).sample(
        (n_samples,)).long().clamp(1, T - 1).to(device)
    t_p = (t_a + deltas).clamp(max=T - 1)

    z_anc = emb_buf[t_a, e_i, ag]
    z_pos = emb_buf[t_p, e_i, ag]
    all_at_t = emb_buf[t_a, e_i]

    mask = torch.ones(n_samples, n, dtype=torch.bool, device=device)
    mask[torch.arange(n_samples, device=device), ag] = False
    z_neg = all_at_t[mask].view(n_samples, n - 1, d)

    z_anc = F.normalize(z_anc, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)

    sim_pos = (z_anc * z_pos).sum(-1, keepdim=True)
    sim_neg = (z_anc.unsqueeze(1) * z_neg).sum(-1)

    logits = torch.cat([sim_pos, sim_neg], dim=1) / temperature
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Reward-conditioned supervised contrastive loss (SupCon over rank)
# ---------------------------------------------------------------------------

def supcon_rank_loss(
    embs:   torch.Tensor,   # (N, emb_dim) — fresh forward pass, has gradients
    labels: torch.Tensor,   # (N,) int64 — 0 = bottom half, 1 = top half
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Supervised contrastive loss where the class label is agent performance rank.

    For each agent embedding, positives are agents in the SAME rank-class
    (top/bottom half by episode return) from OTHER episodes in the buffer.
    Negatives are agents in the opposite rank-class.

    This explicitly trains the encoder to cluster high-contributors together
    and push them away from low-contributors — the representation structure
    needed for role specialisation and high probe accuracy.

    Crucially, the rank labels come from the ENVIRONMENT's per-agent returns
    (hearts collected), not from the shared reward injected into PPO. So even
    when PPO receives shared (averaged) rewards, this loss still has a
    differentiated signal. This is the key difference from inter-agent InfoNCE.

    Reference: Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020.
    """
    N = embs.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=device)
    if labels.min() == labels.max():   # only one class — no signal
        return torch.tensor(0.0, device=device)

    Z = F.normalize(embs, dim=-1)           # (N, d)
    sim = Z @ Z.T / temperature             # (N, N)

    eye      = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye   # (N, N)

    # log-sum-exp over all non-self pairs → denominator
    sim_no_self = sim.masked_fill(eye, -1e9)
    log_denom   = torch.logsumexp(sim_no_self, dim=1)               # (N,)

    n_pos = pos_mask.sum(dim=1).float()
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    # SupCon: for each anchor i, average over its positives p:
    #   loss_i = -( mean_p(sim_ip) - log_denom_i )
    mean_pos_sim = (sim * pos_mask.float()).sum(dim=1) / (n_pos + 1e-8)
    loss = log_denom - mean_pos_sim                                  # (N,)
    return loss[valid].mean()


# ---------------------------------------------------------------------------
# Vectorised MettaGrid environment wrapper
# ---------------------------------------------------------------------------

class MettaGridVecEnv:
    def __init__(self, num_agents: int, num_envs: int, seed: int,
                 max_steps: int = 1024):
        self.num_envs   = num_envs
        self.num_agents = num_agents
        self.max_steps  = max_steps
        self.sims = []
        for i in range(num_envs):
            cfg = make_arena(num_agents=num_agents)
            sim = Simulation(cfg, seed=seed + i)
            self.sims.append(sim)

        obs = self.sims[0]._c_sim.observations()
        self.obs_dim  = int(np.prod(obs.shape[1:]))
        self.n_actions = int(self.sims[0]._c_sim.masks().shape[-1])

        self.episode_steps   = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((num_envs, num_agents), dtype=np.float32)

    def _reset_sim(self, i: int, seed_offset: int = 0):
        self.sims[i].close()
        cfg = make_arena(num_agents=self.num_agents)
        self.sims[i] = Simulation(cfg, seed=seed_offset + i)
        self.episode_steps[i]   = 0
        self.episode_returns[i] = 0.0

    def get_obs(self) -> np.ndarray:
        obs_list = []
        for sim in self.sims:
            o = sim._c_sim.observations().copy()
            obs_list.append(o.reshape(self.num_agents, -1))
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray, global_step: int = 0,
             reward_type: str = "individual"):
        """
        reward_type:
          "individual" — each agent gets its own environment reward
          "shared"     — all agents get the mean of all team rewards
                         (simulates pure gradient-averaging from shared credit)
        """
        all_rewards   = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        all_dones     = np.zeros(self.num_envs, dtype=np.float32)
        all_ep_returns = []

        for i, sim in enumerate(self.sims):
            sim._c_sim.actions()[:] = actions[i].astype(np.int32)
            sim.step()
            self.episode_steps[i] += 1

            rews = sim._c_sim.rewards().copy()  # per-agent

            if reward_type == "shared":
                # All agents get the team-mean reward → identical gradient signal
                team_mean = rews.mean()
                rews = np.full_like(rews, team_mean)

            all_rewards[i] = rews
            self.episode_returns[i] += rews

            terminals   = sim._c_sim.terminals()
            truncations = sim._c_sim.truncations()
            done = bool(
                np.any(terminals) or np.any(truncations) or
                self.episode_steps[i] >= self.max_steps
            )
            all_dones[i] = float(done)

            if done:
                all_ep_returns.append((
                    float(self.episode_returns[i].sum()),
                    self.episode_returns[i].copy()   # per-agent returns for probe
                ))
                self._reset_sim(i, seed_offset=global_step)

        return self.get_obs(), all_rewards, all_dones, all_ep_returns

    def close(self):
        for sim in self.sims:
            try:
                sim.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MAPPO policy network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_dim: int = 128, emb_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.emb_head = nn.Linear(hidden_dim, emb_dim)
        self.actor    = nn.Linear(hidden_dim, n_actions)
        self.critic   = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        return self.actor(h), self.critic(h).squeeze(-1), self.emb_head(h)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        logits, value, emb = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, emb, logits


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T    = rewards.shape[0]
    advs = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        nt    = 1.0 - dones[t]
        nv    = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nt - values[t]
        advs[t] = last = delta + gamma * lam * nt * last
    return advs, advs + values


# ---------------------------------------------------------------------------
# Main training function (single seed)
# ---------------------------------------------------------------------------

def train_one_seed(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    reward_type   = cfg["reward_type"]
    use_cl        = cfg["contrastive"]
    use_reward_cl = cfg["reward_cl"]
    n_agents      = cfg["num_agents"]
    device        = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # Condition label for wandb / file names
    cond = reward_type
    if use_cl:        cond += "_contrastive"
    if use_reward_cl: cond += "_rewardCL"
    run_name = f"paper_reward_{cond}_{n_agents}agents_seed{seed}"

    wandb.init(
        project="representation-collapse",
        entity="tashapais",
        name=run_name,
        config={**cfg, "seed": seed},
        reinit=True,
    )
    print(f"\n[{run_name}] device={device}")

    env = MettaGridVecEnv(
        num_agents=n_agents,
        num_envs=cfg["num_envs"],
        seed=seed * 100,
        max_steps=cfg["max_steps"],
    )
    obs_dim, n_actions = env.obs_dim, env.n_actions
    print(f"[{run_name}] obs_dim={obs_dim}  n_actions={n_actions}")

    policy    = ActorCritic(obs_dim, n_actions, cfg["hidden_dim"], cfg["emb_dim"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"], eps=1e-5)

    T          = cfg["num_steps"]
    E          = cfg["num_envs"]
    total_steps = cfg["total_timesteps"]
    mb_size    = cfg["minibatch_size"]
    epochs     = cfg["update_epochs"]

    obs_buf   = np.zeros((T, E, n_agents, obs_dim), dtype=np.float32)
    act_buf   = np.zeros((T, E, n_agents), dtype=np.int64)
    rew_buf   = np.zeros((T, E, n_agents), dtype=np.float32)
    done_buf  = np.zeros((T, E), dtype=np.float32)
    logp_buf  = np.zeros((T, E, n_agents), dtype=np.float32)
    val_buf   = np.zeros((T, E, n_agents), dtype=np.float32)
    emb_buf   = np.zeros((T, E, n_agents, cfg["emb_dim"]), dtype=np.float32)
    logit_buf = np.zeros((T, E, n_agents, n_actions), dtype=np.float32)

    obs = env.get_obs()
    global_step  = 0
    update_count = 0
    recent_returns: list[float] = []

    eff_rank_hist, exp_ratio_hist, svd_ratio_hist, act_div_hist = [], [], [], []
    t_start = time.time()

    # Probe data collected inline (avoids post-training eval loop that can hang)
    probe_embeddings: list[np.ndarray] = []
    probe_labels:     list[int]        = []

    # Episode buffer for reward-conditioned SupCon.
    # Stores (obs: np.ndarray (n_agents, obs_dim), rank_labels: np.ndarray (n_agents,))
    # for completed episodes. We store obs (not embs) so we can re-forward with
    # gradients during the PPO update phase.
    episode_buffer: deque = deque(maxlen=32)

    while global_step < total_steps:
        # ---- Rollout ----
        for step in range(T):
            obs_buf[step] = obs
            obs_t = torch.tensor(
                obs.reshape(E * n_agents, obs_dim), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                acts, logps, _, vals, embs, lgts = policy.get_action_and_value(obs_t)

            acts_np = acts.cpu().numpy().reshape(E, n_agents)
            obs, rews, dones, ep_rets = env.step(
                acts_np, global_step=global_step, reward_type=reward_type
            )

            act_buf[step]   = acts_np
            rew_buf[step]   = rews
            done_buf[step]  = dones
            logp_buf[step]  = logps.cpu().numpy().reshape(E, n_agents)
            val_buf[step]   = vals.cpu().numpy().reshape(E, n_agents)
            emb_buf[step]   = embs.cpu().numpy().reshape(E, n_agents, -1)
            logit_buf[step] = lgts.cpu().numpy().reshape(E, n_agents, n_actions)
            recent_returns.extend([total_ret for total_ret, _ in ep_rets])

            # Collect probe + episode-buffer data from completed episodes
            embs_np = embs.cpu().numpy().reshape(E, n_agents, -1)
            ep_idx  = 0
            for env_i in range(E):
                if dones[env_i] and ep_idx < len(ep_rets):
                    _, per_agent_rets = ep_rets[ep_idx]
                    ep_idx += 1
                    if len(per_agent_rets) >= 2:
                        # Use argsort so top/bottom halves are always balanced,
                        # even when all returns are equal (e.g. shared reward collapse).
                        half        = n_agents // 2
                        rank_order  = np.argsort(per_agent_rets)
                        rank_labels = np.zeros(n_agents, dtype=np.int64)
                        rank_labels[rank_order[half:]] = 1
                        for a in range(n_agents):
                            probe_embeddings.append(embs_np[env_i, a])
                            probe_labels.append(int(rank_labels[a]))
                        # Store obs at terminal step for reward-conditioned SupCon.
                        if use_reward_cl:
                            episode_buffer.append((
                                obs_buf[step, env_i].copy(),   # (n_agents, obs_dim)
                                rank_labels.copy(),            # (n_agents,) 0/1, always balanced
                            ))

            global_step += E * n_agents

        # ---- Bootstrap value ----
        obs_t = torch.tensor(
            obs.reshape(E * n_agents, obs_dim), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            _, _, _, next_vals, _, _ = policy.get_action_and_value(obs_t)
        next_vals_np = next_vals.cpu().numpy().reshape(E, n_agents)

        # ---- GAE per agent ----
        adv_buf = np.zeros_like(rew_buf)
        ret_buf = np.zeros_like(rew_buf)
        for a in range(n_agents):
            r  = torch.tensor(rew_buf[:, :, a],  dtype=torch.float32, device=device)
            v  = torch.tensor(val_buf[:, :, a],  dtype=torch.float32, device=device)
            d  = torch.tensor(done_buf,           dtype=torch.float32, device=device)
            nv = torch.tensor(next_vals_np[:, a], dtype=torch.float32, device=device)
            adv, ret = compute_gae(r, v, d, nv, cfg["gamma"], cfg["gae_lambda"])
            adv_buf[:, :, a] = adv.cpu().numpy()
            ret_buf[:, :, a] = ret.cpu().numpy()

        # ---- Flatten ----
        n_total  = T * E * n_agents
        flat_obs  = obs_buf.reshape(n_total, obs_dim)
        flat_acts = act_buf.reshape(n_total)
        flat_logp = logp_buf.reshape(n_total)
        flat_adv  = adv_buf.reshape(n_total)
        flat_ret  = ret_buf.reshape(n_total)

        # ---- PPO updates ----
        idx = np.arange(n_total)
        total_pg, total_vl, total_ent, total_cl, total_rcl = 0.0, 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n_total, mb_size):
                mb      = idx[start:start + mb_size]
                mb_obs  = torch.tensor(flat_obs[mb],  dtype=torch.float32, device=device)
                mb_acts = torch.tensor(flat_acts[mb], dtype=torch.long,    device=device)
                mb_logp = torch.tensor(flat_logp[mb], dtype=torch.float32, device=device)
                mb_adv  = torch.tensor(flat_adv[mb],  dtype=torch.float32, device=device)
                mb_ret  = torch.tensor(flat_ret[mb],  dtype=torch.float32, device=device)

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_logp, ent, new_val, _, _ = policy.get_action_and_value(mb_obs, mb_acts)
                ratio = torch.exp(new_logp - mb_logp)
                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(0.8, 1.2)
                ).mean()
                vl   = F.mse_loss(new_val, mb_ret)
                loss = pg + 0.5 * vl - 0.01 * ent.mean()

                cl_val = torch.tensor(0.0, device=device)
                if use_cl:
                    emb_t  = torch.tensor(emb_buf, dtype=torch.float32, device=device)
                    cl_val = inter_agent_infonce_loss(
                        emb_t,
                        temperature=cfg["temperature"],
                        gamma_c=cfg["gamma_c"],
                        n_samples=64,
                        device=device,
                    )
                    loss = loss + cfg["contrastive_coef"] * cl_val

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                total_pg  += pg.item()
                total_vl  += vl.item()
                total_ent += ent.mean().item()
                total_cl  += cl_val.item()
                n_updates  += 1

        # ---- Reward-conditioned SupCon (separate pass, once per PPO update) ----
        # Re-forward the stored terminal observations with gradients so the
        # encoder learns to cluster agents by contribution rank.
        # The rank labels come from environment per-agent returns, NOT from the
        # shared reward used in PPO — so this signal stays differentiated even
        # when reward_type="shared".
        if use_reward_cl and len(episode_buffer) >= 4:
            ep_obs  = np.concatenate([o for o, _ in episode_buffer], axis=0)   # (N*n, obs_dim)
            ep_labs = np.concatenate([l for _, l in episode_buffer], axis=0)   # (N*n,)
            ep_obs_t  = torch.tensor(ep_obs,  dtype=torch.float32, device=device)
            ep_labs_t = torch.tensor(ep_labs, dtype=torch.long,    device=device)
            h         = policy.encoder(ep_obs_t)
            ep_embs   = policy.emb_head(h)
            rcl_val   = supcon_rank_loss(ep_embs, ep_labs_t,
                                         cfg["reward_cl_temperature"], device)
            optimizer.zero_grad()
            (cfg["reward_cl_coef"] * rcl_val).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total_rcl += rcl_val.item()

        # ---- Geometric metrics ----
        all_emb = torch.tensor(
            emb_buf.reshape(-1, cfg["emb_dim"]), dtype=torch.float32
        )
        eff_rank  = compute_effective_rank(all_emb)
        svd_ratio = compute_svd_ratio(all_emb)
        emb_trajs = emb_buf.transpose(1, 2, 0, 3).reshape(
            E * n_agents, T, cfg["emb_dim"]
        )
        exp_ratio = float(np.mean([
            compute_expansion_ratio(torch.tensor(emb_trajs[b], dtype=torch.float32))
            for b in range(min(64, emb_trajs.shape[0]))
        ]))
        lgts_flat   = torch.tensor(
            logit_buf.reshape(T * E, n_agents, n_actions), dtype=torch.float32
        )
        act_diversity = compute_action_diversity(lgts_flat)

        eff_rank_hist.append(eff_rank)
        exp_ratio_hist.append(exp_ratio)
        svd_ratio_hist.append(svd_ratio)
        act_div_hist.append(act_diversity)

        mean_ret = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
        sps      = global_step / max(1.0, time.time() - t_start)
        n_upd    = max(1, n_updates)

        log = {
            "global_step":              global_step,
            "charts/effective_rank":    eff_rank,
            "charts/effrank_per_agent": eff_rank / n_agents,
            "charts/expansion_ratio":   exp_ratio,
            "charts/svd_ratio":         svd_ratio,
            "charts/action_diversity":  act_diversity,
            "charts/mean_return":       mean_ret,
            "charts/sps":               sps,
            "losses/policy":            total_pg / n_upd,
            "losses/value":             total_vl / n_upd,
            "losses/entropy":           total_ent / n_upd,
        }
        if use_cl:
            log["losses/contrastive"] = total_cl / n_upd
        if use_reward_cl:
            log["losses/reward_cl"] = total_rcl
        wandb.log(log, step=global_step)

        update_count += 1
        if update_count % 5 == 0:
            print(
                f"[{run_name}] step={global_step:,}  "
                f"eff_rank={eff_rank:.2f}/{n_agents}({eff_rank/n_agents:.3f})  "
                f"act_div={act_diversity:.4f}  "
                f"exp_ratio={exp_ratio:.3f}  ret={mean_ret:.4f}  sps={sps:.0f}"
            )

    # ---- Final summary metrics ----
    final_eff_rank  = float(np.mean(eff_rank_hist[-10:])) if eff_rank_hist else 0.0
    final_exp_ratio = float(np.mean(exp_ratio_hist[-10:])) if exp_ratio_hist else 0.0
    final_svd_ratio = float(np.mean(svd_ratio_hist[-10:])) if svd_ratio_hist else 0.0
    final_act_div   = float(np.mean(act_div_hist[-10:]))  if act_div_hist   else 0.0
    final_ret  = float(np.mean(recent_returns[-200:])) if recent_returns else 0.0
    win_rate   = float(np.mean([1.0 if r > 0 else 0.0
                                for r in recent_returns[-200:]])) if recent_returns else 0.0

    # ---- Downstream role probe (from inline-collected training episodes) ----
    probe_results = {
        "probe_accuracy": 0.0, "probe_accuracy_std": 0.0,
        "probe_chance": 0.5, "probe_lift": 0.0, "n_samples": 0,
    }
    if len(probe_embeddings) >= 50:
        X   = np.array(probe_embeddings, dtype=np.float32)
        y   = np.array(probe_labels,     dtype=np.int64)
        if len(np.unique(y)) < 2:
            # All labels same class — representations fully collapsed, probe is chance
            probe_results = {
                "probe_accuracy": 0.5, "probe_accuracy_std": 0.0,
                "probe_chance": 0.5,   "probe_lift": 0.0,
                "n_samples": len(y),
            }
        else:
            X_sc   = StandardScaler().fit_transform(X)
            clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=0)
            cv     = min(5, int(len(y) / max(1, np.bincount(y).min())))
            cv     = max(2, cv)
            scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="accuracy")
            chance = float(max(y.mean(), 1 - y.mean()))
            probe_results = {
                "probe_accuracy":     float(scores.mean()),
                "probe_accuracy_std": float(scores.std()),
                "probe_chance":       chance,
                "probe_lift":         float(scores.mean() - chance),
                "n_samples":          len(y),
            }
    print(f"[{run_name}] Probe accuracy: {probe_results['probe_accuracy']:.3f} "
          f"(chance={probe_results['probe_chance']:.3f}, "
          f"lift={probe_results['probe_lift']:+.3f}, "
          f"n={probe_results['n_samples']})")

    wandb.log({
        "final/effective_rank":    final_eff_rank,
        "final/effrank_per_agent": final_eff_rank / n_agents,
        "final/expansion_ratio":   final_exp_ratio,
        "final/svd_ratio":         final_svd_ratio,
        "final/action_diversity":  final_act_div,
        "final/mean_return":       final_ret,
        "final/win_rate":          win_rate,
        "final/probe_accuracy":    probe_results["probe_accuracy"],
        "final/probe_chance":      probe_results["probe_chance"],
        "final/probe_lift":        probe_results["probe_lift"],
    })
    wandb.finish()
    env.close()

    result = {
        "reward_type":        reward_type,
        "contrastive":        use_cl,
        "reward_cl":          use_reward_cl,
        "condition":          cond,
        "seed":               seed,
        "num_agents":         n_agents,
        "run_name":           run_name,
        "final_eff_rank":     round(final_eff_rank, 3),
        "effrank_per_agent":  round(final_eff_rank / n_agents, 3),
        "final_exp_ratio":    round(final_exp_ratio, 3),
        "final_svd_ratio":    round(final_svd_ratio, 3),
        "final_act_div":      round(final_act_div, 4),
        "final_return":       round(final_ret, 4),
        "win_rate":           round(win_rate, 3),
        "probe_accuracy":     round(probe_results["probe_accuracy"], 4),
        "probe_chance":       round(probe_results["probe_chance"], 4),
        "probe_lift":         round(probe_results["probe_lift"], 4),
        "probe_accuracy_std": round(probe_results["probe_accuracy_std"], 4),
    }
    print(f"\n[{run_name}] DONE: {json.dumps(result, indent=2)}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", choices=["individual", "shared"], required=True,
                        help="'individual': per-agent rewards; 'shared': mean team reward")
    parser.add_argument("--contrastive", action="store_true",
                        help="Add inter-agent InfoNCE contrastive loss")
    parser.add_argument("--reward_cl",  action="store_true",
                        help="Add reward-conditioned SupCon loss (rank labels from env returns)")
    parser.add_argument("--gpu",              type=int, default=0)
    parser.add_argument("--num_agents",       type=int, default=18)
    parser.add_argument("--num_seeds",        type=int, default=5)
    parser.add_argument("--total_timesteps",  type=int, default=5_000_000)
    parser.add_argument("--probe_episodes",   type=int, default=200)
    args = parser.parse_args()

    cfg = dict(
        reward_type          = args.reward_type,
        contrastive          = args.contrastive,
        reward_cl            = args.reward_cl,
        num_agents           = args.num_agents,
        gpu                  = args.gpu,
        num_envs             = 4,
        num_steps            = 128,
        max_steps            = 1024,
        total_timesteps      = args.total_timesteps,
        minibatch_size       = 512,
        update_epochs        = 8,
        lr                   = 3e-4,
        gamma                = 0.99,
        gae_lambda           = 0.95,
        hidden_dim           = 128,
        emb_dim              = 64,
        contrastive_coef     = 6.8e-4,
        temperature          = 0.19,
        gamma_c              = 0.977,
        reward_cl_coef       = 0.05,
        reward_cl_temperature= 0.07,
        probe_episodes       = args.probe_episodes,
    )

    cond = args.reward_type
    if args.contrastive: cond += "_contrastive"
    if args.reward_cl:   cond += "_rewardCL"
    all_results = []

    for seed in range(args.num_seeds):
        print(f"\n{'='*70}")
        print(f"  reward_type={args.reward_type}  contrastive={args.contrastive}  "
              f"agents={args.num_agents}  seed={seed}/{args.num_seeds-1}")
        print(f"{'='*70}")
        result = train_one_seed(cfg, seed)
        all_results.append(result)

    out_path = (
        f"/home/ubuntu/metta/v3_experiments/"
        f"results_reward_{cond}_{args.num_agents}agents.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} seeds → {out_path}")

    def stats(key):
        vals = [r[key] for r in all_results]
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*70}")
    print(f"SUMMARY  condition={cond}  agents={args.num_agents}  n={args.num_seeds} seeds")
    er_mean, er_std = stats("effrank_per_agent")
    ad_mean, ad_std = stats("final_act_div")
    wr_mean, wr_std = stats("win_rate")
    pa_mean, pa_std = stats("probe_accuracy")
    pc_mean, _      = stats("probe_chance")
    print(f"  EffRank/n:     {er_mean:.3f} ± {er_std:.3f}")
    print(f"  Act. Div:      {ad_mean:.4f} ± {ad_std:.4f}")
    print(f"  Win Rate:      {wr_mean:.3f} ± {wr_std:.3f}")
    print(f"  Probe Acc:     {pa_mean:.3f} ± {pa_std:.3f}  (chance={pc_mean:.3f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
