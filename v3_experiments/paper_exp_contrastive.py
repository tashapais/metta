"""
Paper Experiment: Contrastive vs Baseline MAPPO — Section 5.3
Compares MAPPO baseline vs MAPPO+InfoNCE contrastive on MettaGrid Arena.
Team size: 18 agents (the team size where baseline completely fails in Exp 2).

Usage:
    # GPU 0: baseline (5 seeds)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3_experiments/paper_exp_contrastive.py --method baseline --gpu 0

    # GPU 1: contrastive (5 seeds)
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python v3_experiments/paper_exp_contrastive.py --method contrastive --gpu 1

wandb project: representation-collapse
Run names: paper_contrastive_baseline_seed{N}  /  paper_contrastive_contrastive_seed{N}
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from mettagrid.builder.envs import make_arena
from mettagrid.simulator.simulator import Simulation

# ---------------------------------------------------------------------------
# Geometric diagnostics
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
    """Mean pairwise KL divergence between agents' action distributions.

    logits: [B, n_agents, n_actions]
    Returns scalar: average KL(π_i || π_j) over all (i≠j) pairs and batch.
    High value → agents take diverse actions (healthy role specialisation).
    Low value  → agents are behaviourally homogeneous (role collapse).
    """
    if logits.shape[1] < 2:
        return 0.0
    probs     = F.softmax(logits, dim=-1)       # [B, n_agents, n_actions]
    log_probs = F.log_softmax(logits, dim=-1)   # [B, n_agents, n_actions]
    # kl[b, i, j] = Σ_a probs[b,i,a] * (log_probs[b,i,a] - log_probs[b,j,a])
    kl_matrix = (
        probs.unsqueeze(2) *
        (log_probs.unsqueeze(2) - log_probs.unsqueeze(1))
    ).sum(-1)  # [B, n_agents, n_agents]
    n = kl_matrix.shape[1]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=logits.device)
    return float(kl_matrix[:, off_diag].mean())


def compute_expansion_ratio(embeddings: torch.Tensor) -> float:
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


# ---------------------------------------------------------------------------
# InfoNCE contrastive loss (temporal positive pairs)
# ---------------------------------------------------------------------------

def inter_agent_infonce_loss(
    emb_buf: torch.Tensor,   # [T, E, n_agents, d]  — full rollout buffer on device
    temperature: float = 0.19,
    gamma_c: float = 0.977,
    n_samples: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Inter-agent InfoNCE contrastive loss.

    Directly targets gradient homogenisation between agents:
      Anchor   : embedding of agent i at timestep t
      Positive : embedding of agent i at nearby timestep t' (same agent, future)
      Negatives: embeddings of all other agents j≠i at the SAME timestep t

    This teaches each agent's representation to be temporally coherent with
    itself while being geometrically distinct from teammates — directly
    counteracting the shared-reward gradient averaging that collapses
    inter-agent representational diversity.
    """
    T, E, n, d = emb_buf.shape
    if n < 2 or T < 2:
        return torch.tensor(0.0, device=device)

    # Sample anchor (t_a, env, agent) indices
    t_a  = torch.randint(0, T - 1,  (n_samples,), device=device)
    e_i  = torch.randint(0, E,       (n_samples,), device=device)
    ag   = torch.randint(0, n,       (n_samples,), device=device)

    # Positive timestep — geometrically distributed future step
    prob   = max(1.0 - gamma_c, 1e-8)
    deltas = torch.distributions.Geometric(torch.tensor(prob)).sample(
        (n_samples,)).long().clamp(1, T - 1).to(device)
    t_p = (t_a + deltas).clamp(max=T - 1)

    # Anchor and positive embeddings: [n_samples, d]
    z_anc = emb_buf[t_a, e_i, ag]       # anchor
    z_pos = emb_buf[t_p, e_i, ag]       # positive (same agent, future)

    # All agents at anchor timestep: [n_samples, n, d]
    all_at_t = emb_buf[t_a, e_i]        # [n_samples, n, d]

    # Negatives: every other agent at the same timestep [n_samples, n-1, d]
    mask = torch.ones(n_samples, n, dtype=torch.bool, device=device)
    mask[torch.arange(n_samples, device=device), ag] = False
    z_neg = all_at_t[mask].view(n_samples, n - 1, d)

    # Normalise
    z_anc = F.normalize(z_anc, dim=-1)          # [n_samples, d]
    z_pos = F.normalize(z_pos, dim=-1)          # [n_samples, d]
    z_neg = F.normalize(z_neg, dim=-1)          # [n_samples, n-1, d]

    # Similarities
    sim_pos = (z_anc * z_pos).sum(-1, keepdim=True)          # [n_samples, 1]
    sim_neg = (z_anc.unsqueeze(1) * z_neg).sum(-1)           # [n_samples, n-1]

    # InfoNCE: positive is index 0
    logits = torch.cat([sim_pos, sim_neg], dim=1) / temperature  # [n_samples, n]
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Vectorised MettaGrid environment wrapper
# ---------------------------------------------------------------------------

class MettaGridVecEnv:
    def __init__(self, num_agents: int, num_envs: int, seed: int, max_steps: int = 1024):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.sims = []
        for i in range(num_envs):
            cfg = make_arena(num_agents=num_agents)
            sim = Simulation(cfg, seed=seed + i)
            self.sims.append(sim)

        # Infer dims from first sim
        obs = self.sims[0]._c_sim.observations()   # (n_agents, H, W) or (n_agents, tokens, feats)
        self.obs_dim = int(np.prod(obs.shape[1:]))  # flatten per-agent
        self.n_actions = int(self.sims[0]._c_sim.masks().shape[-1])

        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((num_envs, num_agents), dtype=np.float32)

    # Reset a single sim by recreating it (Simulation has no reset())
    def _reset_sim(self, i: int, seed_offset: int = 0):
        self.sims[i].close()
        cfg = make_arena(num_agents=self.num_agents)
        self.sims[i] = Simulation(cfg, seed=seed_offset + i)
        self.episode_steps[i] = 0
        self.episode_returns[i] = 0.0

    def get_obs(self) -> np.ndarray:
        obs_list = []
        for sim in self.sims:
            o = sim._c_sim.observations().copy()           # (n_agents, ...)
            obs_list.append(o.reshape(self.num_agents, -1))
        return np.stack(obs_list, axis=0)                  # (num_envs, n_agents, obs_dim)

    def step(self, actions: np.ndarray, global_step: int = 0):
        """actions: (num_envs, num_agents) int32"""
        all_rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        all_dones = np.zeros(self.num_envs, dtype=np.float32)
        all_ep_returns = []

        for i, sim in enumerate(self.sims):
            sim._c_sim.actions()[:] = actions[i].astype(np.int32)
            sim.step()
            self.episode_steps[i] += 1

            rews = sim._c_sim.rewards().copy()
            all_rewards[i] = rews
            self.episode_returns[i] += rews

            terminals = sim._c_sim.terminals()
            truncations = sim._c_sim.truncations()
            done = bool(np.any(terminals) or np.any(truncations) or
                        self.episode_steps[i] >= self.max_steps)
            all_dones[i] = float(done)

            if done:
                all_ep_returns.append(float(self.episode_returns[i].sum()))
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
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, emb_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.emb_head = nn.Linear(hidden_dim, emb_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        emb = self.emb_head(h)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, emb

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
    T = rewards.shape[0]
    advs = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        advs[t] = last_gae = delta + gamma * lam * nonterminal * last_gae
    return advs, advs + values


# ---------------------------------------------------------------------------
# Main training function (single seed)
# ---------------------------------------------------------------------------

def train_one_seed(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    method = cfg["method"]
    n_agents = cfg["num_agents"]
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # ---- wandb run name matches paper convention ----
    run_name = f"paper_contrastive_{method}_{n_agents}agents_seed{seed}"
    wandb.init(
        project="representation-collapse",
        entity="tashapais",
        name=run_name,
        config={**cfg, "seed": seed},
        reinit=True,
    )
    print(f"\n[{run_name}] device={device}  obs_dim=TBD  n_actions=TBD")

    env = MettaGridVecEnv(
        num_agents=n_agents,
        num_envs=cfg["num_envs"],
        seed=seed * 100,
        max_steps=cfg["max_steps"],
    )
    obs_dim, n_actions = env.obs_dim, env.n_actions
    print(f"[{run_name}] obs_dim={obs_dim}  n_actions={n_actions}")

    policy = ActorCritic(obs_dim, n_actions, cfg["hidden_dim"], cfg["emb_dim"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"], eps=1e-5)

    T = cfg["num_steps"]
    E = cfg["num_envs"]
    total_steps = cfg["total_timesteps"]
    mb_size = cfg["minibatch_size"]
    epochs = cfg["update_epochs"]
    use_cl = (method == "contrastive")

    # Storage (T, E, n_agents, ...)
    obs_buf      = np.zeros((T, E, n_agents, obs_dim), dtype=np.float32)
    act_buf      = np.zeros((T, E, n_agents), dtype=np.int64)
    rew_buf      = np.zeros((T, E, n_agents), dtype=np.float32)
    done_buf     = np.zeros((T, E), dtype=np.float32)
    logp_buf     = np.zeros((T, E, n_agents), dtype=np.float32)
    val_buf      = np.zeros((T, E, n_agents), dtype=np.float32)
    emb_buf      = np.zeros((T, E, n_agents, cfg["emb_dim"]), dtype=np.float32)
    logit_buf    = np.zeros((T, E, n_agents, n_actions), dtype=np.float32)

    obs = env.get_obs()
    global_step = 0
    update_count = 0
    recent_returns: list[float] = []

    # Metric history (for final averaging)
    eff_rank_hist, exp_ratio_hist, svd_ratio_hist, act_div_hist = [], [], [], []
    t_start = time.time()

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
            obs, rews, dones, ep_rets = env.step(acts_np, global_step=global_step)

            act_buf[step]   = acts_np
            rew_buf[step]   = rews
            done_buf[step]  = dones
            logp_buf[step]  = logps.cpu().numpy().reshape(E, n_agents)
            val_buf[step]   = vals.cpu().numpy().reshape(E, n_agents)
            emb_buf[step]   = embs.cpu().numpy().reshape(E, n_agents, -1)
            logit_buf[step] = lgts.cpu().numpy().reshape(E, n_agents, n_actions)
            recent_returns.extend(ep_rets)

            global_step += E * n_agents

        # ---- Bootstrap value ----
        obs_t = torch.tensor(
            obs.reshape(E * n_agents, obs_dim), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            _, _, _, next_vals, _, _ = policy.get_action_and_value(obs_t)
        next_vals_np = next_vals.cpu().numpy().reshape(E, n_agents)

        # ---- GAE per agent ----
        adv_buf  = np.zeros_like(rew_buf)
        ret_buf  = np.zeros_like(rew_buf)
        for a in range(n_agents):
            r = torch.tensor(rew_buf[:, :, a], dtype=torch.float32, device=device)
            v = torch.tensor(val_buf[:, :, a], dtype=torch.float32, device=device)
            d = torch.tensor(done_buf, dtype=torch.float32, device=device)
            nv = torch.tensor(next_vals_np[:, a], dtype=torch.float32, device=device)
            adv, ret = compute_gae(r, v, d, nv, cfg["gamma"], cfg["gae_lambda"])
            adv_buf[:, :, a] = adv.cpu().numpy()
            ret_buf[:, :, a] = ret.cpu().numpy()

        # ---- Flatten ----
        n_total = T * E * n_agents
        flat_obs  = obs_buf.reshape(n_total, obs_dim)
        flat_acts = act_buf.reshape(n_total)
        flat_logp = logp_buf.reshape(n_total)
        flat_adv  = adv_buf.reshape(n_total)
        flat_ret  = ret_buf.reshape(n_total)

        # ---- PPO updates ----
        idx = np.arange(n_total)
        total_pg, total_vl, total_ent, total_cl = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n_total, mb_size):
                mb = idx[start:start + mb_size]
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
                vl = F.mse_loss(new_val, mb_ret)

                loss = pg + 0.5 * vl - 0.01 * ent.mean()

                # Inter-agent contrastive loss (applied once per PPO minibatch)
                cl_val = torch.tensor(0.0, device=device)
                if use_cl:
                    emb_t = torch.tensor(emb_buf, dtype=torch.float32, device=device)
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

        # ---- Geometric metrics ----
        all_emb = torch.tensor(emb_buf.reshape(-1, cfg["emb_dim"]), dtype=torch.float32)
        eff_rank  = compute_effective_rank(all_emb)
        svd_ratio = compute_svd_ratio(all_emb)
        # Expansion ratio: reshape emb_buf [T,E,n,d] → [E*n, T, d] per-agent trajectories
        emb_trajs_np = emb_buf.transpose(1, 2, 0, 3).reshape(E * n_agents, T, cfg["emb_dim"])
        exp_ratio = float(np.mean([
            compute_expansion_ratio(torch.tensor(emb_trajs_np[b], dtype=torch.float32))
            for b in range(min(64, emb_trajs_np.shape[0]))
        ]))

        # ---- Behavioural diversity: mean pairwise KL between agents' π(·|o) ----
        # logit_buf: (T, E, n_agents, n_actions) → (T*E, n_agents, n_actions)
        lgts_flat = torch.tensor(
            logit_buf.reshape(T * E, n_agents, n_actions), dtype=torch.float32
        )
        act_diversity = compute_action_diversity(lgts_flat)

        eff_rank_hist.append(eff_rank)
        exp_ratio_hist.append(exp_ratio)
        svd_ratio_hist.append(svd_ratio)
        act_div_hist.append(act_diversity)

        mean_ret = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
        sps = global_step / max(1.0, time.time() - t_start)
        n_upd = max(1, n_updates)

        log = {
            "global_step":                  global_step,
            "charts/effective_rank":        eff_rank,
            "charts/effrank_per_agent":     eff_rank / n_agents,
            "charts/expansion_ratio":       exp_ratio,
            "charts/svd_ratio":             svd_ratio,
            "charts/action_diversity":      act_diversity,
            "charts/mean_return":           mean_ret,
            "charts/sps":                   sps,
            "losses/policy":                total_pg / n_upd,
            "losses/value":                 total_vl / n_upd,
            "losses/entropy":               total_ent / n_upd,
        }
        if use_cl:
            log["losses/contrastive"] = total_cl / n_upd
        wandb.log(log, step=global_step)

        update_count += 1
        if update_count % 5 == 0:
            print(
                f"[{run_name}] step={global_step:,}  "
                f"eff_rank={eff_rank:.2f}/{n_agents}({eff_rank/n_agents:.2f})  "
                f"act_div={act_diversity:.4f}  "
                f"exp_ratio={exp_ratio:.3f}  svd={svd_ratio:.1f}  "
                f"ret={mean_ret:.4f}  sps={sps:.0f}"
            )

    # ---- Final summary ----
    final_eff_rank  = float(np.mean(eff_rank_hist[-10:]))  if eff_rank_hist  else 0.0
    final_exp_ratio = float(np.mean(exp_ratio_hist[-10:])) if exp_ratio_hist else 0.0
    final_svd_ratio = float(np.mean(svd_ratio_hist[-10:])) if svd_ratio_hist else 0.0
    final_act_div   = float(np.mean(act_div_hist[-10:]))   if act_div_hist   else 0.0
    final_ret = float(np.mean(recent_returns[-200:])) if recent_returns else 0.0
    win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in recent_returns[-200:]])) \
               if recent_returns else 0.0

    wandb.log({
        "final/effective_rank":    final_eff_rank,
        "final/effrank_per_agent": final_eff_rank / n_agents,
        "final/expansion_ratio":   final_exp_ratio,
        "final/svd_ratio":         final_svd_ratio,
        "final/action_diversity":  final_act_div,
        "final/mean_return":       final_ret,
        "final/win_rate":          win_rate,
    })
    wandb.finish()
    env.close()

    result = {
        "method":             method,
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
    }
    print(f"\n[{run_name}] DONE: {json.dumps(result, indent=2)}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "contrastive"], required=True)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--num_agents", type=int, default=18,
                        help="Must be divisible by 6 for MettaGrid Arena")
    parser.add_argument("--num_seeds",  type=int, default=5)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    args = parser.parse_args()

    cfg = dict(
        method          = args.method,
        num_agents      = args.num_agents,
        gpu             = args.gpu,
        num_envs        = 4,
        num_steps       = 128,
        max_steps       = 1024,
        total_timesteps = args.total_timesteps,
        minibatch_size  = 512,
        update_epochs   = 8,
        lr              = 3e-4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        hidden_dim      = 128,
        emb_dim         = 64,
        # Contrastive hyperparameters (from existing contrastive_paper_experiments.py)
        contrastive_coef = 6.8e-4,
        temperature      = 0.19,
        gamma_c          = 0.977,
    )

    all_results = []
    for seed in range(args.num_seeds):
        print(f"\n{'='*65}")
        print(f"  method={args.method}  agents={args.num_agents}  seed={seed}/{args.num_seeds-1}")
        print(f"{'='*65}")
        result = train_one_seed(cfg, seed)
        all_results.append(result)

    out_path = f"/home/devuser/metta/v3_experiments/results_contrastive_{args.method}_{args.num_agents}agents.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} seeds → {out_path}")

    ranks  = [r["final_eff_rank"]  for r in all_results]
    exps   = [r["final_exp_ratio"] for r in all_results]
    rets   = [r["final_return"]    for r in all_results]
    print(f"\n{'='*65}")
    print(f"SUMMARY  method={args.method}  agents={args.num_agents}  n={args.num_seeds} seeds")
    print(f"  EffRank:    {np.mean(ranks):.2f} ± {np.std(ranks):.2f}")
    print(f"  ExpRatio:   {np.mean(exps):.3f} ± {np.std(exps):.3f}")
    print(f"  Return:     {np.mean(rets):.4f} ± {np.std(rets):.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
