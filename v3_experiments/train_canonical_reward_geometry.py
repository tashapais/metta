"""Canonical Tribal Village reward-geometry runner.

This runner is intentionally separate from ``paper_exp_reward_type.py``. The
paper reward-geometry section needs fixed 3-way role labels from
``agent_id % 3``, not binary top/bottom return labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.canonical_reward_geometry import (  # noqa: E402
    CANONICAL_EVAL_TRIALS,
    CANONICAL_NUM_AGENTS,
    DEFAULT_ALTAR_LAYER,
    DEFAULT_GOLD_LAYER,
    ROLE_NAMES,
    ROLE_PROBE_CV,
    ROLE_SHAPING_COEFFICIENTS,
    effective_rank,
    fixed_role_probe_accuracy,
    js_action_diversity,
    mix_rewards,
    ordered_kl_action_diversity,
    role_labels,
    role_probe_chance,
    role_shaping_bonuses,
)
from v3_experiments.tribal_event_rewards import (  # noqa: E402
    EVENT_V1_COMMON_COEFFICIENTS,
    EVENT_V1_ROLE_COEFFICIENTS,
    EVENT_V1_ROLE_NAMES,
    EVENT_V2_BREADCRUMB_COMMON_COEFFICIENTS,
    EVENT_V2_BREADCRUMB_ROLE_COEFFICIENTS,
    EVENT_V2_BREADCRUMB_ROLE_NAMES,
    EVENT_V2_BREADCRUMB_TASK_COEFFICIENTS,
    event_v1_reward_design_details,
    event_v1_role_shaping_bonuses,
    event_v2_breadcrumb_reward_design_details,
    event_v2_breadcrumb_role_shaping_bonuses,
)

TRIBAL_VILLAGE_ROOT = REPO_ROOT / "packages" / "tribal_village"
CANONICAL_ENV_DEFINE = "canonicalRewardGeometry"


class CanonicalEnv(Protocol):
    num_agents: int
    num_teams: int
    map_width: int
    map_height: int
    obs_shape: tuple[int, ...]
    action_space_size: int
    build_id: str

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]: ...

    def get_action_stats(self) -> np.ndarray | None: ...

    def get_inventory_snapshot(self) -> np.ndarray | None: ...

    def get_world_stats(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RunnerConfig:
    shared_frac: float
    seed: int
    total_agent_steps: int
    eval_trials: int
    eval_steps: int
    max_steps: int
    num_steps: int
    minibatch_size: int
    update_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    hidden_dim: int
    embedding_dim: int
    reward_design: str
    disable_role_shaping: bool
    separate_encoders: bool
    env_backend: str
    allow_noncanonical_env: bool
    gold_layer: int
    altar_layer: int
    wandb_entity: str | None
    wandb_project: str
    wandb_mode: str
    output: str
    checkpoint_path: str | None
    run_name: str | None
    condition_group: str | None
    device: str
    log_interval: int


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        n_agents: int,
        hidden_dim: int,
        embedding_dim: int,
        separate_encoders: bool,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.separate_encoders = separate_encoders
        obs_dim = int(np.prod(obs_shape))

        if separate_encoders:
            self.encoders = nn.ModuleList(_make_encoder(obs_dim, hidden_dim, embedding_dim) for _ in range(n_agents))
        else:
            self.encoder = _make_encoder(obs_dim, hidden_dim, embedding_dim)
        self.actor = nn.Linear(embedding_dim, n_actions)
        self.critic = nn.Linear(embedding_dim, 1)

    def encode(self, obs: torch.Tensor, agent_ids: torch.Tensor | None = None) -> torch.Tensor:
        flat = obs.reshape(obs.shape[0], -1).float() / 255.0
        if not self.separate_encoders:
            return self.encoder(flat)
        if agent_ids is None:
            raise ValueError("agent_ids are required with separate_encoders")
        out = torch.zeros((flat.shape[0], self.actor.in_features), dtype=flat.dtype, device=flat.device)
        for agent_id in torch.unique(agent_ids).tolist():
            mask = agent_ids == int(agent_id)
            out[mask] = self.encoders[int(agent_id)](flat[mask])
        return out

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        agent_ids: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(obs, agent_ids=agent_ids)
        logits = self.actor(embedding)
        dist = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(embedding).squeeze(-1)
        return action, log_prob, entropy, value, embedding, logits


class MockCanonicalTribalEnv:
    """Fast deterministic backend used only for CLI/tests."""

    num_agents = CANONICAL_NUM_AGENTS
    num_teams = 1
    map_width = 80
    map_height = 80
    obs_shape = (21, 11, 11)
    action_space_size = 56
    build_id = "mock-canonical-tribal-village"

    def __init__(self, max_steps: int, *, gold_layer: int, altar_layer: int) -> None:
        self.max_steps = max_steps
        self.gold_layer = gold_layer
        self.altar_layer = altar_layer
        self._rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        self._step += 1
        labels = role_labels(self.num_agents)
        rewards = 0.01 * (np.asarray(actions) % 3) + np.array([0.1, 0.02, 0.05], dtype=np.float64)[labels]
        return self._observations(), rewards.astype(np.float64), self._step >= self.max_steps

    def close(self) -> None:
        return None

    def get_action_stats(self) -> np.ndarray | None:
        return None

    def get_inventory_snapshot(self) -> np.ndarray | None:
        return None

    def get_world_stats(self) -> np.ndarray | None:
        return None

    def _observations(self) -> np.ndarray:
        obs = self._rng.integers(0, 3, size=(self.num_agents, *self.obs_shape), dtype=np.uint8)
        labels = role_labels(self.num_agents)
        for agent_id, role in enumerate(labels):
            obs[agent_id, :, :, :] = 0
            obs[agent_id, 19, :, :] = self._rng.integers(0, 4, size=self.obs_shape[1:], dtype=np.uint8)
            if role == 0:
                obs[agent_id, self.gold_layer, 3:8, 3:8] = 1
            elif role == 2:
                obs[agent_id, self.altar_layer, 4:7, 4:7] = 1
        return obs


class TribalVillageAdapter:
    def __init__(self, max_steps: int, *, build_canonical: bool = True) -> None:
        if build_canonical:
            _append_nim_define(CANONICAL_ENV_DEFINE)
        _ensure_tribal_village_import_path()
        from tribal_village_env.build import ensure_nim_library_current

        library_path = ensure_nim_library_current()

        from tribal_village_env.environment import TribalVillageEnv

        self._env = TribalVillageEnv(config={"max_steps": max_steps, "render_mode": "ansi", "render_scale": 1})
        self.num_agents = int(self._env.num_agents)
        self.num_teams = 1
        self.map_width = int(self._env.map_width or 0)
        self.map_height = int(self._env.map_height or 0)
        self.obs_shape = tuple(int(x) for x in self._env.single_observation_space.shape)
        self.action_space_size = int(self._env.single_action_space.n)
        self.build_id = f"{library_path.name}:{_git_sha(REPO_ROOT / 'packages' / 'tribal_village')}"

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed)
        return _stack_agent_dict(obs, self.num_agents)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        action_dict = {f"agent_{idx}": np.asarray(int(action), dtype=np.int64) for idx, action in enumerate(actions)}
        obs, rewards, terminated, truncated, _info = self._env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())
        return _stack_agent_dict(obs, self.num_agents), _stack_reward_dict(rewards, self.num_agents), done

    def close(self) -> None:
        self._env.close()

    def get_action_stats(self) -> np.ndarray | None:
        get_stats = getattr(self._env, "get_action_stats", None)
        if get_stats is None:
            return None
        return get_stats()

    def get_inventory_snapshot(self) -> np.ndarray | None:
        get_snapshot = getattr(self._env, "get_inventory_snapshot", None)
        if get_snapshot is None:
            return None
        return get_snapshot()

    def get_world_stats(self) -> np.ndarray | None:
        get_stats = getattr(self._env, "get_world_stats", None)
        if get_stats is None:
            return None
        return get_stats()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RunnerConfig(**vars(args))
    run(config, argv if argv is not None else sys.argv[1:])
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-agent-steps", type=int, default=4_000_000)
    parser.add_argument("--eval-trials", type=int, default=CANONICAL_EVAL_TRIALS)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--reward-design",
        choices=("passive_v0", "event_v1", "event_v2_breadcrumbs"),
        default="passive_v0",
        help="Role-shaping reward design. passive_v0 preserves the old observation shaping.",
    )
    parser.add_argument("--disable-role-shaping", action="store_true")
    parser.add_argument("--separate-encoders", action="store_true")
    parser.add_argument("--env-backend", choices=("tribal", "mock"), default="tribal")
    parser.add_argument("--allow-noncanonical-env", action="store_true")
    parser.add_argument("--gold-layer", type=int, default=DEFAULT_GOLD_LAYER)
    parser.add_argument("--altar-layer", type=int, default=DEFAULT_ALTAR_LAYER)
    parser.add_argument("--wandb-entity", default="tashapais")
    parser.add_argument("--wandb-project", default="representation-collapse")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--run-name")
    parser.add_argument("--condition-group")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args(argv)


def run(config: RunnerConfig, argv: list[str]) -> dict[str, Any]:
    _validate_config(config)
    _seed_everything(config.seed)
    device = torch.device(config.device)

    env = _make_env(config)
    try:
        _validate_env_contract(env, config)
        group = config.condition_group or _condition_group(config)
        run_name = config.run_name or _run_name(group, config.shared_frac, config.seed, config.reward_design)
        output_path = Path(config.output)
        checkpoint_path = (
            Path(config.checkpoint_path) if config.checkpoint_path else _default_checkpoint_path(output_path, run_name)
        )

        policy = ActorCritic(
            env.obs_shape,
            env.action_space_size,
            n_agents=env.num_agents,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            separate_encoders=config.separate_encoders,
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        wandb_run = _init_wandb(config, env, run_name, argv)

        train_metrics = _train_policy(policy, optimizer, env, config, device, wandb_run)
        eval_summary = _evaluate_policy(policy, env, config, device)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "config": asdict(config),
                "env": _env_metadata(env),
                "train_metrics": train_metrics,
                "eval_summary": eval_summary,
            },
            checkpoint_path,
        )

        record = _result_record(
            config,
            env,
            group,
            run_name,
            output_path,
            checkpoint_path,
            train_metrics,
            eval_summary,
            wandb_run,
            argv,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

        if wandb_run is not None:
            wandb_run.log({f"final/{key}": value for key, value in _final_metric_log(record).items()})
            wandb_run.finish()

        print(
            json.dumps(
                {"output": str(output_path), "checkpoint_path": str(checkpoint_path), "run_name": run_name},
                indent=2,
            )
        )
        return record
    finally:
        env.close()


def _train_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    env: CanonicalEnv,
    config: RunnerConfig,
    device: torch.device,
    wandb_run: Any,
) -> dict[str, Any]:
    obs = env.reset(seed=config.seed)
    event_stats = _EventStatsTracker(env)
    global_step = 0
    update = 0
    start_time = time.time()
    recent_returns: list[float] = []
    recent_raw_returns: list[float] = []
    recent_role_shaping_returns: list[float] = []
    recent_individual_returns: list[float] = []
    last_metrics: dict[str, float] = {}

    while global_step < config.total_agent_steps:
        remaining_steps = max(1, math.ceil((config.total_agent_steps - global_step) / env.num_agents))
        rollout_steps = min(config.num_steps, remaining_steps)
        obs_buf: list[np.ndarray] = []
        action_buf: list[np.ndarray] = []
        logprob_buf: list[np.ndarray] = []
        reward_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []
        value_buf: list[np.ndarray] = []
        embedding_buf: list[np.ndarray] = []
        logit_buf: list[np.ndarray] = []

        for _ in range(rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            agent_ids = _agent_ids(env.num_agents, device) if config.separate_encoders else None
            with torch.no_grad():
                actions, logprob, _entropy, values, embeddings, logits = policy.get_action_and_value(
                    obs_tensor, agent_ids=agent_ids
                )
            actions_np = actions.cpu().numpy().astype(np.int64)
            next_obs, env_rewards, done = env.step(actions_np)
            reward_components = _canonical_reward_components(
                obs,
                env_rewards,
                config,
                event_stats_delta=event_stats.delta(env),
            )
            shaped_rewards = reward_components["mixed_rewards"]

            obs_buf.append(obs.copy())
            action_buf.append(actions_np)
            logprob_buf.append(logprob.cpu().numpy())
            reward_buf.append(shaped_rewards)
            done_buf.append(np.full(env.num_agents, float(done), dtype=np.float32))
            value_buf.append(values.cpu().numpy())
            embedding_buf.append(embeddings.cpu().numpy())
            logit_buf.append(logits.cpu().numpy())

            recent_returns.append(float(shaped_rewards.sum()))
            recent_raw_returns.append(float(reward_components["raw_env_rewards"].sum()))
            recent_role_shaping_returns.append(float(reward_components["role_shaping_bonuses"].sum()))
            recent_individual_returns.append(float(reward_components["individual_rewards"].sum()))
            if done:
                obs = env.reset(seed=config.seed + global_step + 1)
                event_stats.reset(env)
            else:
                obs = next_obs
            global_step += env.num_agents

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _a, _lp, _e, next_values, _emb, _logits = policy.get_action_and_value(
                obs_tensor,
                agent_ids=_agent_ids(env.num_agents, device) if config.separate_encoders else None,
            )

        rewards = np.stack(reward_buf)
        dones = np.stack(done_buf)
        values = np.stack(value_buf)
        advantages, returns = _compute_gae(
            rewards,
            values,
            dones,
            next_values.cpu().numpy(),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        flat_obs = np.stack(obs_buf).reshape(-1, *env.obs_shape)
        flat_actions = np.stack(action_buf).reshape(-1)
        flat_logprobs = np.stack(logprob_buf).reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_agent_ids = np.tile(np.arange(env.num_agents), len(obs_buf))

        pg_loss, value_loss, entropy_loss = _ppo_update(
            policy,
            optimizer,
            flat_obs,
            flat_actions,
            flat_logprobs,
            flat_advantages,
            flat_returns,
            flat_agent_ids,
            config,
            device,
        )

        update += 1
        embeddings_arr = np.stack(embedding_buf)
        logits_arr = np.stack(logit_buf)
        flat_embeddings = embeddings_arr.reshape(-1, embeddings_arr.shape[-1])
        flat_logits = logits_arr.reshape(-1, env.num_agents, env.action_space_size)
        effective_rank_value = effective_rank(flat_embeddings)
        last_metrics = {
            "global_step": float(global_step),
            "effective_rank": effective_rank_value,
            "effrank_per_agent": effective_rank_value / env.num_agents,
            "d_act_ordered_kl": ordered_kl_action_diversity(flat_logits),
            "d_act_js": js_action_diversity(flat_logits),
            "mean_return": float(np.mean(recent_returns[-100:])),
            "mean_raw_env_return": float(np.mean(recent_raw_returns[-100:])),
            "mean_role_shaping_return": float(np.mean(recent_role_shaping_returns[-100:])),
            "mean_individual_return": float(np.mean(recent_individual_returns[-100:])),
            "policy_loss": pg_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "sps": float(global_step / max(1e-6, time.time() - start_time)),
        }
        if wandb_run is not None:
            wandb_run.log({f"train/{key}": value for key, value in last_metrics.items()}, step=global_step)
        if config.log_interval > 0 and update % config.log_interval == 0:
            print(
                f"[step={global_step}] effrank/n={last_metrics['effrank_per_agent']:.3f} "
                f"d_act={last_metrics['d_act_ordered_kl']:.4f} return={last_metrics['mean_return']:.3f}"
            )

    return last_metrics


def _ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    flat_obs: np.ndarray,
    flat_actions: np.ndarray,
    flat_logprobs: np.ndarray,
    flat_advantages: np.ndarray,
    flat_returns: np.ndarray,
    flat_agent_ids: np.ndarray,
    config: RunnerConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    total_pg = 0.0
    total_v = 0.0
    total_entropy = 0.0
    n_updates = 0
    idx = np.arange(flat_obs.shape[0])
    batch_size = max(1, min(config.minibatch_size, flat_obs.shape[0]))

    for _ in range(config.update_epochs):
        np.random.shuffle(idx)
        for start in range(0, flat_obs.shape[0], batch_size):
            batch_idx = idx[start : start + batch_size]
            obs_t = torch.as_tensor(flat_obs[batch_idx], dtype=torch.float32, device=device)
            action_t = torch.as_tensor(flat_actions[batch_idx], dtype=torch.long, device=device)
            old_logprob_t = torch.as_tensor(flat_logprobs[batch_idx], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(flat_advantages[batch_idx], dtype=torch.float32, device=device)
            return_t = torch.as_tensor(flat_returns[batch_idx], dtype=torch.float32, device=device)
            agent_ids_t = (
                torch.as_tensor(flat_agent_ids[batch_idx], dtype=torch.long, device=device)
                if config.separate_encoders
                else None
            )

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
            _action, new_logprob, entropy, value, _embedding, _logits = policy.get_action_and_value(
                obs_t, action_t, agent_ids=agent_ids_t
            )
            ratio = (new_logprob - old_logprob_t).exp()
            pg_loss = torch.max(-adv_t * ratio, -adv_t * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)).mean()
            value_loss = F.mse_loss(value, return_t)
            entropy_loss = entropy.mean()
            loss = pg_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_pg += float(pg_loss.item())
            total_v += float(value_loss.item())
            total_entropy += float(entropy_loss.item())
            n_updates += 1

    denom = max(1, n_updates)
    return total_pg / denom, total_v / denom, total_entropy / denom


def _evaluate_policy(
    policy: ActorCritic,
    env: CanonicalEnv,
    config: RunnerConfig,
    device: torch.device,
) -> dict[str, Any]:
    policy.eval()
    trial_metrics: list[dict[str, Any]] = []
    with torch.no_grad():
        for trial in range(config.eval_trials):
            obs = env.reset(seed=config.seed + 100_000 + trial)
            event_stats = _EventStatsTracker(env)
            embeddings: list[np.ndarray] = []
            logits: list[np.ndarray] = []
            returns = np.zeros(env.num_agents, dtype=np.float64)
            raw_env_returns = np.zeros(env.num_agents, dtype=np.float64)
            role_shaping_returns = np.zeros(env.num_agents, dtype=np.float64)
            individual_returns = np.zeros(env.num_agents, dtype=np.float64)
            for step in range(config.eval_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, _logprob, _entropy, _value, emb, logit = policy.get_action_and_value(
                    obs_t,
                    agent_ids=_agent_ids(env.num_agents, device) if config.separate_encoders else None,
                    deterministic=True,
                )
                next_obs, rewards, done = env.step(actions.cpu().numpy())
                reward_components = _canonical_reward_components(
                    obs,
                    rewards,
                    config,
                    event_stats_delta=event_stats.delta(env),
                )
                returns += reward_components["mixed_rewards"]
                raw_env_returns += reward_components["raw_env_rewards"]
                role_shaping_returns += reward_components["role_shaping_bonuses"]
                individual_returns += reward_components["individual_rewards"]
                embeddings.append(emb.cpu().numpy())
                logits.append(logit.cpu().numpy())
                if done:
                    obs = env.reset(seed=config.seed + 200_000 + trial * config.eval_steps + step)
                    event_stats.reset(env)
                else:
                    obs = next_obs

            emb_arr = np.stack(embeddings)
            logit_arr = np.stack(logits)
            probe_acc, probe_meta = fixed_role_probe_accuracy(emb_arr)
            trial_metrics.append(
                {
                    "trial": trial,
                    "effrank_per_agent": effective_rank(emb_arr.reshape(-1, emb_arr.shape[-1])) / env.num_agents,
                    "d_act_ordered_kl": ordered_kl_action_diversity(logit_arr),
                    "d_act_js": js_action_diversity(logit_arr),
                    "role_probe_acc": probe_acc,
                    "role_probe_meta": probe_meta,
                    "mean_return": float(returns.mean()),
                    "total_return": float(returns.sum()),
                    "mean_raw_env_return": float(raw_env_returns.mean()),
                    "total_raw_env_return": float(raw_env_returns.sum()),
                    "mean_role_shaping_return": float(role_shaping_returns.mean()),
                    "total_role_shaping_return": float(role_shaping_returns.sum()),
                    "mean_individual_return": float(individual_returns.mean()),
                    "total_individual_return": float(individual_returns.sum()),
                }
            )
    policy.train()

    summary: dict[str, Any] = {"eval_trial_metrics": trial_metrics}
    for key in (
        "effrank_per_agent",
        "d_act_ordered_kl",
        "d_act_js",
        "role_probe_acc",
        "mean_return",
        "total_return",
        "mean_raw_env_return",
        "total_raw_env_return",
        "mean_role_shaping_return",
        "total_role_shaping_return",
        "mean_individual_return",
        "total_individual_return",
    ):
        values = np.array([float(item[key]) for item in trial_metrics], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_eval_trial_std"] = float(values.std())
    summary["role_probe_chance"] = role_probe_chance()
    summary["role_probe_cv"] = ROLE_PROBE_CV
    return summary


def _result_record(
    config: RunnerConfig,
    env: CanonicalEnv,
    condition_group: str,
    run_name: str,
    output_path: Path,
    checkpoint_path: Path,
    train_metrics: dict[str, Any],
    eval_summary: dict[str, Any],
    wandb_run: Any,
    argv: list[str],
) -> dict[str, Any]:
    command = "uv run python v3_experiments/train_canonical_reward_geometry.py " + " ".join(
        shlex.quote(arg) for arg in argv
    )
    record = {
        "schema_version": "canonical_reward_geometry_v1",
        "run_name": run_name,
        "condition_group": condition_group,
        "environment_backend": config.env_backend,
        "shared_frac": config.shared_frac,
        "seed": config.seed,
        "num_agents": env.num_agents,
        "num_teams": env.num_teams,
        "map_width": env.map_width,
        "map_height": env.map_height,
        "role_assignment": "agent_id % 3",
        "role_names": _reward_design_role_names(config),
        "role_labels": role_labels(env.num_agents).astype(int).tolist(),
        "reward_design": config.reward_design,
        "reward_design_details": _reward_design_details(config),
        "role_shaping_enabled": not config.disable_role_shaping,
        "role_shaping_coefficients": _reward_design_coefficients(config),
        "role_shaping_layers": {"gold_layer": config.gold_layer, "altar_layer": config.altar_layer},
        "separate_encoders": config.separate_encoders,
        "total_agent_steps": config.total_agent_steps,
        "eval_trials": config.eval_trials,
        "eval_steps": config.eval_steps,
        "metta_git_sha": _git_sha(REPO_ROOT),
        "tribal_village_git_sha": _git_sha(REPO_ROOT / "packages" / "tribal_village"),
        "tribal_village_build_id": env.build_id,
        "command": command,
        "wandb_entity": config.wandb_entity,
        "wandb_project": config.wandb_project,
        "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
        "wandb_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "obs_shape": list(env.obs_shape),
        "action_space_size": env.action_space_size,
        "metric_schema": {
            "effrank_per_agent": "effective_rank(flattened_eval_embeddings) / n_agents",
            "d_act_ordered_kl": "mean ordered off-diagonal KL over n_agents * (n_agents - 1) pairs",
            "d_act_js": "mean unordered Jensen-Shannon action diversity",
            "role_probe_acc": "3-way agent_id % 3 role probe with 4-fold held-out-agent CV",
            "role_probe_chance": "1/3",
            "mean_raw_env_return": "eval mean return before role shaping and reward mixing",
            "mean_role_shaping_return": "eval mean role-shaping bonus before reward mixing",
            "mean_individual_return": "eval mean raw plus role-shaping return before reward mixing",
            "mean_return": "eval mean mixed return after shared_frac reward mixing",
        },
        "train_metrics": train_metrics,
        **eval_summary,
    }
    return record


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float64)
    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        next_value = next_values if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _canonical_reward_components(
    obs: np.ndarray,
    env_rewards: np.ndarray,
    config: RunnerConfig,
    *,
    event_stats_delta: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    raw_env_rewards = np.asarray(env_rewards, dtype=np.float64)
    if not config.disable_role_shaping:
        bonuses = _role_shaping_bonuses_for_design(obs, raw_env_rewards.shape[-1], config, event_stats_delta)
    else:
        bonuses = np.zeros_like(raw_env_rewards, dtype=np.float64)
    individual_rewards = raw_env_rewards + bonuses
    return {
        "raw_env_rewards": raw_env_rewards,
        "role_shaping_bonuses": bonuses,
        "individual_rewards": individual_rewards,
        "mixed_rewards": mix_rewards(individual_rewards, config.shared_frac),
    }


def _canonical_rewards(
    obs: np.ndarray,
    env_rewards: np.ndarray,
    config: RunnerConfig,
    *,
    event_stats_delta: np.ndarray | None = None,
) -> np.ndarray:
    return _canonical_reward_components(
        obs,
        env_rewards,
        config,
        event_stats_delta=event_stats_delta,
    )["mixed_rewards"]


def _role_shaping_bonuses_for_design(
    obs: np.ndarray,
    num_agents: int,
    config: RunnerConfig,
    event_stats_delta: np.ndarray | None,
) -> np.ndarray:
    if config.reward_design == "passive_v0":
        return role_shaping_bonuses(
            obs,
            gold_layer=config.gold_layer,
            altar_layer=config.altar_layer,
        )
    if config.reward_design == "event_v1":
        return event_v1_role_shaping_bonuses(event_stats_delta, num_agents=num_agents)
    if config.reward_design == "event_v2_breadcrumbs":
        return event_v2_breadcrumb_role_shaping_bonuses(event_stats_delta, num_agents=num_agents)
    raise ValueError(f"unknown reward design: {config.reward_design}")


def _make_env(config: RunnerConfig) -> CanonicalEnv:
    if config.env_backend == "mock":
        return MockCanonicalTribalEnv(config.max_steps, gold_layer=config.gold_layer, altar_layer=config.altar_layer)
    return TribalVillageAdapter(config.max_steps)


class _EventStatsTracker:
    def __init__(self, env: CanonicalEnv) -> None:
        self._previous = _copy_action_stats(env)

    def reset(self, env: CanonicalEnv) -> None:
        self._previous = _copy_action_stats(env)

    def delta(self, env: CanonicalEnv) -> np.ndarray | None:
        current = _copy_action_stats(env)
        if current is None:
            self._previous = None
            return None
        if self._previous is None:
            self._previous = current
            return None
        delta = current - self._previous
        self._previous = current
        return np.maximum(delta, 0)


def _copy_action_stats(env: CanonicalEnv) -> np.ndarray | None:
    stats = env.get_action_stats()
    if stats is None:
        return None
    return np.asarray(stats, dtype=np.float64).copy()


def _validate_config(config: RunnerConfig) -> None:
    if not 0.0 <= config.shared_frac <= 1.0:
        raise ValueError("--shared-frac must be in [0, 1]")
    if config.total_agent_steps <= 0:
        raise ValueError("--total-agent-steps must be positive")
    if config.eval_trials <= 0:
        raise ValueError("--eval-trials must be positive")
    if config.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive")


def _validate_env_contract(env: CanonicalEnv, config: RunnerConfig) -> None:
    expected = {
        "num_agents": CANONICAL_NUM_AGENTS,
        "num_teams": 1,
        "map_width": 80,
        "map_height": 80,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": getattr(env, key)}
        for key, expected_value in expected.items()
        if getattr(env, key) != expected_value
    }
    if mismatches and not config.allow_noncanonical_env:
        raise RuntimeError(
            "environment does not match canonical reward-geometry protocol: "
            f"{json.dumps(mismatches, sort_keys=True)}. Rebuild Tribal Village with "
            f"-d:{CANONICAL_ENV_DEFINE} or pass --allow-noncanonical-env only for debugging."
        )


def _init_wandb(config: RunnerConfig, env: CanonicalEnv, run_name: str, argv: list[str]) -> Any:
    if config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but wandb is not importable") from exc
    return wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=run_name,
        mode=config.wandb_mode,
        config={
            **asdict(config),
            **_env_metadata(env),
            "command": " ".join(shlex.quote(arg) for arg in argv),
            "metta_git_sha": _git_sha(REPO_ROOT),
        },
    )


def _env_metadata(env: CanonicalEnv) -> dict[str, Any]:
    return {
        "num_agents": env.num_agents,
        "num_teams": env.num_teams,
        "map_width": env.map_width,
        "map_height": env.map_height,
        "obs_shape": list(env.obs_shape),
        "action_space_size": env.action_space_size,
        "tribal_village_build_id": env.build_id,
    }


def _condition_group(config: RunnerConfig) -> str:
    if config.disable_role_shaping:
        return "no_role_shaping"
    if config.separate_encoders:
        return "separate_encoders"
    return "primary"


def _reward_design_role_names(config: RunnerConfig) -> list[str]:
    if config.reward_design == "event_v1":
        return list(EVENT_V1_ROLE_NAMES)
    if config.reward_design == "event_v2_breadcrumbs":
        return list(EVENT_V2_BREADCRUMB_ROLE_NAMES)
    return list(ROLE_NAMES)


def _reward_design_coefficients(config: RunnerConfig) -> dict[str, Any]:
    if config.reward_design == "event_v1":
        return {
            "common": dict(EVENT_V1_COMMON_COEFFICIENTS),
            "roles": {role: dict(coefficients) for role, coefficients in EVENT_V1_ROLE_COEFFICIENTS.items()},
        }
    if config.reward_design == "event_v2_breadcrumbs":
        return {
            "common": dict(EVENT_V2_BREADCRUMB_COMMON_COEFFICIENTS),
            "task_events": dict(EVENT_V2_BREADCRUMB_TASK_COEFFICIENTS),
            "roles": {role: dict(coefficients) for role, coefficients in EVENT_V2_BREADCRUMB_ROLE_COEFFICIENTS.items()},
        }
    return dict(ROLE_SHAPING_COEFFICIENTS)


def _reward_design_details(config: RunnerConfig) -> dict[str, Any]:
    if config.reward_design == "event_v1":
        return event_v1_reward_design_details()
    if config.reward_design == "event_v2_breadcrumbs":
        return event_v2_breadcrumb_reward_design_details()
    return {
        "name": "passive_v0",
        "summary": "Original observation-based role shaping from the reconstructed canonical runner.",
        "role_names": list(ROLE_NAMES),
        "coefficients": dict(ROLE_SHAPING_COEFFICIENTS),
        "observation_layers": {"gold_layer": config.gold_layer, "altar_layer": config.altar_layer},
    }


def _run_name(group: str, shared_frac: float, seed: int, reward_design: str) -> str:
    alpha = str(shared_frac).replace(".", "p")
    reward_part = "" if reward_design == "passive_v0" else f"_{reward_design}"
    return f"canonical_reward_geometry{reward_part}_{group}_alpha{alpha}_seed{seed}"


def _default_checkpoint_path(output_path: Path, run_name: str) -> Path:
    return output_path.parent / "checkpoints" / run_name / "final_model.pt"


def _agent_ids(num_agents: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_agents, dtype=torch.long, device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_encoder(obs_dim: int, hidden_dim: int, embedding_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embedding_dim),
        nn.ReLU(),
    )


def _stack_agent_dict(values: dict[str, Any], num_agents: int) -> np.ndarray:
    return np.stack([np.asarray(values[f"agent_{idx}"]) for idx in range(num_agents)])


def _stack_reward_dict(values: dict[str, float], num_agents: int) -> np.ndarray:
    return np.array([float(values[f"agent_{idx}"]) for idx in range(num_agents)], dtype=np.float64)


def _append_nim_define(define: str) -> None:
    raw = os.environ.get("TRIBAL_VILLAGE_NIM_DEFINES", "")
    current = {part for part in raw.replace(",", " ").split() if part}
    if define not in current:
        current.add(define)
        os.environ["TRIBAL_VILLAGE_NIM_DEFINES"] = " ".join(sorted(current))


def _ensure_tribal_village_import_path() -> None:
    path = str(TRIBAL_VILLAGE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _git_sha(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _final_metric_log(record: dict[str, Any]) -> dict[str, float]:
    keys = ("effrank_per_agent", "d_act_ordered_kl", "d_act_js", "role_probe_acc", "role_probe_chance")
    return {key: float(record[key]) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
