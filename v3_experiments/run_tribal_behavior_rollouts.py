"""Run Tribal Village behavior sanity rollouts for baseline and checkpoint policies."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_experiments.train_canonical_reward_geometry import (  # noqa: E402
    ActorCritic,
    MockCanonicalTribalEnv,
    TribalVillageAdapter,
    _agent_ids,
    _env_metadata,
    _git_sha,
)
from v3_experiments.tribal_behavior import (  # noqa: E402
    BEHAVIOR_SCHEMA_VERSION,
    aggregate_episode_metrics,
    summarize_behavior_rollout,
    validate_behavior_record,
    write_json,
)


@dataclass
class LoadedCheckpointPolicy:
    policy: ActorCritic
    separate_encoders: bool


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_rollouts(args, argv if argv is not None else sys.argv[1:])
    output_path = Path(args.output_dir) / "rollout_metrics.json"
    write_json(output_path, record)
    issues = validate_behavior_record(record)
    if issues:
        raise RuntimeError("invalid behavior rollout record: " + "; ".join(issues))
    print(json.dumps({"output": str(output_path), "episodes": args.episodes}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("no_op", "random", "move_sweep", "use_sweep", "checkpoint"), required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--env-backend", choices=("tribal", "mock"), default="tribal")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample checkpoint actions instead of argmax.")
    parser.add_argument("--allow-noncanonical-env", action="store_true")
    return parser.parse_args(argv)


def run_rollouts(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.policy == "checkpoint" and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required for --policy checkpoint")

    env = _make_behavior_env(args)
    try:
        checkpoint_policy = _load_checkpoint_policy(args, env) if args.policy == "checkpoint" else None
        rng = np.random.default_rng(args.seed)
        output_dir = Path(args.output_dir)
        replay_dir = output_dir / "replays"
        episode_records = []
        replay_paths = []

        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            episode_record, replay_path = _run_episode(
                env,
                args,
                episode=episode,
                seed=episode_seed,
                rng=rng,
                checkpoint_policy=checkpoint_policy,
                replay_dir=replay_dir,
            )
            episode_records.append(episode_record)
            if replay_path is not None:
                replay_paths.append(str(replay_path))

        record = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "policy": args.policy,
            "checkpoint_path": args.checkpoint_path,
            "environment_backend": args.env_backend,
            "seed": args.seed,
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
            "metta_git_sha": _git_sha(REPO_ROOT),
            "tribal_village_git_sha": _git_sha(REPO_ROOT / "packages" / "tribal_village"),
            "command": "uv run python v3_experiments/run_tribal_behavior_rollouts.py "
            + " ".join(shlex.quote(arg) for arg in argv),
            "env_contract": _env_metadata(env),
            "aggregate_behavior_metrics": aggregate_episode_metrics(episode_records),
            "episode_metrics": episode_records,
            "replay_paths": replay_paths,
        }
        return record
    finally:
        env.close()


def _make_behavior_env(args: argparse.Namespace):
    if args.env_backend == "mock":
        return MockCanonicalTribalEnv(args.steps, gold_layer=11, altar_layer=16)
    env = TribalVillageAdapter(args.steps)
    if not args.allow_noncanonical_env:
        mismatches = {
            "num_agents": env.num_agents != 12,
            "action_space_size": env.action_space_size != 56,
            "map_width": env.map_width != 80,
            "map_height": env.map_height != 80,
        }
        bad = {key: getattr(env, key) for key, failed in mismatches.items() if failed}
        if bad:
            raise RuntimeError(f"noncanonical Tribal environment for behavior gate: {bad}")
    return env


def _run_episode(
    env: Any,
    args: argparse.Namespace,
    *,
    episode: int,
    seed: int,
    rng: np.random.Generator,
    checkpoint_policy: LoadedCheckpointPolicy | None,
    replay_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    obs = env.reset(seed=seed)
    actions_by_step: list[np.ndarray] = []
    rewards_by_step: list[np.ndarray] = []
    replay_path = replay_dir / f"episode_{episode:03d}.jsonl" if args.save_replays else None
    replay_handle = None
    if replay_path is not None:
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_handle = replay_path.open("w")

    try:
        for step in range(args.steps):
            actions = _policy_actions(args, env, obs, step, rng, checkpoint_policy)
            next_obs, rewards, done = env.step(actions)
            actions_by_step.append(actions.astype(np.int64, copy=True))
            rewards_by_step.append(np.asarray(rewards, dtype=np.float64).copy())

            if replay_handle is not None:
                frame: dict[str, Any] = {
                    "episode": episode,
                    "step": step,
                    "actions": actions.astype(int).tolist(),
                    "rewards": np.asarray(rewards, dtype=float).tolist(),
                    "done": bool(done),
                }
                if args.render_every > 0 and step % args.render_every == 0:
                    frame["render"] = _render_env(env)
                replay_handle.write(json.dumps(frame, sort_keys=True) + "\n")

            obs = next_obs
            if done:
                break
    finally:
        if replay_handle is not None:
            replay_handle.close()

    action_arr = np.stack(actions_by_step) if actions_by_step else np.zeros((0, env.num_agents), dtype=np.int64)
    reward_arr = np.stack(rewards_by_step) if rewards_by_step else np.zeros((0, env.num_agents), dtype=np.float64)
    metrics = summarize_behavior_rollout(
        action_arr,
        reward_arr,
        action_space_size=env.action_space_size,
        simulator_action_stats=env.get_action_stats(),
    )
    return {
        "episode": episode,
        "seed": seed,
        "behavior_metrics": metrics,
        "replay_path": str(replay_path) if replay_path is not None else None,
    }, replay_path


def _policy_actions(
    args: argparse.Namespace,
    env: Any,
    obs: np.ndarray,
    step: int,
    rng: np.random.Generator,
    checkpoint_policy: LoadedCheckpointPolicy | None,
) -> np.ndarray:
    if args.policy == "no_op":
        return np.zeros(env.num_agents, dtype=np.int64)
    if args.policy == "random":
        return rng.integers(0, env.action_space_size, size=env.num_agents, dtype=np.int64)
    if args.policy == "move_sweep":
        return np.array([8 + ((step + agent_id) % 8) for agent_id in range(env.num_agents)], dtype=np.int64)
    if args.policy == "use_sweep":
        return np.array([24 + ((step + agent_id) % 8) for agent_id in range(env.num_agents)], dtype=np.int64)
    if args.policy == "checkpoint":
        if checkpoint_policy is None:
            raise ValueError("checkpoint policy was not loaded")
        device = next(checkpoint_policy.policy.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _logprob, _entropy, _value, _emb, _logit = checkpoint_policy.policy.get_action_and_value(
                obs_t,
                agent_ids=_agent_ids(env.num_agents, device) if checkpoint_policy.separate_encoders else None,
                deterministic=not args.stochastic,
            )
        return actions.cpu().numpy().astype(np.int64)
    raise ValueError(f"unknown policy {args.policy}")


def _load_checkpoint_policy(args: argparse.Namespace, env: Any) -> LoadedCheckpointPolicy:
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, args.device)
    config = checkpoint.get("config", {})
    env_record = checkpoint.get("env", {})
    checkpoint_action_space = int(env_record.get("action_space_size", env.action_space_size))
    checkpoint_num_agents = int(env_record.get("num_agents", env.num_agents))
    if checkpoint_action_space != env.action_space_size or checkpoint_num_agents != env.num_agents:
        raise ValueError(
            "checkpoint environment contract does not match rollout environment: "
            f"checkpoint agents/actions=({checkpoint_num_agents}, {checkpoint_action_space}), "
            f"env agents/actions=({env.num_agents}, {env.action_space_size})"
        )

    separate_encoders = bool(config.get("separate_encoders", False))
    policy = ActorCritic(
        env.obs_shape,
        env.action_space_size,
        n_agents=env.num_agents,
        hidden_dim=int(config.get("hidden_dim", 256)),
        embedding_dim=int(config.get("embedding_dim", 64)),
        separate_encoders=separate_encoders,
    ).to(args.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return LoadedCheckpointPolicy(policy=policy, separate_encoders=separate_encoders)


def _torch_load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _render_env(env: Any) -> str | None:
    render = getattr(env, "render", None)
    if callable(render):
        return str(render())
    wrapped = getattr(env, "_env", None)
    render = getattr(wrapped, "render", None)
    if callable(render):
        return str(render())
    return None


if __name__ == "__main__":
    raise SystemExit(main())
