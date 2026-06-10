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
    _action_mask_array_from_flags,
    _agent_ids,
    _chain_affordance_extra_verb_names_from_value,
    _env_metadata,
    _git_sha,
)
from v3_experiments.tribal_behavior import (  # noqa: E402
    BEHAVIOR_SCHEMA_VERSION,
    aggregate_episode_metrics,
    inventory_snapshot_to_dict,
    summarize_behavior_rollout,
    validate_behavior_record,
    world_stats_to_dict,
    write_json,
)
from v3_experiments.tribal_event_rewards import (  # noqa: E402
    NAV_AGENT_X,
    NAV_AGENT_Y,
    NAV_DIST_HOME_ASSEMBLER,
    NAV_HOME_ASSEMBLER_X,
    NAV_HOME_ASSEMBLER_Y,
    NAV_INVENTORY_BATTERY,
    NAV_INVENTORY_ORE,
    NAV_NEAREST_CONVERTER_X,
    NAV_NEAREST_CONVERTER_Y,
    NAV_NEAREST_MINE_X,
    NAV_NEAREST_MINE_Y,
)

ACTION_ARGUMENT_COUNT = 8
MOVE_VERB = 1
USE_VERB = 3
ORIENTATION_DELTAS = (
    (0, -1),  # N
    (0, 1),  # S
    (-1, 0),  # W
    (1, 0),  # E
    (-1, -1),  # NW
    (1, -1),  # NE
    (-1, 1),  # SW
    (1, 1),  # SE
)
ORIENTATION_BY_DELTA = {delta: index for index, delta in enumerate(ORIENTATION_DELTAS)}


@dataclass
class LoadedCheckpointPolicy:
    policy: ActorCritic
    separate_encoders: bool
    use_action_mask: bool
    chain_affordance_action_mask: bool
    role_gated_chain_mask: bool
    target_aware_handoff_mask: bool
    chain_affordance_extra_verbs: tuple[str, ...]


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
    parser.add_argument(
        "--policy",
        choices=("no_op", "random", "move_sweep", "use_sweep", "chain_oracle", "checkpoint"),
        required=True,
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--env-backend", choices=("tribal", "mock"), default="tribal")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample checkpoint actions instead of argmax.")
    parser.add_argument(
        "--disable-checkpoint-chain-affordance-action-mask",
        action="store_true",
        help=(
            "When evaluating a checkpoint, ignore its strict v10 chain-affordance "
            "mask and use the environment action mask instead."
        ),
    )
    parser.add_argument(
        "--chain-compass-observation",
        action="store_true",
        help="Use the v7 chain-compass observation planes when loading/evaluating checkpoints.",
    )
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
            "disable_checkpoint_chain_affordance_action_mask": args.disable_checkpoint_chain_affordance_action_mask,
            "checkpoint_uses_action_mask": None if checkpoint_policy is None else checkpoint_policy.use_action_mask,
            "checkpoint_chain_affordance_action_mask": (
                None if checkpoint_policy is None else checkpoint_policy.chain_affordance_action_mask
            ),
            "checkpoint_role_gated_chain_mask": (
                None if checkpoint_policy is None else checkpoint_policy.role_gated_chain_mask
            ),
            "checkpoint_target_aware_handoff_mask": (
                None if checkpoint_policy is None else checkpoint_policy.target_aware_handoff_mask
            ),
            "checkpoint_chain_affordance_extra_verbs": (
                None if checkpoint_policy is None else list(checkpoint_policy.chain_affordance_extra_verbs)
            ),
            "environment_backend": args.env_backend,
            "chain_compass_observation": args.chain_compass_observation,
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
        return MockCanonicalTribalEnv(
            args.steps,
            gold_layer=11,
            altar_layer=16,
            chain_compass_observation=args.chain_compass_observation,
        )
    env = TribalVillageAdapter(args.steps, chain_compass_observation=args.chain_compass_observation)
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
    inventory_initial = env.get_inventory_snapshot()
    world_stats_initial = env.get_world_stats()
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
                if args.snapshot_every > 0 and step % args.snapshot_every == 0:
                    frame["inventory"] = inventory_snapshot_to_dict(env.get_inventory_snapshot())
                    frame["world_stats"] = world_stats_to_dict(env.get_world_stats())
                    frame["navigation"] = _navigation_snapshot_to_list(env)
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
        inventory_initial=inventory_initial,
        inventory_final=env.get_inventory_snapshot(),
        world_stats_initial=world_stats_initial,
        world_stats_final=env.get_world_stats(),
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
    if args.policy == "chain_oracle":
        return _chain_oracle_actions(env)
    if args.policy == "checkpoint":
        if checkpoint_policy is None:
            raise ValueError("checkpoint policy was not loaded")
        device = next(checkpoint_policy.policy.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _logprob, _entropy, _value, _emb, _logit = checkpoint_policy.policy.get_action_and_value(
                obs_t,
                agent_ids=_agent_ids(env.num_agents, device) if checkpoint_policy.separate_encoders else None,
                action_mask=_checkpoint_action_mask(env, checkpoint_policy, device),
                deterministic=not args.stochastic,
            )
        return actions.cpu().numpy().astype(np.int64)
    raise ValueError(f"unknown policy {args.policy}")


def _chain_oracle_actions(env: Any) -> np.ndarray:
    """Follow the ore -> battery -> home-assembler chain from debug snapshots."""

    get_snapshot = getattr(env, "get_navigation_snapshot", None)
    if get_snapshot is None:
        return np.zeros(env.num_agents, dtype=np.int64)
    snapshot = get_snapshot()
    if snapshot is None:
        return np.zeros(env.num_agents, dtype=np.int64)

    mask = _optional_action_mask(env)
    navigation = np.asarray(snapshot, dtype=np.int64)
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for agent_id, row in enumerate(navigation):
        mask_row = None if mask is None else mask[agent_id]
        actions[agent_id] = _chain_oracle_action_for_agent(row, mask_row)
    return actions


def _chain_oracle_action_for_agent(
    navigation_row: np.ndarray,
    action_mask: np.ndarray | None = None,
) -> int:
    target = _chain_target(navigation_row)
    if target is None:
        return _masked_noop_or_first_valid(action_mask)

    agent_x = int(navigation_row[NAV_AGENT_X])
    agent_y = int(navigation_row[NAV_AGENT_Y])
    target_x, target_y = target
    dx = _sign(target_x - agent_x)
    dy = _sign(target_y - agent_y)
    if dx == 0 and dy == 0:
        return _masked_noop_or_first_valid(action_mask)

    if max(abs(target_x - agent_x), abs(target_y - agent_y)) == 1:
        action = _encode_action(USE_VERB, ORIENTATION_BY_DELTA[(dx, dy)])
        if action_mask is None or bool(action_mask[action]):
            return action
        return _masked_noop_or_first_valid(action_mask)

    preferred = _best_masked_move_toward(agent_x, agent_y, target_x, target_y, action_mask)
    if preferred is not None:
        return preferred
    return _encode_action(MOVE_VERB, ORIENTATION_BY_DELTA[(dx, dy)])


def _chain_target(navigation_row: np.ndarray) -> tuple[int, int] | None:
    if int(navigation_row[NAV_INVENTORY_BATTERY]) > 0:
        return _target_if_valid(navigation_row, NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER)
    if int(navigation_row[NAV_INVENTORY_ORE]) > 0:
        return _target_if_valid(navigation_row, NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y)
    return _target_if_valid(navigation_row, NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y)


def _target_if_valid(
    navigation_row: np.ndarray,
    x_index: int,
    y_index: int,
    distance_index: int | None = None,
) -> tuple[int, int] | None:
    target_x = int(navigation_row[x_index])
    target_y = int(navigation_row[y_index])
    if target_x < 0 or target_y < 0:
        return None
    if distance_index is not None and int(navigation_row[distance_index]) < 0:
        return None
    return target_x, target_y


def _best_masked_move_toward(
    agent_x: int,
    agent_y: int,
    target_x: int,
    target_y: int,
    action_mask: np.ndarray | None,
) -> int | None:
    if action_mask is None:
        return None

    best_action = None
    best_distance = None
    for orientation, (delta_x, delta_y) in enumerate(ORIENTATION_DELTAS):
        action = _encode_action(MOVE_VERB, orientation)
        if not bool(action_mask[action]):
            continue
        distance = abs(target_x - (agent_x + delta_x)) + abs(target_y - (agent_y + delta_y))
        if best_distance is None or distance < best_distance:
            best_action = action
            best_distance = distance
    return best_action


def _optional_action_mask(env: Any) -> np.ndarray | None:
    get_mask = getattr(env, "get_action_mask", None)
    if get_mask is None:
        return None
    mask = get_mask()
    if mask is None:
        return None
    return np.asarray(mask, dtype=bool)


def _masked_noop_or_first_valid(action_mask: np.ndarray | None) -> int:
    if action_mask is None or bool(action_mask[0]):
        return 0
    valid = np.flatnonzero(action_mask)
    return int(valid[0]) if valid.size else 0


def _encode_action(verb: int, argument: int) -> int:
    return int(verb * ACTION_ARGUMENT_COUNT + argument)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _load_checkpoint_policy(args: argparse.Namespace, env: Any) -> LoadedCheckpointPolicy:
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, args.device)
    config = checkpoint.get("config", {})
    env_record = checkpoint.get("env", {})
    checkpoint_action_space = int(env_record.get("action_space_size", env.action_space_size))
    checkpoint_num_agents = int(env_record.get("num_agents", env.num_agents))
    checkpoint_obs_shape = tuple(int(x) for x in env_record.get("obs_shape", env.obs_shape))
    if (
        checkpoint_action_space != env.action_space_size
        or checkpoint_num_agents != env.num_agents
        or checkpoint_obs_shape != tuple(env.obs_shape)
    ):
        raise ValueError(
            "checkpoint environment contract does not match rollout environment: "
            f"checkpoint agents/actions=({checkpoint_num_agents}, {checkpoint_action_space}), "
            f"checkpoint obs_shape={checkpoint_obs_shape}, "
            f"env agents/actions=({env.num_agents}, {env.action_space_size}), "
            f"env obs_shape={env.obs_shape}"
        )

    separate_encoders = bool(config.get("separate_encoders", False))
    use_action_mask = bool(config.get("use_action_mask", False))
    chain_affordance_action_mask = _checkpoint_uses_chain_affordance_action_mask(
        config,
        disable_override=args.disable_checkpoint_chain_affordance_action_mask,
    )
    role_gated_chain_mask = config.get("reward_design") in (
        "event_v11_role_gated_chain_handoffs",
        "event_v12_role_gated_depositor_reliability",
        "event_v13_role_gated_depositor_use",
        "event_v14_target_aware_handoffs",
        "event_v15_depositor_final_mile",
    )
    target_aware_handoff_mask = bool(config.get("target_aware_handoff_mask", False)) or (
        config.get("reward_design")
        in ("event_v14_target_aware_handoffs", "event_v15_depositor_final_mile")
        and chain_affordance_action_mask
    )
    chain_affordance_extra_verbs = (
        ()
        if not chain_affordance_action_mask
        else _chain_affordance_extra_verb_names_from_value(config.get("chain_affordance_extra_verbs", ""))
    )
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
    return LoadedCheckpointPolicy(
        policy=policy,
        separate_encoders=separate_encoders,
        use_action_mask=use_action_mask,
        chain_affordance_action_mask=chain_affordance_action_mask,
        role_gated_chain_mask=role_gated_chain_mask,
        target_aware_handoff_mask=target_aware_handoff_mask,
        chain_affordance_extra_verbs=chain_affordance_extra_verbs,
    )


def _checkpoint_uses_chain_affordance_action_mask(
    config: dict[str, Any],
    *,
    disable_override: bool = False,
) -> bool:
    if disable_override:
        return False
    return bool(config.get("chain_affordance_action_mask", False)) or (
        config.get("reward_design")
        in (
            "event_v10_chain_affordance_compass_breadcrumbs",
            "event_v11_role_gated_chain_handoffs",
            "event_v12_role_gated_depositor_reliability",
            "event_v13_role_gated_depositor_use",
            "event_v14_target_aware_handoffs",
            "event_v15_depositor_final_mile",
        )
    )


def _checkpoint_action_mask(
    env: Any,
    checkpoint_policy: LoadedCheckpointPolicy,
    device: torch.device,
) -> torch.Tensor | None:
    if not checkpoint_policy.use_action_mask:
        if checkpoint_policy.chain_affordance_action_mask:
            mask_arr = _action_mask_array_from_flags(
                env,
                use_action_mask=False,
                chain_affordance_action_mask=True,
                chain_affordance_extra_verbs=checkpoint_policy.chain_affordance_extra_verbs,
                role_gated_chain_mask=checkpoint_policy.role_gated_chain_mask,
                target_aware_handoff_mask=checkpoint_policy.target_aware_handoff_mask,
            )
            return None if mask_arr is None else torch.as_tensor(mask_arr, dtype=torch.bool, device=device)
        return None
    mask_arr = _action_mask_array_from_flags(
        env,
        use_action_mask=True,
        chain_affordance_action_mask=checkpoint_policy.chain_affordance_action_mask,
        chain_affordance_extra_verbs=checkpoint_policy.chain_affordance_extra_verbs,
        role_gated_chain_mask=checkpoint_policy.role_gated_chain_mask,
        target_aware_handoff_mask=checkpoint_policy.target_aware_handoff_mask,
    )
    if mask_arr is None:
        raise RuntimeError("checkpoint was trained with action masks but rollout environment returned no mask")
    return torch.as_tensor(mask_arr, dtype=torch.bool, device=device)


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


def _navigation_snapshot_to_list(env: Any) -> list[list[int]] | None:
    get_snapshot = getattr(env, "get_navigation_snapshot", None)
    if get_snapshot is None:
        return None
    snapshot = get_snapshot()
    if snapshot is None:
        return None
    return np.asarray(snapshot, dtype=np.int64).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
