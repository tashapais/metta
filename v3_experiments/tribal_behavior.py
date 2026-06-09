"""Behavior metrics for Tribal Village policy sanity checks."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ACTION_ARGUMENT_COUNT = 8
SIMULATOR_STAT_COLUMNS = (
    "action_invalid",
    "action_noop",
    "action_move",
    "action_attack",
    "action_use",
    "action_swap",
    "action_put",
    "action_plant",
    "resource_water",
    "resource_wheat",
    "resource_wood",
    "resource_ore",
    "craft_battery",
    "craft_spear",
    "craft_lantern",
    "craft_armor",
    "craft_bread",
    "deposit_heart",
    "put_armor",
    "put_bread",
    "tumor_kill",
    "spawner_kill",
    "agent_kill",
    "death",
    "respawn",
    "lantern_plant",
    "put_ore",
    "put_battery",
)
ACTION_STAT_COLUMNS = SIMULATOR_STAT_COLUMNS
INVENTORY_COLUMNS = (
    "ore",
    "battery",
    "water",
    "wheat",
    "wood",
    "spear",
    "lantern",
    "armor",
    "bread",
)
WORLD_STAT_COLUMNS = (
    "current_step",
    "live_agents",
    "dead_agents",
    "assemblers",
    "assembler_hearts",
    "mines",
    "converters",
    "spawners",
    "tumors",
    "planted_lanterns",
    "production_buildings",
    "things",
)
ACTION_STAT_TO_VERB = {
    "action_noop": "noop",
    "action_move": "move",
    "action_attack": "attack",
    "action_use": "use",
    "action_swap": "swap",
    "action_put": "put",
    "action_plant": "plant",
}
BASE_VERB_NAMES = (
    "noop",
    "move",
    "attack",
    "use",
    "swap",
    "put",
    "plant",
    "plant_resource",
)

BEHAVIOR_SCHEMA_VERSION = "tribal_behavior_rollout_v1"


def action_verb_names(action_space_size: int, *, argument_count: int = ACTION_ARGUMENT_COUNT) -> list[str]:
    verb_count = max(1, int(math.ceil(action_space_size / argument_count)))
    names = list(BASE_VERB_NAMES[:verb_count])
    while len(names) < verb_count:
        names.append(f"verb_{len(names)}")
    return names


def decode_action_verbs(actions: np.ndarray, *, argument_count: int = ACTION_ARGUMENT_COUNT) -> np.ndarray:
    return np.asarray(actions, dtype=np.int64) // argument_count


def summarize_behavior_rollout(
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    action_space_size: int,
    simulator_action_stats: np.ndarray | None = None,
    inventory_initial: np.ndarray | None = None,
    inventory_final: np.ndarray | None = None,
    world_stats_initial: np.ndarray | None = None,
    world_stats_final: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize a rollout from action and reward arrays.

    ``actions`` is shaped ``[steps, agents]`` and ``rewards`` is shaped
    ``[steps, agents]``. Simulator stats are optional cumulative per-agent action
    counters from the Nim environment, shaped ``[agents, 8]``.
    """

    actions_arr = np.asarray(actions, dtype=np.int64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    if actions_arr.ndim != 2:
        raise ValueError("actions must have shape [steps, agents]")
    if rewards_arr.shape != actions_arr.shape:
        raise ValueError("rewards must have the same shape as actions")

    steps, num_agents = actions_arr.shape
    verb_names = action_verb_names(action_space_size)
    verbs = decode_action_verbs(actions_arr)
    flat_actions = actions_arr.reshape(-1)
    flat_verbs = verbs.reshape(-1)
    total_action_count = int(flat_actions.size)

    action_counts = _count_named_values(flat_actions, [str(idx) for idx in range(action_space_size)])
    verb_counts = _count_named_values(flat_verbs, verb_names)
    joint_actions = [tuple(int(value) for value in row) for row in actions_arr]
    repeated_joint_streak_max = _max_repeated_streak(joint_actions)
    action_entropy = _entropy_from_counts(Counter(int(value) for value in flat_actions))
    verb_entropy = _entropy_from_counts(Counter(int(value) for value in flat_verbs))
    stats_summary = _summarize_simulator_action_stats(simulator_action_stats, verb_names)
    inventory_summary = _summarize_inventory(inventory_initial, inventory_final)
    world_summary = _summarize_world_stats(world_stats_initial, world_stats_final)

    return {
        "steps": int(steps),
        "num_agents": int(num_agents),
        "action_space_size": int(action_space_size),
        "action_argument_count": ACTION_ARGUMENT_COUNT,
        "action_verb_names": verb_names,
        "total_action_count": total_action_count,
        "unique_action_count": int(np.unique(flat_actions).size),
        "unique_joint_action_count": len(set(joint_actions)),
        "repeated_joint_action_streak_max": repeated_joint_streak_max,
        "action_entropy": action_entropy,
        "verb_entropy": verb_entropy,
        "noop_fraction": float(np.mean(flat_actions == 0)) if total_action_count else 0.0,
        "move_verb_fraction": float(np.mean(flat_verbs == 1)) if total_action_count else 0.0,
        "non_noop_fraction": float(np.mean(flat_actions != 0)) if total_action_count else 0.0,
        "action_counts": action_counts,
        "action_attempts_by_verb": verb_counts,
        "raw_env_reward_total": float(rewards_arr.sum()),
        "raw_env_reward_mean_per_agent": float(rewards_arr.sum(axis=0).mean()) if steps else 0.0,
        "raw_env_reward_mean_per_step": float(rewards_arr.sum(axis=1).mean()) if steps else 0.0,
        "raw_env_reward_positive_count": int(np.sum(rewards_arr > 0.0)),
        "raw_env_reward_negative_count": int(np.sum(rewards_arr < 0.0)),
        "simulator_action_stats_available": simulator_action_stats is not None,
        "inventory_snapshot_available": inventory_final is not None,
        "world_stats_available": world_stats_final is not None,
        **stats_summary,
        **inventory_summary,
        **world_summary,
    }


def aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")

    metric_keys = (
        "raw_env_reward_total",
        "raw_env_reward_mean_per_agent",
        "raw_env_reward_mean_per_step",
        "unique_action_count",
        "unique_joint_action_count",
        "repeated_joint_action_streak_max",
        "action_entropy",
        "verb_entropy",
        "noop_fraction",
        "move_verb_fraction",
        "non_noop_fraction",
        "task_event_count",
    )
    aggregate = {"episodes": len(episodes)}
    for key in metric_keys:
        values = np.asarray([float(ep["behavior_metrics"][key]) for ep in episodes], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(values.mean())
        aggregate[f"{key}_min"] = float(values.min())
        aggregate[f"{key}_max"] = float(values.max())

    aggregate["simulator_action_stats_available"] = all(
        bool(ep["behavior_metrics"].get("simulator_action_stats_available")) for ep in episodes
    )
    aggregate["action_attempts_by_verb_total"] = _sum_counter_dicts(
        ep["behavior_metrics"]["action_attempts_by_verb"] for ep in episodes
    )
    aggregate["action_successes_by_verb_total"] = _sum_counter_dicts(
        ep["behavior_metrics"]["action_successes_by_verb"] for ep in episodes
    )
    aggregate["invalid_actions_by_verb_total"] = _sum_counter_dicts(
        ep["behavior_metrics"]["invalid_actions_by_verb"] for ep in episodes
    )
    for key in (
        "resource_pickups_by_type",
        "crafting_outputs_by_type",
        "deposits_by_type",
        "handoffs_by_type",
        "combat_events",
        "lifecycle_events",
        "inventory_delta_by_type",
        "inventory_final_totals_by_type",
    ):
        aggregate[f"{key}_total"] = _sum_counter_dicts(ep["behavior_metrics"][key] for ep in episodes)
    return aggregate


def validate_behavior_record(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required_top = {
        "schema_version",
        "policy",
        "environment_backend",
        "seed",
        "episodes",
        "steps_per_episode",
        "env_contract",
        "aggregate_behavior_metrics",
        "episode_metrics",
    }
    for key in sorted(required_top):
        if key not in record:
            issues.append(f"missing top-level field: {key}")

    if record.get("schema_version") != BEHAVIOR_SCHEMA_VERSION:
        issues.append(f"schema_version must be {BEHAVIOR_SCHEMA_VERSION!r}")

    episode_metrics = record.get("episode_metrics", [])
    if not isinstance(episode_metrics, list) or not episode_metrics:
        issues.append("episode_metrics must be a non-empty list")
    else:
        for index, episode in enumerate(episode_metrics):
            metrics = episode.get("behavior_metrics") if isinstance(episode, dict) else None
            if not isinstance(metrics, dict):
                issues.append(f"episode_metrics[{index}].behavior_metrics missing")
                continue
            for key in (
                "steps",
                "num_agents",
                "action_space_size",
                "unique_action_count",
                "unique_joint_action_count",
                "action_attempts_by_verb",
                "raw_env_reward_total",
                "simulator_action_stats_available",
                "task_event_count",
                "resource_pickups_by_type",
                "crafting_outputs_by_type",
                "inventory_final_totals_by_type",
                "world_stats_final",
            ):
                if key not in metrics:
                    issues.append(f"episode_metrics[{index}].behavior_metrics missing {key}")
            for key, value in metrics.items():
                if isinstance(value, float) and not math.isfinite(value):
                    issues.append(f"episode_metrics[{index}].behavior_metrics.{key} is not finite")

    return issues


def write_json(path: Path, record: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def inventory_snapshot_to_dict(snapshot: np.ndarray | None) -> list[dict[str, int]] | None:
    if snapshot is None:
        return None
    snapshot_arr = np.asarray(snapshot, dtype=np.int64)
    if snapshot_arr.ndim != 2 or snapshot_arr.shape[1] != len(INVENTORY_COLUMNS):
        raise ValueError("inventory snapshot must have shape [agents, 9]")
    return [{name: int(row[index]) for index, name in enumerate(INVENTORY_COLUMNS)} for row in snapshot_arr]


def world_stats_to_dict(snapshot: np.ndarray | None) -> dict[str, int] | None:
    if snapshot is None:
        return None
    snapshot_arr = np.asarray(snapshot, dtype=np.int64).reshape(-1)
    if snapshot_arr.shape[0] != len(WORLD_STAT_COLUMNS):
        raise ValueError("world stats snapshot must have shape [12]")
    return {name: int(snapshot_arr[index]) for index, name in enumerate(WORLD_STAT_COLUMNS)}


def _summarize_simulator_action_stats(stats: np.ndarray | None, verb_names: list[str]) -> dict[str, Any]:
    zero_by_verb = {name: 0 for name in verb_names}
    if stats is None:
        return {
            "action_successes_by_verb": dict(zero_by_verb),
            "invalid_actions_by_verb": {"invalid": 0},
            "simulator_action_stats_by_agent": None,
            "resource_pickups_by_type": _zero_keys("water", "wheat", "wood", "ore"),
            "crafting_outputs_by_type": _zero_keys("battery", "spear", "lantern", "armor", "bread"),
            "deposits_by_type": _zero_keys("heart"),
            "handoffs_by_type": _zero_keys("ore", "battery", "armor", "bread"),
            "combat_events": _zero_keys("tumor_kill", "spawner_kill", "agent_kill"),
            "lifecycle_events": _zero_keys("death", "respawn", "lantern_plant"),
            "task_event_count": 0,
        }

    stats_arr = np.asarray(stats, dtype=np.int64)
    if stats_arr.ndim != 2 or stats_arr.shape[1] != len(SIMULATOR_STAT_COLUMNS):
        raise ValueError(f"simulator_action_stats must have shape [agents, {len(SIMULATOR_STAT_COLUMNS)}]")

    totals = stats_arr.sum(axis=0)
    successes = dict(zero_by_verb)
    invalids = {"invalid": 0}
    totals_by_column = {name: int(totals[index]) for index, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    for stat_name, verb_name in ACTION_STAT_TO_VERB.items():
        if verb_name in successes:
            successes[verb_name] = totals_by_column[stat_name]
    invalids["invalid"] = totals_by_column["action_invalid"]

    resources = {
        "water": totals_by_column["resource_water"],
        "wheat": totals_by_column["resource_wheat"],
        "wood": totals_by_column["resource_wood"],
        "ore": totals_by_column["resource_ore"],
    }
    crafting = {
        "battery": totals_by_column["craft_battery"],
        "spear": totals_by_column["craft_spear"],
        "lantern": totals_by_column["craft_lantern"],
        "armor": totals_by_column["craft_armor"],
        "bread": totals_by_column["craft_bread"],
    }
    deposits = {"heart": totals_by_column["deposit_heart"]}
    handoffs = {
        "ore": totals_by_column["put_ore"],
        "battery": totals_by_column["put_battery"],
        "armor": totals_by_column["put_armor"],
        "bread": totals_by_column["put_bread"],
    }
    combat = {
        "tumor_kill": totals_by_column["tumor_kill"],
        "spawner_kill": totals_by_column["spawner_kill"],
        "agent_kill": totals_by_column["agent_kill"],
    }
    lifecycle = {
        "death": totals_by_column["death"],
        "respawn": totals_by_column["respawn"],
        "lantern_plant": totals_by_column["lantern_plant"],
    }
    task_event_count = sum(resources.values()) + sum(crafting.values()) + sum(deposits.values())
    task_event_count += sum(handoffs.values()) + sum(combat.values()) + lifecycle["lantern_plant"]

    return {
        "action_successes_by_verb": successes,
        "invalid_actions_by_verb": invalids,
        "resource_pickups_by_type": resources,
        "crafting_outputs_by_type": crafting,
        "deposits_by_type": deposits,
        "handoffs_by_type": handoffs,
        "combat_events": combat,
        "lifecycle_events": lifecycle,
        "task_event_count": int(task_event_count),
        "simulator_action_stats_by_agent": [
            {name: int(row[index]) for index, name in enumerate(SIMULATOR_STAT_COLUMNS)} for row in stats_arr
        ],
    }


def _summarize_inventory(initial: np.ndarray | None, final: np.ndarray | None) -> dict[str, Any]:
    final_by_agent = inventory_snapshot_to_dict(final)
    if final is None:
        return {
            "inventory_initial_by_agent": None,
            "inventory_final_by_agent": None,
            "inventory_initial_totals_by_type": _zero_keys(*INVENTORY_COLUMNS),
            "inventory_final_totals_by_type": _zero_keys(*INVENTORY_COLUMNS),
            "inventory_delta_by_type": _zero_keys(*INVENTORY_COLUMNS),
        }

    final_arr = np.asarray(final, dtype=np.int64)
    if initial is None:
        initial_arr = np.zeros_like(final_arr)
    else:
        initial_arr = np.asarray(initial, dtype=np.int64)
        if initial_arr.shape != final_arr.shape:
            raise ValueError("initial and final inventory snapshots must have the same shape")

    initial_totals = {name: int(initial_arr[:, index].sum()) for index, name in enumerate(INVENTORY_COLUMNS)}
    final_totals = {name: int(final_arr[:, index].sum()) for index, name in enumerate(INVENTORY_COLUMNS)}
    deltas = {name: final_totals[name] - initial_totals[name] for name in INVENTORY_COLUMNS}
    return {
        "inventory_initial_by_agent": inventory_snapshot_to_dict(initial),
        "inventory_final_by_agent": final_by_agent,
        "inventory_initial_totals_by_type": initial_totals,
        "inventory_final_totals_by_type": final_totals,
        "inventory_delta_by_type": deltas,
    }


def _summarize_world_stats(initial: np.ndarray | None, final: np.ndarray | None) -> dict[str, Any]:
    initial_dict = world_stats_to_dict(initial)
    final_dict = world_stats_to_dict(final)
    if final_dict is None:
        return {
            "world_stats_initial": initial_dict,
            "world_stats_final": None,
            "world_stats_delta": {},
        }
    initial_for_delta = initial_dict or {key: 0 for key in final_dict}
    return {
        "world_stats_initial": initial_dict,
        "world_stats_final": final_dict,
        "world_stats_delta": {key: final_dict[key] - int(initial_for_delta.get(key, 0)) for key in final_dict},
    }


def _count_named_values(values: np.ndarray, names: list[str]) -> dict[str, int]:
    counter = Counter(int(value) for value in values)
    out: dict[str, int] = {}
    for index, name in enumerate(names):
        out[name] = int(counter.get(index, 0))
    for value, count in sorted(counter.items()):
        if value < 0 or value >= len(names):
            out[str(value)] = int(count)
    return out


def _entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def _max_repeated_streak(values: list[tuple[int, ...]]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    previous = values[0]
    for value in values[1:]:
        if value == previous:
            current += 1
        else:
            best = max(best, current)
            current = 1
            previous = value
    return max(best, current)


def _sum_counter_dicts(dicts: Any) -> dict[str, int]:
    total: Counter[str] = Counter()
    for item in dicts:
        total.update({str(key): int(value) for key, value in item.items()})
    return dict(sorted(total.items()))


def _zero_keys(*keys: str) -> dict[str, int]:
    return {key: 0 for key in keys}
