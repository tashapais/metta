import json

import numpy as np
import pytest

from v3_experiments.audit_tribal_versions import audit_package_contract, compare_contracts
from v3_experiments.run_tribal_behavior_rollouts import _chain_oracle_action_for_agent
from v3_experiments.run_tribal_behavior_rollouts import main as rollout_main
from v3_experiments.summarize_tribal_behavior_outputs import summarize_record
from v3_experiments.tribal_behavior import summarize_behavior_rollout, validate_behavior_record
from v3_experiments.tribal_event_rewards import (
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
    NAVIGATION_SNAPSHOT_COLUMNS,
)


def test_summarize_behavior_rollout_tracks_actions_rewards_and_simulator_stats():
    actions = np.array(
        [
            [0, 9],
            [0, 9],
            [16, 31],
        ],
        dtype=np.int64,
    )
    rewards = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.5],
            [-0.25, 0.0],
        ],
        dtype=np.float64,
    )
    simulator_stats = np.zeros((2, 26), dtype=np.int32)
    simulator_stats[0, 1] = 2  # noop
    simulator_stats[0, 3] = 1  # attack
    simulator_stats[0, 11] = 2  # ore pickups
    simulator_stats[0, 12] = 1  # battery crafts
    simulator_stats[0, 17] = 1  # heart deposits
    simulator_stats[0, 20] = 1  # tumor kills
    simulator_stats[1, 0] = 1  # invalid
    simulator_stats[1, 2] = 2  # move
    simulator_stats[1, 18] = 1  # armor handoff
    inventory_initial = np.zeros((2, 9), dtype=np.int32)
    inventory_final = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    world_initial = np.array([0, 2, 0, 1, 5, 1, 1, 1, 0, 0, 0, 10], dtype=np.int32)
    world_final = np.array([3, 2, 0, 1, 6, 1, 1, 1, 0, 0, 0, 10], dtype=np.int32)

    metrics = summarize_behavior_rollout(
        actions,
        rewards,
        action_space_size=56,
        simulator_action_stats=simulator_stats,
        inventory_initial=inventory_initial,
        inventory_final=inventory_final,
        world_stats_initial=world_initial,
        world_stats_final=world_final,
    )

    assert metrics["steps"] == 3
    assert metrics["num_agents"] == 2
    assert metrics["unique_action_count"] == 4
    assert metrics["unique_joint_action_count"] == 2
    assert metrics["repeated_joint_action_streak_max"] == 2
    assert metrics["noop_fraction"] == pytest.approx(2 / 6)
    assert metrics["raw_env_reward_total"] == pytest.approx(1.25)
    assert metrics["action_attempts_by_verb"]["noop"] == 2
    assert metrics["action_attempts_by_verb"]["move"] == 2
    assert metrics["action_attempts_by_verb"]["attack"] == 1
    assert metrics["action_attempts_by_verb"]["use"] == 1
    assert metrics["action_successes_by_verb"]["noop"] == 2
    assert metrics["action_successes_by_verb"]["move"] == 2
    assert metrics["action_successes_by_verb"]["attack"] == 1
    assert metrics["invalid_actions_by_verb"]["invalid"] == 1
    assert metrics["resource_pickups_by_type"]["ore"] == 2
    assert metrics["crafting_outputs_by_type"]["battery"] == 1
    assert metrics["deposits_by_type"]["heart"] == 1
    assert metrics["handoffs_by_type"]["armor"] == 1
    assert metrics["combat_events"]["tumor_kill"] == 1
    assert metrics["task_event_count"] == 6
    assert metrics["inventory_delta_by_type"] == {
        "armor": 1,
        "battery": 1,
        "bread": 0,
        "lantern": 0,
        "ore": 1,
        "spear": 0,
        "water": 0,
        "wheat": 0,
        "wood": 0,
    }
    assert metrics["world_stats_delta"]["assembler_hearts"] == 1


def test_audit_package_contract_detects_coworld_incompatible_shape(tmp_path):
    trained = tmp_path / "trained"
    coworld = tmp_path / "coworld"
    _write_minimal_tribal_sources(trained, action_verbs=7, houses=8, agents_per_house=6, canonical=True)
    _write_minimal_tribal_sources(coworld, action_verbs=8, houses=8, agents_per_house=6, canonical=False)
    (coworld / "tribal_village_env" / "coworld").mkdir(parents=True)
    (coworld / "tribal_village_env" / "coworld" / "server.py").write_text("")
    (coworld / "src" / "equipment.nim").write_text("")

    trained_contract = audit_package_contract(
        trained,
        label="trained",
        canonical_reward_geometry=True,
    )
    coworld_contract = audit_package_contract(
        coworld,
        label="coworld",
        canonical_reward_geometry=False,
    )
    differences = compare_contracts(trained_contract, coworld_contract)

    assert trained_contract["num_agents"] == 12
    assert trained_contract["action_space_size"] == 56
    assert coworld_contract["num_agents"] == 48
    assert coworld_contract["action_space_size"] == 64
    assert {item["field"] for item in differences} >= {
        "num_agents",
        "action_space_size",
        "has_coworld_runtime",
        "has_equipment_module",
    }


def test_run_tribal_behavior_rollouts_mock_noop_writes_valid_record(tmp_path):
    output_dir = tmp_path / "rollouts"

    assert (
        rollout_main(
            [
                "--env-backend",
                "mock",
                "--policy",
                "no_op",
                "--episodes",
                "1",
                "--steps",
                "3",
                "--seed",
                "7",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    record = json.loads((output_dir / "rollout_metrics.json").read_text())
    assert validate_behavior_record(record) == []
    assert record["policy"] == "no_op"
    assert record["env_contract"]["action_space_size"] == 56
    metrics = record["episode_metrics"][0]["behavior_metrics"]
    assert metrics["unique_action_count"] == 1
    assert metrics["repeated_joint_action_streak_max"] == 3
    assert metrics["noop_fraction"] == pytest.approx(1.0)
    assert metrics["task_event_count"] == 0
    assert metrics["world_stats_final"] is None
    summary = summarize_record(output_dir / "rollout_metrics.json", record)
    assert summary["flags"] == ["no_task_events", "single_joint_action", "mostly_noop"]


def test_chain_oracle_follows_current_inventory_target():
    row = np.zeros(len(NAVIGATION_SNAPSHOT_COLUMNS), dtype=np.int64)
    row[[NAV_AGENT_X, NAV_AGENT_Y]] = [5, 5]
    row[[NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y]] = [3, 6]
    row[[NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y]] = [2, 5]
    row[[NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y]] = [7, 4]
    row[NAV_DIST_HOME_ASSEMBLER] = 3

    assert _chain_oracle_action_for_agent(row) == 13  # move NE toward mine.

    row[NAV_INVENTORY_ORE] = 1
    assert _chain_oracle_action_for_agent(row) == 10  # move W toward converter.

    row[NAV_INVENTORY_BATTERY] = 1
    row[NAV_INVENTORY_ORE] = 0
    assert _chain_oracle_action_for_agent(row) == 14  # move SW toward home assembler.


def test_chain_oracle_uses_adjacent_target_when_valid():
    row = np.zeros(len(NAVIGATION_SNAPSHOT_COLUMNS), dtype=np.int64)
    row[[NAV_AGENT_X, NAV_AGENT_Y]] = [4, 5]
    row[[NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y]] = [3, 6]
    row[NAV_DIST_HOME_ASSEMBLER] = 1
    row[NAV_INVENTORY_BATTERY] = 1

    assert _chain_oracle_action_for_agent(row) == 30  # use SW at home assembler.

    mask = np.ones(56, dtype=bool)
    mask[30] = False
    assert _chain_oracle_action_for_agent(row, mask) == 0


def _write_minimal_tribal_sources(root, *, action_verbs, houses, agents_per_house, canonical):
    (root / "src").mkdir(parents=True)
    (root / "src" / "common.nim").write_text(
        f"""
const
  ActionVerbCount* = {action_verbs}
  ActionArgumentCount* = 8
"""
    )
    canonical_block = (
        """
when defined(canonicalRewardGeometry):
  discard
"""
        if canonical
        else ""
    )
    (root / "src" / "environment.nim").write_text(
        f"""
{canonical_block}
const
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomBorder* = 0
  MapRoomWidth* = 192
  MapRoomHeight* = 108
  MapRoomObjectsHouses* = {houses}
  MapAgentsPerHouse* = {agents_per_house}
  ObservationLayers* = 21
  ObservationWidth* = 11
  ObservationHeight* = 11
"""
    )
