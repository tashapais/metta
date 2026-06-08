import json

import numpy as np
import pytest
import torch

from v3_experiments.canonical_reward_geometry import (
    ROLE_PROBE_CV,
    audit_fixed_role_probe_records,
    effective_rank,
    fixed_role_probe_accuracy,
    js_action_diversity,
    mix_rewards,
    ordered_kl_action_diversity,
    role_labels,
    role_probe_chance,
    role_shaping_bonuses,
    summarize_probe_audit,
)
from v3_experiments.run_tribal_behavior_rollouts import _checkpoint_uses_chain_affordance_action_mask
from v3_experiments.train_canonical_reward_geometry import (
    CHAIN_COMPASS_CENTER_VALUE,
    CHAIN_COMPASS_NEGATIVE_VALUE,
    CHAIN_COMPASS_OBSERVATION_PLANES,
    CHAIN_COMPASS_POSITIVE_VALUE,
    CHAIN_COMPASS_STAGE_BATTERY_VALUE,
    CHAIN_COMPASS_STAGE_EMPTY_VALUE,
    CHAIN_COMPASS_STAGE_ORE_VALUE,
    _action_mask_array_from_flags,
    _augment_chain_compass_observation,
)
from v3_experiments.train_canonical_reward_geometry import (
    main as train_canonical_main,
)
from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS
from v3_experiments.tribal_event_rewards import (
    ACTION_ARGUMENT_COUNT,
    EVENT_V1_ROLE_NAMES,
    EVENT_V2_BREADCRUMB_ROLE_NAMES,
    EVENT_V3_NAVIGATION_ROLE_NAMES,
    EVENT_V4_HEART_CHAIN_ROLE_NAMES,
    EVENT_V5_NAVIGATION_CHAIN_ROLE_NAMES,
    EVENT_V6_ORACLE_CHAIN_ROLE_NAMES,
    EVENT_V7_CHAIN_COMPASS_ROLE_NAMES,
    EVENT_V8_CLEAN_CHAIN_COMPASS_OFFCHAIN_PENALTIES,
    EVENT_V8_CLEAN_CHAIN_COMPASS_ROLE_NAMES,
    EVENT_V9_POTENTIAL_CHAIN_CLOSENESS_SCALES,
    EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES,
    EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    EVENT_V9_POTENTIAL_CHAIN_STAGE_OFFSETS,
    EVENT_V10_CHAIN_AFFORDANCE_COMPASS_ROLE_NAMES,
    MOVE_VERB,
    NAV_AGENT_X,
    NAV_AGENT_Y,
    NAV_DIST_HOME_ASSEMBLER,
    NAV_DIST_NEAREST_CONVERTER,
    NAV_DIST_NEAREST_MINE,
    NAV_HOME_ASSEMBLER_X,
    NAV_HOME_ASSEMBLER_Y,
    NAV_INVENTORY_BATTERY,
    NAV_INVENTORY_ORE,
    NAV_NEAREST_CONVERTER_X,
    NAV_NEAREST_CONVERTER_Y,
    NAV_NEAREST_MINE_X,
    NAV_NEAREST_MINE_Y,
    NAVIGATION_SNAPSHOT_COLUMNS,
    USE_VERB,
    event_v1_reward_design_details,
    event_v1_role_shaping_bonuses,
    event_v2_breadcrumb_reward_design_details,
    event_v2_breadcrumb_role_shaping_bonuses,
    event_v3_navigation_reward_design_details,
    event_v3_navigation_role_shaping_bonuses,
    event_v4_heart_chain_reward_design_details,
    event_v4_heart_chain_role_shaping_bonuses,
    event_v5_navigation_chain_reward_design_details,
    event_v5_navigation_chain_role_shaping_bonuses,
    event_v6_oracle_chain_reward_design_details,
    event_v6_oracle_chain_role_shaping_bonuses,
    event_v7_chain_compass_reward_design_details,
    event_v8_clean_chain_compass_reward_design_details,
    event_v8_clean_chain_compass_role_shaping_bonuses,
    event_v9_potential_chain_compass_reward_design_details,
    event_v9_potential_chain_compass_role_shaping_bonuses,
    event_v10_chain_affordance_compass_reward_design_details,
)
from v3_experiments.validate_canonical_reward_geometry_results import validate_record


def test_role_labels_are_balanced_interleaved_roles():
    labels = role_labels(num_agents=12, num_roles=3)

    assert labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert np.bincount(labels).tolist() == [4, 4, 4]
    assert role_probe_chance(3) == pytest.approx(1 / 3)


def test_role_labels_reject_unbalanced_assignment():
    with pytest.raises(ValueError, match="balanced roles"):
        role_labels(num_agents=10, num_roles=3)


def test_mix_rewards_preserves_individual_and_shared_extremes():
    rewards = np.array([[1.0, 2.0, 7.0], [3.0, 3.0, 9.0]])

    np.testing.assert_allclose(mix_rewards(rewards, 0.0), rewards)
    np.testing.assert_allclose(mix_rewards(rewards, 1.0), np.array([[10 / 3, 10 / 3, 10 / 3], [5, 5, 5]]))
    np.testing.assert_allclose(mix_rewards(rewards, 0.5), 0.5 * rewards + 0.5 * np.array([[10 / 3], [5]]))


def test_mix_rewards_rejects_invalid_shared_fraction():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_rewards([1.0, 2.0], -0.1)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_rewards([1.0, 2.0], 1.1)


def test_role_shaping_bonuses_follow_fixed_roles():
    obs = np.zeros((12, 21, 11, 11), dtype=np.uint8)
    obs[0, 11, 0, 0] = 1
    obs[0, 11, 0, 1] = 1
    obs[1, 11, 1, 1] = 1
    obs[2, 16, 2, 0:3] = 1

    bonuses = role_shaping_bonuses(obs, gold_layer=11, altar_layer=16)

    assert bonuses[0] == pytest.approx(0.6)
    assert bonuses[1] == pytest.approx(0.0)
    assert bonuses[2] == pytest.approx(6.0)
    assert bonuses[4] == pytest.approx(0.5)


def test_event_v1_role_shaping_bonuses_follow_success_events():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    stats[0, stat_index["resource_ore"]] = 2
    stats[0, stat_index["action_use"]] = 2
    stats[1, stat_index["craft_battery"]] = 1
    stats[1, stat_index["deposit_heart"]] = 1
    stats[1, stat_index["put_armor"]] = 1
    stats[1, stat_index["action_use"]] = 2
    stats[1, stat_index["action_put"]] = 1
    stats[2, stat_index["tumor_kill"]] = 1
    stats[2, stat_index["lantern_plant"]] = 1
    stats[2, stat_index["action_attack"]] = 1
    stats[2, stat_index["action_plant"]] = 1
    stats[2, stat_index["action_invalid"]] = 1
    stats[3, stat_index["craft_battery"]] = 1
    stats[3, stat_index["action_use"]] = 1

    bonuses = event_v1_role_shaping_bonuses(stats)

    assert bonuses[0] == pytest.approx(0.20 * 2 + 0.02 * 2)
    assert bonuses[1] == pytest.approx(0.45 + 0.80 + 0.30 + 0.02 * 3)
    assert bonuses[2] == pytest.approx(0.90 + 0.45 + 0.02 * 2 - 0.01)
    assert bonuses[3] == pytest.approx(0.02)
    np.testing.assert_allclose(event_v1_role_shaping_bonuses(None), np.zeros(12))


def test_event_v1_reward_design_details_are_serializable():
    details = event_v1_reward_design_details()

    assert details["name"] == "event_v1"
    assert details["role_names"] == list(EVENT_V1_ROLE_NAMES)
    assert details["coworld_role_sources"]["defender_territory"]


def test_event_v2_breadcrumbs_pay_generic_and_role_specific_events():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    stats[0, stat_index["resource_ore"]] = 1
    stats[0, stat_index["action_use"]] = 1
    stats[1, stat_index["resource_ore"]] = 1
    stats[1, stat_index["action_use"]] = 1
    stats[1, stat_index["craft_battery"]] = 1
    stats[2, stat_index["tumor_kill"]] = 1
    stats[2, stat_index["action_attack"]] = 1
    stats[3, stat_index["action_noop"]] = 2
    stats[4, stat_index["action_invalid"]] = 3

    bonuses = event_v2_breadcrumb_role_shaping_bonuses(stats)

    assert bonuses[0] == pytest.approx(0.08 + 0.10 + 0.10)
    assert bonuses[1] == pytest.approx(0.08 + 0.10 + 0.25 + 0.20)
    assert bonuses[2] == pytest.approx(0.05 + 0.70 + 0.30)
    assert bonuses[3] == pytest.approx(-0.004)
    assert bonuses[4] == pytest.approx(-0.003)
    np.testing.assert_allclose(event_v2_breadcrumb_role_shaping_bonuses(None), np.zeros(12))


def test_event_v2_breadcrumb_reward_design_details_are_serializable():
    details = event_v2_breadcrumb_reward_design_details()

    assert details["name"] == "event_v2_breadcrumbs"
    assert details["role_names"] == list(EVENT_V2_BREADCRUMB_ROLE_NAMES)
    assert details["task_event_coefficients"]["resource_ore"] > 0
    assert details["coworld_role_sources"]["supplier"]


def test_event_v3_navigation_breadcrumbs_cap_movement_and_penalize_invalid_use_spam():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    totals = np.zeros_like(stats)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    stats[0, stat_index["action_move"]] = 1
    totals[0, stat_index["action_move"]] = 40
    stats[0, stat_index["action_use"]] = 1
    stats[0, stat_index["resource_wood"]] = 1
    stats[1, stat_index["action_invalid"]] = 2
    stats[2, stat_index["action_use"]] = 1
    stats[2, stat_index["resource_wheat"]] = 1
    stats[3, stat_index["action_move"]] = 1
    totals[3, stat_index["action_move"]] = 41

    bonuses = event_v3_navigation_role_shaping_bonuses(stats, totals)

    assert bonuses[0] == pytest.approx(0.004 + 0.05 + 0.45 + 0.20)
    assert bonuses[1] == pytest.approx(-0.008)
    assert bonuses[2] == pytest.approx(0.05 + 0.45)
    assert bonuses[3] == pytest.approx(0.0)
    np.testing.assert_allclose(event_v3_navigation_role_shaping_bonuses(None), np.zeros(12))


def test_event_v3_navigation_reward_design_details_are_serializable():
    details = event_v3_navigation_reward_design_details()

    assert details["name"] == "event_v3_navigation_breadcrumbs"
    assert details["role_names"] == list(EVENT_V3_NAVIGATION_ROLE_NAMES)
    assert details["common_caps"]["action_move"] == 40
    assert details["task_event_coefficients"]["deposit_heart"] > details["task_event_coefficients"]["resource_ore"]


def test_event_v4_heart_chain_prioritizes_deposits_over_handoffs():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    totals = np.zeros_like(stats)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    stats[0, stat_index["action_move"]] = 1
    totals[0, stat_index["action_move"]] = 80
    stats[0, stat_index["action_use"]] = 1
    stats[0, stat_index["resource_ore"]] = 1
    stats[1, stat_index["action_use"]] = 2
    stats[1, stat_index["action_put"]] = 1
    stats[1, stat_index["craft_battery"]] = 1
    stats[1, stat_index["deposit_heart"]] = 1
    stats[1, stat_index["put_armor"]] = 1
    stats[2, stat_index["action_attack"]] = 1
    stats[2, stat_index["tumor_kill"]] = 1
    stats[3, stat_index["action_move"]] = 1
    totals[3, stat_index["action_move"]] = 81
    stats[4, stat_index["action_invalid"]] = 2

    bonuses = event_v4_heart_chain_role_shaping_bonuses(stats, totals)

    assert bonuses[0] == pytest.approx(0.003 + 0.03 + 1.20 + 0.80)
    assert bonuses[1] == pytest.approx(0.03 * 2 + 0.005 + 6.00 + 20.00 + 0.02 + 4.00 + 10.00 + 0.02)
    assert bonuses[2] == pytest.approx(0.03 + 1.00 + 0.50)
    assert bonuses[3] == pytest.approx(0.0)
    assert bonuses[4] == pytest.approx(-0.012)
    assert bonuses[1] > 15 * bonuses[0]
    np.testing.assert_allclose(event_v4_heart_chain_role_shaping_bonuses(None), np.zeros(12))


def test_event_v4_heart_chain_reward_design_details_are_serializable():
    details = event_v4_heart_chain_reward_design_details()

    assert details["name"] == "event_v4_heart_chain_breadcrumbs"
    assert details["role_names"] == list(EVENT_V4_HEART_CHAIN_ROLE_NAMES)
    assert details["common_caps"]["action_move"] == 80
    assert details["task_event_coefficients"]["deposit_heart"] > 3 * details["task_event_coefficients"]["craft_battery"]
    assert details["task_event_coefficients"]["craft_battery"] > details["task_event_coefficients"]["craft_armor"]


def test_event_v5_navigation_chain_pays_inventory_conditioned_progress():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    totals = np.zeros_like(stats)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    stats[0, stat_index["action_move"]] = 1
    stats[1, stat_index["action_move"]] = 1
    stats[2, stat_index["action_move"]] = 1
    stats[3, stat_index["action_move"]] = 1
    stats[4, stat_index["action_invalid"]] = 1

    before = np.zeros((12, len(NAVIGATION_SNAPSHOT_COLUMNS)), dtype=np.float64)
    after = before.copy()
    before[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = 10
    after[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = 10
    before[0, NAV_INVENTORY_ORE] = 1
    before[0, NAV_DIST_NEAREST_CONVERTER] = 9
    after[0, NAV_DIST_NEAREST_CONVERTER] = 7
    before[1, NAV_INVENTORY_BATTERY] = 1
    before[1, NAV_DIST_HOME_ASSEMBLER] = 6
    after[1, NAV_DIST_HOME_ASSEMBLER] = 3
    before[2, NAV_DIST_NEAREST_MINE] = 4
    after[2, NAV_DIST_NEAREST_MINE] = 3
    before[3, NAV_INVENTORY_BATTERY] = 1
    before[3, NAV_DIST_HOME_ASSEMBLER] = 4
    after[3, NAV_DIST_HOME_ASSEMBLER] = 5
    before[5, NAV_INVENTORY_BATTERY] = 1
    before[5, NAV_DIST_HOME_ASSEMBLER] = 2
    after[5, NAV_DIST_HOME_ASSEMBLER] = 1

    bonuses = event_v5_navigation_chain_role_shaping_bonuses(
        stats,
        totals,
        navigation_before=before,
        navigation_after=after,
    )

    assert bonuses[0] == pytest.approx(0.001 + 0.30 * 2)
    assert bonuses[1] == pytest.approx(0.001 + 0.80 * 3)
    assert bonuses[2] == pytest.approx(0.001 + 0.02)
    assert bonuses[3] == pytest.approx(0.001)
    assert bonuses[4] == pytest.approx(-0.006)
    assert bonuses[5] == pytest.approx(0.80 + 0.15)
    np.testing.assert_allclose(
        event_v5_navigation_chain_role_shaping_bonuses(None, navigation_before=before, navigation_after=after),
        np.zeros(12),
    )


def test_event_v5_navigation_chain_reward_design_details_are_serializable():
    details = event_v5_navigation_chain_reward_design_details()

    assert details["name"] == "event_v5_navigation_chain_breadcrumbs"
    assert details["role_names"] == list(EVENT_V5_NAVIGATION_CHAIN_ROLE_NAMES)
    assert details["navigation_snapshot_columns"] == list(NAVIGATION_SNAPSHOT_COLUMNS)
    assert (
        details["navigation_progress_coefficients"]["battery_to_home_assembler"]
        > (details["navigation_progress_coefficients"]["ore_to_converter"])
    )
    assert details["task_event_coefficients"]["deposit_heart"] > details["task_event_coefficients"]["craft_battery"]


def test_event_v6_oracle_chain_pays_only_chain_progress_and_actions():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    totals = np.zeros_like(stats)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    actions = np.zeros(12, dtype=np.int64)
    action_mask = np.ones((12, 56), dtype=bool)

    before = np.zeros((12, len(NAVIGATION_SNAPSHOT_COLUMNS)), dtype=np.float64)
    after = before.copy()
    before[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = 10
    after[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = 10

    before[0, [NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y, NAV_DIST_NEAREST_MINE]] = [7, 4, 3]
    before[0, [NAV_AGENT_X, NAV_AGENT_Y]] = [5, 5]
    after[0, NAV_DIST_NEAREST_MINE] = 1
    stats[0, stat_index["action_move"]] = 1
    totals[0, stat_index["action_move"]] = 1
    actions[0] = 13  # move NE toward mine.

    before[1, [NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y, NAV_DIST_NEAREST_MINE]] = [7, 4, 1]
    before[1, [NAV_AGENT_X, NAV_AGENT_Y]] = [6, 5]
    stats[1, stat_index["action_use"]] = 1
    stats[1, stat_index["resource_ore"]] = 1
    actions[1] = 29  # use NE at mine.

    before[2, [NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y, NAV_DIST_NEAREST_CONVERTER]] = [2, 5, 2]
    before[2, [NAV_AGENT_X, NAV_AGENT_Y]] = [4, 5]
    before[2, NAV_INVENTORY_ORE] = 1
    after[2, NAV_DIST_NEAREST_CONVERTER] = 1
    stats[2, stat_index["action_move"]] = 1
    totals[2, stat_index["action_move"]] = 1
    actions[2] = 10  # move W toward converter.

    before[3, [NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y, NAV_DIST_NEAREST_CONVERTER]] = [3, 5, 1]
    before[3, [NAV_AGENT_X, NAV_AGENT_Y]] = [4, 5]
    before[3, NAV_INVENTORY_ORE] = 1
    stats[3, stat_index["action_use"]] = 1
    stats[3, stat_index["craft_battery"]] = 1
    actions[3] = 26  # use W at converter.

    before[4, [NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER]] = [3, 6, 3]
    before[4, [NAV_AGENT_X, NAV_AGENT_Y]] = [5, 5]
    before[4, NAV_INVENTORY_BATTERY] = 1
    after[4, NAV_DIST_HOME_ASSEMBLER] = 1
    stats[4, stat_index["action_move"]] = 1
    totals[4, stat_index["action_move"]] = 1
    actions[4] = 14  # move SW toward home assembler.

    before[5, [NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER]] = [3, 6, 1]
    before[5, [NAV_AGENT_X, NAV_AGENT_Y]] = [4, 5]
    before[5, NAV_INVENTORY_BATTERY] = 1
    stats[5, stat_index["action_use"]] = 1
    stats[5, stat_index["deposit_heart"]] = 1
    actions[5] = 30  # use SW at home assembler.

    stats[6, stat_index["action_invalid"]] = 1

    before[7, [NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER]] = [3, 6, 1]
    before[7, [NAV_AGENT_X, NAV_AGENT_Y]] = [4, 5]
    before[7, NAV_INVENTORY_BATTERY] = 1
    actions[7] = 30
    action_mask[7, 30] = False

    stats[8, stat_index["resource_water"]] = 9
    stats[8, stat_index["craft_armor"]] = 3
    stats[8, stat_index["tumor_kill"]] = 2

    bonuses = event_v6_oracle_chain_role_shaping_bonuses(
        stats,
        totals,
        navigation_before=before,
        navigation_after=after,
        actions=actions,
        action_mask=action_mask,
    )

    assert bonuses[0] == pytest.approx(0.08 * 2 + 0.05)
    assert bonuses[1] == pytest.approx(1.00 + 1.00)
    assert bonuses[2] == pytest.approx(0.80 + 0.05)
    assert bonuses[3] == pytest.approx(8.00 + 1.00)
    assert bonuses[4] == pytest.approx(1.50 * 2 + 0.50 + 0.05)
    assert bonuses[5] == pytest.approx(40.00 + 1.00)
    assert bonuses[6] == pytest.approx(-0.02)
    assert bonuses[7] == pytest.approx(0.0)
    assert bonuses[8] == pytest.approx(0.0)
    np.testing.assert_allclose(
        event_v6_oracle_chain_role_shaping_bonuses(None, navigation_before=before, navigation_after=after),
        np.zeros(12),
    )


def test_event_v6_oracle_chain_reward_design_details_are_serializable():
    details = event_v6_oracle_chain_reward_design_details()

    assert details["name"] == "event_v6_oracle_chain_breadcrumbs"
    assert details["role_names"] == list(EVENT_V6_ORACLE_CHAIN_ROLE_NAMES)
    assert set(details["task_event_coefficients"]) == {"craft_battery", "deposit_heart", "resource_ore"}
    assert details["oracle_action_coefficients"]["use_chain_target"] > 0
    assert details["navigation_snapshot_columns"] == list(NAVIGATION_SNAPSHOT_COLUMNS)


def test_event_v7_chain_compass_reward_design_details_are_serializable():
    details = event_v7_chain_compass_reward_design_details()

    assert details["name"] == "event_v7_chain_compass_breadcrumbs"
    assert details["role_names"] == list(EVENT_V7_CHAIN_COMPASS_ROLE_NAMES)
    assert set(details["task_event_coefficients"]) == {"craft_battery", "deposit_heart", "resource_ore"}
    assert details["oracle_action_coefficients"]["use_chain_target"] > 0
    assert details["observation_breadcrumbs"]["planes"] == list(CHAIN_COMPASS_OBSERVATION_PLANES)


def test_event_v8_clean_chain_compass_penalizes_off_chain_task_events():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}

    stats[0, stat_index["resource_ore"]] = 1
    stats[0, stat_index["craft_battery"]] = 1
    stats[0, stat_index["deposit_heart"]] = 1

    stats[1, stat_index["resource_water"]] = 2
    stats[1, stat_index["resource_wheat"]] = 3
    stats[1, stat_index["resource_wood"]] = 4
    stats[1, stat_index["craft_armor"]] = 1
    stats[1, stat_index["craft_bread"]] = 1
    stats[1, stat_index["craft_lantern"]] = 1
    stats[1, stat_index["craft_spear"]] = 1
    stats[1, stat_index["put_armor"]] = 1
    stats[1, stat_index["put_bread"]] = 1
    stats[1, stat_index["tumor_kill"]] = 1
    stats[1, stat_index["spawner_kill"]] = 1
    stats[1, stat_index["agent_kill"]] = 1
    stats[1, stat_index["lantern_plant"]] = 1

    stats[2, stat_index["resource_ore"]] = 1
    stats[2, stat_index["resource_water"]] = 1
    stats[2, stat_index["craft_armor"]] = 1

    bonuses = event_v8_clean_chain_compass_role_shaping_bonuses(stats)

    assert bonuses[0] == pytest.approx(1.0 + 8.0 + 40.0)
    off_chain_total = sum(
        stats[1, stat_index[name]] * coefficient
        for name, coefficient in EVENT_V8_CLEAN_CHAIN_COMPASS_OFFCHAIN_PENALTIES.items()
    )
    assert bonuses[1] == pytest.approx(off_chain_total)
    assert bonuses[2] == pytest.approx(1.0 - 0.10 - 1.00)
    np.testing.assert_allclose(event_v8_clean_chain_compass_role_shaping_bonuses(None), np.zeros(12))


def test_event_v8_clean_chain_compass_reward_design_details_are_serializable():
    details = event_v8_clean_chain_compass_reward_design_details()

    assert details["name"] == "event_v8_clean_chain_compass_breadcrumbs"
    assert details["role_names"] == list(EVENT_V8_CLEAN_CHAIN_COMPASS_ROLE_NAMES)
    assert set(details["task_event_coefficients"]) == {"craft_battery", "deposit_heart", "resource_ore"}
    assert details["oracle_action_coefficients"]["use_chain_target"] > 0
    assert details["observation_breadcrumbs"]["planes"] == list(CHAIN_COMPASS_OBSERVATION_PLANES)
    assert details["off_chain_penalty_coefficients"]["resource_water"] < 0
    assert details["off_chain_penalty_coefficients"]["craft_armor"] < 0


def test_event_v9_potential_chain_compass_removes_explicit_event_penalties():
    stats = np.zeros((12, len(SIMULATOR_STAT_COLUMNS)), dtype=np.float64)
    stat_index = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}
    before = np.zeros((12, len(NAVIGATION_SNAPSHOT_COLUMNS)), dtype=np.float64)
    after = before.copy()
    before[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = -1
    after[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = -1

    stats[0, stat_index["resource_ore"]] = 1
    stats[0, stat_index["craft_battery"]] = 1
    stats[0, stat_index["deposit_heart"]] = 1
    before[0, NAV_DIST_NEAREST_MINE] = 10
    after[0, NAV_DIST_NEAREST_MINE] = 8

    stats[1, stat_index["resource_water"]] = 2
    stats[1, stat_index["resource_wheat"]] = 3
    stats[1, stat_index["resource_wood"]] = 4
    stats[1, stat_index["craft_armor"]] = 1
    stats[1, stat_index["action_noop"]] = 8
    stats[1, stat_index["action_invalid"]] = 9

    before[2, NAV_INVENTORY_BATTERY] = 1
    before[2, NAV_DIST_HOME_ASSEMBLER] = 2
    after[2, NAV_INVENTORY_BATTERY] = 1
    after[2, NAV_DIST_HOME_ASSEMBLER] = 4

    bonuses = event_v9_potential_chain_compass_role_shaping_bonuses(
        stats,
        navigation_before=before,
        navigation_after=after,
    )

    mine_phi_before = _test_v9_potential("empty_to_mine", 10)
    mine_phi_after = _test_v9_potential("empty_to_mine", 8)
    expected_chain = 1.0 + 8.0 + 40.0 + EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA * mine_phi_after - mine_phi_before
    assert bonuses[0] == pytest.approx(expected_chain)
    assert bonuses[1] == pytest.approx(0.0)

    home_phi_before = _test_v9_potential("battery_to_home_assembler", 2)
    home_phi_after = _test_v9_potential("battery_to_home_assembler", 4)
    assert bonuses[2] == pytest.approx(EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA * home_phi_after - home_phi_before)


def test_event_v9_potential_chain_compass_reward_design_details_are_serializable():
    details = event_v9_potential_chain_compass_reward_design_details()

    assert details["name"] == "event_v9_potential_chain_compass_breadcrumbs"
    assert details["role_names"] == list(EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES)
    assert set(details["task_event_coefficients"]) == {"craft_battery", "deposit_heart", "resource_ore"}
    assert details["potential_shaping"]["formula"] == "F(s,s') = gamma * Phi(s') - Phi(s)"
    assert details["negative_reward_coefficients"] == {}
    assert details["observation_breadcrumbs"]["planes"] == list(CHAIN_COMPASS_OBSERVATION_PLANES)


def test_event_v10_chain_affordance_compass_reward_design_details_are_serializable():
    details = event_v10_chain_affordance_compass_reward_design_details()

    assert details["name"] == "event_v10_chain_affordance_compass_breadcrumbs"
    assert details["role_names"] == list(EVENT_V10_CHAIN_AFFORDANCE_COMPASS_ROLE_NAMES)
    assert details["negative_reward_coefficients"] == {}
    assert details["action_affordance_curriculum"]["enabled"] is True
    assert details["action_affordance_curriculum"]["reward_penalties_added"] is False


def test_chain_compass_observation_tracks_inventory_conditioned_target():
    obs = np.zeros((3, 21, 11, 11), dtype=np.uint8)
    navigation = np.zeros((3, len(NAVIGATION_SNAPSHOT_COLUMNS)), dtype=np.float64)
    navigation[:, [NAV_AGENT_X, NAV_AGENT_Y]] = [5, 5]

    navigation[0, [NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y]] = [7, 4]

    navigation[1, [NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y]] = [2, 5]
    navigation[1, NAV_INVENTORY_ORE] = 1

    navigation[2, [NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER]] = [4, 6, 1]
    navigation[2, NAV_INVENTORY_BATTERY] = 1

    augmented = _augment_chain_compass_observation(obs, navigation)
    extra = augmented[:, 21:, :, :]
    dx = CHAIN_COMPASS_OBSERVATION_PLANES["chain_target_dx_sign"]
    dy = CHAIN_COMPASS_OBSERVATION_PLANES["chain_target_dy_sign"]
    stage = CHAIN_COMPASS_OBSERVATION_PLANES["chain_inventory_stage"]
    adjacent = CHAIN_COMPASS_OBSERVATION_PLANES["chain_target_adjacent"]

    assert augmented.shape == (3, 26, 11, 11)
    assert np.all(extra[0, dx] == CHAIN_COMPASS_POSITIVE_VALUE)
    assert np.all(extra[0, dy] == CHAIN_COMPASS_NEGATIVE_VALUE)
    assert np.all(extra[0, stage] == CHAIN_COMPASS_STAGE_EMPTY_VALUE)
    assert np.all(extra[0, adjacent] == 0)

    assert np.all(extra[1, dx] == CHAIN_COMPASS_NEGATIVE_VALUE)
    assert np.all(extra[1, dy] == CHAIN_COMPASS_CENTER_VALUE)
    assert np.all(extra[1, stage] == CHAIN_COMPASS_STAGE_ORE_VALUE)

    assert np.all(extra[2, dx] == CHAIN_COMPASS_NEGATIVE_VALUE)
    assert np.all(extra[2, dy] == CHAIN_COMPASS_POSITIVE_VALUE)
    assert np.all(extra[2, stage] == CHAIN_COMPASS_STAGE_BATTERY_VALUE)
    assert np.all(extra[2, adjacent] == CHAIN_COMPASS_POSITIVE_VALUE)


def test_chain_affordance_action_mask_allows_moves_and_current_target_use_only():
    navigation = np.zeros((3, len(NAVIGATION_SNAPSHOT_COLUMNS)), dtype=np.float64)
    navigation[:, [NAV_AGENT_X, NAV_AGENT_Y]] = [5, 5]
    navigation[:, [NAV_DIST_HOME_ASSEMBLER, NAV_DIST_NEAREST_CONVERTER, NAV_DIST_NEAREST_MINE]] = [4, 4, 4]

    navigation[0, [NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y, NAV_DIST_NEAREST_MINE]] = [6, 5, 1]

    navigation[1, [NAV_NEAREST_CONVERTER_X, NAV_NEAREST_CONVERTER_Y, NAV_DIST_NEAREST_CONVERTER]] = [7, 5, 2]
    navigation[1, NAV_INVENTORY_ORE] = 1

    navigation[2, [NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER]] = [4, 6, 1]
    navigation[2, NAV_INVENTORY_BATTERY] = 1

    base_mask = np.ones((3, 56), dtype=bool)
    env = _MaskEnv(base_mask, navigation)
    mask = _action_mask_array_from_flags(env, use_action_mask=True, chain_affordance_action_mask=True)
    assert mask is not None

    move_actions = [MOVE_VERB * ACTION_ARGUMENT_COUNT + orientation for orientation in range(8)]
    use_east = USE_VERB * ACTION_ARGUMENT_COUNT + 3
    use_southwest = USE_VERB * ACTION_ARGUMENT_COUNT + 6
    attack_north = 2 * ACTION_ARGUMENT_COUNT

    assert mask[0, move_actions].all()
    assert mask[0, use_east]
    assert not mask[0, attack_north]
    assert mask[0].sum() == 9

    assert mask[1, move_actions].all()
    assert mask[1].sum() == 8

    assert mask[2, move_actions].all()
    assert mask[2, use_southwest]
    assert mask[2].sum() == 9

    base_mask[0, use_east] = False
    mask_without_use = _action_mask_array_from_flags(env, use_action_mask=True, chain_affordance_action_mask=True)
    assert mask_without_use is not None
    assert not mask_without_use[0, use_east]
    assert mask_without_use[0].sum() == 8

    no_chain_navigation = navigation.copy()
    no_chain_navigation[0, [NAV_NEAREST_MINE_X, NAV_NEAREST_MINE_Y, NAV_DIST_NEAREST_MINE]] = [0, 0, 0]
    no_chain_base_mask = np.zeros((3, 56), dtype=bool)
    no_chain_base_mask[:, 0] = True
    no_chain_base_mask[0, 5 * ACTION_ARGUMENT_COUNT] = True
    no_chain_env = _MaskEnv(no_chain_base_mask, no_chain_navigation)
    no_chain_mask = _action_mask_array_from_flags(
        no_chain_env,
        use_action_mask=True,
        chain_affordance_action_mask=True,
    )
    assert no_chain_mask is not None
    assert no_chain_mask[0, 0]
    assert not no_chain_mask[0, 5 * ACTION_ARGUMENT_COUNT]
    assert no_chain_mask[0].sum() == 1


def test_effective_rank_uses_entropy_of_singular_values():
    assert effective_rank(np.eye(4)) == pytest.approx(4.0)


def test_fixed_role_probe_uses_agent_generalization_folds():
    pytest.importorskip("sklearn")
    labels = role_labels()
    embeddings = np.repeat(np.eye(3, dtype=np.float64)[labels][None, :, :], repeats=2, axis=0)

    accuracy, meta = fixed_role_probe_accuracy(embeddings)

    assert accuracy == pytest.approx(1.0)
    assert meta["cv"] == ROLE_PROBE_CV
    assert meta["chance"] == pytest.approx(1 / 3)
    assert meta["fold_agents"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def test_ordered_kl_action_diversity_uses_ordered_off_diagonal_pairs():
    # Two samples, three agents, two actions.
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            [[1.5, -1.0], [-1.0, 1.5], [0.2, -0.2]],
        ]
    )

    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    log_probs = np.log(probs)
    expected_terms = []
    for sample in range(logits.shape[0]):
        for i in range(logits.shape[1]):
            for j in range(logits.shape[1]):
                if i == j:
                    continue
                expected_terms.append((probs[sample, i] * (log_probs[sample, i] - log_probs[sample, j])).sum())

    assert ordered_kl_action_diversity(logits) == pytest.approx(np.mean(expected_terms))


def test_js_action_diversity_is_zero_for_identical_action_distributions():
    logits = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

    assert ordered_kl_action_diversity(logits) == pytest.approx(0.0)
    assert js_action_diversity(logits) == pytest.approx(0.0)


def test_fixed_role_probe_audit_flags_binary_or_stale_probe_records():
    issues = audit_fixed_role_probe_records(
        [
            {
                "run_name": "paper_reward_individual_12agents_seed0",
                "seed": 0,
                "probe_chance": 0.9534,
                "probe_lift": -0.2867,
            },
            {
                "run_name": "canonical_tribal_alpha0_seed0",
                "seed": 0,
                "probe_chance": 1 / 3,
                "probe_lift": 0.6,
            },
        ]
    )

    assert summarize_probe_audit(issues) == {
        "non_fixed_role_probe_chance": 1,
        "negative_probe_lift": 1,
    }


def test_validator_accepts_smoke_record(tmp_path):
    checkpoint_path = tmp_path / "final_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    record = _valid_record(checkpoint_path)

    assert validate_record(record, path=tmp_path / "result.json", allow_smoke=True) == []


def test_validator_accepts_custom_smoke_condition_group(tmp_path):
    checkpoint_path = tmp_path / "final_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    record = _valid_record(checkpoint_path)
    record["condition_group"] = "stage1_event_v3_mask_alpha0"

    assert validate_record(record, path=tmp_path / "result.json", allow_smoke=True) == []


def test_validator_rejects_custom_full_condition_group(tmp_path):
    checkpoint_path = tmp_path / "final_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    record = _valid_record(checkpoint_path)
    record["condition_group"] = "stage1_event_v3_mask_alpha0"

    issues = validate_record(record, path=tmp_path / "result.json")

    assert "condition_group" in [issue.field for issue in issues]


def test_validator_rejects_noncanonical_probe_chance(tmp_path):
    checkpoint_path = tmp_path / "final_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    record = _valid_record(checkpoint_path)
    record["role_probe_chance"] = 0.5

    issues = validate_record(record, path=tmp_path / "result.json", allow_smoke=True)

    assert [issue.field for issue in issues] == ["role_probe_chance"]


def test_train_canonical_reward_geometry_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--shared-frac",
                "0.8",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    assert output_path.exists()
    assert checkpoint_path.exists()
    record = json.loads(output_path.read_text())
    assert record["environment_backend"] == "mock"
    assert record["num_agents"] == 12
    assert record["map_width"] == 80
    assert record["role_probe_chance"] == pytest.approx(1 / 3)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v1_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v1_result.json"
    checkpoint_path = tmp_path / "event_v1_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v1",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v1"
    assert record["role_names"] == list(EVENT_V1_ROLE_NAMES)
    assert record["reward_design_details"]["coworld_role_sources"]["supplier"]
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v2_breadcrumb_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v2_result.json"
    checkpoint_path = tmp_path / "event_v2_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v2_breadcrumbs",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v2_breadcrumbs"
    assert record["role_names"] == list(EVENT_V2_BREADCRUMB_ROLE_NAMES)
    assert record["reward_design_details"]["task_event_coefficients"]["craft_battery"] > 0
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v3_navigation_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v3_result.json"
    checkpoint_path = tmp_path / "event_v3_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v3_navigation_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v3_navigation_breadcrumbs"
    assert record["role_names"] == list(EVENT_V3_NAVIGATION_ROLE_NAMES)
    assert record["use_action_mask"] is True
    assert record["reward_design_details"]["common_caps"]["action_move"] == 40
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v4_heart_chain_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v4_result.json"
    checkpoint_path = tmp_path / "event_v4_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v4_heart_chain_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v4_heart_chain_breadcrumbs"
    assert record["role_names"] == list(EVENT_V4_HEART_CHAIN_ROLE_NAMES)
    assert record["use_action_mask"] is True
    assert record["reward_design_details"]["common_caps"]["action_move"] == 80
    assert record["reward_design_details"]["task_event_coefficients"]["craft_battery"] == pytest.approx(6.0)
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v6_oracle_chain_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v6_result.json"
    checkpoint_path = tmp_path / "event_v6_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v6_oracle_chain_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v6_oracle_chain_breadcrumbs"
    assert record["role_names"] == list(EVENT_V6_ORACLE_CHAIN_ROLE_NAMES)
    assert record["use_action_mask"] is True
    assert set(record["reward_design_details"]["task_event_coefficients"]) == {
        "craft_battery",
        "deposit_heart",
        "resource_ore",
    }
    assert record["reward_design_details"]["oracle_action_coefficients"]["use_chain_target"] > 0
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v7_chain_compass_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v7_result.json"
    checkpoint_path = tmp_path / "event_v7_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v7_chain_compass_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v7_chain_compass_breadcrumbs"
    assert record["role_names"] == list(EVENT_V7_CHAIN_COMPASS_ROLE_NAMES)
    assert record["chain_compass_observation"] is True
    assert record["obs_shape"] == [26, 11, 11]
    assert record["reward_design_details"]["observation_breadcrumbs"]["planes"] == list(
        CHAIN_COMPASS_OBSERVATION_PLANES
    )
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v8_clean_chain_compass_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v8_result.json"
    checkpoint_path = tmp_path / "event_v8_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v8_clean_chain_compass_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v8_clean_chain_compass_breadcrumbs"
    assert record["role_names"] == list(EVENT_V8_CLEAN_CHAIN_COMPASS_ROLE_NAMES)
    assert record["chain_compass_observation"] is True
    assert record["obs_shape"] == [26, 11, 11]
    assert record["reward_design_details"]["off_chain_penalty_coefficients"]["resource_water"] < 0
    assert record["role_shaping_coefficients"]["off_chain_penalties"]["craft_armor"] < 0
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v9_potential_chain_compass_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v9_result.json"
    checkpoint_path = tmp_path / "event_v9_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v9_potential_chain_compass_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v9_potential_chain_compass_breadcrumbs"
    assert record["role_names"] == list(EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES)
    assert record["chain_compass_observation"] is True
    assert record["obs_shape"] == [26, 11, 11]
    assert record["reward_design_details"]["negative_reward_coefficients"] == {}
    assert record["role_shaping_coefficients"]["common"] == {}
    assert record["role_shaping_coefficients"]["negative_reward_coefficients"] == {}
    assert record["mean_role_shaping_return"] == pytest.approx(0.0)
    assert validate_record(record, path=output_path, allow_smoke=True) == []


def test_train_canonical_reward_geometry_event_v10_chain_affordance_mock_smoke(tmp_path):
    pytest.importorskip("sklearn")
    output_path = tmp_path / "event_v10_result.json"
    checkpoint_path = tmp_path / "event_v10_final_model.pt"

    assert (
        train_canonical_main(
            [
                "--env-backend",
                "mock",
                "--reward-design",
                "event_v10_chain_affordance_compass_breadcrumbs",
                "--use-action-mask",
                "--shared-frac",
                "0.0",
                "--seed",
                "0",
                "--total-agent-steps",
                "24",
                "--eval-trials",
                "1",
                "--eval-steps",
                "1",
                "--num-steps",
                "2",
                "--minibatch-size",
                "12",
                "--update-epochs",
                "1",
                "--hidden-dim",
                "8",
                "--embedding-dim",
                "6",
                "--output",
                str(output_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--wandb-mode",
                "disabled",
                "--log-interval",
                "0",
            ]
        )
        == 0
    )

    record = json.loads(output_path.read_text())
    assert record["reward_design"] == "event_v10_chain_affordance_compass_breadcrumbs"
    assert record["role_names"] == list(EVENT_V10_CHAIN_AFFORDANCE_COMPASS_ROLE_NAMES)
    assert record["chain_compass_observation"] is True
    assert record["chain_affordance_action_mask"] is True
    assert record["obs_shape"] == [26, 11, 11]
    assert record["reward_design_details"]["action_affordance_curriculum"]["enabled"] is True
    assert record["role_shaping_coefficients"]["chain_affordance_action_mask"]["reward_penalties_added"] is False
    assert validate_record(record, path=output_path, allow_smoke=True) == []

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["chain_affordance_action_mask"] is True
    assert checkpoint["config"]["chain_compass_observation"] is True


def test_checkpoint_chain_affordance_mask_infers_legacy_v10_config():
    assert _checkpoint_uses_chain_affordance_action_mask(
        {
            "reward_design": "event_v10_chain_affordance_compass_breadcrumbs",
            "chain_affordance_action_mask": False,
        }
    )
    assert _checkpoint_uses_chain_affordance_action_mask({"chain_affordance_action_mask": True})
    assert not _checkpoint_uses_chain_affordance_action_mask(
        {
            "reward_design": "event_v9_potential_chain_compass_breadcrumbs",
            "chain_affordance_action_mask": False,
        }
    )


def _test_v9_potential(stage_name: str, distance: float) -> float:
    closeness = 80.0 - min(80.0, max(0.0, distance))
    return EVENT_V9_POTENTIAL_CHAIN_STAGE_OFFSETS[stage_name] + (
        EVENT_V9_POTENTIAL_CHAIN_CLOSENESS_SCALES[stage_name] * closeness
    )


class _MaskEnv:
    num_agents = 3
    action_space_size = 56

    def __init__(self, action_mask: np.ndarray, navigation: np.ndarray | None) -> None:
        self._action_mask = action_mask
        self._navigation = navigation

    def get_action_mask(self) -> np.ndarray:
        return self._action_mask

    def get_navigation_snapshot(self) -> np.ndarray | None:
        return self._navigation


def _valid_record(checkpoint_path):
    return {
        "schema_version": "canonical_reward_geometry_v1",
        "run_name": "canonical_reward_geometry_primary_alpha0p8_seed0",
        "condition_group": "primary",
        "environment_backend": "mock",
        "shared_frac": 0.8,
        "seed": 0,
        "num_agents": 12,
        "num_teams": 1,
        "map_width": 80,
        "map_height": 80,
        "role_assignment": "agent_id % 3",
        "role_names": ["gatherer", "explorer", "guardian"],
        "role_labels": role_labels().astype(int).tolist(),
        "reward_design": "passive_v0",
        "reward_design_details": {"name": "passive_v0"},
        "role_shaping_enabled": True,
        "separate_encoders": False,
        "total_agent_steps": 24,
        "eval_trials": 1,
        "eval_steps": 1,
        "metta_git_sha": "a" * 40,
        "tribal_village_git_sha": "b" * 40,
        "tribal_village_build_id": "mock-canonical-tribal-village",
        "command": "uv run python v3_experiments/train_canonical_reward_geometry.py --shared-frac 0.8",
        "wandb_entity": "tashapais",
        "wandb_project": "representation-collapse",
        "wandb_run_id": None,
        "wandb_url": None,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(checkpoint_path.parent / "result.json"),
        "obs_shape": [21, 11, 11],
        "action_space_size": 56,
        "metric_schema": {
            "effrank_per_agent": "effective rank divided by n_agents",
            "d_act_ordered_kl": "ordered off-diagonal KL",
            "d_act_js": "Jensen-Shannon diversity",
            "role_probe_acc": "3-way fixed-role probe",
            "role_probe_chance": "1/3",
        },
        "train_metrics": {},
        "effrank_per_agent": 0.5,
        "d_act_ordered_kl": 0.1,
        "d_act_js": 0.01,
        "role_probe_acc": 0.75,
        "role_probe_chance": 1 / 3,
        "role_probe_cv": ROLE_PROBE_CV,
        "eval_trial_metrics": [
            {
                "trial": 0,
                "effrank_per_agent": 0.5,
                "d_act_ordered_kl": 0.1,
                "d_act_js": 0.01,
                "role_probe_acc": 0.75,
                "role_probe_meta": {"cv": ROLE_PROBE_CV, "chance": 1 / 3},
                "mean_return": 1.0,
                "total_return": 12.0,
            }
        ],
    }
