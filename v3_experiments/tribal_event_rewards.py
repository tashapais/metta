"""Event-based Tribal Village reward shaping definitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from v3_experiments.canonical_reward_geometry import CANONICAL_NUM_AGENTS, role_labels
from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS

EVENT_V1_ROLE_NAMES = ("supplier", "crafter_logistics", "defender_territory")
EVENT_V2_BREADCRUMB_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V3_NAVIGATION_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V4_HEART_CHAIN_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V5_NAVIGATION_CHAIN_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V6_ORACLE_CHAIN_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V7_CHAIN_COMPASS_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V8_CLEAN_CHAIN_COMPASS_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V10_CHAIN_AFFORDANCE_COMPASS_ROLE_NAMES = EVENT_V1_ROLE_NAMES
EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES = ("supplier", "crafter_logistics", "depositor")
EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V14_TARGET_AWARE_HANDOFF_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V16_HANDOFF_RELIABILITY_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V17_DEPOSITOR_STAGING_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V18_HANDOFF_RENDEZVOUS_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V20_HOME_STAGED_HANDOFF_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES
EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ROLE_NAMES = EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES

NAV_AGENT_X = 0
NAV_AGENT_Y = 1
NAV_HOME_ASSEMBLER_X = 2
NAV_HOME_ASSEMBLER_Y = 3
NAV_NEAREST_CONVERTER_X = 4
NAV_NEAREST_CONVERTER_Y = 5
NAV_NEAREST_MINE_X = 6
NAV_NEAREST_MINE_Y = 7
NAV_DIST_HOME_ASSEMBLER = 8
NAV_DIST_NEAREST_CONVERTER = 9
NAV_DIST_NEAREST_MINE = 10
NAV_INVENTORY_ORE = 11
NAV_INVENTORY_BATTERY = 12
NAVIGATION_SNAPSHOT_COLUMNS = (
    "agent_x",
    "agent_y",
    "home_assembler_x",
    "home_assembler_y",
    "nearest_converter_x",
    "nearest_converter_y",
    "nearest_mine_x",
    "nearest_mine_y",
    "dist_home_assembler",
    "dist_nearest_converter",
    "dist_nearest_mine",
    "inventory_ore",
    "inventory_battery",
)

EVENT_V1_COMMON_COEFFICIENTS = {
    "action_attack": 0.02,
    "action_use": 0.02,
    "action_put": 0.02,
    "action_plant": 0.02,
    "action_invalid": -0.01,
}

EVENT_V1_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_water": 0.20,
        "resource_wheat": 0.20,
        "resource_wood": 0.20,
        "resource_ore": 0.20,
    },
    "crafter_logistics": {
        "craft_battery": 0.45,
        "craft_spear": 0.45,
        "craft_lantern": 0.45,
        "craft_armor": 0.45,
        "craft_bread": 0.45,
        "deposit_heart": 0.80,
        "put_armor": 0.30,
        "put_bread": 0.30,
    },
    "defender_territory": {
        "tumor_kill": 0.90,
        "spawner_kill": 0.90,
        "agent_kill": 0.40,
        "lantern_plant": 0.45,
    },
}

EVENT_V1_COWORLD_ROLE_SOURCES = {
    "supplier": [
        "Hearter ore collection",
        "Armorer/Hunter wood collection",
        "Baker/Lighter wheat collection",
        "Farmer water and planting loop",
    ],
    "crafter_logistics": [
        "Hearter battery and assembler workflow",
        "Armorer armor production and teammate handoff",
        "Baker bread production and teammate handoff",
        "Lighter lantern production",
        "Hunter spear production",
    ],
    "defender_territory": [
        "Hunter tumor/spawner combat",
        "Lighter lantern planting and territory protection",
    ],
}

EVENT_V2_BREADCRUMB_COMMON_COEFFICIENTS = {
    "action_use": 0.08,
    "action_put": 0.05,
    "action_attack": 0.05,
    "action_plant": 0.05,
    "action_swap": 0.01,
    "action_noop": -0.002,
    "action_invalid": -0.001,
}

EVENT_V2_BREADCRUMB_TASK_COEFFICIENTS = {
    "resource_water": 0.10,
    "resource_wheat": 0.10,
    "resource_wood": 0.10,
    "resource_ore": 0.10,
    "craft_battery": 0.25,
    "craft_spear": 0.25,
    "craft_lantern": 0.25,
    "craft_armor": 0.25,
    "craft_bread": 0.25,
    "deposit_heart": 0.70,
    "put_armor": 0.25,
    "put_bread": 0.25,
    "tumor_kill": 0.70,
    "spawner_kill": 0.70,
    "agent_kill": 0.25,
    "lantern_plant": 0.35,
}

EVENT_V2_BREADCRUMB_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_water": 0.10,
        "resource_wheat": 0.10,
        "resource_wood": 0.10,
        "resource_ore": 0.10,
    },
    "crafter_logistics": {
        "craft_battery": 0.20,
        "craft_spear": 0.20,
        "craft_lantern": 0.20,
        "craft_armor": 0.20,
        "craft_bread": 0.20,
        "deposit_heart": 0.30,
        "put_armor": 0.15,
        "put_bread": 0.15,
    },
    "defender_territory": {
        "tumor_kill": 0.30,
        "spawner_kill": 0.30,
        "agent_kill": 0.15,
        "lantern_plant": 0.15,
    },
}

EVENT_V3_NAVIGATION_COMMON_COEFFICIENTS = {
    "action_move": 0.004,
    "action_use": 0.05,
    "action_put": 0.05,
    "action_attack": 0.05,
    "action_plant": 0.05,
    "action_swap": 0.005,
    "action_noop": -0.004,
    "action_invalid": -0.004,
}

EVENT_V3_NAVIGATION_COMMON_CAPS = {
    "action_move": 40,
}

EVENT_V3_NAVIGATION_TASK_COEFFICIENTS = {
    "resource_water": 0.45,
    "resource_wheat": 0.45,
    "resource_wood": 0.45,
    "resource_ore": 0.45,
    "craft_battery": 1.00,
    "craft_spear": 1.00,
    "craft_lantern": 1.00,
    "craft_armor": 1.00,
    "craft_bread": 1.00,
    "deposit_heart": 2.00,
    "put_armor": 0.75,
    "put_bread": 0.75,
    "tumor_kill": 2.00,
    "spawner_kill": 2.00,
    "agent_kill": 0.75,
    "lantern_plant": 1.00,
}

EVENT_V3_NAVIGATION_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_water": 0.20,
        "resource_wheat": 0.20,
        "resource_wood": 0.20,
        "resource_ore": 0.20,
    },
    "crafter_logistics": {
        "craft_battery": 0.35,
        "craft_spear": 0.35,
        "craft_lantern": 0.35,
        "craft_armor": 0.35,
        "craft_bread": 0.35,
        "deposit_heart": 0.50,
        "put_armor": 0.25,
        "put_bread": 0.25,
    },
    "defender_territory": {
        "tumor_kill": 0.50,
        "spawner_kill": 0.50,
        "agent_kill": 0.25,
        "lantern_plant": 0.30,
    },
}

EVENT_V4_HEART_CHAIN_COMMON_COEFFICIENTS = {
    "action_move": 0.003,
    "action_use": 0.03,
    "action_put": 0.005,
    "action_attack": 0.03,
    "action_plant": 0.02,
    "action_swap": 0.001,
    "action_noop": -0.006,
    "action_invalid": -0.006,
}

EVENT_V4_HEART_CHAIN_COMMON_CAPS = {
    "action_move": 80,
}

EVENT_V4_HEART_CHAIN_TASK_COEFFICIENTS = {
    "resource_water": 0.08,
    "resource_wheat": 0.08,
    "resource_wood": 0.08,
    "resource_ore": 1.20,
    "craft_battery": 6.00,
    "craft_spear": 0.10,
    "craft_lantern": 0.10,
    "craft_armor": 0.10,
    "craft_bread": 0.10,
    "deposit_heart": 20.00,
    "put_armor": 0.02,
    "put_bread": 0.02,
    "tumor_kill": 1.00,
    "spawner_kill": 1.00,
    "agent_kill": 0.20,
    "lantern_plant": 0.20,
}

EVENT_V4_HEART_CHAIN_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_water": 0.05,
        "resource_wheat": 0.05,
        "resource_wood": 0.05,
        "resource_ore": 0.80,
    },
    "crafter_logistics": {
        "craft_battery": 4.00,
        "craft_spear": 0.10,
        "craft_lantern": 0.10,
        "craft_armor": 0.10,
        "craft_bread": 0.10,
        "deposit_heart": 10.00,
        "put_armor": 0.02,
        "put_bread": 0.02,
    },
    "defender_territory": {
        "tumor_kill": 0.50,
        "spawner_kill": 0.50,
        "agent_kill": 0.10,
        "lantern_plant": 0.10,
    },
}

EVENT_V5_NAVIGATION_CHAIN_COMMON_COEFFICIENTS = {
    "action_move": 0.001,
    "action_use": 0.02,
    "action_put": 0.002,
    "action_attack": 0.02,
    "action_plant": 0.01,
    "action_swap": 0.001,
    "action_noop": -0.006,
    "action_invalid": -0.006,
}

EVENT_V5_NAVIGATION_CHAIN_COMMON_CAPS = {
    "action_move": 120,
}

EVENT_V5_NAVIGATION_CHAIN_TASK_COEFFICIENTS = {
    "resource_water": 0.04,
    "resource_wheat": 0.04,
    "resource_wood": 0.04,
    "resource_ore": 0.80,
    "craft_battery": 4.00,
    "craft_spear": 0.05,
    "craft_lantern": 0.05,
    "craft_armor": 0.05,
    "craft_bread": 0.05,
    "deposit_heart": 24.00,
    "put_armor": 0.01,
    "put_bread": 0.01,
    "tumor_kill": 0.60,
    "spawner_kill": 0.60,
    "agent_kill": 0.10,
    "lantern_plant": 0.10,
}

EVENT_V5_NAVIGATION_CHAIN_PROGRESS_COEFFICIENTS = {
    "toward_mine_empty": 0.02,
    "ore_to_converter": 0.30,
    "battery_to_home_assembler": 0.80,
    "battery_adjacent_home_assembler": 0.15,
}

EVENT_V5_NAVIGATION_CHAIN_PROGRESS_CAPS = {
    "toward_mine_empty": 80,
    "ore_to_converter": 80,
    "battery_to_home_assembler": 80,
    "battery_adjacent_home_assembler": 12,
}

EVENT_V5_NAVIGATION_CHAIN_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_water": 0.03,
        "resource_wheat": 0.03,
        "resource_wood": 0.03,
        "resource_ore": 0.40,
    },
    "crafter_logistics": {
        "craft_battery": 2.00,
        "craft_spear": 0.05,
        "craft_lantern": 0.05,
        "craft_armor": 0.05,
        "craft_bread": 0.05,
        "deposit_heart": 12.00,
        "put_armor": 0.01,
        "put_bread": 0.01,
    },
    "defender_territory": {
        "tumor_kill": 0.30,
        "spawner_kill": 0.30,
        "agent_kill": 0.05,
        "lantern_plant": 0.05,
    },
}

EVENT_V6_ORACLE_CHAIN_COMMON_COEFFICIENTS = {
    "action_noop": -0.010,
    "action_invalid": -0.020,
}

EVENT_V6_ORACLE_CHAIN_TASK_COEFFICIENTS = {
    "resource_ore": 1.00,
    "craft_battery": 8.00,
    "deposit_heart": 40.00,
}

EVENT_V6_ORACLE_CHAIN_PROGRESS_COEFFICIENTS = {
    "toward_mine_empty": 0.08,
    "ore_to_converter": 0.80,
    "battery_to_home_assembler": 1.50,
    "battery_adjacent_home_assembler": 0.50,
}

EVENT_V6_ORACLE_CHAIN_PROGRESS_CAPS = {
    "toward_mine_empty": 160,
    "ore_to_converter": 160,
    "battery_to_home_assembler": 160,
    "battery_adjacent_home_assembler": 24,
}

EVENT_V6_ORACLE_CHAIN_ACTION_COEFFICIENTS = {
    "move_toward_chain_target": 0.05,
    "use_chain_target": 1.00,
}

EVENT_V6_ORACLE_CHAIN_ACTION_CAPS = {
    "move_toward_chain_target": 160,
    "use_chain_target": 80,
}

EVENT_V6_ORACLE_CHAIN_ROLE_COEFFICIENTS = {role: {} for role in EVENT_V6_ORACLE_CHAIN_ROLE_NAMES}

EVENT_V11_ROLE_GATED_CHAIN_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_ore": 2.0,
        "put_ore": 6.0,
    },
    "crafter_logistics": {
        "craft_battery": 10.0,
        "put_battery": 12.0,
    },
    "depositor": {
        "deposit_heart": 50.0,
    },
}

EVENT_V11_ROLE_GATED_CHAIN_STAGE_OFFSETS = {
    "supplier_empty_to_mine": 0.0,
    "supplier_ore_to_converter": 4.0,
    "crafter_empty_to_mine": 0.0,
    "crafter_ore_to_converter": 5.0,
    "crafter_battery_to_home": 8.0,
    "depositor_empty_to_home": 2.0,
    "depositor_battery_to_home": 10.0,
}

EVENT_V11_ROLE_GATED_CHAIN_CLOSENESS_SCALES = {
    "supplier_empty_to_mine": 0.04,
    "supplier_ore_to_converter": 0.08,
    "crafter_empty_to_mine": 0.03,
    "crafter_ore_to_converter": 0.10,
    "crafter_battery_to_home": 0.08,
    "depositor_empty_to_home": 0.05,
    "depositor_battery_to_home": 0.12,
}

EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_ore": 2.0,
        "put_ore": 0.5,
        "put_ore_to_crafter": 10.0,
    },
    "crafter_logistics": {
        "receive_ore_from_supplier": 4.0,
        "craft_battery": 10.0,
        "put_battery": 0.5,
        "put_battery_to_depositor": 16.0,
    },
    "depositor": {
        "receive_battery_from_crafter": 14.0,
        "deposit_heart": 70.0,
    },
}

EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_STAGE_OFFSETS = {
    "supplier_empty_to_mine": 0.0,
    "supplier_ore_to_converter": 4.0,
    "crafter_empty_to_mine": 0.0,
    "crafter_ore_to_converter": 5.0,
    "crafter_battery_to_home": 8.0,
    "depositor_empty_to_home": 4.0,
    "depositor_battery_to_home": 14.0,
}

EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_CLOSENESS_SCALES = {
    "supplier_empty_to_mine": 0.04,
    "supplier_ore_to_converter": 0.08,
    "crafter_empty_to_mine": 0.03,
    "crafter_ore_to_converter": 0.10,
    "crafter_battery_to_home": 0.08,
    "depositor_empty_to_home": 0.10,
    "depositor_battery_to_home": 0.25,
}

EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_COEFFICIENTS = {
    "supplier": {
        "resource_ore": 2.0,
        "put_ore": 6.0,
        "put_ore_to_crafter": 2.0,
    },
    "crafter_logistics": {
        "craft_battery": 10.0,
        "put_battery": 12.0,
        "put_battery_to_depositor": 4.0,
    },
    "depositor": {
        "receive_battery_from_crafter": 8.0,
        "deposit_heart": 70.0,
    },
}

EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ACTION_COEFFICIENTS = {
    "depositor_use_home_assembler": 2.0,
}

EVENT_V14_TARGET_AWARE_HANDOFF_ROLE_COEFFICIENTS = EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_COEFFICIENTS

EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_COEFFICIENTS = EVENT_V14_TARGET_AWARE_HANDOFF_ROLE_COEFFICIENTS

EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_COEFFICIENTS = {
    "depositor_battery_move_toward_home": 0.50,
    "depositor_battery_arrive_adjacent_home": 4.00,
}

EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_CAPS = {
    "depositor_battery_move_toward_home": 160,
    "depositor_battery_arrive_adjacent_home": 160,
}

EVENT_V16_HANDOFF_RELIABILITY_ROLE_COEFFICIENTS = {
    "supplier": dict(EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_COEFFICIENTS["supplier"]),
    "crafter_logistics": {
        "craft_battery": 10.0,
        "put_battery": 12.0,
        "put_battery_to_depositor": 16.0,
    },
    "depositor": {
        "receive_battery_from_crafter": 14.0,
        "deposit_heart": 80.0,
    },
}

EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS = {
    "depositor_use_home_assembler": 6.0,
    "depositor_battery_move_toward_home": 0.75,
    "depositor_battery_arrive_adjacent_home": 6.00,
}

EVENT_V16_HANDOFF_RELIABILITY_ACTION_CAPS = dict(EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_CAPS)

EVENT_V17_DEPOSITOR_STAGING_ROLE_COEFFICIENTS = EVENT_V16_HANDOFF_RELIABILITY_ROLE_COEFFICIENTS

EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS = {
    **EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS,
    "depositor_empty_move_toward_home": 0.25,
    "depositor_empty_arrive_adjacent_home": 2.00,
}

EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS = {
    **EVENT_V16_HANDOFF_RELIABILITY_ACTION_CAPS,
    "depositor_empty_move_toward_home": 160,
    "depositor_empty_arrive_adjacent_home": 160,
}

EVENT_V18_HANDOFF_RENDEZVOUS_ROLE_COEFFICIENTS = EVENT_V17_DEPOSITOR_STAGING_ROLE_COEFFICIENTS

EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS = {
    **EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS,
    "crafter_battery_move_toward_empty_depositor": 0.50,
    "crafter_battery_arrive_adjacent_empty_depositor": 4.00,
    "depositor_empty_move_toward_battery_crafter": 0.35,
    "depositor_empty_arrive_adjacent_battery_crafter": 3.00,
}

EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_CAPS = {
    **EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS,
    "crafter_battery_move_toward_empty_depositor": 160,
    "crafter_battery_arrive_adjacent_empty_depositor": 160,
    "depositor_empty_move_toward_battery_crafter": 160,
    "depositor_empty_arrive_adjacent_battery_crafter": 160,
}

EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_COEFFICIENTS = EVENT_V17_DEPOSITOR_STAGING_ROLE_COEFFICIENTS

EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS = {
    **EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS,
    "crafter_battery_move_toward_home": 0.75,
    "crafter_battery_arrive_adjacent_home": 6.00,
}

EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_CAPS = {
    **EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS,
    "crafter_battery_move_toward_home": 160,
    "crafter_battery_arrive_adjacent_home": 160,
}

EVENT_V20_HOME_STAGED_HANDOFF_ROLE_COEFFICIENTS = EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_COEFFICIENTS

EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS = {
    **EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS,
    "crafter_battery_put_to_home_staged_depositor": 8.00,
}

EVENT_V20_HOME_STAGED_HANDOFF_ACTION_CAPS = {
    **EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_CAPS,
    "crafter_battery_put_to_home_staged_depositor": 160,
}

EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ROLE_COEFFICIENTS = EVENT_V20_HOME_STAGED_HANDOFF_ROLE_COEFFICIENTS

EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS = {
    **EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS,
    "crafter_battery_put_to_home_staged_depositor": 20.00,
    "crafter_battery_move_toward_home_staged_depositor": 0.75,
    "crafter_battery_arrive_adjacent_home_staged_depositor": 6.00,
}

EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_CAPS = {
    **EVENT_V20_HOME_STAGED_HANDOFF_ACTION_CAPS,
    "crafter_battery_move_toward_home_staged_depositor": 160,
    "crafter_battery_arrive_adjacent_home_staged_depositor": 160,
}

EVENT_V8_CLEAN_CHAIN_COMPASS_OFFCHAIN_PENALTIES = {
    "resource_water": -0.10,
    "resource_wheat": -0.10,
    "resource_wood": -0.10,
    "craft_spear": -1.00,
    "craft_lantern": -1.00,
    "craft_armor": -1.00,
    "craft_bread": -1.00,
    "put_armor": -0.50,
    "put_bread": -0.50,
    "tumor_kill": -0.25,
    "spawner_kill": -0.25,
    "agent_kill": -0.25,
    "lantern_plant": -0.25,
}

EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA = 0.99
EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE = 80.0
EVENT_V9_POTENTIAL_CHAIN_STAGE_OFFSETS = {
    "empty_to_mine": 0.0,
    "ore_to_converter": 4.0,
    "battery_to_home_assembler": 10.0,
}
EVENT_V9_POTENTIAL_CHAIN_CLOSENESS_SCALES = {
    "empty_to_mine": 0.04,
    "ore_to_converter": 0.08,
    "battery_to_home_assembler": 0.12,
}

ACTION_ARGUMENT_COUNT = 8
MOVE_VERB = 1
USE_VERB = 3
PUT_VERB = 5
ORIENTATION_DELTAS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)
ORIENTATION_BY_DELTA = {delta: index for index, delta in enumerate(ORIENTATION_DELTAS)}

_STAT_INDEX = {name: idx for idx, name in enumerate(SIMULATOR_STAT_COLUMNS)}


def event_v1_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    *,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return per-agent shaping bonuses from successful simulator event deltas.

    ``event_stats_delta`` is the per-step difference of the cumulative
    ``SIMULATOR_STAT_COLUMNS`` counters exposed by the Nim environment. Missing
    stats return zeros so the mock backend and older libraries remain usable.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    _add_agent_coefficients(bonuses, stats, EVENT_V1_COMMON_COEFFICIENTS)
    _add_role_coefficients(bonuses, stats, EVENT_V1_ROLE_NAMES, EVENT_V1_ROLE_COEFFICIENTS)
    return bonuses


def event_v2_breadcrumb_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    *,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return exploratory breadcrumb rewards for discovering task events.

    This is intentionally not the final paper reward. It pays every agent a
    role-agnostic bonus for successful task events, then adds role-specific
    bonuses on top. The purpose is to help PPO discover the event surface before
    rerunning a cleaner role-specialized reward.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    _add_agent_coefficients(bonuses, stats, EVENT_V2_BREADCRUMB_COMMON_COEFFICIENTS)
    _add_agent_coefficients(bonuses, stats, EVENT_V2_BREADCRUMB_TASK_COEFFICIENTS)
    _add_role_coefficients(
        bonuses,
        stats,
        EVENT_V2_BREADCRUMB_ROLE_NAMES,
        EVENT_V2_BREADCRUMB_ROLE_COEFFICIENTS,
    )
    return bonuses


def event_v3_navigation_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return navigation-first breadcrumbs that reject blind invalid use spam.

    ``event_v2_breadcrumbs`` proved that task-event rewards can fire, but PPO
    found a bad local optimum: repeat one ``use`` direction forever and collect
    the rare accidental resource pickup. This design keeps the task-event
    breadcrumbs, adds a capped successful-movement bonus so agents can discover
    objects, and makes invalid actions expensive enough that fixed use spam is
    not competitive.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)
    totals = _validate_event_stats_total(event_stats_total, num_agents)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    uncapped_common = {
        name: coefficient
        for name, coefficient in EVENT_V3_NAVIGATION_COMMON_COEFFICIENTS.items()
        if name not in EVENT_V3_NAVIGATION_COMMON_CAPS
    }
    capped_common = {
        name: coefficient
        for name, coefficient in EVENT_V3_NAVIGATION_COMMON_COEFFICIENTS.items()
        if name in EVENT_V3_NAVIGATION_COMMON_CAPS
    }
    _add_agent_coefficients(bonuses, stats, uncapped_common)
    _add_capped_agent_coefficients(bonuses, stats, totals, capped_common, EVENT_V3_NAVIGATION_COMMON_CAPS)
    _add_agent_coefficients(bonuses, stats, EVENT_V3_NAVIGATION_TASK_COEFFICIENTS)
    _add_role_coefficients(
        bonuses,
        stats,
        EVENT_V3_NAVIGATION_ROLE_NAMES,
        EVENT_V3_NAVIGATION_ROLE_COEFFICIENTS,
    )
    return bonuses


def event_v4_heart_chain_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return heart-chain breadcrumbs for ore -> battery -> assembler deposits.

    ``event_v3_navigation_breadcrumbs`` plus action masking made agents produce
    task events, but long runs plateaued on resource pickup and easy handoff
    loops with zero heart deposits. This design keeps capped movement and low
    invalid-action tolerance while making the heart chain dominate shaping.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)
    totals = _validate_event_stats_total(event_stats_total, num_agents)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    uncapped_common = {
        name: coefficient
        for name, coefficient in EVENT_V4_HEART_CHAIN_COMMON_COEFFICIENTS.items()
        if name not in EVENT_V4_HEART_CHAIN_COMMON_CAPS
    }
    capped_common = {
        name: coefficient
        for name, coefficient in EVENT_V4_HEART_CHAIN_COMMON_COEFFICIENTS.items()
        if name in EVENT_V4_HEART_CHAIN_COMMON_CAPS
    }
    _add_agent_coefficients(bonuses, stats, uncapped_common)
    _add_capped_agent_coefficients(bonuses, stats, totals, capped_common, EVENT_V4_HEART_CHAIN_COMMON_CAPS)
    _add_agent_coefficients(bonuses, stats, EVENT_V4_HEART_CHAIN_TASK_COEFFICIENTS)
    _add_role_coefficients(
        bonuses,
        stats,
        EVENT_V4_HEART_CHAIN_ROLE_NAMES,
        EVENT_V4_HEART_CHAIN_ROLE_COEFFICIENTS,
    )
    return bonuses


def event_v5_navigation_chain_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return heart-chain rewards with inventory-conditioned navigation progress.

    ``event_v4_heart_chain_breadcrumbs`` discovered ore and sometimes battery
    production, but policies still failed to return battery carriers to their
    home assembler. This design keeps terminal task-event rewards and adds
    dense, capped progress rewards for the next useful target in the chain:
    empty-handed agents toward mines, ore carriers toward converters, and
    battery carriers toward the home assembler.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)
    totals = _validate_event_stats_total(event_stats_total, num_agents)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    uncapped_common = {
        name: coefficient
        for name, coefficient in EVENT_V5_NAVIGATION_CHAIN_COMMON_COEFFICIENTS.items()
        if name not in EVENT_V5_NAVIGATION_CHAIN_COMMON_CAPS
    }
    capped_common = {
        name: coefficient
        for name, coefficient in EVENT_V5_NAVIGATION_CHAIN_COMMON_COEFFICIENTS.items()
        if name in EVENT_V5_NAVIGATION_CHAIN_COMMON_CAPS
    }
    _add_agent_coefficients(bonuses, stats, uncapped_common)
    _add_capped_agent_coefficients(bonuses, stats, totals, capped_common, EVENT_V5_NAVIGATION_CHAIN_COMMON_CAPS)
    _add_agent_coefficients(bonuses, stats, EVENT_V5_NAVIGATION_CHAIN_TASK_COEFFICIENTS)
    _add_role_coefficients(
        bonuses,
        stats,
        EVENT_V5_NAVIGATION_CHAIN_ROLE_NAMES,
        EVENT_V5_NAVIGATION_CHAIN_ROLE_COEFFICIENTS,
    )
    _add_navigation_progress_bonuses(bonuses, navigation_before, navigation_after, totals)
    return bonuses


def event_v6_oracle_chain_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return chain-only breadcrumbs aligned to a simple ore-to-heart oracle.

    ``event_v5_navigation_chain_breadcrumbs`` still rewarded enough off-chain
    activity that policies gathered water/wheat/wood, fought tumors, and rarely
    completed battery deposits. This debug design focuses shaping on the single
    ore -> battery -> home-assembler chain and pays a small dense bonus only
    when the chosen action matches a mask-valid oracle move/use action.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is None:
        return np.zeros(num_agents, dtype=np.float64)
    totals = _validate_event_stats_total(event_stats_total, num_agents)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    _add_agent_coefficients(bonuses, stats, EVENT_V6_ORACLE_CHAIN_COMMON_COEFFICIENTS)
    _add_agent_coefficients(bonuses, stats, EVENT_V6_ORACLE_CHAIN_TASK_COEFFICIENTS)
    _add_navigation_progress_bonuses(
        bonuses,
        navigation_before,
        navigation_after,
        totals,
        progress_coefficients=EVENT_V6_ORACLE_CHAIN_PROGRESS_COEFFICIENTS,
        progress_caps=EVENT_V6_ORACLE_CHAIN_PROGRESS_CAPS,
    )
    _add_chain_oracle_action_bonuses(bonuses, navigation_before, actions, action_mask, totals)
    return bonuses


def event_v8_clean_chain_compass_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v7 chain-compass rewards with small off-chain event penalties.

    ``event_v7_chain_compass_breadcrumbs`` produced real heart deposits, but
    behavior gates still showed frequent water/wheat/wood collection, non-chain
    crafting, and combat. This v8 debug design keeps the successful chain
    incentives intact while charging bounded penalties for those off-chain
    successful task events.
    """

    bonuses = event_v6_oracle_chain_role_shaping_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        num_agents=num_agents,
    )
    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    if stats is not None:
        _add_agent_coefficients(bonuses, stats, EVENT_V8_CLEAN_CHAIN_COMPASS_OFFCHAIN_PENALTIES)
    return bonuses


def event_v9_potential_chain_compass_role_shaping_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v7 chain-compass rewards with potential-based chain progress.

    This v9 debug design removes explicit noop/invalid and off-chain penalty
    coefficients. Dense navigation shaping is instead the potential difference
    ``gamma * Phi(s') - Phi(s)`` over the ore -> battery -> heart chain state.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    totals = _validate_event_stats_total(event_stats_total, num_agents)

    bonuses = np.zeros(num_agents, dtype=np.float64)
    if stats is not None:
        _add_agent_coefficients(bonuses, stats, EVENT_V6_ORACLE_CHAIN_TASK_COEFFICIENTS)
    _add_chain_potential_bonuses(bonuses, navigation_before, navigation_after, gamma=gamma)
    _add_chain_oracle_action_bonuses(bonuses, navigation_before, actions, action_mask, totals)
    return bonuses


def event_v11_role_gated_chain_handoff_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return rewards for a handoff-required ore -> battery -> heart chain.

    This is a diagnostic curriculum, not a negative-reward patch. The training
    action mask splits the chain across fixed roles: suppliers mine ore and hand
    it off, crafters convert ore into batteries and hand those off, and
    depositors turn batteries into hearts at the home assembler.
    """

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    bonuses = np.zeros(num_agents, dtype=np.float64)
    if stats is not None:
        _add_role_coefficients(
            bonuses,
            stats,
            EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES,
            EVENT_V11_ROLE_GATED_CHAIN_ROLE_COEFFICIENTS,
        )
    _add_role_gated_chain_potential_bonuses(bonuses, navigation_before, navigation_after, gamma=gamma)
    return bonuses


def event_v12_role_gated_depositor_reliability_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v11 rewards with correct-recipient handoffs and stronger depositor breadcrumbs."""

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    bonuses = np.zeros(num_agents, dtype=np.float64)
    if stats is not None:
        _add_role_coefficients(
            bonuses,
            stats,
            EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_NAMES,
            EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_COEFFICIENTS,
        )
    _add_role_gated_chain_potential_bonuses(
        bonuses,
        navigation_before,
        navigation_after,
        gamma=gamma,
        stage_offsets=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_STAGE_OFFSETS,
        closeness_scales=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_CLOSENESS_SCALES,
    )
    return bonuses


def event_v13_role_gated_depositor_use_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v11-style handoff rewards plus depositor receive/use breadcrumbs."""

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    bonuses = np.zeros(num_agents, dtype=np.float64)
    if stats is not None:
        _add_role_coefficients(
            bonuses,
            stats,
            EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_NAMES,
            EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_COEFFICIENTS,
        )
    _add_role_gated_chain_potential_bonuses(
        bonuses,
        navigation_before,
        navigation_after,
        gamma=gamma,
        stage_offsets=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_STAGE_OFFSETS,
        closeness_scales=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_CLOSENESS_SCALES,
    )
    _add_depositor_home_use_breadcrumb(bonuses, navigation_before, actions, action_mask)
    return bonuses


def event_v14_target_aware_handoff_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v13 positive rewards under the v14 target-aware handoff mask."""

    return event_v13_role_gated_depositor_use_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )


def event_v15_depositor_final_mile_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v14 rewards plus positive-only depositor battery-to-home breadcrumbs."""

    bonuses = event_v14_target_aware_handoff_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_depositor_final_mile_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v16_handoff_reliability_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v15-style rewards with stronger positive final handoff breadcrumbs."""

    stats = _validate_event_stats_delta(event_stats_delta, num_agents)
    bonuses = np.zeros(num_agents, dtype=np.float64)
    if stats is not None:
        _add_role_coefficients(
            bonuses,
            stats,
            EVENT_V16_HANDOFF_RELIABILITY_ROLE_NAMES,
            EVENT_V16_HANDOFF_RELIABILITY_ROLE_COEFFICIENTS,
        )
    _add_role_gated_chain_potential_bonuses(
        bonuses,
        navigation_before,
        navigation_after,
        gamma=gamma,
        stage_offsets=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_STAGE_OFFSETS,
        closeness_scales=EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_CLOSENESS_SCALES,
    )
    _add_depositor_home_use_breadcrumb(
        bonuses,
        navigation_before,
        actions,
        action_mask,
        coefficient=EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS["depositor_use_home_assembler"],
    )
    _add_depositor_final_mile_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
        action_coefficients=EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS,
        action_caps=EVENT_V16_HANDOFF_RELIABILITY_ACTION_CAPS,
    )
    return bonuses


def event_v17_depositor_staging_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v16 rewards plus positive staging breadcrumbs for empty depositors."""

    bonuses = event_v16_handoff_reliability_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_depositor_empty_staging_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v18_handoff_rendezvous_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v16 rewards plus positive rendezvous breadcrumbs for final handoff."""

    bonuses = event_v16_handoff_reliability_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_handoff_rendezvous_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v19_crafter_home_delivery_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v17 rewards plus positive crafter battery-to-home breadcrumbs."""

    bonuses = event_v17_depositor_staging_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_crafter_battery_home_delivery_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v20_home_staged_handoff_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v19 rewards plus a positive masked-put breadcrumb near home."""

    bonuses = event_v19_crafter_home_delivery_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_crafter_home_staged_handoff_breadcrumb(
        bonuses,
        navigation_before,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v21_home_handoff_rendezvous_bonuses(
    event_stats_delta: np.ndarray | None,
    event_stats_total: np.ndarray | None = None,
    *,
    navigation_before: np.ndarray | None = None,
    navigation_after: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    gamma: float = EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
    num_agents: int = CANONICAL_NUM_AGENTS,
) -> np.ndarray:
    """Return v20 rewards plus home-staged final-handoff rendezvous breadcrumbs."""

    bonuses = event_v19_crafter_home_delivery_bonuses(
        event_stats_delta,
        event_stats_total,
        navigation_before=navigation_before,
        navigation_after=navigation_after,
        actions=actions,
        action_mask=action_mask,
        gamma=gamma,
        num_agents=num_agents,
    )
    _add_crafter_home_staged_handoff_breadcrumb(
        bonuses,
        navigation_before,
        actions,
        action_mask,
        event_stats_total,
        coefficient=EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
            "crafter_battery_put_to_home_staged_depositor"
        ],
        cap=EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_CAPS[
            "crafter_battery_put_to_home_staged_depositor"
        ],
    )
    _add_crafter_home_staged_depositor_rendezvous_breadcrumbs(
        bonuses,
        navigation_before,
        navigation_after,
        actions,
        action_mask,
        event_stats_total,
    )
    return bonuses


def event_v1_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the event-v1 reward design."""

    return {
        "name": "event_v1",
        "summary": "Coworld-derived three-role shaping paid only on simulator event deltas.",
        "role_names": list(EVENT_V1_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V1_COMMON_COEFFICIENTS),
        "role_coefficients": {role: dict(coeffs) for role, coeffs in EVENT_V1_ROLE_COEFFICIENTS.items()},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v2_breadcrumb_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the breadcrumb reward design."""

    return {
        "name": "event_v2_breadcrumbs",
        "summary": (
            "Exploratory discovery reward: all roles get task-event breadcrumbs, "
            "with Coworld-derived role bonuses layered on top."
        ),
        "role_names": list(EVENT_V2_BREADCRUMB_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V2_BREADCRUMB_COMMON_COEFFICIENTS),
        "task_event_coefficients": dict(EVENT_V2_BREADCRUMB_TASK_COEFFICIENTS),
        "role_coefficients": {role: dict(coeffs) for role, coeffs in EVENT_V2_BREADCRUMB_ROLE_COEFFICIENTS.items()},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v3_navigation_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the navigation reward design."""

    return {
        "name": "event_v3_navigation_breadcrumbs",
        "summary": (
            "Exploratory reward with capped successful movement, stronger task-event "
            "breadcrumbs, and a larger invalid-action penalty to avoid fixed use spam."
        ),
        "role_names": list(EVENT_V3_NAVIGATION_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V3_NAVIGATION_COMMON_COEFFICIENTS),
        "common_caps": dict(EVENT_V3_NAVIGATION_COMMON_CAPS),
        "task_event_coefficients": dict(EVENT_V3_NAVIGATION_TASK_COEFFICIENTS),
        "role_coefficients": {role: dict(coeffs) for role, coeffs in EVENT_V3_NAVIGATION_ROLE_COEFFICIENTS.items()},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v4_heart_chain_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the heart-chain reward design."""

    return {
        "name": "event_v4_heart_chain_breadcrumbs",
        "summary": (
            "Exploratory reward that keeps capped navigation but prioritizes "
            "the ore -> battery -> assembler-heart chain over easy handoff loops."
        ),
        "role_names": list(EVENT_V4_HEART_CHAIN_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V4_HEART_CHAIN_COMMON_COEFFICIENTS),
        "common_caps": dict(EVENT_V4_HEART_CHAIN_COMMON_CAPS),
        "task_event_coefficients": dict(EVENT_V4_HEART_CHAIN_TASK_COEFFICIENTS),
        "role_coefficients": {role: dict(coeffs) for role, coeffs in EVENT_V4_HEART_CHAIN_ROLE_COEFFICIENTS.items()},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v5_navigation_chain_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the navigation-chain design."""

    return {
        "name": "event_v5_navigation_chain_breadcrumbs",
        "summary": (
            "Exploratory reward that adds inventory-conditioned progress toward "
            "mine, converter, and home-assembler targets to close the "
            "ore -> battery -> heart-deposit loop."
        ),
        "role_names": list(EVENT_V5_NAVIGATION_CHAIN_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V5_NAVIGATION_CHAIN_COMMON_COEFFICIENTS),
        "common_caps": dict(EVENT_V5_NAVIGATION_CHAIN_COMMON_CAPS),
        "task_event_coefficients": dict(EVENT_V5_NAVIGATION_CHAIN_TASK_COEFFICIENTS),
        "navigation_progress_coefficients": dict(EVENT_V5_NAVIGATION_CHAIN_PROGRESS_COEFFICIENTS),
        "navigation_progress_caps": dict(EVENT_V5_NAVIGATION_CHAIN_PROGRESS_CAPS),
        "navigation_snapshot_columns": list(NAVIGATION_SNAPSHOT_COLUMNS),
        "role_coefficients": {
            role: dict(coeffs) for role, coeffs in EVENT_V5_NAVIGATION_CHAIN_ROLE_COEFFICIENTS.items()
        },
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v6_oracle_chain_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the oracle-chain design."""

    return {
        "name": "event_v6_oracle_chain_breadcrumbs",
        "summary": (
            "Debug reward that strips off-chain task bonuses and adds mask-valid "
            "oracle-action breadcrumbs for the ore -> battery -> heart-deposit chain."
        ),
        "role_names": list(EVENT_V6_ORACLE_CHAIN_ROLE_NAMES),
        "common_coefficients": dict(EVENT_V6_ORACLE_CHAIN_COMMON_COEFFICIENTS),
        "task_event_coefficients": dict(EVENT_V6_ORACLE_CHAIN_TASK_COEFFICIENTS),
        "navigation_progress_coefficients": dict(EVENT_V6_ORACLE_CHAIN_PROGRESS_COEFFICIENTS),
        "navigation_progress_caps": dict(EVENT_V6_ORACLE_CHAIN_PROGRESS_CAPS),
        "oracle_action_coefficients": dict(EVENT_V6_ORACLE_CHAIN_ACTION_COEFFICIENTS),
        "oracle_action_caps": dict(EVENT_V6_ORACLE_CHAIN_ACTION_CAPS),
        "navigation_snapshot_columns": list(NAVIGATION_SNAPSHOT_COLUMNS),
        "role_coefficients": {role: dict(coeffs) for role, coeffs in EVENT_V6_ORACLE_CHAIN_ROLE_COEFFICIENTS.items()},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v7_chain_compass_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the chain-compass design."""

    details = event_v6_oracle_chain_reward_design_details()
    details.update(
        {
            "name": "event_v7_chain_compass_breadcrumbs",
            "summary": (
                "Debug reward that keeps the v6 chain-only oracle breadcrumbs "
                "and adds observation planes for the current ore -> battery -> "
                "heart-deposit target."
            ),
            "role_names": list(EVENT_V7_CHAIN_COMPASS_ROLE_NAMES),
            "observation_breadcrumbs": {
                "planes": [
                    "chain_target_dx_sign",
                    "chain_target_dy_sign",
                    "chain_inventory_stage",
                    "chain_target_closeness",
                    "chain_target_adjacent",
                ],
                "source": "navigation snapshot exposed by the canonical Tribal Village build",
                "purpose": (
                    "Make the currently shaped chain target observable to the "
                    "feed-forward policy instead of relying on hidden global map state."
                ),
            },
        }
    )
    return details


def event_v8_clean_chain_compass_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the clean chain-compass design."""

    details = event_v7_chain_compass_reward_design_details()
    details.update(
        {
            "name": "event_v8_clean_chain_compass_breadcrumbs",
            "summary": (
                "Debug reward that keeps the v7 chain-compass intervention "
                "and adds small negative coefficients for successful off-chain "
                "task events observed in v7 behavior gates."
            ),
            "role_names": list(EVENT_V8_CLEAN_CHAIN_COMPASS_ROLE_NAMES),
            "off_chain_penalty_coefficients": dict(EVENT_V8_CLEAN_CHAIN_COMPASS_OFFCHAIN_PENALTIES),
        }
    )
    return details


def event_v9_potential_chain_compass_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v9 potential-chain design."""

    return {
        "name": "event_v9_potential_chain_compass_breadcrumbs",
        "summary": (
            "Debug reward that keeps the chain-compass observation and positive "
            "ore -> battery -> heart event rewards, removes explicit negative "
            "event coefficients, and replaces clipped distance rewards with "
            "potential-based chain progress."
        ),
        "role_names": list(EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES),
        "common_coefficients": {},
        "task_event_coefficients": dict(EVENT_V6_ORACLE_CHAIN_TASK_COEFFICIENTS),
        "potential_shaping": {
            "formula": "F(s,s') = gamma * Phi(s') - Phi(s)",
            "default_gamma": EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
            "max_distance": EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE,
            "stage_offsets": dict(EVENT_V9_POTENTIAL_CHAIN_STAGE_OFFSETS),
            "target_closeness_scales": dict(EVENT_V9_POTENTIAL_CHAIN_CLOSENESS_SCALES),
            "stages": [
                "empty_to_mine",
                "ore_to_converter",
                "battery_to_home_assembler",
            ],
        },
        "oracle_action_coefficients": dict(EVENT_V6_ORACLE_CHAIN_ACTION_COEFFICIENTS),
        "oracle_action_caps": dict(EVENT_V6_ORACLE_CHAIN_ACTION_CAPS),
        "negative_reward_coefficients": {},
        "observation_breadcrumbs": {
            "planes": [
                "chain_target_dx_sign",
                "chain_target_dy_sign",
                "chain_inventory_stage",
                "chain_target_closeness",
                "chain_target_adjacent",
            ],
            "source": "navigation snapshot exposed by the canonical Tribal Village build",
            "purpose": (
                "Expose the currently shaped chain target to the feed-forward "
                "policy while keeping final evaluation in the full Tribal world."
            ),
        },
        "navigation_snapshot_columns": list(NAVIGATION_SNAPSHOT_COLUMNS),
        "role_coefficients": {role: {} for role in EVENT_V9_POTENTIAL_CHAIN_COMPASS_ROLE_NAMES},
        "coworld_role_sources": {role: list(sources) for role, sources in EVENT_V1_COWORLD_ROLE_SOURCES.items()},
        "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
    }


def event_v10_chain_affordance_compass_reward_design_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v10 affordance curriculum."""

    details = event_v9_potential_chain_compass_reward_design_details()
    details.update(
        {
            "name": "event_v10_chain_affordance_compass_breadcrumbs",
            "summary": (
                "Debug curriculum that keeps the v9 potential-chain reward and "
                "chain-compass observation, then applies a chain-affordance "
                "action mask so PPO can move freely but can only use the current "
                "ore -> battery -> heart target."
            ),
            "role_names": list(EVENT_V10_CHAIN_AFFORDANCE_COMPASS_ROLE_NAMES),
            "action_affordance_curriculum": {
                "enabled": True,
                "allowed_verbs": ["move", "use_current_chain_target"],
                "blocked_successes": [
                    "off-chain resource use",
                    "put handoffs",
                    "attack combat",
                    "plant lantern",
                    "swap",
                ],
                "reward_penalties_added": False,
                "purpose": (
                    "Remove off-chain affordances during the cleanup ramp without adding negative reward terms."
                ),
            },
        }
    )
    return details


def event_v11_role_gated_chain_handoff_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v11 specialization diagnostic."""

    details = event_v10_chain_affordance_compass_reward_design_details()
    details.update(
        {
            "name": "event_v11_role_gated_chain_handoffs",
            "summary": (
                "Diagnostic curriculum that makes the ore -> battery -> heart chain "
                "mechanically require cross-role handoff: suppliers can mine and "
                "pass ore, crafters can craft and pass batteries, and depositors "
                "can deposit batteries at the home assembler."
            ),
            "role_names": list(EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES),
            "task_event_coefficients": {},
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V11_ROLE_GATED_CHAIN_ROLE_COEFFICIENTS.items()
            },
            "potential_shaping": {
                "formula": "F(s,s') = gamma * Phi_role(s') - Phi_role(s)",
                "default_gamma": EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
                "max_distance": EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE,
                "stage_offsets": dict(EVENT_V11_ROLE_GATED_CHAIN_STAGE_OFFSETS),
                "target_closeness_scales": dict(EVENT_V11_ROLE_GATED_CHAIN_CLOSENESS_SCALES),
            },
            "action_affordance_curriculum": {
                "enabled": True,
                "allowed_verbs": [
                    "move",
                    "supplier_use_mine",
                    "supplier_put_ore",
                    "crafter_use_converter",
                    "crafter_put_battery",
                    "depositor_use_assembler",
                ],
                "blocked_successes": [
                    "single-agent mine->craft->deposit completion",
                    "off-chain resource use",
                    "attack combat",
                    "plant lantern",
                    "swap",
                ],
                "reward_penalties_added": False,
                "purpose": (
                    "Make role coordination necessary before rerunning fixed-role representation geometry."
                ),
            },
            "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
        }
    )
    return details


def event_v12_role_gated_depositor_reliability_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v12 depositor reliability diagnostic."""

    details = event_v11_role_gated_chain_handoff_details()
    details.update(
        {
            "name": "event_v12_role_gated_depositor_reliability",
            "summary": (
                "V11 diagnostic curriculum with positive-only, correct-recipient handoff rewards "
                "and stronger depositor receive/home breadcrumbs."
            ),
            "role_names": list(EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs)
                for role, coeffs in EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_ROLE_COEFFICIENTS.items()
            },
            "potential_shaping": {
                "formula": "F(s,s') = gamma * Phi_role(s') - Phi_role(s)",
                "default_gamma": EVENT_V9_POTENTIAL_CHAIN_DEFAULT_GAMMA,
                "max_distance": EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE,
                "stage_offsets": dict(EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_STAGE_OFFSETS),
                "target_closeness_scales": dict(EVENT_V12_ROLE_GATED_DEPOSITOR_RELIABILITY_CLOSENESS_SCALES),
            },
            "v12_changes": {
                "correct_recipient_handoff_counters": [
                    "put_ore_to_crafter",
                    "put_battery_to_depositor",
                    "receive_ore_from_supplier",
                    "receive_battery_from_crafter",
                ],
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Improve depositor battery-to-home completion while measuring whether "
                    "handoffs are going to the intended role."
                ),
            },
            "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
        }
    )
    return details


def event_v13_role_gated_depositor_use_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v13 depositor-use diagnostic."""

    details = event_v12_role_gated_depositor_reliability_details()
    details.update(
        {
            "name": "event_v13_role_gated_depositor_use",
            "summary": (
                "V11-style generic handoff acquisition with correct-recipient diagnostics, "
                "depositor receive credit, stronger depositor home potential, and a small "
                "positive breadcrumb for the final valid depositor home-assembler use action."
            ),
            "role_names": list(EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs)
                for role, coeffs in EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": dict(EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ACTION_COEFFICIENTS),
            "v13_changes": {
                "restores_v11_generic_handoff_rewards": True,
                "keeps_correct_recipient_metrics": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "final_action_breadcrumb": (
                    "Depositor receives a small positive reward only when it already has a battery, "
                    "is adjacent to the home assembler, and chooses the matching valid use action."
                ),
            },
            "simulator_stat_columns": list(SIMULATOR_STAT_COLUMNS),
        }
    )
    return details


def event_v14_target_aware_handoff_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v14 target-aware handoff diagnostic."""

    details = event_v13_role_gated_depositor_use_details()
    details.update(
        {
            "name": "event_v14_target_aware_handoffs",
            "summary": (
                "V13 positive rewards with a stricter role-gated action affordance surface: "
                "supplier put actions are exposed only toward adjacent crafters, and crafter "
                "put actions are exposed only when they would hand a battery to an adjacent depositor."
            ),
            "role_names": list(EVENT_V14_TARGET_AWARE_HANDOFF_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V14_TARGET_AWARE_HANDOFF_ROLE_COEFFICIENTS.items()
            },
            "v14_changes": {
                "changes_rewards": False,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Remove wrong-recipient handoff actions from the curriculum while leaving movement, "
                    "timing, and final use choices to the learned policy."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["allowed_verbs"] = [
        "move",
        "supplier_use_mine",
        "supplier_put_ore_to_adjacent_crafter",
        "crafter_use_converter",
        "crafter_put_battery_to_adjacent_depositor",
        "depositor_use_assembler",
    ]
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v15_depositor_final_mile_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v15 depositor final-mile diagnostic."""

    details = event_v14_target_aware_handoff_details()
    details.update(
        {
            "name": "event_v15_depositor_final_mile",
            "summary": (
                "V14 target-aware handoff curriculum plus positive-only final-mile "
                "breadcrumbs for battery-carrying depositors: valid movement that "
                "reduces distance to the home assembler and a capped adjacent-home "
                "arrival bonus."
            ),
            "role_names": list(EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_COEFFICIENTS.items()
            },
            "final_mile_action_coefficients": dict(EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_COEFFICIENTS),
            "final_mile_action_caps": dict(EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_CAPS),
            "v15_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Address Stage-22 evidence where depositors received batteries "
                    "but remained far from home and never issued the final use action."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v16_handoff_reliability_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v16 handoff reliability diagnostic."""

    details = event_v15_depositor_final_mile_details()
    details.update(
        {
            "name": "event_v16_handoff_reliability",
            "summary": (
                "V15 target-aware final-mile curriculum with stronger positive-only "
                "battery handoff, depositor receive, and valid home-assembler use "
                "breadcrumbs to make completed heart deposits reproducible."
            ),
            "role_names": list(EVENT_V16_HANDOFF_RELIABILITY_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V16_HANDOFF_RELIABILITY_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ]
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V16_HANDOFF_RELIABILITY_ACTION_COEFFICIENTS.items()
                if name != "depositor_use_home_assembler"
            },
            "final_mile_action_caps": dict(EVENT_V16_HANDOFF_RELIABILITY_ACTION_CAPS),
            "v16_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Address Stage-23 v15 evidence: the full chain is possible, "
                    "but battery-to-depositor handoff and final depositor use remain rare."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v17_depositor_staging_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v17 depositor staging diagnostic."""

    details = event_v16_handoff_reliability_details()
    details.update(
        {
            "name": "event_v17_depositor_staging",
            "summary": (
                "V16 handoff reliability plus positive-only staging breadcrumbs "
                "for empty depositors to move toward and arrive adjacent to the "
                "home assembler before they receive a battery."
            ),
            "role_names": list(EVENT_V17_DEPOSITOR_STAGING_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V17_DEPOSITOR_STAGING_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ]
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS.items()
                if name != "depositor_use_home_assembler"
            },
            "final_mile_action_caps": dict(EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS),
            "v17_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Address Stage-24 v16 evidence: completed deposits are correct when "
                    "depositors receive batteries, but many crafters still end episodes "
                    "holding batteries because an adjacent depositor is not reliably staged."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v18_handoff_rendezvous_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v18 final-handoff rendezvous diagnostic."""

    details = event_v17_depositor_staging_details()
    details.update(
        {
            "name": "event_v18_handoff_rendezvous",
            "summary": (
                "V16 handoff reliability plus positive-only rendezvous breadcrumbs "
                "for the final battery handoff: battery-carrying crafters move "
                "toward empty depositors, and empty depositors move toward "
                "battery-carrying crafters when such a crafter exists."
            ),
            "role_names": list(EVENT_V18_HANDOFF_RENDEZVOUS_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V18_HANDOFF_RENDEZVOUS_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ]
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS.items()
                if name != "depositor_use_home_assembler"
            },
            "final_mile_action_caps": dict(EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_CAPS),
            "v18_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Address Stage-25 v17 evidence: agents learn ore transfer and "
                    "battery crafting, but many episodes fail because crafters hold "
                    "batteries without reaching an empty depositor for the final handoff."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v19_crafter_home_delivery_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v19 crafter home-delivery diagnostic."""

    details = event_v17_depositor_staging_details()
    details.update(
        {
            "name": "event_v19_crafter_home_delivery",
            "summary": (
                "V17 depositor home staging plus positive-only breadcrumbs for "
                "battery-carrying crafters to move toward and arrive adjacent "
                "to the home assembler, where target-aware battery handoff can occur."
            ),
            "role_names": list(EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ]
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS.items()
                if name != "depositor_use_home_assembler"
            },
            "final_mile_action_caps": dict(EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_CAPS),
            "v19_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "purpose": (
                    "Address Stage-26 v18 evidence: peer rendezvous reduced final "
                    "completion and could pull recipients away from home. Keep "
                    "depositor home staging and instead reward crafters for "
                    "bringing batteries to the home handoff area."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v20_home_staged_handoff_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v20 home-staged handoff diagnostic."""

    details = event_v19_crafter_home_delivery_details()
    details.update(
        {
            "name": "event_v20_home_staged_handoff",
            "summary": (
                "V19 crafter home delivery plus a positive-only oracle-action "
                "breadcrumb for battery-carrying crafters that select a "
                "target-aware mask-valid put action near the home assembler."
            ),
            "role_names": list(EVENT_V20_HOME_STAGED_HANDOFF_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V20_HOME_STAGED_HANDOFF_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ],
                "crafter_battery_put_to_home_staged_depositor": EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS[
                    "crafter_battery_put_to_home_staged_depositor"
                ],
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS.items()
                if name
                not in {
                    "depositor_use_home_assembler",
                    "crafter_battery_put_to_home_staged_depositor",
                }
            },
            "final_mile_action_caps": dict(EVENT_V20_HOME_STAGED_HANDOFF_ACTION_CAPS),
            "v20_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "changes_action_permissions": False,
                "purpose": (
                    "Address Stage-27 v19 evidence: crafters can make batteries, "
                    "but deterministic rollouts selected no battery-to-depositor "
                    "handoffs. Reward the existing target-aware put affordance "
                    "when it is selected near a home-staged depositor."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def event_v21_home_handoff_rendezvous_details() -> dict[str, Any]:
    """Return a JSON-serializable description of the v21 final-rendezvous diagnostic."""

    details = event_v20_home_staged_handoff_details()
    details.update(
        {
            "name": "event_v21_home_handoff_rendezvous",
            "summary": (
                "V20 home-staged handoff with stronger positive-only credit for "
                "the final mask-valid put action, plus capped movement/arrival "
                "breadcrumbs for battery-carrying crafters to become adjacent to "
                "an empty depositor already staged near the home assembler."
            ),
            "role_names": list(EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ROLE_NAMES),
            "role_coefficients": {
                role: dict(coeffs) for role, coeffs in EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ROLE_COEFFICIENTS.items()
            },
            "oracle_action_coefficients": {
                "depositor_use_home_assembler": EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
                    "depositor_use_home_assembler"
                ],
                "crafter_battery_put_to_home_staged_depositor": EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
                    "crafter_battery_put_to_home_staged_depositor"
                ],
            },
            "final_mile_action_coefficients": {
                name: coefficient
                for name, coefficient in EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS.items()
                if name
                not in {
                    "depositor_use_home_assembler",
                    "crafter_battery_put_to_home_staged_depositor",
                }
            },
            "final_mile_action_caps": dict(EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_CAPS),
            "v21_changes": {
                "changes_rewards": True,
                "target_aware_handoff_mask": True,
                "reward_penalties_added": False,
                "scripted_policy_added": False,
                "changes_action_permissions": False,
                "purpose": (
                    "Address Stage-28 v20 evidence: crafters reliably make and "
                    "hold batteries, but almost never take the final put action. "
                    "Keep the home-staged structure and reward approaching an "
                    "already staged empty depositor before the target-aware handoff."
                ),
            },
        }
    )
    details["action_affordance_curriculum"]["target_aware_handoffs"] = True
    return details


def _validate_event_stats_delta(event_stats_delta: np.ndarray | None, num_agents: int) -> np.ndarray | None:
    if event_stats_delta is None:
        return None

    stats = np.asarray(event_stats_delta, dtype=np.float64)
    if stats.ndim != 2:
        raise ValueError("event_stats_delta must have shape [agents, stat_columns]")
    if stats.shape[0] != num_agents:
        raise ValueError(f"event_stats_delta has {stats.shape[0]} agents, expected {num_agents}")
    if stats.shape[1] != len(SIMULATOR_STAT_COLUMNS):
        raise ValueError(f"event_stats_delta has {stats.shape[1]} columns, expected {len(SIMULATOR_STAT_COLUMNS)}")
    return np.maximum(stats, 0.0)


def _validate_event_stats_total(event_stats_total: np.ndarray | None, num_agents: int) -> np.ndarray | None:
    if event_stats_total is None:
        return None

    stats = np.asarray(event_stats_total, dtype=np.float64)
    if stats.ndim != 2:
        raise ValueError("event_stats_total must have shape [agents, stat_columns]")
    if stats.shape[0] != num_agents:
        raise ValueError(f"event_stats_total has {stats.shape[0]} agents, expected {num_agents}")
    if stats.shape[1] != len(SIMULATOR_STAT_COLUMNS):
        raise ValueError(f"event_stats_total has {stats.shape[1]} columns, expected {len(SIMULATOR_STAT_COLUMNS)}")
    return np.maximum(stats, 0.0)


def _add_agent_coefficients(
    bonuses: np.ndarray,
    stats: np.ndarray,
    coefficients: dict[str, float],
) -> None:
    for stat_name, coefficient in coefficients.items():
        bonuses += coefficient * stats[:, _STAT_INDEX[stat_name]]


def _add_capped_agent_coefficients(
    bonuses: np.ndarray,
    stats: np.ndarray,
    totals: np.ndarray | None,
    coefficients: dict[str, float],
    caps: dict[str, int],
) -> None:
    if totals is None:
        _add_agent_coefficients(bonuses, stats, coefficients)
        return

    for stat_name, coefficient in coefficients.items():
        index = _STAT_INDEX[stat_name]
        eligible = totals[:, index] <= caps[stat_name]
        bonuses[eligible] += coefficient * stats[eligible, index]


def _add_role_coefficients(
    bonuses: np.ndarray,
    stats: np.ndarray,
    role_names: tuple[str, ...],
    coefficients_by_role: dict[str, dict[str, float]],
) -> None:
    labels = role_labels(stats.shape[0], len(role_names))
    for role_id, role_name in enumerate(role_names):
        mask = labels == role_id
        if not np.any(mask):
            continue
        for stat_name, coefficient in coefficients_by_role[role_name].items():
            bonuses[mask] += coefficient * stats[mask, _STAT_INDEX[stat_name]]


def _add_navigation_progress_bonuses(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    event_stats_total: np.ndarray | None,
    *,
    progress_coefficients: dict[str, float] | None = None,
    progress_caps: dict[str, int] | None = None,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    if before is None or after is None:
        return
    coefficients = progress_coefficients or EVENT_V5_NAVIGATION_CHAIN_PROGRESS_COEFFICIENTS
    caps = progress_caps or EVENT_V5_NAVIGATION_CHAIN_PROGRESS_CAPS

    ore_before = before[:, NAV_INVENTORY_ORE] > 0
    battery_before = before[:, NAV_INVENTORY_BATTERY] > 0
    empty_before = ~ore_before & ~battery_before

    progress_terms = {
        "toward_mine_empty": (
            empty_before,
            _positive_distance_progress(before, after, NAV_DIST_NEAREST_MINE),
        ),
        "ore_to_converter": (
            ore_before,
            _positive_distance_progress(before, after, NAV_DIST_NEAREST_CONVERTER),
        ),
        "battery_to_home_assembler": (
            battery_before,
            _positive_distance_progress(before, after, NAV_DIST_HOME_ASSEMBLER),
        ),
        "battery_adjacent_home_assembler": (
            battery_before & (before[:, NAV_DIST_HOME_ASSEMBLER] > 1) & (after[:, NAV_DIST_HOME_ASSEMBLER] == 1),
            np.ones(bonuses.shape[0], dtype=np.float64),
        ),
    }
    for name, (eligible, progress) in progress_terms.items():
        capped = _navigation_progress_cap_eligible(
            event_stats_total,
            caps[name],
            bonuses.shape[0],
        )
        bonuses[eligible & capped] += coefficients[name] * progress[eligible & capped]


def _add_chain_potential_bonuses(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    *,
    gamma: float,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    if before is None or after is None:
        return

    bonuses += gamma * _chain_potential_values(after) - _chain_potential_values(before)


def _chain_potential_values(navigation: np.ndarray) -> np.ndarray:
    potentials = np.zeros(navigation.shape[0], dtype=np.float64)
    for agent_id, row in enumerate(navigation):
        stage_name, distance = _chain_potential_stage_and_distance(row)
        if stage_name is None or distance is None:
            continue
        clipped_distance = max(0.0, min(EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE, distance))
        closeness = EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE - clipped_distance
        potentials[agent_id] = (
            EVENT_V9_POTENTIAL_CHAIN_STAGE_OFFSETS[stage_name]
            + EVENT_V9_POTENTIAL_CHAIN_CLOSENESS_SCALES[stage_name] * closeness
        )
    return potentials


def _chain_potential_stage_and_distance(navigation_row: np.ndarray) -> tuple[str | None, float | None]:
    if int(navigation_row[NAV_INVENTORY_BATTERY]) > 0:
        return "battery_to_home_assembler", _valid_navigation_distance(
            navigation_row,
            NAV_DIST_HOME_ASSEMBLER,
        )
    if int(navigation_row[NAV_INVENTORY_ORE]) > 0:
        return "ore_to_converter", _valid_navigation_distance(
            navigation_row,
            NAV_DIST_NEAREST_CONVERTER,
        )
    return "empty_to_mine", _valid_navigation_distance(navigation_row, NAV_DIST_NEAREST_MINE)


def _add_role_gated_chain_potential_bonuses(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    *,
    gamma: float,
    stage_offsets: dict[str, float] = EVENT_V11_ROLE_GATED_CHAIN_STAGE_OFFSETS,
    closeness_scales: dict[str, float] = EVENT_V11_ROLE_GATED_CHAIN_CLOSENESS_SCALES,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    if before is None or after is None:
        return
    bonuses += gamma * _role_gated_chain_potential_values(
        after,
        stage_offsets=stage_offsets,
        closeness_scales=closeness_scales,
    ) - _role_gated_chain_potential_values(
        before,
        stage_offsets=stage_offsets,
        closeness_scales=closeness_scales,
    )


def _role_gated_chain_potential_values(
    navigation: np.ndarray,
    *,
    stage_offsets: dict[str, float],
    closeness_scales: dict[str, float],
) -> np.ndarray:
    potentials = np.zeros(navigation.shape[0], dtype=np.float64)
    labels = role_labels(navigation.shape[0], len(EVENT_V11_ROLE_GATED_CHAIN_ROLE_NAMES))
    for agent_id, row in enumerate(navigation):
        stage_name, distance = _role_gated_chain_stage_and_distance(row, int(labels[agent_id]))
        if stage_name is None or distance is None:
            continue
        clipped_distance = max(0.0, min(EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE, distance))
        closeness = EVENT_V9_POTENTIAL_CHAIN_MAX_DISTANCE - clipped_distance
        potentials[agent_id] = (
            stage_offsets[stage_name]
            + closeness_scales[stage_name] * closeness
        )
    return potentials


def _role_gated_chain_stage_and_distance(
    navigation_row: np.ndarray,
    role_id: int,
) -> tuple[str | None, float | None]:
    has_ore = int(navigation_row[NAV_INVENTORY_ORE]) > 0
    has_battery = int(navigation_row[NAV_INVENTORY_BATTERY]) > 0
    if role_id == 0:
        if has_ore:
            return "supplier_ore_to_converter", _valid_navigation_distance(
                navigation_row,
                NAV_DIST_NEAREST_CONVERTER,
            )
        return "supplier_empty_to_mine", _valid_navigation_distance(navigation_row, NAV_DIST_NEAREST_MINE)
    if role_id == 1:
        if has_battery:
            return "crafter_battery_to_home", _valid_navigation_distance(
                navigation_row,
                NAV_DIST_HOME_ASSEMBLER,
            )
        if has_ore:
            return "crafter_ore_to_converter", _valid_navigation_distance(
                navigation_row,
                NAV_DIST_NEAREST_CONVERTER,
            )
        return "crafter_empty_to_mine", _valid_navigation_distance(navigation_row, NAV_DIST_NEAREST_MINE)
    if has_battery:
        return "depositor_battery_to_home", _valid_navigation_distance(
            navigation_row,
            NAV_DIST_HOME_ASSEMBLER,
        )
    return "depositor_empty_to_home", _valid_navigation_distance(navigation_row, NAV_DIST_HOME_ASSEMBLER)


def _add_depositor_home_use_breadcrumb(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    *,
    coefficient: float = EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ACTION_COEFFICIENTS["depositor_use_home_assembler"],
) -> None:
    navigation = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if navigation is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(navigation.shape[0], len(EVENT_V13_ROLE_GATED_DEPOSITOR_USE_ROLE_NAMES))
    for agent_id, row in enumerate(navigation):
        if int(labels[agent_id]) != 2 or int(row[NAV_INVENTORY_BATTERY]) <= 0:
            continue
        target = _target_if_valid(row, NAV_HOME_ASSEMBLER_X, NAV_HOME_ASSEMBLER_Y, NAV_DIST_HOME_ASSEMBLER)
        if target is None:
            continue
        agent_x = int(row[NAV_AGENT_X])
        agent_y = int(row[NAV_AGENT_Y])
        target_x, target_y = target
        dx_raw = target_x - agent_x
        dy_raw = target_y - agent_y
        if max(abs(dx_raw), abs(dy_raw)) != 1:
            continue
        dx = _sign(dx_raw)
        dy = _sign(dy_raw)
        if dx == 0 and dy == 0:
            continue
        action = _encode_action(USE_VERB, ORIENTATION_BY_DELTA[(dx, dy)])
        if actions_arr[agent_id] == action and _mask_allows(mask, agent_id, action):
            bonuses[agent_id] += coefficient


def _add_depositor_final_mile_breadcrumbs(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
    *,
    action_coefficients: dict[str, float] = EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_COEFFICIENTS,
    action_caps: dict[str, int] = EVENT_V15_DEPOSITOR_FINAL_MILE_ACTION_CAPS,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if before is None or after is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(before.shape[0], len(EVENT_V15_DEPOSITOR_FINAL_MILE_ROLE_NAMES))
    move_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        action_caps["depositor_battery_move_toward_home"],
        bonuses.shape[0],
    )
    arrival_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        action_caps["depositor_battery_arrive_adjacent_home"],
        bonuses.shape[0],
    )

    progress = _positive_distance_progress(before, after, NAV_DIST_HOME_ASSEMBLER)
    for agent_id, row in enumerate(before):
        if int(labels[agent_id]) != 2 or int(row[NAV_INVENTORY_BATTERY]) <= 0:
            continue
        action = int(actions_arr[agent_id])
        if action // ACTION_ARGUMENT_COUNT != MOVE_VERB or not _mask_allows(mask, agent_id, action):
            continue
        if progress[agent_id] > 0.0 and move_cap[agent_id]:
            bonuses[agent_id] += action_coefficients["depositor_battery_move_toward_home"] * progress[agent_id]
        if (
            row[NAV_DIST_HOME_ASSEMBLER] > 1
            and after[agent_id, NAV_DIST_HOME_ASSEMBLER] == 1
            and arrival_cap[agent_id]
        ):
            bonuses[agent_id] += action_coefficients["depositor_battery_arrive_adjacent_home"]


def _add_depositor_empty_staging_breadcrumbs(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if before is None or after is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(before.shape[0], len(EVENT_V17_DEPOSITOR_STAGING_ROLE_NAMES))
    move_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS["depositor_empty_move_toward_home"],
        bonuses.shape[0],
    )
    arrival_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V17_DEPOSITOR_STAGING_ACTION_CAPS["depositor_empty_arrive_adjacent_home"],
        bonuses.shape[0],
    )

    progress = _positive_distance_progress(before, after, NAV_DIST_HOME_ASSEMBLER)
    for agent_id, row in enumerate(before):
        if int(labels[agent_id]) != 2 or int(row[NAV_INVENTORY_BATTERY]) > 0:
            continue
        action = int(actions_arr[agent_id])
        if action // ACTION_ARGUMENT_COUNT != MOVE_VERB or not _mask_allows(mask, agent_id, action):
            continue
        if progress[agent_id] > 0.0 and move_cap[agent_id]:
            bonuses[agent_id] += (
                EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS["depositor_empty_move_toward_home"]
                * progress[agent_id]
            )
        if (
            row[NAV_DIST_HOME_ASSEMBLER] > 1
            and after[agent_id, NAV_DIST_HOME_ASSEMBLER] == 1
            and arrival_cap[agent_id]
        ):
            bonuses[agent_id] += EVENT_V17_DEPOSITOR_STAGING_ACTION_COEFFICIENTS[
                "depositor_empty_arrive_adjacent_home"
            ]


def _add_handoff_rendezvous_breadcrumbs(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if before is None or after is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(before.shape[0], len(EVENT_V18_HANDOFF_RENDEZVOUS_ROLE_NAMES))

    battery_crafters = [
        agent_id
        for agent_id, row in enumerate(before)
        if int(labels[agent_id]) == 1 and int(row[NAV_INVENTORY_BATTERY]) > 0
    ]
    empty_depositors = [
        agent_id
        for agent_id, row in enumerate(before)
        if int(labels[agent_id]) == 2 and int(row[NAV_INVENTORY_BATTERY]) <= 0
    ]
    if not battery_crafters:
        _add_depositor_empty_staging_breadcrumbs(
            bonuses,
            navigation_before,
            navigation_after,
            actions,
            action_mask,
            event_stats_total,
        )
        return
    if not empty_depositors:
        return

    action_coefficients = EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS
    move_caps = {
        name: _action_cap_eligible(event_stats_total, "action_move", cap, bonuses.shape[0])
        for name, cap in EVENT_V18_HANDOFF_RENDEZVOUS_ACTION_CAPS.items()
    }

    for agent_id in battery_crafters:
        target = _nearest_peer_xy(before, agent_id, empty_depositors)
        if target is None:
            continue
        _add_peer_rendezvous_move_bonus(
            bonuses,
            before,
            after,
            actions_arr,
            mask,
            agent_id,
            target,
            move_coefficient=action_coefficients["crafter_battery_move_toward_empty_depositor"],
            arrival_coefficient=action_coefficients["crafter_battery_arrive_adjacent_empty_depositor"],
            move_cap=move_caps["crafter_battery_move_toward_empty_depositor"],
            arrival_cap=move_caps["crafter_battery_arrive_adjacent_empty_depositor"],
        )

    for agent_id in empty_depositors:
        target = _nearest_peer_xy(before, agent_id, battery_crafters)
        if target is None:
            continue
        _add_peer_rendezvous_move_bonus(
            bonuses,
            before,
            after,
            actions_arr,
            mask,
            agent_id,
            target,
            move_coefficient=action_coefficients["depositor_empty_move_toward_battery_crafter"],
            arrival_coefficient=action_coefficients["depositor_empty_arrive_adjacent_battery_crafter"],
            move_cap=move_caps["depositor_empty_move_toward_battery_crafter"],
            arrival_cap=move_caps["depositor_empty_arrive_adjacent_battery_crafter"],
        )


def _nearest_peer_xy(
    navigation: np.ndarray,
    agent_id: int,
    peer_ids: list[int],
) -> tuple[int, int] | None:
    if not peer_ids:
        return None
    agent_x = int(navigation[agent_id, NAV_AGENT_X])
    agent_y = int(navigation[agent_id, NAV_AGENT_Y])
    best_peer = min(
        peer_ids,
        key=lambda peer_id: abs(agent_x - int(navigation[peer_id, NAV_AGENT_X]))
        + abs(agent_y - int(navigation[peer_id, NAV_AGENT_Y])),
    )
    return int(navigation[best_peer, NAV_AGENT_X]), int(navigation[best_peer, NAV_AGENT_Y])


def _add_peer_rendezvous_move_bonus(
    bonuses: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray | None,
    agent_id: int,
    target_xy: tuple[int, int],
    *,
    move_coefficient: float,
    arrival_coefficient: float,
    move_cap: np.ndarray,
    arrival_cap: np.ndarray,
) -> None:
    action = int(actions[agent_id])
    if action // ACTION_ARGUMENT_COUNT != MOVE_VERB or not _mask_allows(action_mask, agent_id, action):
        return

    before_distance = _manhattan_distance_to_xy(before[agent_id], target_xy)
    after_distance = _manhattan_distance_to_xy(after[agent_id], target_xy)
    progress = before_distance - after_distance
    if progress > 0 and move_cap[agent_id]:
        bonuses[agent_id] += move_coefficient * progress

    if (
        _chebyshev_distance_to_xy(before[agent_id], target_xy) > 1
        and _chebyshev_distance_to_xy(after[agent_id], target_xy) == 1
        and arrival_cap[agent_id]
    ):
        bonuses[agent_id] += arrival_coefficient


def _manhattan_distance_to_xy(navigation_row: np.ndarray, target_xy: tuple[int, int]) -> int:
    target_x, target_y = target_xy
    return abs(int(navigation_row[NAV_AGENT_X]) - target_x) + abs(int(navigation_row[NAV_AGENT_Y]) - target_y)


def _chebyshev_distance_to_xy(navigation_row: np.ndarray, target_xy: tuple[int, int]) -> int:
    target_x, target_y = target_xy
    return max(abs(int(navigation_row[NAV_AGENT_X]) - target_x), abs(int(navigation_row[NAV_AGENT_Y]) - target_y))


def _add_crafter_battery_home_delivery_breadcrumbs(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if before is None or after is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(before.shape[0], len(EVENT_V19_CRAFTER_HOME_DELIVERY_ROLE_NAMES))
    move_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_CAPS["crafter_battery_move_toward_home"],
        bonuses.shape[0],
    )
    arrival_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_CAPS["crafter_battery_arrive_adjacent_home"],
        bonuses.shape[0],
    )

    progress = _positive_distance_progress(before, after, NAV_DIST_HOME_ASSEMBLER)
    for agent_id, row in enumerate(before):
        if int(labels[agent_id]) != 1 or int(row[NAV_INVENTORY_BATTERY]) <= 0:
            continue
        action = int(actions_arr[agent_id])
        if action // ACTION_ARGUMENT_COUNT != MOVE_VERB or not _mask_allows(mask, agent_id, action):
            continue
        if progress[agent_id] > 0.0 and move_cap[agent_id]:
            bonuses[agent_id] += (
                EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS["crafter_battery_move_toward_home"]
                * progress[agent_id]
            )
        if (
            row[NAV_DIST_HOME_ASSEMBLER] > 1
            and after[agent_id, NAV_DIST_HOME_ASSEMBLER] == 1
            and arrival_cap[agent_id]
        ):
            bonuses[agent_id] += EVENT_V19_CRAFTER_HOME_DELIVERY_ACTION_COEFFICIENTS[
                "crafter_battery_arrive_adjacent_home"
            ]


def _add_crafter_home_staged_handoff_breadcrumb(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
    *,
    coefficient: float = EVENT_V20_HOME_STAGED_HANDOFF_ACTION_COEFFICIENTS[
        "crafter_battery_put_to_home_staged_depositor"
    ],
    cap: int = EVENT_V20_HOME_STAGED_HANDOFF_ACTION_CAPS["crafter_battery_put_to_home_staged_depositor"],
) -> None:
    navigation = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    if navigation is None or actions_arr is None or mask is None:
        return
    labels = role_labels(navigation.shape[0], len(EVENT_V20_HOME_STAGED_HANDOFF_ROLE_NAMES))
    put_cap = _action_cap_eligible(
        event_stats_total,
        "action_put",
        cap,
        bonuses.shape[0],
    )

    for agent_id, row in enumerate(navigation):
        if int(labels[agent_id]) != 1 or int(row[NAV_INVENTORY_BATTERY]) <= 0:
            continue
        home_distance = _valid_navigation_distance(row, NAV_DIST_HOME_ASSEMBLER)
        if home_distance is None or home_distance > 2:
            continue
        action = int(actions_arr[agent_id])
        if action // ACTION_ARGUMENT_COUNT != PUT_VERB:
            continue
        if put_cap[agent_id] and _mask_allows(mask, agent_id, action):
            bonuses[agent_id] += coefficient


def _add_crafter_home_staged_depositor_rendezvous_breadcrumbs(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    navigation_after: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
) -> None:
    before = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    after = _validate_navigation_snapshot(navigation_after, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if before is None or after is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])
    labels = role_labels(before.shape[0], len(EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ROLE_NAMES))

    home_staged_depositors = [
        agent_id
        for agent_id, row in enumerate(before)
        if int(labels[agent_id]) == 2
        and int(row[NAV_INVENTORY_BATTERY]) <= 0
        and (distance := _valid_navigation_distance(row, NAV_DIST_HOME_ASSEMBLER)) is not None
        and distance <= 2
    ]
    if not home_staged_depositors:
        return

    battery_crafters = [
        agent_id
        for agent_id, row in enumerate(before)
        if int(labels[agent_id]) == 1 and int(row[NAV_INVENTORY_BATTERY]) > 0
    ]
    if not battery_crafters:
        return

    move_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_CAPS[
            "crafter_battery_move_toward_home_staged_depositor"
        ],
        bonuses.shape[0],
    )
    arrival_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_CAPS[
            "crafter_battery_arrive_adjacent_home_staged_depositor"
        ],
        bonuses.shape[0],
    )

    for agent_id in battery_crafters:
        target = _nearest_peer_xy(before, agent_id, home_staged_depositors)
        if target is None:
            continue
        _add_peer_rendezvous_move_bonus(
            bonuses,
            before,
            after,
            actions_arr,
            mask,
            agent_id,
            target,
            move_coefficient=EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
                "crafter_battery_move_toward_home_staged_depositor"
            ],
            arrival_coefficient=EVENT_V21_HOME_HANDOFF_RENDEZVOUS_ACTION_COEFFICIENTS[
                "crafter_battery_arrive_adjacent_home_staged_depositor"
            ],
            move_cap=move_cap,
            arrival_cap=arrival_cap,
        )


def _valid_navigation_distance(navigation_row: np.ndarray, column: int) -> float | None:
    distance = float(navigation_row[column])
    if distance < 0:
        return None
    return distance


def _add_chain_oracle_action_bonuses(
    bonuses: np.ndarray,
    navigation_before: np.ndarray | None,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    event_stats_total: np.ndarray | None,
) -> None:
    navigation = _validate_navigation_snapshot(navigation_before, bonuses.shape[0])
    actions_arr = _validate_actions(actions, bonuses.shape[0])
    if navigation is None or actions_arr is None:
        return
    mask = _validate_action_mask(action_mask, bonuses.shape[0])

    move_cap = _action_cap_eligible(
        event_stats_total,
        "action_move",
        EVENT_V6_ORACLE_CHAIN_ACTION_CAPS["move_toward_chain_target"],
        bonuses.shape[0],
    )
    use_cap = _action_cap_eligible(
        event_stats_total,
        "action_use",
        EVENT_V6_ORACLE_CHAIN_ACTION_CAPS["use_chain_target"],
        bonuses.shape[0],
    )

    for agent_id, row in enumerate(navigation):
        target = _chain_target(row)
        if target is None:
            continue
        agent_x = int(row[NAV_AGENT_X])
        agent_y = int(row[NAV_AGENT_Y])
        target_x, target_y = target
        dx = _sign(target_x - agent_x)
        dy = _sign(target_y - agent_y)
        if dx == 0 and dy == 0:
            continue

        if max(abs(target_x - agent_x), abs(target_y - agent_y)) == 1:
            action = _encode_action(USE_VERB, ORIENTATION_BY_DELTA[(dx, dy)])
            if actions_arr[agent_id] == action and use_cap[agent_id] and _mask_allows(mask, agent_id, action):
                bonuses[agent_id] += EVENT_V6_ORACLE_CHAIN_ACTION_COEFFICIENTS["use_chain_target"]
            continue

        action = _best_masked_move_toward(
            agent_x, agent_y, target_x, target_y, None if mask is None else mask[agent_id]
        )
        if action is None:
            action = _encode_action(MOVE_VERB, ORIENTATION_BY_DELTA[(dx, dy)])
        if actions_arr[agent_id] == action and move_cap[agent_id]:
            bonuses[agent_id] += EVENT_V6_ORACLE_CHAIN_ACTION_COEFFICIENTS["move_toward_chain_target"]


def _validate_actions(actions: np.ndarray | None, num_agents: int) -> np.ndarray | None:
    if actions is None:
        return None
    actions_arr = np.asarray(actions, dtype=np.int64)
    if actions_arr.shape != (num_agents,):
        raise ValueError(f"actions must have shape [{num_agents}]")
    return actions_arr


def _validate_action_mask(action_mask: np.ndarray | None, num_agents: int) -> np.ndarray | None:
    if action_mask is None:
        return None
    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != num_agents:
        raise ValueError(f"action_mask must have shape [{num_agents}, action_space_size]")
    return mask


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
        if not _mask_allows_single(action_mask, action):
            continue
        distance = abs(target_x - (agent_x + delta_x)) + abs(target_y - (agent_y + delta_y))
        if best_distance is None or distance < best_distance:
            best_action = action
            best_distance = distance
    return best_action


def _mask_allows(mask: np.ndarray | None, agent_id: int, action: int) -> bool:
    if mask is None:
        return True
    return _mask_allows_single(mask[agent_id], action)


def _mask_allows_single(mask_row: np.ndarray, action: int) -> bool:
    return action < mask_row.shape[0] and bool(mask_row[action])


def _action_cap_eligible(
    event_stats_total: np.ndarray | None,
    stat_name: str,
    cap: int,
    num_agents: int,
) -> np.ndarray:
    if event_stats_total is None:
        return np.ones(num_agents, dtype=bool)
    return event_stats_total[:, _STAT_INDEX[stat_name]] <= cap


def _encode_action(verb: int, argument: int) -> int:
    return int(verb * ACTION_ARGUMENT_COUNT + argument)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _validate_navigation_snapshot(navigation: np.ndarray | None, num_agents: int) -> np.ndarray | None:
    if navigation is None:
        return None

    snapshot = np.asarray(navigation, dtype=np.float64)
    if snapshot.ndim != 2:
        raise ValueError("navigation snapshot must have shape [agents, navigation_columns]")
    if snapshot.shape[0] != num_agents:
        raise ValueError(f"navigation snapshot has {snapshot.shape[0]} agents, expected {num_agents}")
    if snapshot.shape[1] != len(NAVIGATION_SNAPSHOT_COLUMNS):
        raise ValueError(
            f"navigation snapshot has {snapshot.shape[1]} columns, expected {len(NAVIGATION_SNAPSHOT_COLUMNS)}"
        )
    return snapshot


def _positive_distance_progress(before: np.ndarray, after: np.ndarray, column: int) -> np.ndarray:
    valid = (before[:, column] >= 0) & (after[:, column] >= 0)
    progress = before[:, column] - after[:, column]
    return np.where(valid & (progress > 0), progress, 0.0)


def _navigation_progress_cap_eligible(event_stats_total: np.ndarray | None, cap: int, num_agents: int) -> np.ndarray:
    if event_stats_total is None:
        return np.ones(num_agents, dtype=bool)
    return event_stats_total[:, _STAT_INDEX["action_move"]] <= cap
