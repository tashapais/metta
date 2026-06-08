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
