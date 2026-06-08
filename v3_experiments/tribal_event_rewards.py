"""Event-based Tribal Village reward shaping definitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from v3_experiments.canonical_reward_geometry import CANONICAL_NUM_AGENTS, role_labels
from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS

EVENT_V1_ROLE_NAMES = ("supplier", "crafter_logistics", "defender_territory")

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

    if event_stats_delta is None:
        return np.zeros(num_agents, dtype=np.float64)

    stats = np.asarray(event_stats_delta, dtype=np.float64)
    if stats.ndim != 2:
        raise ValueError("event_stats_delta must have shape [agents, stat_columns]")
    if stats.shape[0] != num_agents:
        raise ValueError(f"event_stats_delta has {stats.shape[0]} agents, expected {num_agents}")
    if stats.shape[1] != len(SIMULATOR_STAT_COLUMNS):
        raise ValueError(f"event_stats_delta has {stats.shape[1]} columns, expected {len(SIMULATOR_STAT_COLUMNS)}")

    stats = np.maximum(stats, 0.0)
    bonuses = np.zeros(num_agents, dtype=np.float64)
    for stat_name, coefficient in EVENT_V1_COMMON_COEFFICIENTS.items():
        bonuses += coefficient * stats[:, _STAT_INDEX[stat_name]]

    labels = role_labels(num_agents, len(EVENT_V1_ROLE_NAMES))
    for role_id, role_name in enumerate(EVENT_V1_ROLE_NAMES):
        mask = labels == role_id
        if not np.any(mask):
            continue
        for stat_name, coefficient in EVENT_V1_ROLE_COEFFICIENTS[role_name].items():
            bonuses[mask] += coefficient * stats[mask, _STAT_INDEX[stat_name]]
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
