from v3_experiments.analyze_tribal_behavior_roles import analyze_rollout_record
from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS


def test_behavior_role_analysis_flags_full_chain_not_fixed_roles():
    record = _record(
        [
            {"resource_ore": 10, "craft_battery": 8, "deposit_heart": 6},
            {"resource_ore": 9, "craft_battery": 7, "deposit_heart": 5},
            {"resource_ore": 11, "craft_battery": 9, "deposit_heart": 6},
            {"resource_ore": 10, "craft_battery": 8, "deposit_heart": 5},
            {"resource_ore": 9, "craft_battery": 8, "deposit_heart": 6},
            {"resource_ore": 10, "craft_battery": 7, "deposit_heart": 5},
        ]
    )

    analysis = analyze_rollout_record(record, specialization_threshold=0.6, min_chain_events=10)

    assert analysis["agents_with_all_chain_stages"] == 6
    assert analysis["specialized_agent_count"] == 0
    assert analysis["fixed_role_behavior_separation"] is False


def test_behavior_role_analysis_detects_fixed_role_stage_separation():
    record = _record(
        [
            {"resource_ore": 20, "craft_battery": 1, "deposit_heart": 1},
            {"resource_ore": 1, "craft_battery": 20, "deposit_heart": 1},
            {"resource_ore": 1, "craft_battery": 1, "deposit_heart": 20},
            {"resource_ore": 18, "craft_battery": 1, "deposit_heart": 1},
            {"resource_ore": 1, "craft_battery": 18, "deposit_heart": 1},
            {"resource_ore": 1, "craft_battery": 1, "deposit_heart": 18},
        ]
    )

    analysis = analyze_rollout_record(record, specialization_threshold=0.6, min_chain_events=10)

    assert analysis["specialized_agent_count"] == 6
    assert analysis["dominant_stage_counts"] == {"ore": 2, "battery": 2, "heart": 2}
    assert analysis["fixed_role_behavior_separation"] is True


def _record(agent_rows):
    return {
        "policy": "checkpoint",
        "seed": 0,
        "episodes": 1,
        "steps_per_episode": 10,
        "episode_metrics": [
            {
                "behavior_metrics": {
                    "simulator_action_stats_by_agent": [
                        {column: row.get(column, 0) for column in SIMULATOR_STAT_COLUMNS} for row in agent_rows
                    ]
                }
            }
        ],
    }
