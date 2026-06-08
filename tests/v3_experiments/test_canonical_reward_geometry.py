import json

import numpy as np
import pytest

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
from v3_experiments.train_canonical_reward_geometry import main as train_canonical_main
from v3_experiments.tribal_behavior import SIMULATOR_STAT_COLUMNS
from v3_experiments.tribal_event_rewards import (
    EVENT_V1_ROLE_NAMES,
    event_v1_reward_design_details,
    event_v1_role_shaping_bonuses,
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
