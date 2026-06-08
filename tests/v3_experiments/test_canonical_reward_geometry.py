import numpy as np
import pytest

from v3_experiments.canonical_reward_geometry import (
    audit_fixed_role_probe_records,
    js_action_diversity,
    mix_rewards,
    ordered_kl_action_diversity,
    role_labels,
    role_probe_chance,
    summarize_probe_audit,
)


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
