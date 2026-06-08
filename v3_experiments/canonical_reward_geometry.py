"""Canonical helpers for the MAPPO shared-reward geometry reruns.

This module deliberately separates the fixed-role Tribal Village protocol from
the later reward-ablation scripts that use binary top/bottom return probes.
The paper's role-probe result should only use labels from ``agent_id % 3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

CANONICAL_NUM_AGENTS = 12
CANONICAL_NUM_ROLES = 3
CANONICAL_SHARED_FRACS = (0.0, 0.8, 1.0)
CANONICAL_TRAINING_SEEDS = tuple(range(5))
CANONICAL_EVAL_TRIALS = 10
ROLE_NAMES = ("gatherer", "explorer", "guardian")
ROLE_PROBE_CV = "agent_generalization_4fold"
DEFAULT_GOLD_LAYER = 11
DEFAULT_ALTAR_LAYER = 16
ROLE_SHAPING_COEFFICIENTS = {
    "gatherer_visible_gold": 0.3,
    "explorer_no_gold_or_altar": 0.5,
    "guardian_visible_altar": 2.0,
}


@dataclass(frozen=True)
class ProbeAuditIssue:
    """A compact issue record for result-stream audits."""

    kind: str
    message: str
    run_name: str | None = None
    seed: int | None = None


def role_labels(num_agents: int = CANONICAL_NUM_AGENTS, num_roles: int = CANONICAL_NUM_ROLES) -> np.ndarray:
    """Return fixed Tribal role labels from interleaved ``agent_id % num_roles`` assignment."""

    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if num_roles <= 1:
        raise ValueError("num_roles must be greater than 1")
    if num_agents % num_roles != 0:
        raise ValueError("canonical role-probe labels require balanced roles")
    return np.arange(num_agents, dtype=np.int64) % num_roles


def role_probe_chance(num_roles: int = CANONICAL_NUM_ROLES) -> float:
    """Return the chance accuracy for the balanced fixed-role probe."""

    if num_roles <= 1:
        raise ValueError("num_roles must be greater than 1")
    return 1.0 / float(num_roles)


def mix_rewards(individual_rewards: Sequence[float] | np.ndarray, shared_frac: float) -> np.ndarray:
    """Mix per-agent rewards with the team mean along the last axis.

    ``shared_frac=0`` returns individual rewards. ``shared_frac=1`` returns the
    same team-mean reward for every agent.
    """

    if not 0.0 <= shared_frac <= 1.0:
        raise ValueError("shared_frac must be in [0, 1]")
    rewards = np.asarray(individual_rewards, dtype=np.float64)
    if rewards.shape == ():
        raise ValueError("individual_rewards must include an agent axis")
    team_mean = rewards.mean(axis=-1, keepdims=True)
    return (1.0 - shared_frac) * rewards + shared_frac * team_mean


def role_shaping_bonuses(
    observations: Sequence[float] | np.ndarray,
    *,
    gold_layer: int = DEFAULT_GOLD_LAYER,
    altar_layer: int = DEFAULT_ALTAR_LAYER,
) -> np.ndarray:
    """Compute fixed-role shaping bonuses from agent observations.

    The canonical pilot described gold and altar visibility. The current
    Tribal Village wrapper exposes those as configurable observation layers so
    reconstructed runs can bind the closest available object layers explicitly.
    """

    obs = np.asarray(observations)
    if obs.ndim != 4:
        raise ValueError("observations must have shape [agents, layers, width, height]")
    num_agents = obs.shape[0]
    labels = role_labels(num_agents)
    bonuses = np.zeros(num_agents, dtype=np.float64)

    gold_visible = _visible_count(obs, gold_layer)
    altar_visible = _visible_count(obs, altar_layer)

    bonuses[labels == 0] = ROLE_SHAPING_COEFFICIENTS["gatherer_visible_gold"] * gold_visible[labels == 0]
    bonuses[labels == 1] = ROLE_SHAPING_COEFFICIENTS["explorer_no_gold_or_altar"] * (
        (gold_visible[labels == 1] + altar_visible[labels == 1]) == 0
    )
    bonuses[labels == 2] = ROLE_SHAPING_COEFFICIENTS["guardian_visible_altar"] * altar_visible[labels == 2]
    return bonuses


def effective_rank(values: Sequence[float] | np.ndarray) -> float:
    """Return entropy effective rank for a 2D embedding matrix."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must have shape [samples, features]")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        return 1.0
    singular_values = np.linalg.svd(arr, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-10]
    if singular_values.size == 0:
        return 1.0
    normalized = singular_values / singular_values.sum()
    entropy = -(normalized * np.log(normalized + 1e-12)).sum()
    return float(np.exp(entropy))


def fixed_role_probe_accuracy(embeddings: Sequence[float] | np.ndarray) -> tuple[float, dict[str, object]]:
    """Evaluate role separability with 4-fold agent-generalization CV.

    Expected shape is ``[steps, agents, embedding_dim]``. Each fold holds out
    one gatherer, one explorer, and one guardian, matching the pilot probe note
    of training on 9 agents and testing on 3 held-out agents.
    """

    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("embeddings must have shape [steps, agents, embedding_dim]")
    if arr.shape[0] < 1:
        raise ValueError("embeddings must contain at least one step")
    num_agents = arr.shape[1]
    labels_by_agent = role_labels(num_agents)
    agents_per_role = num_agents // CANONICAL_NUM_ROLES

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - dependency failure is surfaced in smoke tests.
        raise RuntimeError("fixed_role_probe_accuracy requires scikit-learn") from exc

    fold_scores: list[float] = []
    fold_agents: list[list[int]] = []
    agent_ids = np.arange(num_agents)
    flat_embeddings = arr.reshape(-1, arr.shape[-1])
    flat_labels = np.tile(labels_by_agent, arr.shape[0])
    flat_agent_ids = np.tile(agent_ids, arr.shape[0])

    for fold in range(agents_per_role):
        held_out = np.array([fold * CANONICAL_NUM_ROLES + role for role in range(CANONICAL_NUM_ROLES)])
        train_mask = ~np.isin(flat_agent_ids, held_out)
        test_mask = np.isin(flat_agent_ids, held_out)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=0),
        )
        clf.fit(flat_embeddings[train_mask], flat_labels[train_mask])
        fold_scores.append(float(clf.score(flat_embeddings[test_mask], flat_labels[test_mask])))
        fold_agents.append(held_out.astype(int).tolist())

    return float(np.mean(fold_scores)), {
        "cv": ROLE_PROBE_CV,
        "fold_scores": fold_scores,
        "fold_agents": fold_agents,
        "chance": role_probe_chance(),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def ordered_kl_action_diversity(logits: Sequence[float] | np.ndarray) -> float:
    """Mean ordered off-diagonal KL divergence between agents' policies.

    Expected shape is ``[samples, agents, actions]``. A single sample with shape
    ``[agents, actions]`` is also accepted. The denominator is
    ``samples * agents * (agents - 1)``.
    """

    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("logits must have shape [samples, agents, actions] or [agents, actions]")
    if arr.shape[1] < 2:
        return 0.0
    probs = _softmax(arr)
    log_probs = np.log(np.clip(probs, 1e-12, None))
    kl_matrix = (probs[:, :, None, :] * (log_probs[:, :, None, :] - log_probs[:, None, :, :])).sum(axis=-1)
    off_diag = ~np.eye(arr.shape[1], dtype=bool)
    return float(kl_matrix[:, off_diag].mean())


def js_action_diversity(logits: Sequence[float] | np.ndarray) -> float:
    """Mean unordered Jensen-Shannon divergence between agents' policies."""

    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("logits must have shape [samples, agents, actions] or [agents, actions]")
    n_agents = arr.shape[1]
    if n_agents < 2:
        return 0.0
    probs = _softmax(arr)
    values: list[float] = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            p = probs[:, i, :]
            q = probs[:, j, :]
            m = 0.5 * (p + q)
            p_log = np.log(np.clip(p, 1e-12, None))
            q_log = np.log(np.clip(q, 1e-12, None))
            m_log = np.log(np.clip(m, 1e-12, None))
            js = 0.5 * ((p * (p_log - m_log)).sum(axis=-1) + (q * (q_log - m_log)).sum(axis=-1))
            values.extend(js.tolist())
    return float(np.mean(values))


def audit_fixed_role_probe_records(
    records: Iterable[Mapping[str, object]],
    *,
    expected_chance: float = role_probe_chance(),
    tolerance: float = 1e-6,
) -> list[ProbeAuditIssue]:
    """Audit result records before they are used as fixed-role probe evidence."""

    issues: list[ProbeAuditIssue] = []
    for record in records:
        run_name = _str_or_none(record.get("run_name"))
        seed = _int_or_none(record.get("seed"))
        chance = record.get("probe_chance")
        if chance is None:
            issues.append(ProbeAuditIssue("missing_probe_chance", "record has no probe_chance", run_name, seed))
            continue
        chance_float = float(chance)
        if abs(chance_float - expected_chance) > tolerance:
            issues.append(
                ProbeAuditIssue(
                    "non_fixed_role_probe_chance",
                    f"probe_chance={chance_float:.4f}; fixed-role probe chance should be {expected_chance:.4f}",
                    run_name,
                    seed,
                )
            )
        lift = record.get("probe_lift")
        if lift is not None and float(lift) < -tolerance:
            issues.append(
                ProbeAuditIssue(
                    "negative_probe_lift",
                    f"probe_lift={float(lift):.4f}; result is below its own stored baseline",
                    run_name,
                    seed,
                )
            )
    return issues


def summarize_probe_audit(issues: Sequence[ProbeAuditIssue]) -> dict[str, int]:
    """Count audit issues by kind for machine-readable reports."""

    summary: dict[str, int] = {}
    for issue in issues:
        summary[issue.kind] = summary.get(issue.kind, 0) + 1
    return summary


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _visible_count(observations: np.ndarray, layer: int) -> np.ndarray:
    if layer < 0 or layer >= observations.shape[1]:
        return np.zeros(observations.shape[0], dtype=np.float64)
    return (observations[:, layer, :, :] > 0).sum(axis=(1, 2)).astype(np.float64)
