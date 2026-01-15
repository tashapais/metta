"""Regret-based learning progress curriculum algorithm."""

from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithmConfig, CurriculumTask
from .regret_algorithm_base import RegretAlgorithmBase


class RegretLearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for regret learning progress."""

    type: str = "regret_learning_progress"

    optimal_value: float = 1.0
    regret_ema_timescale: float = 0.001
    use_bidirectional: bool = True
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.05
    invert_regret_progress: bool = True
    min_samples_for_lp: int = 2
    memory: int = 25
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False

    def algorithm_type(self) -> str:
        return "regret_learning_progress"

    def create(self, num_tasks: int) -> "RegretLearningProgressAlgorithm":
        return RegretLearningProgressAlgorithm(num_tasks, self)


class RegretLearningProgressAlgorithm(RegretAlgorithmBase):
    """Prioritize tasks where regret decreases fastest."""

    def __init__(self, num_tasks: int, hypers: RegretLearningProgressConfig):
        super().__init__(num_tasks, hypers, use_task_tracker=True)
        self.hypers: RegretLearningProgressConfig = hypers

        if hypers.use_bidirectional:
            self._init_bidirectional_regret_tracking()

    def _init_bidirectional_regret_tracking(self):
        """Initialize bidirectional regret tracking."""
        self._regret_outcomes: Dict[int, List[float]] = {}
        self._r_fast: Optional[np.ndarray] = None
        self._r_slow: Optional[np.ndarray] = None
        self._task_dist: Optional[np.ndarray] = None
        self._stale_dist = True
        self._update_mask: np.ndarray = np.array([])
        self._task_ids: List[int] = []

    def _score_task(self, task_id: int) -> float:
        """Calculate regret learning progress score for a task."""
        if self.hypers.use_bidirectional and self._stale_dist:
            self._invalidate_score_cache()

        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.task_tracker.get_task_stats(task_id)

        if not task_stats or task_stats["completion_count"] < self.hypers.min_samples_for_lp:
            score = self.hypers.exploration_bonus
        elif self.hypers.use_bidirectional:
            score = self._compute_bidirectional_score(task_id)
        else:
            regret_progress = self.regret_tracker.get_regret_progress(task_id)
            if regret_progress is None:
                score = self.hypers.exploration_bonus
            else:
                score = max(0.0, -regret_progress) if self.hypers.invert_regret_progress else abs(regret_progress)
                score += self._exploration_boost(task_stats["completion_count"])

        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _compute_bidirectional_score(self, task_id: int) -> float:
        """Compute bidirectional learning progress score."""
        if task_id not in self._regret_outcomes or len(self._regret_outcomes[task_id]) < 2:
            return self.hypers.exploration_bonus

        if self._task_dist is None or self._stale_dist:
            self._calculate_regret_task_distribution()

        task_indices = self._task_ids or sorted(self._regret_outcomes.keys())
        if task_id in task_indices and self._task_dist is not None:
            task_idx = task_indices.index(task_id)
            if task_idx < len(self._task_dist):
                return float(self._task_dist[task_idx])

        return self.hypers.exploration_bonus

    def _update_regret_progress(self, update_ema: bool = True):
        """Update bidirectional regret progress tracking."""
        if not self._regret_outcomes:
            return

        task_ids = sorted(self._regret_outcomes.keys())
        if not task_ids:
            return
        num_tasks = len(task_ids)

        mean_regrets = np.array(
            [np.mean(self._regret_outcomes[task_id]) if self._regret_outcomes[task_id] else 0.0 for task_id in task_ids]
        )

        self._update_mask = np.array([len(self._regret_outcomes[task_id]) >= 2 for task_id in task_ids])
        old_task_ids = self._task_ids
        old_index = {task_id: idx for idx, task_id in enumerate(old_task_ids)}
        self._task_ids = task_ids

        new_r_fast = np.zeros(num_tasks)
        new_r_slow = np.zeros(num_tasks)

        if self._r_fast is None or self._r_slow is None:
            new_r_fast = mean_regrets.copy()
            new_r_slow = mean_regrets.copy()
        else:
            for idx, task_id in enumerate(task_ids):
                old_idx = old_index.get(task_id)
                if old_idx is not None:
                    new_r_fast[idx] = self._r_fast[old_idx]
                    new_r_slow[idx] = self._r_slow[old_idx]
                else:
                    new_r_fast[idx] = mean_regrets[idx]
                    new_r_slow[idx] = mean_regrets[idx]

        self._r_fast = new_r_fast
        self._r_slow = new_r_slow

        if update_ema and np.any(self._update_mask):
            self._r_fast[self._update_mask] = mean_regrets[
                self._update_mask
            ] * self.hypers.regret_ema_timescale + self._r_fast[self._update_mask] * (
                1.0 - self.hypers.regret_ema_timescale
            )
            slow_timescale = self.hypers.regret_ema_timescale * 0.2
            self._r_slow[self._update_mask] = mean_regrets[self._update_mask] * slow_timescale + self._r_slow[
                self._update_mask
            ] * (1.0 - slow_timescale)

        self._stale_dist = True

    def _regret_learning_progress(self) -> np.ndarray:
        """Calculate regret learning progress as slow - fast."""
        if self._r_fast is None or self._r_slow is None:
            return np.array([])

        regret_lp = self._r_slow - self._r_fast

        lp = np.abs(regret_lp)

        if self.hypers.invert_regret_progress:
            improvement_bonus = np.maximum(regret_lp, 0) * 0.2
            lp = lp + improvement_bonus

        return lp

    def _calculate_regret_task_distribution(self):
        """Calculate task distribution based on regret learning progress."""
        if not self._regret_outcomes:
            self._task_dist = np.array([])
            self._stale_dist = False
            return

        num_tasks = len(self._regret_outcomes)
        task_dist = np.ones(num_tasks) / num_tasks

        regret_lp = self._regret_learning_progress()
        if regret_lp.size == 0:
            self._task_dist = task_dist
            self._stale_dist = False
            return

        mask = regret_lp > 0
        subprobs = regret_lp[mask] if np.any(mask) else regret_lp

        subprobs = subprobs - np.mean(subprobs)
        std = np.std(subprobs)
        if std > 0:
            subprobs = subprobs / std
        subprobs = 1 / (1 + np.exp(-np.clip(subprobs, -500, 500)))

        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            subprobs = np.ones_like(subprobs) / len(subprobs)

        if np.any(mask):
            task_dist = np.zeros(len(regret_lp))
            task_dist[mask] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False

    def _get_task_stats(self, task_id: int):
        return self.task_tracker.get_task_stats(task_id)

    def _get_tracked_task_ids(self):
        return self.task_tracker.get_all_tracked_tasks()

    def _eviction_fraction(self) -> float:
        return 0.4

    def _after_task_evicted(self, task_id: int) -> None:
        if self.hypers.use_bidirectional:
            self._regret_outcomes.pop(task_id, None)
            self._update_regret_progress(update_ema=False)
            self._stale_dist = True
            self._invalidate_score_cache()

    def _after_update_task_performance(self, task_id: int, score: float) -> None:
        if not self.hypers.use_bidirectional:
            return

        regret = self.regret_tracker.compute_regret(score)
        self._regret_outcomes.setdefault(task_id, []).append(regret)
        self._regret_outcomes[task_id] = self._regret_outcomes[task_id][-self.hypers.memory :]
        self._update_regret_progress(update_ema=True)
        self._stale_dist = True
        self._invalidate_score_cache()

    def _after_task_created(self, task: CurriculumTask) -> None:
        if self.hypers.use_bidirectional:
            self._stale_dist = True
            self._invalidate_score_cache()

    def _detailed_stats_prefix(self) -> str:
        return "regret_lp"

    def _get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed regret learning progress statistics."""
        if not self._regret_outcomes:
            return {
                "num_tracked_tasks": 0.0,
                "mean_regret_lp": 0.0,
                "num_decreasing_regret": 0.0,
                "num_increasing_regret": 0.0,
            }

        stats = {
            "num_tracked_tasks": float(len(self._regret_outcomes)),
        }

        if self._r_fast is not None and self._r_slow is not None:
            regret_lp = self._regret_learning_progress()
            if len(regret_lp) > 0:
                stats["mean_regret_lp"] = float(np.mean(regret_lp))

                # Count tasks with decreasing vs increasing regret
                regret_changes = self._r_slow - self._r_fast
                stats["num_decreasing_regret"] = float(np.sum(regret_changes > 0))
                stats["num_increasing_regret"] = float(np.sum(regret_changes < 0))
            else:
                stats["mean_regret_lp"] = 0.0
                stats["num_decreasing_regret"] = 0.0
                stats["num_increasing_regret"] = 0.0
        else:
            stats["mean_regret_lp"] = 0.0
            stats["num_decreasing_regret"] = 0.0
            stats["num_increasing_regret"] = 0.0

        return stats

    def _get_extra_state(self) -> Dict[str, Any]:
        if not self.hypers.use_bidirectional:
            return {}
        return {
            "regret_outcomes": {k: v for k, v in self._regret_outcomes.items()},
            "r_fast": self._r_fast.tolist() if self._r_fast is not None else None,
            "r_slow": self._r_slow.tolist() if self._r_slow is not None else None,
            "task_dist": self._task_dist.tolist() if self._task_dist is not None else None,
            "stale_dist": self._stale_dist,
            "update_mask": self._update_mask.tolist(),
            "task_ids": list(self._task_ids),
        }

    def _load_extra_state(self, state: Dict[str, Any]) -> None:
        if "regret_outcomes" not in state:
            return
        self._regret_outcomes = state["regret_outcomes"]
        self._r_fast = np.array(state["r_fast"]) if state["r_fast"] is not None else None
        self._r_slow = np.array(state["r_slow"]) if state["r_slow"] is not None else None
        self._task_dist = np.array(state["task_dist"]) if state["task_dist"] is not None else None
        self._stale_dist = state["stale_dist"]
        self._update_mask = np.array(state["update_mask"])
        self._task_ids = state.get("task_ids", sorted(self._regret_outcomes.keys()))
